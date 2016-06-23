#include <iostream>
#include <cstdio>
#include <iterator>

#include <opengm/unittests/test.hxx>

#ifdef WITH_CPLEX
#include <opengm/inference/auxiliary/lp_solver/lp_solver_cplex.hxx>
#endif

#ifdef WITH_GUROBI
#include <opengm/inference/auxiliary/lp_solver/lp_solver_gurobi.hxx>
#endif

template <class SOLVER_TYPE, class SOLVER_VALUE_TYPE, class PARAMETER1_TYPE, class PARAMETER1_VALUE_TYPE, class PARAMETER2_TYPE, class PARAMETER2_VALUE_TYPE>
void testLPSolver(const SOLVER_VALUE_TYPE expectedInfinityValue, const PARAMETER1_TYPE timingParameter, const PARAMETER1_VALUE_TYPE maxTime, const PARAMETER2_TYPE threadParameter, const PARAMETER2_VALUE_TYPE maxNumThreads);

int main(int argc, char** argv){
  try{
   std::cout << "LP Solver test... " << std::endl;

#ifdef WITH_CPLEX
   std::cout << "Test CPLEX Solver" << std::endl;
   testLPSolver<opengm::LPSolverCplex>(IloInfinity, IloCplex::TiLim, 1.0, IloCplex::Threads, 1);
#endif

#ifdef WITH_GUROBI
   std::cout << "Test Gurobi Solver" << std::endl;
   testLPSolver<opengm::LPSolverGurobi>(GRB_INFINITY, GRB_DoubleParam_TimeLimit, 1.0, GRB_IntParam_Threads, 1);
#endif

   std::cout << "done..." << std::endl;
   return 0;
  }
  catch(std::exception & e)
  {
	  std::cerr << "Unexpected exception: " << e.what() << std::endl;
	  return 1;
  }
}

template <class SOLVER_TYPE, class SOLVER_VALUE_TYPE, class PARAMETER1_TYPE, class PARAMETER1_VALUE_TYPE, class PARAMETER2_TYPE, class PARAMETER2_VALUE_TYPE>
void testLPSolver(const SOLVER_VALUE_TYPE expectedInfinityValue, const PARAMETER1_TYPE timingParameter, const PARAMETER1_VALUE_TYPE maxTime, const PARAMETER2_TYPE threadParameter, const PARAMETER2_VALUE_TYPE maxNumThreads) {
   // create solver
   std::cout << "  * create solver" << std::endl;
   SOLVER_TYPE lpSolverMinimize;
   SOLVER_TYPE lpSolverMaximize;
   SOLVER_TYPE ilpSolverMinimize;
   SOLVER_TYPE ilpSolverMaximize;

   // test infinity
   std::cout << "  * test infinity" << std::endl;
   OPENGM_TEST_EQUAL(SOLVER_TYPE::infinity(), expectedInfinityValue);

   // add variables
   std::cout << "  * add variables" << std::endl;
   lpSolverMinimize.addContinuousVariables(3, 0.0, 1.0);
   lpSolverMaximize.addContinuousVariables(3, 0.0, 1.0);
   ilpSolverMinimize.addIntegerVariables(2, 0.0, 1.0);
   ilpSolverMinimize.addBinaryVariables(1);
   ilpSolverMaximize.addIntegerVariables(2, 0.0, 1.0);
   ilpSolverMaximize.addBinaryVariables(1);

   // set objective
   std::cout << "  * set objective" << std::endl;
   lpSolverMinimize.setObjective(SOLVER_TYPE::Minimize);
   lpSolverMaximize.setObjective(SOLVER_TYPE::Maximize);
   ilpSolverMinimize.setObjective(SOLVER_TYPE::Minimize);
   ilpSolverMaximize.setObjective(SOLVER_TYPE::Maximize);

   // set objective function
   std::cout << "  * set objective function" << std::endl;
   typename SOLVER_TYPE::SolverValueType objectiveFunctionValues[] = {1.0, 1.0, 1.0};
   typename SOLVER_TYPE::SolverIndexType objectiveFunctionIndices[] = {0, 1, 2};
   lpSolverMinimize.setObjectiveValue(objectiveFunctionValues, objectiveFunctionValues + 3);
   lpSolverMaximize.setObjectiveValue(0, 1.0);
   lpSolverMaximize.setObjectiveValue(objectiveFunctionIndices + 1, objectiveFunctionIndices + 3, objectiveFunctionValues + 1);
   ilpSolverMinimize.setObjectiveValue(objectiveFunctionValues, objectiveFunctionValues + 3);
   ilpSolverMaximize.setObjectiveValue(0, 1.0);
   ilpSolverMaximize.setObjectiveValue(objectiveFunctionIndices + 1, objectiveFunctionIndices + 3, objectiveFunctionValues + 1);

   // add constraints
   std::cout << "  * add constraints" << std::endl;
   typename SOLVER_TYPE::SolverValueType constraintValues[] = {1.0, 1.0, 1.0};
   typename SOLVER_TYPE::SolverIndexType constraintIndices[] = {0, 1, 2};
   lpSolverMinimize.addEqualityConstraint(constraintIndices, constraintIndices + 3, constraintValues, 1.0, "Equality_Sum_Constraint");
   lpSolverMaximize.addLessEqualConstraint(constraintIndices, constraintIndices + 3, constraintValues, 1.0, "Less_Equal_Sum_Constraint");
   lpSolverMaximize.addGreaterEqualConstraint(constraintIndices, constraintIndices + 3, constraintValues, 1.0, "Greater_Equal_Sum_Constraint");
   ilpSolverMinimize.addEqualityConstraint(constraintIndices, constraintIndices + 3, constraintValues, 1.0, "Equality_Sum_Constraint");
   ilpSolverMaximize.addLessEqualConstraint(constraintIndices, constraintIndices + 3, constraintValues, 1.0, "Less_Equal_Sum_Constraint");
   ilpSolverMaximize.addGreaterEqualConstraint(constraintIndices, constraintIndices + 3, constraintValues, 1.0, "Greater_Equal_Sum_Constraint");

   // finalize
   std::cout << "  * finalize" << std::endl;
   typename SOLVER_TYPE::SolverTimingType timing = -1.0;
   lpSolverMinimize.addConstraintsFinished();
   lpSolverMaximize.addConstraintsFinished(timing);
   OPENGM_TEST(timing >= 0);
   ilpSolverMinimize.addConstraintsFinished();
   timing = -1.0;
   ilpSolverMaximize.addConstraintsFinished(timing);
   OPENGM_TEST(timing >= 0);

   // set parameter
   std::cout << "  * set parameter" << std::endl;
   lpSolverMinimize.setParameter(timingParameter, maxTime);
   lpSolverMaximize.setParameter(timingParameter, maxTime);
   ilpSolverMinimize.setParameter(timingParameter, maxTime);
   ilpSolverMaximize.setParameter(timingParameter, maxTime);

   lpSolverMinimize.setParameter(threadParameter, maxNumThreads);
   lpSolverMaximize.setParameter(threadParameter, maxNumThreads);
   ilpSolverMinimize.setParameter(threadParameter, maxNumThreads);
   ilpSolverMaximize.setParameter(threadParameter, maxNumThreads);

   // solve problem
   std::cout << "  * solve problem" << std::endl;
   lpSolverMinimize.solve();
   timing = -1.0;
   lpSolverMaximize.solve(timing);
   OPENGM_TEST(timing >= 0);
   OPENGM_TEST(timing <= maxTime);
   ilpSolverMinimize.solve();
   timing = -1.0;
   ilpSolverMaximize.solve(timing);
   OPENGM_TEST(timing >= 0);
   OPENGM_TEST(timing <= maxTime);

   // get solution
   std::cout << "  * get solution" << std::endl;
   typename SOLVER_TYPE::SolverSolutionIteratorType solutionBegin = lpSolverMinimize.solutionBegin();
   typename SOLVER_TYPE::SolverSolutionIteratorType solutionEnd = lpSolverMinimize.solutionEnd();
   OPENGM_TEST_EQUAL(std::distance(solutionBegin, solutionEnd), 3);
   typename SOLVER_TYPE::SolverValueType solutionSum = 0.0;
   for(typename SOLVER_TYPE::SolverIndexType i = 0; i < 3; ++i) {
      solutionSum += *solutionBegin;
      OPENGM_TEST_EQUAL(lpSolverMinimize.solution(i), *solutionBegin);
      ++solutionBegin;
   }
   OPENGM_TEST(solutionBegin == solutionEnd);
   OPENGM_TEST_EQUAL(solutionSum, 1.0);

   solutionBegin = lpSolverMaximize.solutionBegin();
   solutionEnd = lpSolverMaximize.solutionEnd();
   OPENGM_TEST_EQUAL(std::distance(solutionBegin, solutionEnd), 3);
   solutionSum = 0.0;
   for(typename SOLVER_TYPE::SolverIndexType i = 0; i < 3; ++i) {
      solutionSum += *solutionBegin;
      OPENGM_TEST_EQUAL(lpSolverMaximize.solution(i), *solutionBegin);
      ++solutionBegin;
   }
   OPENGM_TEST(solutionBegin == solutionEnd);
   OPENGM_TEST_EQUAL(solutionSum, 1.0);

   solutionBegin = ilpSolverMinimize.solutionBegin();
   solutionEnd = ilpSolverMinimize.solutionEnd();
   OPENGM_TEST_EQUAL(std::distance(solutionBegin, solutionEnd), 3);
   solutionSum = 0.0;
   for(typename SOLVER_TYPE::SolverIndexType i = 0; i < 3; ++i) {
      solutionSum += *solutionBegin;
      OPENGM_TEST_EQUAL(ilpSolverMinimize.solution(i), *solutionBegin);
      ++solutionBegin;
   }
   OPENGM_TEST(solutionBegin == solutionEnd);
   OPENGM_TEST_EQUAL(solutionSum, 1.0);

   solutionBegin = ilpSolverMaximize.solutionBegin();
   solutionEnd = ilpSolverMaximize.solutionEnd();
   OPENGM_TEST_EQUAL(std::distance(solutionBegin, solutionEnd), 3);
   solutionSum = 0.0;
   for(typename SOLVER_TYPE::SolverIndexType i = 0; i < 3; ++i) {
      solutionSum += *solutionBegin;
      OPENGM_TEST_EQUAL(ilpSolverMaximize.solution(i), *solutionBegin);
      ++solutionBegin;
   }
   OPENGM_TEST(solutionBegin == solutionEnd);
   OPENGM_TEST_EQUAL(solutionSum, 1.0);


   OPENGM_TEST_EQUAL(lpSolverMinimize.objectiveFunctionValue(), 1.0);
   OPENGM_TEST_EQUAL(lpSolverMaximize.objectiveFunctionValue(), 1.0);
   OPENGM_TEST_EQUAL(ilpSolverMinimize.objectiveFunctionValue(), 1.0);
   OPENGM_TEST_EQUAL(ilpSolverMaximize.objectiveFunctionValue(), 1.0);

   // check bound
   std::cout << "  * check bound" << std::endl;
   OPENGM_TEST(lpSolverMinimize.objectiveFunctionValueBound() >= 1.0);
   OPENGM_TEST(lpSolverMaximize.objectiveFunctionValueBound() <= 1.0);
   OPENGM_TEST(ilpSolverMinimize.objectiveFunctionValueBound() >= 1.0);
   OPENGM_TEST(ilpSolverMaximize.objectiveFunctionValueBound() <= 1.0);

   // test model export (check only for creation of file)
   std::cout << "  * test model export (check only for creation of file)" << std::endl;
   lpSolverMinimize.exportModel("lpSolverMinimizeModel.lp");
   OPENGM_TEST(!std::remove("lpSolverMinimizeModel.lp"));
   lpSolverMaximize.exportModel("lpSolverMaximizeModel.lp");
   OPENGM_TEST(!std::remove("lpSolverMaximizeModel.lp"));
   ilpSolverMinimize.exportModel("ilpSolverMinimizeModel.lp");
   OPENGM_TEST(!std::remove("ilpSolverMinimizeModel.lp"));
   ilpSolverMaximize.exportModel("ilpSolverMaximizeModel.lp");
   OPENGM_TEST(!std::remove("ilpSolverMaximizeModel.lp"));
}
