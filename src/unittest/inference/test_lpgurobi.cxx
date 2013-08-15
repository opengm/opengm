#ifdef WITH_GUROBI
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>


#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>


#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/bruteforce.hxx>



#include <opengm/inference/auxiliary/lp_solver/lp_solver_gurobi.hxx>
#include <opengm/inference/lp_gurobi.hxx>

#endif


#include <iostream>

int main(){
#ifdef WITH_GUROBI
   {
      typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

      typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
      typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
      typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
      typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;


      opengm::InferenceBlackBoxTester<SumGmType> sumTester;
      sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 5));
      sumTester.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::PASS, 5));
      sumTester.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::PASS, 20));
      sumTester.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::PASS, 5));
 
      opengm::InferenceBlackBoxTester<SumGmType> sumTesterOpt;
      sumTesterOpt.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::OPTIMAL, 5));
      sumTesterOpt.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::OPTIMAL, 5));
      sumTesterOpt.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
      sumTesterOpt.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::OPTIMAL, 5));


      opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
      prodTester.addTest(new ProdGridTest(4, 4, 2, true, true, ProdGridTest::RANDOM, opengm::PASS, 5));

 
      opengm::InferenceBlackBoxTester<ProdGmType> prodTesterOpt;
      prodTesterOpt.addTest(new ProdGridTest(4, 4, 2, true, true, ProdGridTest::RANDOM, opengm::OPTIMAL, 5));



      std::cout << "Gurobi Tests"<<std::endl;
      {
         std::cout << "  * Minimization/Adder LP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LpSolverGurobi LpSolver;
         typedef opengm::LPGurobi<GmType, opengm::Minimizer,LpSolver>    Gurobi;
         Gurobi::Parameter para;
         para.relaxation_        = Gurobi::FirstOrder;
         para.integerConstraint_ = false;
         sumTester.test<Gurobi>(para);
         para.relaxation_        = Gurobi::FirstOrder2;
         para.integerConstraint_ = false;
         sumTester.test<Gurobi>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Minimization/Adder ILP ... RELAXATION 1"<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LpSolverGurobi LpSolver;
         typedef opengm::LPGurobi<GmType, opengm::Minimizer,LpSolver>    Gurobi;
         Gurobi::Parameter para;
         para.relaxation_        = Gurobi::FirstOrder;
         para.integerConstraint_ = true;
         para.integerConstraintFactorVar_ = true;
         sumTesterOpt.test<Gurobi>(para);
         para.relaxation_        = Gurobi::FirstOrder2;
         para.integerConstraint_ = true;
         para.integerConstraintFactorVar_ = true;
         std::cout << "  * Minimization/Adder ILP ... RELAXATION 2"<<std::endl;
         sumTesterOpt.test<Gurobi>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximization/Adder LP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LpSolverGurobi LpSolver;
         typedef opengm::LPGurobi<GmType, opengm::Maximizer,LpSolver>    Gurobi;
         Gurobi::Parameter para;
         para.integerConstraint_ = false;
         sumTester.test<Gurobi>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximization/Adder ILP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LpSolverGurobi LpSolver;
         typedef opengm::LPGurobi<GmType, opengm::Maximizer,LpSolver>    Gurobi;
         Gurobi::Parameter para;
         para.integerConstraint_ = true;
         para.integerConstraintFactorVar_ = true;
         sumTesterOpt.test<Gurobi>(para);
         std::cout << " OK!"<<std::endl;
      }     








      {
         std::cout << "  * Minimization/Multiplier LP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier > GmType;
         typedef opengm::LpSolverGurobi LpSolver;
         typedef opengm::LPGurobi<GmType, opengm::Minimizer,LpSolver>    Gurobi;
         Gurobi::Parameter para;
         para.integerConstraint_ = false;
         prodTester.test<Gurobi>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Minimization/Multiplier ILP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier > GmType;
         typedef opengm::LpSolverGurobi LpSolver;
         typedef opengm::LPGurobi<GmType, opengm::Minimizer,LpSolver>    Gurobi;
         Gurobi::Parameter para;
         para.integerConstraint_ = true;
         para.integerConstraintFactorVar_ = true;
         prodTester.test<Gurobi>(para);
         std::cout << " OK!"<<std::endl;
      }

      {
         std::cout << "  * Maximization/Multiplier LP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier > GmType;
         typedef opengm::LpSolverGurobi LpSolver;
         typedef opengm::LPGurobi<GmType, opengm::Maximizer,LpSolver>    Gurobi;
         Gurobi::Parameter para;
         para.integerConstraint_ = false;
         prodTester.test<Gurobi>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximization/Multiplier ILP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier > GmType;
         typedef opengm::LpSolverGurobi LpSolver;
         typedef opengm::LPGurobi<GmType, opengm::Maximizer,LpSolver>    Gurobi;
         Gurobi::Parameter para;
         para.integerConstraint_ = true;
         para.integerConstraintFactorVar_ = true;
         prodTester.test<Gurobi>(para);
         std::cout << " OK!"<<std::endl;
      }     
      std::cout << "done!"<<std::endl;
   }
#else
   std::cout << "LpGurobiTest test is disabled (compiled without LpCplex) "<< std::endl;
#endif
   return 0;
}

