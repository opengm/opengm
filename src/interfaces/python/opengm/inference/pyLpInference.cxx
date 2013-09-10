
#include <boost/python.hpp>
#include <stdexcept>
#include <stddef.h>
#include <string>
#include "nifty_iterator.hxx"
#include "inf_def_visitor.hxx"
#include "lp_def_suite.hxx"
#include "export_typedes.hxx"


#include <opengm/inference/lp_inference.hxx>
#include <param/lp_inference_param.hxx>


#ifdef WITH_CPLEX
#include <opengm/inference/auxiliary/lp_solver/lp_solver_cplex.hxx>
#endif 

#ifdef WITH_GUROBI
#include <opengm/inference/auxiliary/lp_solver/lp_solver_gurobi.hxx>
#endif 

// export function
template<class GM, class ACC>
void export_lp_inference() {

   import_array();
   using namespace boost::python;
   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;

   #ifdef WITH_CPLEX
   {
      append_subnamespace("solver");

      // setup 
      InfSetup setup;
      setup.algType    = "linear-programming";
      setup.guarantees = "global optimal";
      setup.examples   = ">>> parameter = opengm.InfParam(integerConstraint=True)\n\n"
                         ">>> inference = opengm.inference.LpCplex2(gm=gm,accumulator='minimizer',parameter=parameter)\n\n"
                         "\n\n";  
      setup.dependencies = "This algorithm needs the IBM CPLEX Optimizer, compile OpenGM with CMake-Flag ``WITH_CPLEX`` set to ``ON`` ";
      // export parameter
      typedef typename GM::ValueType ValueType;
      typedef typename GM::IndexType IndexType;

      typedef opengm::LpSolverCplex LpSolver;
      typedef opengm::LPInference<GM, opengm::Minimizer,LpSolver>    PyLPSolver;




      const std::string enumName1=std::string("_LpCplex2Relaxation")+srName;
      enum_<typename PyLPSolver::Relaxation> (enumName1.c_str())
         .value("firstOrder",  PyLPSolver::FirstOrder)
         .value("firstOrder2",   PyLPSolver::FirstOrder2)
      ;


      exportInfParam<PyLPSolver>("_LpCplex2");
      // export inference
      class_< PyLPSolver>("_LpCplex2",init<const GM & >())  
      .def(InfSuite<PyLPSolver,false,true,false>(std::string("LpCplex2"),setup))
      .def(LpInferenceSuite<PyLPSolver>())
      ;
      
   }
   #endif

   #ifdef WITH_GUROBI
   {
      append_subnamespace("solver");

      // setup 
      InfSetup setup;
      setup.algType    = "linear-programming";
      setup.guarantees = "global optimal";
      setup.examples   = ">>> parameter = opengm.InfParam(integerConstraint=True)\n\n"
                         ">>> inference = opengm.inference.LpGurobi(gm=gm,accumulator='minimizer',parameter=parameter)\n\n"
                         "\n\n";  
      setup.dependencies = "This algorithm needs the Gurobi Optimizer, compile OpenGM with CMake-Flag ``WITH_GUROBI`` set to ``ON`` ";
      // export parameter
      typedef typename GM::ValueType ValueType;
      typedef typename GM::IndexType IndexType;

      typedef opengm::LpSolverGurobi LpSolver;
      typedef opengm::LPInference<GM, opengm::Minimizer,LpSolver>    PyLPSolver;


      const std::string enumName1=std::string("_LpGurobiRelaxation")+srName;
      enum_<typename PyLPSolver::Relaxation> (enumName1.c_str())
         .value("firstOrder",  PyLPSolver::FirstOrder)
         .value("firstOrder2",   PyLPSolver::FirstOrder2)
      ;


      exportInfParam<PyLPSolver>("_LpGurobi");
      // export inference
      class_< PyLPSolver>("_LpGurobi",init<const GM & >())  
      .def(InfSuite<PyLPSolver,false,true,false>(std::string("LpGurobi"),setup))
      .def(LpInferenceSuite<PyLPSolver>())
      ;
      
   }
   #endif

}
// explicit template instantiation for the supported semi-rings
template void export_lp_inference<opengm::python::GmAdder, opengm::Minimizer>();
