//#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleInference
#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <boost/python.hpp>
#include <stddef.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/inference/inference.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/integrator.hxx>

#include "pyInference.hxx"
#include "pyIcm.hxx"
#include "pyGraphcut.hxx"
#include "pyTrbp.hxx"
#include "pyBp.hxx"
#include "pyLoc.hxx"
#include "pyAstar.hxx"

#include "pyGibbs.hxx"
#include "pyBruteforce.hxx"
#include "pyLazyflipper.hxx"
#include "pyDynp.hxx"

#include  "pyAe.hxx"
#include  "pyAbSwap.hxx"
#include  "pyPartitionMove.hxx"


#include "pyDualDecompositionSubGradient.hxx"
//#include "pyDualDecompositionMerit.hxx"
#ifdef WITH_CONICBUNDLE
   //#include "pyDualDecompositionBundle.hxx"
#endif
#ifdef WITH_LIBDAI
#include  "pyLibdai.hxx" 
#endif

#ifdef WITH_CPLEX
#include "pyCplex.hxx"
#include "pyMultiCut.hxx"
#endif

#ifdef WITH_TRWS
#include "pyTrws.hxx"
#endif

#ifdef WITH_MRF
#include "pyMrf.hxx"
#endif

#ifdef WITH_FASTPD
#include "pyFastPD.hxx"
#endif

#include "pyQpbo.hxx"

#ifdef WITH_QPBO
#include "pyQpbo.hxx"
#include "pyMQpbo.hxx"
#include "pyReducedInference.hxx"
#include  "pyAeFusion.hxx"
#endif
//#include "pySwendsenWang.hxx"

#include "pyLpInference.hxx"

#include "converter.hxx"
#include "export_typedes.hxx"


using namespace boost::python;


BOOST_PYTHON_MODULE_INIT(_inference) {

   Py_Initialize();
   PyEval_InitThreads();
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   
   std::string adderString="adder";
   std::string multiplierString="multiplier";
   std::string minimizerString="minimizer";
   std::string maximizerString="maximizer";
   std::string integratorString="integrator";
   std::string substring,submoduleName,subsubmoduleName,subsubstring;
   docstring_options doc_options(true,true,false);
   scope current;
   std::string currentScopeName(extract<const char*>(current.attr("__name__")));
   currentScopeName="inference";
   //import_array();
   export_inference();
   //adder
   {
      substring=adderString;
      submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      submodule.attr("__package__")=submoduleName.c_str();
      // Switch the scope to the submodule, add methods and classes.
      scope submoduleScope = submodule;
      // minimizer
      {
         subsubstring=minimizerString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         //subsubmodule.attr("__package__")=subsubmoduleName.c_str();
         // Switch the scope to the submodule, add methods and classes.
         scope subsubmoduleScope = subsubmodule;

         export_icm<GmAdder,opengm::Minimizer>();
         export_bp<GmAdder,opengm::Minimizer>();
         export_trbp<GmAdder,opengm::Minimizer>();
         export_astar<GmAdder,opengm::Minimizer>();
         export_gibbs<GmAdder,opengm::Minimizer>();
         
         export_dual_decomposition_subgradient<GmAdder,opengm::Minimizer>();
         #ifdef WITH_CONICBUNDLE
            //export_dual_decomposition_bundle<GmAdder,opengm::Minimizer>();
         #endif
         export_lazyflipper<GmAdder,opengm::Minimizer>();
         export_loc<GmAdder,opengm::Minimizer>();
         export_bruteforce<GmAdder,opengm::Minimizer>();
         export_graphcut<GmAdder,opengm::Minimizer>();
         export_abswap<GmAdder,opengm::Minimizer>();
         export_ae<GmAdder,opengm::Minimizer>();
         export_dynp<GmAdder,opengm::Minimizer>();
         export_partition_move<GmAdder,opengm::Minimizer>();
         
         //export_qpbo<GmAdder,opengm::Minimizer>();
         #ifdef WITH_QPBO
         export_reduced_inference<GmAdder,opengm::Minimizer>();
         export_ae_fusion<GmAdder,opengm::Minimizer>();
         export_mqpbo<GmAdder,opengm::Minimizer>();
         export_qpbo_external<GmAdder,opengm::Minimizer>();
         #endif
         #ifdef WITH_TRWS
         export_trws<GmAdder,opengm::Minimizer>();
         #endif
         
         #ifdef WITH_CPLEX
         export_cplex<GmAdder,opengm::Minimizer>();
         export_multicut<GmAdder,opengm::Minimizer>();
         #endif

         export_lp_inference<GmAdder,opengm::Minimizer>();

         #ifdef WITH_LIBDAI
         export_libdai_inference<GmAdder,opengm::Minimizer>();
         #endif

         #ifdef WITH_MRF
         export_mrf<GmAdder,opengm::Minimizer>();
         #endif

         #ifdef WITH_FASTPD
         export_fast_pd<GmAdder,opengm::Minimizer>();
         #endif
      }
      // maximizer
      {
         subsubstring=maximizerString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         //subsubmodule.attr("__package__")=subsubmoduleName.c_str();
         // Switch the scope to the submodule, add methods and classes.
         scope subsubmoduleScope = subsubmodule;
         export_icm<GmAdder,opengm::Maximizer>();
         export_bp<GmAdder,opengm::Maximizer>();
         export_trbp<GmAdder,opengm::Maximizer>();
         export_astar<GmAdder,opengm::Maximizer>();
         export_lazyflipper<GmAdder,opengm::Maximizer>();
         export_loc<GmAdder,opengm::Maximizer>();
         export_bruteforce<GmAdder,opengm::Maximizer>();
         export_dynp<GmAdder,opengm::Maximizer>();
      }
      // integrator
      {
         subsubstring=integratorString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         //subsubmodule.attr("__package__")=subsubmoduleName.c_str();
         scope subsubmoduleScope = subsubmodule;

         export_bp<GmAdder,opengm::Integrator>();
         export_trbp<GmAdder,opengm::Integrator>();
         //export_dynp<GmMultiplier,opengm::Maximizer>();
      }
   }
   //multiplier
   {
      substring=multiplierString;
      submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      submodule.attr("__package__")=submoduleName.c_str();
      // Switch the scope to the submodule, add methods and classes.
      scope submoduleScope = submodule;
      // minimizer
      {
         subsubstring=minimizerString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         //subsubmodule.attr("__package__")=subsubmoduleName.c_str();
         // Switch the scope to the submodule, add methods and classes.
         scope subsubmoduleScope = subsubmodule;
         export_icm<GmMultiplier,opengm::Minimizer>();
         export_bp<GmMultiplier,opengm::Minimizer>();
         export_trbp<GmMultiplier,opengm::Minimizer>();
         export_astar<GmMultiplier,opengm::Minimizer>();
         export_lazyflipper<GmMultiplier,opengm::Minimizer>();
         export_loc<GmMultiplier,opengm::Minimizer>();
         export_bruteforce<GmMultiplier,opengm::Minimizer>();
         export_dynp<GmMultiplier,opengm::Minimizer>();
      }
      // maximizer
      {
         subsubstring=maximizerString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         //subsubmodule.attr("__package__")=subsubmoduleName.c_str();
         // Switch the scope to the submodule, add methods and classes.
         scope subsubmoduleScope = subsubmodule;
         export_icm<GmMultiplier,opengm::Maximizer>();
         export_bp<GmMultiplier,opengm::Maximizer>();
         export_trbp<GmMultiplier,opengm::Maximizer>();
         export_astar<GmMultiplier,opengm::Maximizer>();
         export_gibbs<GmMultiplier,opengm::Maximizer>();
         export_lazyflipper<GmMultiplier,opengm::Maximizer>();
         export_loc<GmMultiplier,opengm::Maximizer>();
         export_bruteforce<GmMultiplier,opengm::Maximizer>();
         export_dynp<GmMultiplier,opengm::Maximizer>();
      }
      // integrator
      {
         subsubstring=integratorString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         //subsubmodule.attr("__package__")=subsubmoduleName.c_str();
         scope subsubmoduleScope = subsubmodule;

         export_bp<GmMultiplier,opengm::Integrator>();
         export_trbp<GmMultiplier,opengm::Integrator>();
         //export_dynp<GmMultiplier,opengm::Maximizer>();
      }
   }
}
