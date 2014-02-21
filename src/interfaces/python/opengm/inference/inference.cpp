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
#include "pyAstar.hxx"


//#include "pyGibbs.hxx"
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


#ifdef WITH_AD3
#include "pyAd3.hxx"
#include "pyLoc.hxx"
#endif


#include "pyQpbo.hxx"

#ifdef WITH_QPBO
#include "pyQpbo.hxx"
#include "pyMQpbo.hxx"
#include "pyReducedInference.hxx"
#include  "pyAeFusion.hxx"
#endif

//#include "pyPbp.hxx"
//#include "pySelfFusion.hxx"

#include "pyFusionMoves.hxx"

//#include "pySwendsenWang.hxx"

//#include "pyLpInference.hxx"


#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>



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

         export_icm<opengm::python::GmAdder,opengm::Minimizer>();
         //export_pbp<opengm::python::GmAdder,opengm::Minimizer>();
         export_bp<opengm::python::GmAdder,opengm::Minimizer>();
         export_trbp<opengm::python::GmAdder,opengm::Minimizer>();
         export_astar<opengm::python::GmAdder,opengm::Minimizer>();
         //export_gibbs<opengm::python::GmAdder,opengm::Minimizer>();
         
         export_dual_decomposition_subgradient<opengm::python::GmAdder,opengm::Minimizer>();

         //export_self_fusion<opengm::python::GmAdder,opengm::Minimizer>();
         export_fusion_moves<opengm::python::GmAdder,opengm::Minimizer>();
         
         #ifdef WITH_CONICBUNDLE
            //export_dual_decomposition_bundle<opengm::python::GmAdder,opengm::Minimizer>();
         #endif
         export_lazyflipper<opengm::python::GmAdder,opengm::Minimizer>();
         #ifdef WITH_AD3
         export_loc<opengm::python::GmAdder,opengm::Minimizer>();
         #endif
         export_bruteforce<opengm::python::GmAdder,opengm::Minimizer>();
         export_graphcut<opengm::python::GmAdder,opengm::Minimizer>();
         export_abswap<opengm::python::GmAdder,opengm::Minimizer>();
         export_ae<opengm::python::GmAdder,opengm::Minimizer>();
         export_dynp<opengm::python::GmAdder,opengm::Minimizer>();
         export_partition_move<opengm::python::GmAdder,opengm::Minimizer>();
         
         //export_qpbo<opengm::python::GmAdder,opengm::Minimizer>();
         #ifdef WITH_QPBO
         export_reduced_inference<opengm::python::GmAdder,opengm::Minimizer>();
         export_ae_fusion<opengm::python::GmAdder,opengm::Minimizer>();
         export_mqpbo<opengm::python::GmAdder,opengm::Minimizer>();
         export_qpbo_external<opengm::python::GmAdder,opengm::Minimizer>();
         #endif
         #ifdef WITH_TRWS
         export_trws<opengm::python::GmAdder,opengm::Minimizer>();
         #endif
         
         #ifdef WITH_CPLEX
         export_cplex<opengm::python::GmAdder,opengm::Minimizer>();
         export_multicut<opengm::python::GmAdder,opengm::Minimizer>();
         #endif

         //export_lp_inference<opengm::python::GmAdder,opengm::Minimizer>();

         #ifdef WITH_LIBDAI
         export_libdai_inference<opengm::python::GmAdder,opengm::Minimizer>();
         #endif

         #ifdef WITH_MRF
         export_mrf<opengm::python::GmAdder,opengm::Minimizer>();
         #endif

         #ifdef WITH_FASTPD
         export_fast_pd<opengm::python::GmAdder,opengm::Minimizer>();
         #endif

         #ifdef WITH_AD3
         export_ad3<opengm::python::GmAdder,opengm::Minimizer>();
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
         export_icm<opengm::python::GmAdder,opengm::Maximizer>();
         export_bp<opengm::python::GmAdder,opengm::Maximizer>();
         export_trbp<opengm::python::GmAdder,opengm::Maximizer>();
         export_astar<opengm::python::GmAdder,opengm::Maximizer>();
         export_lazyflipper<opengm::python::GmAdder,opengm::Maximizer>();
         #ifdef WITH_AD3
         export_loc<opengm::python::GmAdder,opengm::Maximizer>();
         #endif
         export_bruteforce<opengm::python::GmAdder,opengm::Maximizer>();
         export_dynp<opengm::python::GmAdder,opengm::Maximizer>();

         #ifdef WITH_AD3
         export_ad3<opengm::python::GmAdder,opengm::Maximizer>();
         #endif
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

         export_bp<opengm::python::GmAdder,opengm::Integrator>();
         export_trbp<opengm::python::GmAdder,opengm::Integrator>();
         //export_dynp<opengm::python::GmMultiplier,opengm::Maximizer>();
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
         export_icm<opengm::python::GmMultiplier,opengm::Minimizer>();
         export_bp<opengm::python::GmMultiplier,opengm::Minimizer>();
         export_trbp<opengm::python::GmMultiplier,opengm::Minimizer>();
         export_astar<opengm::python::GmMultiplier,opengm::Minimizer>();
         export_lazyflipper<opengm::python::GmMultiplier,opengm::Minimizer>();
         //export_loc<opengm::python::GmMultiplier,opengm::Minimizer>();
         export_bruteforce<opengm::python::GmMultiplier,opengm::Minimizer>();
         export_dynp<opengm::python::GmMultiplier,opengm::Minimizer>();
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
         export_icm<opengm::python::GmMultiplier,opengm::Maximizer>();
         export_bp<opengm::python::GmMultiplier,opengm::Maximizer>();
         export_trbp<opengm::python::GmMultiplier,opengm::Maximizer>();
         export_astar<opengm::python::GmMultiplier,opengm::Maximizer>();
         //export_gibbs<opengm::python::GmMultiplier,opengm::Maximizer>();
         export_lazyflipper<opengm::python::GmMultiplier,opengm::Maximizer>();
         //export_loc<opengm::python::GmMultiplier,opengm::Maximizer>();
         export_bruteforce<opengm::python::GmMultiplier,opengm::Maximizer>();
         export_dynp<opengm::python::GmMultiplier,opengm::Maximizer>();
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

         export_bp<opengm::python::GmMultiplier,opengm::Integrator>();
         export_trbp<opengm::python::GmMultiplier,opengm::Integrator>();
         //export_dynp<opengm::python::GmMultiplier,opengm::Maximizer>();
      }
   }
}
