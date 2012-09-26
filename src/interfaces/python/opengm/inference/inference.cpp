//#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleInference
#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/inference/inference.hxx>

#include "pyInference.hxx"
#include "pyIcm.hxx"
#include "pyGraphcut.hxx"
#include "pyMpBased.hxx"
#include "pyGibbs.hxx"
#include "pyBruteforce.hxx"
#include "pyLazyflipper.hxx"
#include "pyDynp.hxx"
#include  "pyAe.hxx"
#include  "pyAbSwap.hxx"

#ifdef WITH_LIBDAI
#include  "external/pyLibdai.hxx" 
#endif


#include "converter.hxx"
#include "export_typedes.hxx"



using namespace boost::python;


BOOST_PYTHON_MODULE_INIT(_inference) {
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   
   std::string adderString="adder";
   std::string multiplierString="multiplier";
   std::string minimizerString="minimizer";
   std::string maximizerString="maximizer";
   std::string substring,submoduleName,subsubmoduleName,subsubstring;
   docstring_options doc_options(true,true,false);
   scope current;
   std::string currentScopeName(extract<const char*>(current.attr("__name__")));
   //import_array();
   export_inference();
   //adder
   {
      substring=adderString;
      submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      // Switch the scope to the submodule, add methods and classes.
      scope submoduleScope = submodule;
      // minimizer
      {
         subsubstring=minimizerString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         // Switch the scope to the submodule, add methods and classes.
         scope subsubmoduleScope = subsubmodule;
         export_icm<GmAdder,opengm::Minimizer>();
         export_mp_based<GmAdder,opengm::Minimizer>();
         //export_bp<GmAdder,opengm::Minimizer>();
         //export_trbp<GmAdder,opengm::Minimizer>();
         //export_astar<GmAdder,opengm::Minimizer>();
         export_gibbs<GmAdder,opengm::Minimizer>();
         //export_loc<GmAdder,opengm::Minimizer>();
         export_bruteforce<GmAdder,opengm::Minimizer>();
         export_graphcut<GmAdder,opengm::Minimizer>();
         export_ae<GmAdder,opengm::Minimizer>();
         export_abswap<GmAdder,opengm::Minimizer>();
         export_dynp<GmAdder,opengm::Minimizer>();
         export_lazyflipper<GmAdder,opengm::Minimizer>();
         #ifdef WITH_LIBDAI
         export_libdai_inference<GmAdder,opengm::Minimizer>();
         #endif
      }
      // maximizer
      {
         subsubstring=maximizerString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         // Switch the scope to the submodule, add methods and classes.
         scope subsubmoduleScope = subsubmodule;
         export_icm<GmAdder,opengm::Maximizer>();
         //export_bp<GmAdder,opengm::Maximizer>();
         //export_trbp<GmAdder,opengm::Maximizer>();
         //export_astar<GmAdder,opengm::Maximizer>();
         //export_gibbs<GmAdder,opengm::Maximizer>();
         export_mp_based<GmAdder,opengm::Maximizer>();         
         //export_loc<GmAdder,opengm::Maximizer>();
         export_bruteforce<GmAdder,opengm::Maximizer>();
         //export_graphcut<GmAdder,opengm::Maximizer>();
         export_dynp<GmAdder,opengm::Maximizer>();
         export_lazyflipper<GmAdder,opengm::Maximizer>();
         #ifdef WITH_LIBDAI
         export_libdai_inference<GmAdder,opengm::Maximizer>();
         #endif
      }
   }
   //multiplier
   {
      substring=multiplierString;
      submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      // Switch the scope to the submodule, add methods and classes.
      scope submoduleScope = submodule;
      // minimizer
      {
         subsubstring=minimizerString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         // Switch the scope to the submodule, add methods and classes.
         scope subsubmoduleScope = subsubmodule;
         export_icm<GmMultiplier,opengm::Minimizer>();
         //export_bp<GmMultiplier,opengm::Minimizer>();
         //export_trbp<GmMultiplier,opengm::Minimizer>();
         //export_astar<GmMultiplier,opengm::Minimizer>();
         //export_loc<GmMultiplier,opengm::Minimizer>();
         export_mp_based<GmMultiplier,opengm::Minimizer>();
         export_bruteforce<GmMultiplier,opengm::Minimizer>();
         //export_gibbs<GmMultiplier,opengm::Minimizer>();
         export_dynp<GmMultiplier,opengm::Minimizer>();
         export_lazyflipper<GmMultiplier,opengm::Minimizer>();
         #ifdef WITH_LIBDAI
         export_libdai_inference<GmMultiplier,opengm::Minimizer>();
         #endif
      }
      // maximizer
      {
         subsubstring=maximizerString;
         subsubmoduleName = currentScopeName + std::string(".") + substring  + std::string(".") + subsubstring ;
         // Create the submodule, and attach it to the current scope.
         object subsubmodule(borrowed(PyImport_AddModule(subsubmoduleName.c_str())));
         submoduleScope.attr(subsubstring.c_str()) = subsubmodule;
         // Switch the scope to the submodule, add methods and classes.
         scope subsubmoduleScope = subsubmodule;
         export_icm<GmMultiplier,opengm::Maximizer>();
         //export_bp<GmMultiplier,opengm::Maximizer>();
         //export_trbp<GmMultiplier,opengm::Maximizer>();
         //export_astar<GmMultiplier,opengm::Maximizer>();
         export_mp_based<GmMultiplier,opengm::Maximizer>();
         export_gibbs<GmMultiplier,opengm::Maximizer>();
         //export_loc<GmMultiplier,opengm::Maximizer>();
         export_bruteforce<GmMultiplier,opengm::Maximizer>();
         export_dynp<GmMultiplier,opengm::Maximizer>();
         export_lazyflipper<GmMultiplier,opengm::Maximizer>();
         #ifdef WITH_LIBDAI
         export_libdai_inference<GmMultiplier,opengm::Maximizer>();
         #endif
      }
   }
}
