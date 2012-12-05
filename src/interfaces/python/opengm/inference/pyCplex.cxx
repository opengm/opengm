#ifdef WITH_CPLEX
#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include "export_typedes.hxx"

#include <opengm/inference/lpcplex.hxx>

// to print parameter as string
template< class PARAM>
std::string cplexParamAsString(const PARAM & param) {
   std::string p=" ";
   return p;
}


// export function
template<class GM, class ACC>
void export_cplex() {
   using namespace boost::python;
   // import numpy c-api
   import_array();
   // Inference typedefs
   typedef opengm::LPCplex<GM, ACC> PyLPCplex;
   typedef typename PyLPCplex::Parameter PyLPCplexParameter;
   typedef typename PyLPCplex::VerboseVisitorType PyLPCplexVerboseVisitor;

   // export inference parameter
   class_<PyLPCplexParameter > ("LPCplexParameter", init<  >() )
      .def_readwrite("numberOfThreads", &PyLPCplexParameter::numberOfThreads_)
      .def_readwrite("cutUp", &PyLPCplexParameter::cutUp_)
      .def_readwrite("epGap", &PyLPCplexParameter::epGap_)
      .def("__str__", &cplexParamAsString<PyLPCplexParameter>)
      ;
   // export inference verbose visitor via macro
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLPCplexVerboseVisitor, "LPCplexVerboseVisitor");
   // export inference via macro
   OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(PyLPCplex, "LPCplex",   
   "TODO:\n\n"
   );
}
// explicit template instantiation for the supported semi-rings
template void export_cplex<GmAdder, opengm::Minimizer>();
#endif