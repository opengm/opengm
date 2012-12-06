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



namespace pycplex{

   template<class PARAM>
   inline void set
   (
      PARAM & p,
      const bool integerConstraint,
      const int numberOfThreads,
      const double cutUp,
      const double epGap,
      const double timeLimit
   ){
      p.integerConstraint_=integerConstraint;
      p.numberOfThreads_=numberOfThreads;
      p.cutUp_=cutUp;
      p.epGap_=epGap;
      p.timeLimit_=timeLimit;
   }

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
      .def_readwrite("integerConstraint", &PyLPCplexParameter::integerConstraint_)
      .def_readwrite("numberOfThreads", &PyLPCplexParameter::numberOfThreads_)
      .def_readwrite("cutUp", &PyLPCplexParameter::cutUp_)
      .def_readwrite("epGap", &PyLPCplexParameter::epGap_)
      .def_readwrite("timeLimit",&PyLPCplexParameter::timeLimit_)
      .def("__str__", &cplexParamAsString<PyLPCplexParameter>)
      .def ("set", &pycplex::set<PyLPCplexParameter>, 
      (
         arg("integerConstraint")=false,
         arg("numberOfThreads")=0,
         arg("cutUp")=1.0e+75,
         arg("epGap")=0,
         arg("timeLimit")=1e+75
      ),
         "Set the parameters values.\n\n"
         "All values of the parameter have a default value.\n\n"
         "Args:\n\n"
         "TODO..\n\n"
      )
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