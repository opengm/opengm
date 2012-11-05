#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

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

template<class V,class I>
struct Constraint{
   typedef V ValueType;
   typedef I IndexType;
   std::vector<IndexType> ilpVars_;
   std::vector<ValueType> coefficients_;
   ValueType lowerBound_;
   ValueType upperBound_;
};

template<class I>
struct GmVarToLpVar{
   template<class GM>
   GmVarToLpVar(const GM & gm){
      IndexType lpVarBegin=0;
      for(IndexType v=0;v<gm.numberOfVariables();++v){
         ilpVarBegin_[v]=lpVarBegin;
         lpVarBegin+=gm.numberOfLabels(v);
      }
   }
   std::vector<IndexType> ilpVarBegin_;
};



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
   class_<PyLPCplexParameter > ("LPCplexParameter", init< typename PyLPCplex::MoveType > (args("moveType")))
      .def(init<>())
      .def_readwrite("numberOfThreads", &PyLPCplexParameter::numberOfThreads_)
      .def_readwrite("cutUp", &PyLPCplexParameter::cutUp_)
      .def_readwrite("epGap", &PyLPCplexParameter::epGap_)
      .def("__str__", &cplexParamAsString<PyLPCplexParameter>)
      ;
   // export inference verbose visitor via macro
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLPCplexVerboseVisitor, "LPCplexVerboseVisitor");
   // export inference via macro
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyLPCplex, "LPCplex");
}
// explicit template instantiation for the supported semi-rings
template void export_LPCplex<GmAdder, opengm::Minimizer>();
