#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include "../export_typedes.hxx"
using namespace boost::python;

// Py Inference Types 
namespace pydynp{
   template<class PARAM>
   inline void set(PARAM & p){}
}

template<class GM,class ACC>
void export_dynp(){
   import_array(); 
   typedef opengm::DynamicProgramming<GM, ACC>  PyDynamicProgramming;
   typedef typename PyDynamicProgramming::Parameter PyDynamicProgrammingParameter;
   typedef typename PyDynamicProgramming::VerboseVisitorType PyDynamicProgrammingVerboseVisitor;

   class_<PyDynamicProgrammingParameter > ( "DynamicProgrammingParameter" , init< > ())
   .def("set",&pydynp::set<PyDynamicProgrammingParameter>)
   ;
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyDynamicProgrammingVerboseVisitor,"DynamicProgrammingVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_NO_RESET_EXPORTER(PyDynamicProgramming,"DynamicProgramming");
}

template void export_dynp<GmAdder,opengm::Minimizer>();
template void export_dynp<GmAdder,opengm::Maximizer>();
template void export_dynp<GmMultiplier,opengm::Minimizer>();
template void export_dynp<GmMultiplier,opengm::Maximizer>();