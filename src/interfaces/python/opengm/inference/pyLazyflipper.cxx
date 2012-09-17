#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/lazyflipper.hxx>
#include "../export_typedes.hxx"
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
using namespace boost::python;

namespace layzflipper{
   template<class PARAM>
   inline void set 
   (
      PARAM & p,
      const size_t maxSubgraphSize
   ) {
      p.maxSubgraphSize_=maxSubgraphSize;
   } 
}

template<class GM,class ACC>
void export_lazyflipper(){
   import_array();
   // Py Inference Types 
   typedef opengm::LazyFlipper<GM, ACC>  PyLazyFlipper;
   typedef typename PyLazyFlipper::Parameter PyLazyFlipperParameter;
   typedef typename PyLazyFlipper::VerboseVisitorType PyLazyFlipperVerboseVisitor;
   
   class_<PyLazyFlipperParameter > ( "LazyFlipperParameter" , init< const size_t > (args("maxSubGraphSize")))
   .def(init<>())
   .def_readwrite("maxSubgraphSize", &PyLazyFlipperParameter::maxSubgraphSize_)
   .def ("set", &layzflipper::set<PyLazyFlipperParameter>, 
      (
      arg("maxSubgraphSize")=2
      ) 
   ) 
   ;

   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLazyFlipperVerboseVisitor,"LazyFlipperVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyLazyFlipper,"LazyFlipper");
}

template void export_lazyflipper<GmAdder,opengm::Minimizer>();
template void export_lazyflipper<GmAdder,opengm::Maximizer>();
template void export_lazyflipper<GmMultiplier,opengm::Minimizer>();
template void export_lazyflipper<GmMultiplier,opengm::Maximizer>();
