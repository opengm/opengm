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

#include <opengm/inference/icm.hxx>

// to print parameter as string
template<class ICM, class PARAM>
std::string icmParamAsString(const PARAM & param) {
   std::string s = "[ moveType=";
   if (param.moveType_ == ICM::SINGLE_VARIABLE)
      s.append("variable");
   else
      s.append("factor ]");
   return s;
}

namespace pyicm{   
   template<class PARAM,class INF>
   typename pyenums::IcmMoveType getMoveType(const PARAM & p){
      if(p.moveType_==INF::SINGLE_VARIABLE)
         return pyenums::SINGLE_VARIABLE;
      else
         return pyenums::FACTOR;
   }
   template<class PARAM,class INF>
   void setMoveType( PARAM & p,const pyenums::IcmMoveType h){
      if(h==pyenums::SINGLE_VARIABLE)
         p.moveType_=INF::SINGLE_VARIABLE;
      else
         p.moveType_=INF::FACTOR;
   }
}
// export function
template<class GM, class ACC>
void export_icm() {
   using namespace boost::python;
   // import numpy c-api
   import_array();
   // Inference typedefs
   typedef opengm::ICM<GM, ACC> PyICM;
   typedef typename PyICM::Parameter PyICMParameter;
   typedef typename PyICM::VerboseVisitorType PyICMVerboseVisitor;
   // export inference parameter
   class_<PyICMParameter > ("IcmParameter", init< typename PyICM::MoveType > (args("moveType")))
      .def(init<>())
      .add_property("moveType", 
           &pyicm::getMoveType<PyICMParameter,PyICM>, pyicm::setMoveType<PyICMParameter,PyICM>)
      .def ("set", &pyicm::setMoveType<PyICMParameter,PyICM>, (arg("moveType")=pyenums::SINGLE_VARIABLE) ) 
      .def("__str__", &icmParamAsString<PyICM, PyICMParameter>)
      ;
   // export inference verbose visitor via macro
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyICMVerboseVisitor, "IcmVerboseVisitor");
   // export inference via macro
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyICM, "Icm");
}
// explicit template instantiation for the supported semi-rings
template void export_icm<GmAdder, opengm::Minimizer>();
template void export_icm<GmAdder, opengm::Maximizer>();
template void export_icm<GmMultiplier, opengm::Minimizer>();
template void export_icm<GmMultiplier, opengm::Maximizer>();
