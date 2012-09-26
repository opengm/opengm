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
      .add_property("moveType", &pyicm::getMoveType<PyICMParameter,PyICM>, pyicm::setMoveType<PyICMParameter,PyICM>,
      "moveType can be:\n\n"
      "-``opengm.IcmMoveType.variable`` :  move only one variable at once optimaly (default) \n\n"
      "-``opengm.IcmMoveType.factor`` :   move all variable of a factor at once optimaly \n"
      )
      .def ("set", &pyicm::setMoveType<PyICMParameter,PyICM>, (arg("moveType")=pyenums::SINGLE_VARIABLE),
      "Set the parameters values.\n\n"
      "All values of the parameter have a default value.\n\n"
      "Args:\n\n"
      "  moveType: moveType can be:\n\n"
      "     -``opengm.IcmMoveType.variable`` :  move only one variable at once optimaly (default) \n\n"
      "     -``opengm.IcmMoveType.factor`` :   move all variable of a factor at once optimaly \n\n"
      "  bound: AStar objective bound.\n\n"
      "     A good bound will speedup inference (default = neutral value)\n\n"
      "  maxHeapSize: Maximum size of the heap which is used while inference (default=3000000)\n\n"
      "  numberOfOpt: Select which n best states should be searched for while inference (default=1):\n\n"
      "Returns:\n"
      "  None\n\n"
      ) 
      .def("__str__", &icmParamAsString<PyICM, PyICMParameter>,
      "Get the infernce paramteres values as string"
      )
      ;
   // export inference verbose visitor via macro
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyICMVerboseVisitor, "IcmVerboseVisitor");
   // export inference via macro
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyICM, "Icm",
   "Iterated Conditional Modes Algorithm (ICM):\n\n"
   "cite: J. E. Besag: \"`On the Statistical Analysis of Dirty Pictures <http://webdocs.cs.ualberta.ca/~nray1/CMPUT617/Inference/Besag.pdf>`_\"," 
   "Journal of the Royal Statistical Society, Series B 48(3):259-302, 1986.\n\n"
   "limitations: -\n\n"
   "guarantees: -\n"
   );
}
// explicit template instantiation for the supported semi-rings
template void export_icm<GmAdder, opengm::Minimizer>();
template void export_icm<GmAdder, opengm::Maximizer>();
template void export_icm<GmMultiplier, opengm::Minimizer>();
template void export_icm<GmMultiplier, opengm::Maximizer>();
