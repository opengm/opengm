#ifndef ICM_PARAM
#define ICM_PARAM

#include <string>
#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/icm.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterICM{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterICM<INFERENCE> SelfType;


   // to print parameter as string
   //static std::string asString(const Parameter & param) {
   //   std::string s = "[ moveType=";
   //   if (param.moveType_ == INFERENCE::SINGLE_VARIABLE)
   //      s.append("variable");
   //   else
   //      s.append("factor ]");
   //   return s;
   //}

   static typename opengm::python::pyenums::IcmMoveType getMoveType(const Parameter & p){
      if(p.moveType_==INFERENCE::SINGLE_VARIABLE)
         return opengm::python::pyenums::SINGLE_VARIABLE;
      else
         return opengm::python::pyenums::FACTOR;
   }

   static void set( Parameter & p,const opengm::python::pyenums::IcmMoveType h){
      if(h==opengm::python::pyenums::SINGLE_VARIABLE)
         p.moveType_=INFERENCE::SINGLE_VARIABLE;
      else
         p.moveType_=INFERENCE::FACTOR;
   }


   void static exportInfParam(const std::string & className){
      class_<Parameter > (className.c_str(), init< typename INFERENCE::MoveType > (args("moveType")))
         .def(init<>())
         .add_property("moveType", &SelfType::getMoveType, SelfType::set,
         "moveType can be:\n\n"
         "  -``'variable'`` :  move only one variable at once optimaly (default) \n\n"
         "  -``'factor'`` :   move all variable of a factor at once optimaly \n"
         )
         .def ("set", &SelfType::set, (boost::python::arg("moveType")=opengm::python::pyenums::SINGLE_VARIABLE),
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
         //.def("__str__", &SelfType::asString,
         //"Get the infernce paramteres values as string"
         //)
         ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::ICM<GM,ACC> >  : public  InfParamExporterICM<opengm::ICM< GM,ACC> > {

};

#endif