#ifndef EMPTY_PARAM
#define EMPTY_PARAM

#include "param_exporter_base.hxx"
//solver specific
//--

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterEmpty{
public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterEmpty<INFERENCE> SelfType;

   inline static void set
   (
      Parameter & p
   ){
      
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter >( className.c_str(),DefaultParamStr::classDocStr().c_str(), init<>( DefaultParamStr::emptyConstructorDocStr().c_str() )) 
         .def ("set", &SelfType::set, 
         "Set the parameters values.\n\n"
         "Args:\n\n"
         "  none\n\n"
         "Returns:\n"
         "  None\n\n"
         )
      ; 
   }
};

#endif