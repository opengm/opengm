#ifndef GRAPH_CUT_PARAM
#define GRAPH_CUT_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/graphcut.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterGraphCut{

public:
    typedef typename INFERENCE::ValueType ValueType;
    typedef typename INFERENCE::Parameter Parameter;
    typedef InfParamExporterGraphCut<INFERENCE> SelfType;

    inline static void set
    (
        Parameter & p,
        double scale     
    ){
        p.scale_=scale;
    }

    void static exportInfParam(const std::string & className){
        
        class_<Parameter >( className.c_str(),DefaultParamStr::classDocStr().c_str(), init<>( DefaultParamStr::emptyConstructorDocStr().c_str() )) 
          .def(init< ValueType>())
          .def ("set", &SelfType::set, 
           ( 
                boost::python::arg("scale")=1.0 
           ) , 
          "Set the parameters values.\n\n"
          "All values of the parameter have a default value.\n\n"
          "Args:\n\n"
          "  scale: rescale the objective function.(default=1)\n\n"
          "     This is only usefull if the min-st-cut uses \n\n"
          "     integral value types.\n\n"
          "     This will be supported in the next release.\n\n"
          "Returns:\n"
          "  None\n\n"
          ) 
          .def_readwrite("scale", & Parameter::scale_,
          "rescale the objective function.\n\n"
          "  This is only usefull if the min-st-cut uses \n\n"
          "  integral value types.\n\n"\
          "  This will be supported in one of the next releases.\n\n")
          ; 
    }
};

template<class GM,class ACC,class MIN_ST_CUT>
class InfParamExporter<opengm::GraphCut<GM,ACC,MIN_ST_CUT> >  
: public  InfParamExporterGraphCut<opengm::GraphCut< GM,ACC,MIN_ST_CUT> > {

};

#endif