#ifndef DYNAMIC_PROGRAMMING_PARAM
#define DYNAMIC_PROGRAMMING_PARAM

#include "empty_param.hxx"
//solver specific
#include <opengm/inference/dynamicprogramming.hxx>

//template<class INFERENCE>
//class InfParamExporterEmpty;


template<class GM,class ACC>
class InfParamExporter<opengm::DynamicProgramming<GM,ACC> >  
: public  InfParamExporterEmpty<opengm::DynamicProgramming< GM,ACC> > {

};


#endif