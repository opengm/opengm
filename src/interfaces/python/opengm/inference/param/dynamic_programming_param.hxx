#ifndef DYNAMIC_PROGRAMMING_PARAM
#define DYNAMIC_PROGRAMMING_PARAM

#include "empty_param.hxx"
//solver specific
#include <opengm/inference/dynamicprogramming.hxx>

template<class DEPTH,class INFERENCE>
class InfParamExporterEmpty;


template<class DEPTH,class GM,class ACC>
class InfParamExporter<DEPTH,opengm::DynamicProgramming<GM,ACC> >  : public  InfParamExporterEmpty<DEPTH,opengm::DynamicProgramming< GM,ACC> > {

};


#endif