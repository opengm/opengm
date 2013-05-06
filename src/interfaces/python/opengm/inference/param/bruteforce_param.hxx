#ifndef BRUTEFORCE_PARAM
#define BRUTEFORCE_PARAM

#include "empty_param.hxx"
//solver specific
#include <opengm/inference/bruteforce.hxx>

template<class INFERENCE>
class InfParamExporterEmpty;

template<class GM,class ACC>
class InfParamExporter<opengm::Bruteforce<GM,ACC> >  
: public  InfParamExporterEmpty<opengm::Bruteforce< GM,ACC> > {

};


#endif