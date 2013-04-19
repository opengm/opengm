#ifndef BRUTEFORCE_PARAM
#define BRUTEFORCE_PARAM

#include "empty_param.hxx"
//solver specific
#include <opengm/inference/bruteforce.hxx>

template<class DEPTH,class INFERENCE>
class InfParamExporterEmpty;


template<class DEPTH,class GM,class ACC>
class InfParamExporter<DEPTH,opengm::Bruteforce<GM,ACC> >  : public  InfParamExporterEmpty<DEPTH,opengm::Bruteforce< GM,ACC> > {

};


#endif