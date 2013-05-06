#ifndef QPBO_PARAM
#define QPBO_PARAM

#include "empty_param.hxx"
//solver specific
#include <opengm/inference/qpbo.hxx>

template<class INFERENCE>
class InfParamExporterEmpty;

template<class GM,class ACC>
class InfParamExporter<opengm::QPBO<GM,ACC> >  
: public  InfParamExporterEmpty<opengm::QPBO< GM,ACC> > {

};


#endif