#ifndef QPBO_PARAM
#define QPBO_PARAM

#include "empty_param.hxx"
//solver specific
#include <opengm/inference/qpbo.hxx>

template<class DEPTH,class INFERENCE>
class InfParamExporterEmpty;


template<class DEPTH,class GM,class ACC>
class InfParamExporter<DEPTH,opengm::QPBO<GM,ACC> >  : public  InfParamExporterEmpty<DEPTH,opengm::QPBO< GM,ACC> > {

};


#endif