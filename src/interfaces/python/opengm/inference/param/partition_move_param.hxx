#ifndef PARTITION_MOVE_PARAM
#define PARTITION_MOVE_PARAM

#include "empty_param.hxx"
//solver specific
#include <opengm/inference/partition-move.hxx>

template<class INFERENCE>
class InfParamExporterEmpty;

template<class GM,class ACC>
class InfParamExporter<opengm::PartitionMove<GM,ACC> >  
: public  InfParamExporterEmpty<opengm::PartitionMove< GM,ACC> > {

};


#endif