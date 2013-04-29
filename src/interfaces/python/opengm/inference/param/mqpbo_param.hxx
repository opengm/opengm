#ifndef MQPBO_EXTERNA_PARAM
#define MQPBO_EXTERNA_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/mqpbo.hxx>

using namespace boost::python;

template<class DEPTH,class INFERENCE>
class InfParamExporterMQpbo{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterMQpbo<DEPTH,INFERENCE> SelfType;


   inline static void set 
   (
      Parameter & p,
      const bool useKovtunsMethod,
      const bool probing,
      const bool strongPersistency,
      const size_t rounds,
      const typename INFERENCE::PermutationType permutationType
   ) {
         p.useKovtunsMethod_=useKovtunsMethod;
         p.probing_=probing;
         p.strongPersistency_=strongPersistency;
         p.rounds_=rounds;
         p.permutationType_=permutationType;
   }


   void static exportInfParam(const std::string & className,const std::vector<std::string> & subInfParamNames){
      class_<Parameter > ( className.c_str(),init<>())
         .def ("set", &SelfType::set, 
            (
               arg("useKovtunsMethod")=true ,
               arg("useProbing")=false ,
               arg("strongPersistency")=false ,
               arg("rounds")=0 ,
               arg("permutationType")=INFERENCE::NONE
            )
         )
         .def_readwrite("useKovtunsMethod",  &Parameter::useKovtunsMethod_,   "use Kovtuns Method")
         .def_readwrite("useProbeing",       &Parameter::probing_,            "use probing")
         .def_readwrite("strongPersistency", &Parameter::strongPersistency_,  "use strong persitency")
         .def_readwrite("rounds",            &Parameter::rounds_,             "rounds of MQPBO")
         .def_readwrite("permutationType",   &Parameter::permutationType_,    "permutation used for label-ordering")
         .def_readwrite("useProbeing",       &Parameter::probing_,            "use probing")
      ;
   }
};

template<class DEPTH,class GM,class ACC>
class InfParamExporter<DEPTH,opengm::MQPBO<GM,ACC> >  : public  InfParamExporterMQpbo<DEPTH,opengm::MQPBO< GM,ACC> > {

};

#endif