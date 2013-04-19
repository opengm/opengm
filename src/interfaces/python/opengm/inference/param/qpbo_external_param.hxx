#ifndef QPBO_EXTERNA_PARAM
#define QPBO_EXTERNA_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/external/qpbo.hxx>

using namespace boost::python;

template<class DEPTH,class INFERENCE>
class InfParamExporterQpboExternal{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterQpboExternal<DEPTH,INFERENCE> SelfType;


   inline static void set 
   (
      Parameter & p,
      const bool strongPersistency,
      const bool useImproveing,
      const bool useProbeing
   ) {
      p.strongPersistency_=strongPersistency;
      p.useImproveing_=useImproveing;
      p.useProbeing_=useProbeing;
   }


   void static exportInfParam(const std::string & className,const std::vector<std::string> & subInfParamNames){
      class_<Parameter > ( className.c_str(),init<>())
         .def ("set", &SelfType::set, 
            (
               arg("strongPersistency")=true,
               arg("useImproveing")=false,
               arg("useProbeing")=false
            )
         )
         .def_readwrite("strongPersistency", &Parameter::strongPersistency_, "use strong persitency")
         .def_readwrite("useImproveing",     &Parameter::useImproveing_,     "use improveing to get better solutions")
         .def_readwrite("useProbeing",       &Parameter::useProbeing_,       "use probing")
      ;


   }
};

template<class DEPTH,class GM>
class InfParamExporter<DEPTH,opengm::external::QPBO<GM> >  : public  InfParamExporterQpboExternal<DEPTH,opengm::external::QPBO< GM> > {

};

#endif