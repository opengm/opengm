#ifndef DUAL_DECOMPOSITION_PARAM
#define DUAL_DECOMPOSITION_PARAM

#include "param_exporter_base.hxx"
//solver specific

#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/self_fusion.hxx>


using namespace boost::python;

template<class INFERENCE>
class InfParamExporterSelfFusion{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef typename INFERENCE::ToFuseInferenceType SubInfType;
   typedef typename SubInfType::Parameter SubInfParameter;

   typedef InfParamExporterSelfFusion<INFERENCE> SelfType;


   static void set(
        Parameter & p,
        const size_t fuseNth,
        const typename INFERENCE::FusionSolver fusionSolver,
        const SubInfParameter & infParam,
        const opengm::UInt64Type maxSubgraphSize,
        const bool reducedInf ,
        const bool tentacles ,
        const bool connectedComponents,
        const double fusionTimeLimit,
        const size_t numStopIt
   ){
      p.fuseNth_ = fuseNth;
      p.fusionSolver_ = fusionSolver;
      p.infParam_ = infParam;
      p.maxSubgraphSize_ = maxSubgraphSize; 
      p.reducedInf_ = reducedInf;
      p.connectedComponents_ = connectedComponents;
      p.tentacles_ = tentacles;
      p.fusionTimeLimit_ = fusionTimeLimit;
      p.numStopIt_ = numStopIt;
   }





    void static exportInfParam(const std::string & className){
    class_<Parameter > ( className.c_str(),init<>() ) 

        .def_readwrite("fuseNth",&Parameter::fuseNth_,"fuse each nth step")
        .def_readwrite("fusionSolver",&Parameter::fusionSolver_,
           "type of solver for the fusion move inference (default = 'qpbo') : \n\n"
           "  * 'qpbo'\n\n"
           "  * 'cplex'\n\n"
           "  * 'lf'\n\n"
        )
        .def_readwrite("infParam",&Parameter::infParam_,"parameter of proposal generator inference")
        .def_readwrite("maxSubgraphSize",&Parameter::maxSubgraphSize_,"subgraphsize if lf is used (default=2)")
        .def_readwrite("reducedInf",&Parameter::reducedInf_,"use reduced inference (default=false)")
        .def_readwrite("connectedComponents",&Parameter::connectedComponents_,"if reduced inference is used, use connected components (default=false)")
        .def_readwrite("tentacles",&Parameter::tentacles_,"if reduced inference is used,  eliminate tentacles (default=false)")
        .def_readwrite("fusionTimeLimit",&Parameter::fusionTimeLimit_, "time limit for each fusion move step")
        .def_readwrite("numStopIt",&Parameter::numStopIt_,"stop after n not successful iterations")
    .def ("set", &SelfType::set,
      (
        boost::python::arg("fuseNth")=1,
        boost::python::arg("fusionSolver")=INFERENCE::QpboFusion,
        boost::python::arg("infParam")=SubInfParameter(),
        boost::python::arg("maxSubgraphSize")=2,
        boost::python::arg("reducedInf")=false,
        boost::python::arg("connectedComponents")=false,
        boost::python::arg("tentacles")=false,
        boost::python::arg("fusionTimeLimit")=100.0,
        boost::python::arg("numStopIt")=10
      ) 
    )
   ; 
   }
};

template<class SUB_INF>
class InfParamExporter<opengm::SelfFusion<SUB_INF> > 
 : public  InfParamExporterSelfFusion<opengm::SelfFusion< SUB_INF> > {

};

#endif

