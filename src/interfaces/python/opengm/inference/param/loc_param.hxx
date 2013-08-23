#ifndef LOC_PARAM
#define LOC_PARAM
#ifdef  WITH_AD3
#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/loc.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterLOC{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterLOC<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const double phi,
      const size_t maxRadius,
      const size_t maxIterations,
      const size_t ad3Threshold,
      const size_t autoStop
   ){
      p.phi_=phi;
      p.maxRadius_=maxRadius;
      p.maxIterations_=maxIterations;
      p.ad3Threshold_=ad3Threshold;
      p.stopAfterNBadIterations_=autoStop;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str( ) , init< double ,size_t,size_t,size_t > (args("phi,maxRadius,maxIteration,ad3Threshold")))
      .def(init<>())
      .def_readwrite("phi", &Parameter::phi_,
      "Open parameter in (truncated) geometric distribution.\n"
      "The subgraph radius is sampled from that distribution"
      )
      
      .def_readwrite("maxRadius", &Parameter::maxRadius_,
      "Maximum subgraph radius.\n\n"
      "The subgraph radius is in [0,maxRadius]"
      )
      .def_readwrite("steps", &Parameter::maxIterations_,
      "Number of iterations. \n"
      "If steps is zero a suitable number is choosen)"
      )
      .def_readwrite("ad3Threshold", &Parameter::ad3Threshold_,
      "If there are more variables in the subgraph than ``ad3Threshold`` ,\n"
      "AD3 is used to optimise the subgraph, otherwise Bruteforce is used."
      )
      .def_readwrite("autoStop", &Parameter::stopAfterNBadIterations_,
      "If there are more than ``autoStop`` iterations without improvement ,\n"
      "inference is terminated. if autoStop==0 , autoStop will be set to gm.numberOfVariables()."
      )
      .def ("set", & SelfType::set, 
      (
         boost::python::arg("phi")=0.3,
         boost::python::arg("maxRadius")=0,
         boost::python::arg("steps")=0,
         boost::python::arg("ad3Threshold")=3,
         boost::python::arg("autoStop")=0
      )
      );
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::LOC<GM,ACC> >  : public  InfParamExporterLOC<opengm::LOC< GM,ACC> > {

};

#endif
#endif