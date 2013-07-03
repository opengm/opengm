#ifndef LOC_PARAM
#define LOC_PARAM

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
      const size_t aStarThreshold
   ){
      p.phi_=phi;
      p.maxRadius_=maxRadius;
      p.maxIterations_=maxIterations;
      p.aStarThreshold_=aStarThreshold;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str( ) , init< double ,size_t,size_t,size_t > (args("phi,maxRadius,maxIteration,aStarThreshold")))
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
      .def_readwrite("aStarThreshold", &Parameter::aStarThreshold_,
      "If there are more variables in the subgraph than ``aStarThreshold`` ,\n"
      "AStar is used to optimise the subgraph, otherwise Bruteforce is used."
      )
      .def ("set", & SelfType::set, 
      (
         arg("phi")=0.5,
         arg("maxRadius")=5,
         arg("steps")=0,
         arg("aStarThreshold")=10
      )
      );
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::LOC<GM,ACC> >  : public  InfParamExporterLOC<opengm::LOC< GM,ACC> > {

};

#endif