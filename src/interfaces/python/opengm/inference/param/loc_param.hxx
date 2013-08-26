#ifndef LOC_PARAM
#define LOC_PARAM
#ifdef  WITH_AD3

#include <string>

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
      const std::string solver,
      const double phi,
      const size_t maxRadius,
      const double pFastHeuristic,
      const size_t maxIterations,
      const size_t autoStop,
      const size_t maxSubgraphSize
   ){
      p.solver_=solver;
      p.phi_=phi;
      p.maxRadius_=maxRadius;
      p.pFastHeuristic_=pFastHeuristic;
      p.maxIterations_=maxIterations;
      p.stopAfterNBadIterations_=autoStop;
      p.maxSubgraphSize_=maxSubgraphSize;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str( ) , init<  > () )

      .def_readwrite("solver", &Parameter::solver_,
      "solver used for the subproblems.\n"
      "must be \"ad3\" , \"astar\" ,\"dp\" "
      )

      .def_readwrite("phi", &Parameter::phi_,
      "Open parameter in (truncated) geometric distribution.\n"
      "The subgraph radius is sampled from that distribution"
      )
      
      .def_readwrite("pFastHeuristic", &Parameter::pFastHeuristic_,
      "Probability of fast heuristic in [0,1].\n\n"
      "0 means no fast heurisitc,1 means pure fast heuristc, and values between 0 and 1 are a mix of both"
      )

      .def_readwrite("maxRadius", &Parameter::maxRadius_,
      "Maximum subgraph radius.\n\n"
      "The subgraph radius is in [2,maxRadius]"
      )


      .def_readwrite("steps", &Parameter::maxIterations_,
      "Number of iterations. \n"
      "If steps is zero a suitable number is choosen)"
      )
      .def_readwrite("autoStop", &Parameter::stopAfterNBadIterations_,
      "If there are more than ``autoStop`` iterations without improvement ,\n"
      "inference is terminated. if autoStop==0 , autoStop will be set to gm.numberOfVariables()."
      )

      .def_readwrite("maxSubgraphSize", &Parameter::maxSubgraphSize_,
      "maxSubgraphSize which is allowed (0 means any),\n"
      )

      
      .def ("set", & SelfType::set, 
      (
         boost::python::arg("solver")=std::string("ad3"),
         boost::python::arg("phi")=0.3,
         boost::python::arg("maxRadius")=0,
         boost::python::arg("pFastHeuristic")=0.997,
         boost::python::arg("steps")=0,
         boost::python::arg("autoStop")=0,
         boost::python::arg("maxSubgraphSize")=0
      )
      );
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::LOC<GM,ACC> >  : public  InfParamExporterLOC<opengm::LOC< GM,ACC> > {

};

#endif
#endif