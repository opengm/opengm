#ifndef OPENGM_GCG_PARAM_HXX
#define OPENGM_GCG_PARAM_HXX

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/cgc.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterHcm{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterHcm<INFERENCE> SelfType;

   inline static void set 
   (
      Parameter &  p,
      const bool   planar,
      const size_t maxIterations,
      const bool useBookkeeping,
      const double threshold,
      const bool startFromThreshold,
      const bool doCutMove,
      const bool doGlueCutMove 
   ) {
      p.planar_          = planar;
      p.maxIterations_   = maxIterations;
      p.useBookkeeping_  = useBookkeeping;
      p.threshold_       = threshold;
      p.startFromThreshold_ = startFromThreshold;
      p.doCutMove_ = doCutMove;
      p.doGlueCutMove_ = doGlueCutMove;
   } 

   void static exportInfParam(const std::string & className){
      boost::python::class_<Parameter > ( className.c_str() , init< > ())

      .def_readwrite("planar",          &Parameter::planar_,         "is model planar")
      .def_readwrite("maxIterations", &Parameter::maxIterations_," ")
      .def_readwrite("useBookkeeping",  &Parameter::useBookkeeping_, " use useBookkeeping")
      .def_readwrite("threshold",       &Parameter::threshold_,      " threshold")
      .def_readwrite("startFromThreshold", &Parameter::startFromThreshold_, "start from threshold")
      .def_readwrite("doCutMove", &Parameter::doCutMove_, "do  the cut move")
      .def_readwrite("doGlueCutMove", &Parameter::doGlueCutMove_, "do  the glue and cut move")

      .def ("set", &SelfType::set, 
         (
            boost::python::arg("planar")             = true,
            boost::python::arg("maxIterations")      = 1,
            boost::python::arg("useBookkeeping")     = true,
            boost::python::arg("threshold")          = 0.0,
            boost::python::arg("startFromThreshold") = true,
            boost::python::arg("doCutMove") = true,
            boost::python::arg("doGlueCutMove") = true
         ) 
      )
   ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::CGC<GM,ACC> >  : public  InfParamExporterHcm<opengm::CGC< GM,ACC> > {

};

#endif /* OPENGM_GCG_PARAM_HXX */
