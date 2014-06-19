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
      const std::string illustrationOut
   ) {
      p.planar_          = planar;
      p.maxIterations_   = maxIterations;
      p.useBookkeeping_  = useBookkeeping;
      p.threshold_       = threshold;
      p.illustrationOut_ = illustrationOut;
   } 

   void static exportInfParam(const std::string & className){
      boost::python::class_<Parameter > ( className.c_str() , init< > ())

      .def_readwrite("planar",          &Parameter::planar_,         "is model planar")
      .def_readwrite("maxIterations", &Parameter::maxIterations_," ")
      .def_readwrite("useBookkeeping",  &Parameter::useBookkeeping_, " use useBookkeeping")
      .def_readwrite("threshold",       &Parameter::threshold_,      " threshold")
      .def_readwrite("illustrationOut", &Parameter::illustrationOut_," write out file for illustrations (for figures)")
      
      .def ("set", &SelfType::set, 
         (
            boost::python::arg("planar")          = true,
            boost::python::arg("maxIterations")   = 1,
            boost::python::arg("useBookkeeping")  = true,
            boost::python::arg("threshold")       = 0.0,
            boost::python::arg("illustrationOut") = ""
         ) 
      )
   ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::CGC<GM,ACC> >  : public  InfParamExporterHcm<opengm::CGC< GM,ACC> > {

};

#endif /* OPENGM_GCG_PARAM_HXX */
