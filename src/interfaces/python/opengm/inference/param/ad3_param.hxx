#ifndef AD3_PARAM
#define AD3_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/external/ad3.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterAD3{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterAD3<INFERENCE> SelfType;

   inline static void set 
   (
      Parameter & p,
      const typename INFERENCE::SolverType   solverType,
      const double                           eta,
      const bool                             adaptEta,
      opengm::UInt64Type                     steps,
      const double                           residualThreshold,
      const int                              verbosity          
   ) {
      p.solverType_=solverType;
      p.eta_=eta;
      p.adaptEta_=adaptEta;
      p.steps_=steps;
      p.residualThreshold_=residualThreshold;
      p.verbosity_=verbosity;
   } 

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str() , init< > ())
      .def_readwrite("solverType", &Parameter::solverType_,"solverType can be:\n\n"
         "  -``'ad3_lp'``  : ad3 with naive rounding of fractal solutions\n\n"
         "  -``'ad3_ilp'`` : exact ad3 with branch and bound optimization \n\n"
         "  -``'psdd_lp'`` : pssd with naive rounding of fractal solutions"
      )   
      .def_readwrite("eta", &Parameter::eta_,"eta of ad3 (see ad3 doc)")
      .def_readwrite("adaptEta", &Parameter::adaptEta_,"adapt eta  ad3 (see ad3 doc)")
      .def_readwrite("steps", &Parameter::steps_,"maximum iterations in ad3 (see ad3 doc)")
      .def_readwrite("residualThreshold", &Parameter::residualThreshold_,"residualThreshold of ad3 (see ad3 doc)")
      .def_readwrite("verbose", &Parameter::verbosity_,"verbosity level of ad3 (see ad3 doc)")

      .def ("set", &SelfType::set, 
         (
            boost::python::arg("solverType")=INFERENCE::AD3_LP,
            boost::python::arg("eta")=0.1,
            boost::python::arg("adaptEta")=true,
            boost::python::arg("steps")=1000,
            boost::python::arg("residualThreshold")=1e-6,
            boost::python::arg("verbose")=0
         ) 
      )
   ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::external::AD3Inf<GM,ACC> >  : public  InfParamExporterAD3<opengm::external::AD3Inf< GM,ACC> > {

};

#endif