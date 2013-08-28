#ifndef OPENGM_MRF_PARAM
#define OPENGM_MRF_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/external/mrflib.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterMrfLib{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterMrfLib<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const typename Parameter::InferenceType inferenceType,
      const typename Parameter::EnergyType energyType,
      const size_t numberOfIterations
   ){
      p.inferenceType_=inferenceType;
      p.energyType_=energyType;
      p.numberOfIterations_=numberOfIterations;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str( ) , init< > ())
      .def_readwrite("inferenceType", &Parameter::inferenceType_,
      "MrfLib algorithm:\n\n"
      "   * ``'icm'``\n\n"
      "   * ``'expansion'``\n\n"
      "   * ``'swap'``\n\n"
      "   * ``'maxProdBp'``\n\n"
      "   * ``'trws'``\n\n"
      "   * ``'bps'``\n\n"
      )
      .def_readwrite("energyType", &Parameter::energyType_,
      "MrfLib energy type:\n\n"
      "   * ``'view'``\n\n"
      "   * ``'tables'``\n\n"
      "   * ``'tl1'``\n\n"
      "   * ``'tl2'``\n\n"
      "   * ``'weightedTable'``\n\n"
      )
      .def_readwrite("steps", &Parameter::numberOfIterations_,
      "Number of iterations. \n"
      )
      .def ("set", & SelfType::set, 
      (
         boost::python::arg("inferenceType")=Parameter::ICM,
         boost::python::arg("energyType")=Parameter::VIEW,
         boost::python::arg("steps")=1000
      ) 
      );
   }
};

template<class GM>
class InfParamExporter<opengm::external::MRFLIB<GM> >  : public  InfParamExporterMrfLib<opengm::external::MRFLIB< GM> > {

};

#endif