#ifndef TRWS_EXTERNAL_PARAM
#define TRWS_EXTERNAL_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/external/trws.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterTrwsExternal{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterTrwsExternal<INFERENCE> SelfType;


   static void set 
   (
      Parameter & p,
      const size_t numberOfIterations,
      const bool useRandomStart,
      const bool useZeroStart,
      const bool doBPS,
      typename Parameter::EnergyType energyType,
      const double tolerance
   ) {
      p.numberOfIterations_=numberOfIterations;
      p.useRandomStart_=useRandomStart;
      p.useZeroStart_=useZeroStart;
      p.doBPS_=doBPS;
      p.energyType_=energyType;
      p.tolerance_=tolerance;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str(),init<>())
         .def ("set", &SelfType::set, 
            (
               boost::python::arg("steps")=1000,
               boost::python::arg("useRandomStart")=false,
               boost::python::arg("useZeroStart")=false,
               boost::python::arg("doBPS")=false,
               boost::python::arg("energyType")=Parameter::VIEW,
               boost::python::arg("tolerance")=double(0.0)
            )
         )
         .def_readwrite("steps",          &Parameter::numberOfIterations_, "number of iterations")
         .def_readwrite("useRandomStart", &Parameter::useRandomStart_,     "use a random starting point")
         .def_readwrite("useZeroStart",   &Parameter::useZeroStart_,       "use zero als label for all variables as starting point")
         .def_readwrite("doBPS",          &Parameter::doBPS_,              "use BPS while inference")
         .def_readwrite("energyType",     &Parameter::energyType_,         
            "type of the value table :\n\n"
            "   - ``'view'``   :  view to an existing value table\n\n"
            "   - ``'tables'`` :  copy to dense value tables\n\n"
            "   - ``'tl1'``    :  l1 value table\n\n"
            "   - ``'tl2'``    :  l2 value table"
         )
         .def_readwrite("tolerance",      &Parameter::tolerance_,          "termination criterion")
      ;
   }
};

template<class GM>
class InfParamExporter<opengm::external::TRWS<GM> >  
: public  InfParamExporterTrwsExternal<opengm::external::TRWS<GM> > {

};

#endif