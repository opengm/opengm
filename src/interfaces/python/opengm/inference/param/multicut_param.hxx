#ifdef WITH_CPLEX
#ifndef MULTICUT_PARAM
#define MULTICUT_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/multicut.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterMulticut{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterMulticut<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const int numThreads,
      const bool verbose,
      const bool verboseCPLEX,
      const double cutUp,
      const double timeOut,
      const size_t maximalNumberOfConstraintsPerRound,
      const double edgeRoundingValue,
      //const MWCRounding MWCRounding,
      const size_t reductionMode
   ){
      p.numThreads_=numThreads;
      p.verbose_=verbose;
      p.verboseCPLEX_=verboseCPLEX;
      p.cutUp_=cutUp;
      p.timeOut_=timeOut;
      p.maximalNumberOfConstraintsPerRound_=maximalNumberOfConstraintsPerRound;
      p.edgeRoundingValue_=edgeRoundingValue;
      p.MWCRounding_=Parameter::NEAREST;
      p.reductionMode_=reductionMode;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > (className.c_str(), init<  >() )


         .def_readwrite("numThreads", &Parameter::numThreads_,"numThreads number of threads that should be used (default = 0 [automatic])")
         .def_readwrite("verbose", &Parameter::verbose_,"verbose output")
         .def_readwrite("verboseCPLEX", &Parameter::verboseCPLEX_,"verbose cplex output")
         .def_readwrite("cutUp", &Parameter::cutUp_,"cutUp value which the optima at least has (helps to cut search-tree")
         .def_readwrite("timeOut", &Parameter::timeOut_,"maximum time in sec")
         .def_readwrite("maximalNumberOfConstraintsPerRound", &Parameter::maximalNumberOfConstraintsPerRound_,"maximal mumber of constraints per round ")
         .def_readwrite("edgeRoundingValue", &Parameter::edgeRoundingValue_,"edge Rounding Value")
         //.def_readwrite("MWCRounding", &Parameter::MWCRounding_,"multiway cut rounding ")
         .def_readwrite("reductionMode", &Parameter::reductionMode_," reductionMode")
         .def ("set", &SelfType::set, 
            (
               boost::python::arg("numThreads")                         =0,
               boost::python::arg("verbose")                            =false,
               boost::python::arg("verboseCPLEX")                       =false,
               boost::python::arg("cutUp")                              =1.0e+75,
               boost::python::arg("timeOut")                            =36000000,
               boost::python::arg("maximalNumberOfConstraintsPerRound") =1000000,
               boost::python::arg("edgeRoundingValue")                  =0.00000001,
               boost::python::arg("reductionMode")                      =3
            )
         )
         ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::Multicut<GM,ACC> >  : public  InfParamExporterMulticut<opengm::Multicut< GM,ACC> > {

};

#endif
#endif
