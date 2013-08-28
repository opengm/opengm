#ifndef LP_CPLEX_PARAM
#define LP_CPLEX_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/lpcplex.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterLpCplex{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterLpCplex<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const bool integerConstraint,
      const int numberOfThreads,
      const double cutUp,
      const double epGap,
      const double timeLimit
   ){
      p.integerConstraint_=integerConstraint;
      p.numberOfThreads_=numberOfThreads;
      p.cutUp_=cutUp;
      p.epGap_=epGap;
      p.timeLimit_=timeLimit;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > (className.c_str(), init<  >() )
         .def_readwrite("integerConstraint", &Parameter::integerConstraint_,"use interger constraint to solve ILP or solve LP")
         .def_readwrite("numberOfThreads", &Parameter::numberOfThreads_,"number of threads used by cplex")
         .def_readwrite("cutUp", &Parameter::cutUp_, "cut up")
         .def_readwrite("epGap", &Parameter::epGap_, "ep-Gap")
         .def_readwrite("timeLimit",&Parameter::timeLimit_,"time limit for inference in sec.")
         .def ("set", &SelfType::set, 
            (
               boost::python::arg("integerConstraint")=false,
               boost::python::arg("numberOfThreads")=0,
               boost::python::arg("cutUp")=1.0e+75,
               boost::python::arg("epGap")=0,
               boost::python::arg("timeLimit")=1e+75
            )
         )
         ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::LPCplex<GM,ACC> >  : public  InfParamExporterLpCplex<opengm::LPCplex< GM,ACC> > {

};

#endif