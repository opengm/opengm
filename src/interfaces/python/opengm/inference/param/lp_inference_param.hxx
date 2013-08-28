#ifndef LP_CPLEX_PARAM
#define LP_CPLEX_PARAM

#include "param_exporter_base.hxx"


//solver specific
#include <opengm/inference/lp_inference.hxx>

//lp solver specific
#ifdef WITH_CPLEX
#include <opengm/inference/auxiliary/lp_solver/lp_solver_cplex.hxx>
#endif

#ifdef WITH_GUROBI
#include <opengm/inference/auxiliary/lp_solver/lp_solver_gurobi.hxx>
#endif

using namespace boost::python;






#ifdef WITH_CPLEX
template<class INFERENCE>
class InfParamExporterLpInferenceCplex{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterLpInferenceCplex<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const bool integerConstraint,
      const bool integerConstraintFactorVar,
      const typename INFERENCE::Relaxation relaxation
      //const int numberOfThreads,
      //const double cutUp,
      //const double epGap,
      //const double timeLimit
   ){
      p.integerConstraint_=integerConstraint;
      p.integerConstraintFactorVar_=integerConstraintFactorVar;
      p.relaxation_=relaxation;
      //p.lpSolverParameter_.numberOfThreads_=numberOfThreads;
     // p.lpSolverParameter_.cutUp_=cutUp;
     // p.lpSolverParameter_.epGap_=epGap;
     // p.lpSolverParameter_.timeLimit_=timeLimit;
   }


   

   void static exportInfParam(const std::string & className){
      class_<Parameter > (className.c_str(), init<  >() )
         .def_readwrite("integerConstraint", &Parameter::integerConstraint_,"use interger constraint to solve ILP ")
         .def_readwrite("integerConstraintFactorVar", &Parameter::integerConstraintFactorVar_,"use interger constraint lp variables from factors")\
         .def_readwrite("relaxation", &Parameter::relaxation_,
            "relaxation can be:\n\n"
            "  -``'firstOrder'``   : default first order relaxation (default)\n\n"
            "  -``'firstOrder2'``  : experimental 1 order relaxation "
         )
         //.def_readwrite("numberOfThreads", &Parameter::lpSolverParameter_.numberOfThreads_,"number of threads used by cplex")
         //.def_readwrite("cutUp", &Parameter::lpSolverParameter_::cutUp_, "cut up")
         //.def_readwrite("epGap", &Parameter::lpSolverParameter_::epGap_, "ep-Gap")
         //.def_readwrite("timeLimit",&Parameter::lpSolverParameter_::timeLimit_,"time limit for inference in sec.")
         .def ("set", &SelfType::set, 
            (
               boost::python::arg("integerConstraint")=false,
               boost::python::arg("integerConstraintFactorVar")=false,
               boost::python::arg("relaxation")=INFERENCE::FirstOrder
               //arg("numberOfThreads")=0,
               //arg("cutUp")=1.0e+75,
               //arg("epGap")=0,
               //arg("timeLimit")=1e+75
            )
         )
         ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::LPInference<GM,ACC,opengm::LpSolverCplex> >  : 
   public  InfParamExporterLpInferenceCplex< opengm::LPInference<GM,ACC,opengm::LpSolverCplex>  > {

};
#endif




#ifdef WITH_GUROBI

template<class INFERENCE>
class InfParamExporterLpInferenceGurobi{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterLpInferenceGurobi<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const bool integerConstraint,
      const bool integerConstraintFactorVar,
      const typename INFERENCE::Relaxation relaxation
      //const int numberOfThreads,
      //const double cutUp,
      //const double epGap,
      //const double timeLimit
   ){
      p.integerConstraint_=integerConstraint;
      p.integerConstraintFactorVar_=integerConstraintFactorVar;
      p.relaxation_=relaxation;
      //p.lpSolverParameter_.numberOfThreads_=numberOfThreads;
     // p.lpSolverParameter_.cutUp_=cutUp;
     // p.lpSolverParameter_.epGap_=epGap;
     // p.lpSolverParameter_.timeLimit_=timeLimit;
   }


   

   void static exportInfParam(const std::string & className){
      class_<Parameter > (className.c_str(), init<  >() )
         .def_readwrite("integerConstraint", &Parameter::integerConstraint_,"use interger constraint to solve ILP ")
         .def_readwrite("integerConstraintFactorVar", &Parameter::integerConstraintFactorVar_,"use interger constraint lp variables from factors")\
         .def_readwrite("relaxation", &Parameter::relaxation_,
            "relaxation can be:\n\n"
            "  -``'firstOrder'``   : default first order relaxation (default)\n\n"
            "  -``'firstOrder2'``  : experimental 1 order relaxation "
         )
         //.def_readwrite("numberOfThreads", &Parameter::lpSolverParameter_.numberOfThreads_,"number of threads used by cplex")
         //.def_readwrite("cutUp", &Parameter::lpSolverParameter_::cutUp_, "cut up")
         //.def_readwrite("epGap", &Parameter::lpSolverParameter_::epGap_, "ep-Gap")
         //.def_readwrite("timeLimit",&Parameter::lpSolverParameter_::timeLimit_,"time limit for inference in sec.")
         .def ("set", &SelfType::set, 
            (
               boost::python::arg("integerConstraint")=false,
               boost::python::arg("integerConstraintFactorVar")=false,
               boost::python::arg("relaxation")=INFERENCE::FirstOrder
               //arg("numberOfThreads")=0,
               //arg("cutUp")=1.0e+75,
               //arg("epGap")=0,
               //arg("timeLimit")=1e+75
            )
         )
         ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::LPInference<GM,ACC,opengm::LpSolverGurobi> >  : 
   public  InfParamExporterLpInferenceGurobi< opengm::LPInference<GM,ACC,opengm::LpSolverGurobi>  > {

};
#endif



#endif