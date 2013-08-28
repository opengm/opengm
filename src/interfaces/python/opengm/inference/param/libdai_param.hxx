#ifdef WITH_LIBDAI
#ifndef OPENGM_LIBDAI_PARAM
#define OPENGM_LIBDAI_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/external/libdai/bp.hxx>
#include <opengm/inference/external/libdai/fractional_bp.hxx>
#include <opengm/inference/external/libdai/tree_reweighted_bp.hxx>
#include <opengm/inference/external/libdai/double_loop_generalized_bp.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>
#include <opengm/inference/external/libdai/gibbs.hxx>


using namespace boost::python;

template<class INFERENCE>
class InfParamExporterLibdaiDecMap{

public:
   typedef typename INFERENCE::ValueType                 ValueType;
   typedef typename INFERENCE::Parameter                 Parameter;
   typedef typename INFERENCE::SubInferenceParameter     SubInferenceParameter;
   typedef InfParamExporterLibdaiDecMap< INFERENCE> SelfType;

   static void set
   (
      Parameter  & p,
      const SubInferenceParameter & subInfParam=SubInferenceParameter(),
      const size_t reinit=10000,
      const size_t verbose=0
   ){
      p.subInferenceParam_=subInfParam;
      p.reinit_=reinit;
      p.verbose_=verbose;
   }

   void static exportInfParam(const std::string & className){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def ("set", &set, 
         (
            boost::python::arg("subInfParam")=SubInferenceParameter(),
            boost::python::arg("reinit")=1,
            boost::python::arg("verbose")=0
         ) 
      )
      .def_readwrite("subInfParam",   &Parameter::subInferenceParam_,"sub-inference parameter")
      .def_readwrite("reinit",  &Parameter::reinit_, "number of reinitializations")
      .def_readwrite("verbose", &Parameter::verbose_,"verbose level")  
   ; 
   }
};


template<class INFERENCE>
class InfParamExporterLibdaiDoubleLoopGBP{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef InfParamExporterLibdaiDoubleLoopGBP<INFERENCE> SelfType;


   static void set
   (
      Parameter  & p,
      const bool doubleloop,
      const typename INFERENCE::Clusters clusters,
      const size_t loopdepth,
      const typename INFERENCE::Init init,
      const size_t maxiter,
      const double tolerance,
      const size_t verbose
   ){
      p.doubleloop_=doubleloop;
      p.clusters_=clusters;
      p.loopdepth_=loopdepth;
      p.init_=init;
      p.maxiter_=maxiter;
      p.tolerance_=tolerance;
      p.verbose_=verbose;
   }
   

   void static exportInfParam(const std::string & className){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def ("set", &set, 
         (
            boost::python::arg("doubleloop")=true,
            boost::python::arg("clusters")=INFERENCE::MIN,
            boost::python::arg("loopdepth")=3,
            boost::python::arg("init")=INFERENCE::UNIFORM,
            boost::python::arg("steps")=10000,
            boost::python::arg("tolerance")=1e-9,
            boost::python::arg("verbose")=0
         ) 
      )
      .def_readwrite("doubleloop",   &Parameter::doubleloop_,"use double loop?")
      .def_readwrite("clusters",  &Parameter::clusters_, 
         "cluster type:\n\n"
         "   * ``'min'``\n\n"
         "   * ``'bethe'``\n\n"
         "   * ``'delta'``\n\n"
         "   * ``'loop'``\n\n"
      )
      .def_readwrite("loopdepth", &Parameter::loopdepth_,"depth of loop")
      .def_readwrite("init", &Parameter::init_,
         "initialization type:\n\n"
         "   * ``'uniform'``\n\n"
         "   * ``'random'``\n\n"
      )
      .def_readwrite("steps",   &Parameter::maxiter_,"message passing steps")
      .def_readwrite("tolerance",  &Parameter::tolerance_, "tolerance (termination criterion)")
      .def_readwrite("verbose", &Parameter::verbose_,"verbose level")  
   ; 
   }
};


template<class INFERENCE>
class InfParamExporterLibdaiGibbs{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef InfParamExporterLibdaiGibbs<INFERENCE> SelfType;


   static void set
   (
      Parameter  & p,
      const size_t maxiter=10000,
      const size_t burnin=100,
      const size_t restart=10000,
      const size_t verbose=0
   ){
      p.maxiter_=maxiter;
      p.burnin_=burnin;
      p.restart_=restart;
      p.verbose_=verbose;
   }

   void static exportInfParam(const std::string & className){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def ("set", &set, 
         (
            boost::python::arg("steps")=10000,
            boost::python::arg("burnin")=100,
            boost::python::arg("restart")=10000,
            boost::python::arg("verbose")=0
         ) 
      )
      .def_readwrite("steps",   &Parameter::maxiter_,"sampling steps")
      .def_readwrite("burnin",  &Parameter::burnin_, "burn-in steps")
      .def_readwrite("restart", &Parameter::restart_,"restart after n steps")
      .def_readwrite("verbose", &Parameter::verbose_,"verbose level")  
   ; 
   }
};

template<class INFERENCE>
class InfParamExporterLibdaiJunctionTree{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef InfParamExporterLibdaiJunctionTree<INFERENCE> SelfType;


   static void set
   (
      Parameter  & p,
      typename INFERENCE::UpdateRule updateRule,
      typename INFERENCE::Heuristic heuristic,
      const size_t verbose
   ){
      p.updateRule_=updateRule;
      p.heuristic_=heuristic;
      p.verbose_=verbose;
   }

   void static exportInfParam(const std::string & className){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def ("set", &set, 
         (
            boost::python::arg("updateRule")=INFERENCE::HUGIN,
            boost::python::arg("heuristic")=INFERENCE::MINWEIGHT,
            boost::python::arg("verbose")=0
         ) 
      )
      .def_readwrite("updateRule",  &Parameter::updateRule_,
         "update rule of junctuion tree:\n\n"
         "   * ``'hugin'``\n\n"
         "   * ``'shsh'``\n\n"
      )
      .def_readwrite("heuristic",   &Parameter::heuristic_,
         "heuristic of junction tree :\n\n"
         "   * ``'minfill'``\n\n"
         "   * ``'weightedminfill'``\n\n"
         "   * ``'minweight'``\n\n"
         "   * ``'minneighbors'``\n\n"
      )
      .def_readwrite("verbose",     &Parameter::verbose_,"verbose level")  
   ; 
   }
};


template<class INFERENCE>
class InfParamExporterLibdaiBp{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef InfParamExporterLibdaiBp<INFERENCE> SelfType;


   static void set(
      Parameter  & p,
      const size_t maxIterations,
      const double damping,
      const double tolerance,
      const typename INFERENCE::UpdateRule updateRule,
      const size_t verbose
   ){
      p.maxIterations_=maxIterations;
      p.damping_=damping;
      p.tolerance_=tolerance;
      p.updateRule_=updateRule;
      p.verbose_=verbose;
      p.logDomain_=0;
      p.updateRule_=updateRule;
   }


   void static exportInfParam(const std::string & className){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def ("set", &SelfType::set,
         (
            boost::python::arg("steps")=100,
            boost::python::arg("damping")=0.0,
            boost::python::arg("tolerance")=0.0,
            boost::python::arg("updateRule")=INFERENCE::PARALL,
            boost::python::arg("verbose")=0
         )
      )
      .def_readwrite("steps",       &Parameter::maxIterations_ ,"Number of message passing updates")
      .def_readwrite("damping",     &Parameter::damping_       ,"Damping must be in [0,1]")
      .def_readwrite("tolerance",   &Parameter::tolerance_     ,"Tolerance (termination criterion)")
      .def_readwrite("updateRule",  &Parameter::updateRule_    ,
         "message passing update rule :\n\n"
         "   * ``'parall'`` \n\n"
         "   * ``'seqfix'`` \n\n"
         "   * ``'seqrnd'`` \n\n"
         "   * ``'seqmax'`` \n\n"
      ) 
      .def_readwrite("verbose",     &Parameter::verbose_       ,"verbose level")  
   ; 
   }
};

template<class INFERENCE>
class InfParamExporterLibdaiTrwBp{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef InfParamExporterLibdaiTrwBp<INFERENCE> SelfType;


   static void set(
      Parameter  & p,
      const size_t maxIterations,
      const double damping,
      const double tolerance,
      const size_t ntrees,
      const typename INFERENCE::UpdateRule updateRule,
      const size_t verbose
   ){
      p.maxIterations_=maxIterations;
      p.damping_=damping;
      p.tolerance_=tolerance;
      p.ntrees_=ntrees;
      p.updateRule_=updateRule;
      p.verbose_=verbose;
      p.logDomain_=0;
      p.updateRule_=updateRule;
   }


   void static exportInfParam(const std::string & className){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def ("set", &SelfType::set,
         (
            boost::python::arg("steps")=100,
            boost::python::arg("damping")=0.0,
            boost::python::arg("tolerance")=0.0,
            boost::python::arg("ntrees")=0,
            boost::python::arg("updateRule")=INFERENCE::PARALL,
            boost::python::arg("verbose")=0
         )
      )
      .def_readwrite("steps",       &Parameter::maxIterations_ ,"Number of message passing updates")
      .def_readwrite("damping",     &Parameter::damping_       ,"Damping must be in [0,1]")
      .def_readwrite("tolerance",   &Parameter::tolerance_     ,"tolerance (termination criterion)")
      .def_readwrite("ntrees",      &Parameter::ntrees_        ,"number of threes")
      .def_readwrite("updateRule",  &Parameter::updateRule_    ,
         "message passing update rule :\n\n"
         "   * ``'parall'`` \n\n"
         "   * ``'seqfix'`` \n\n"
         "   * ``'seqrnd'`` \n\n"
         "   * ``'seqmax'`` \n\n"
      ) 
      .def_readwrite("verbose",     &Parameter::verbose_       ,"verbose level")  
   ; 
   }
};

// decl map
template<class SUB_INF>
class InfParamExporter<opengm::external::libdai::DecMap<SUB_INF > >
 : public  InfParamExporterLibdaiDecMap<opengm::external::libdai::DecMap<SUB_INF> > {
};



// double loop generalized bp
template<class GM,class ACC>
class InfParamExporter<opengm::external::libdai::DoubleLoopGeneralizedBP<GM,ACC> > 
 : public  InfParamExporterLibdaiDoubleLoopGBP<opengm::external::libdai::DoubleLoopGeneralizedBP<GM,ACC> > {
};
// gibbs
template<class GM,class ACC>
class InfParamExporter<opengm::external::libdai::Gibbs<GM,ACC> > 
 : public  InfParamExporterLibdaiGibbs<opengm::external::libdai::Gibbs<GM,ACC> > {
};

// junction tree
template<class GM,class ACC>
class InfParamExporter<opengm::external::libdai::JunctionTree<GM,ACC> > 
 : public  InfParamExporterLibdaiJunctionTree<opengm::external::libdai::JunctionTree<GM,ACC> > {
};

// bp
template<class GM,class ACC>
class InfParamExporter<opengm::external::libdai::Bp<GM,ACC> > 
 : public  InfParamExporterLibdaiBp<opengm::external::libdai::Bp<GM,ACC> > {
};

// fractional bp
template<class GM,class ACC>
class InfParamExporter<opengm::external::libdai::FractionalBp<GM,ACC> > 
 : public  InfParamExporterLibdaiBp<opengm::external::libdai::FractionalBp<GM,ACC> > {
};

// trwbp bp
template<class GM,class ACC>
class InfParamExporter<opengm::external::libdai::TreeReweightedBp<GM,ACC> > 
 : public  InfParamExporterLibdaiTrwBp<opengm::external::libdai::TreeReweightedBp<GM,ACC> > {
};


#endif
#endif