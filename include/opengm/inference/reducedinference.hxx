#pragma once
#ifndef OPENGM_REDUCEDINFERENCE_HXX
#define OPENGM_REDUCEDINFERENCE_HXX

#include <vector>
#include <set>
#include <map>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/datastructures/partition.hxx"

#include "opengm/inference/external/qpbo.hxx"
#include "opengm/inference/mqpbo.hxx"
#include "opengm/inference/fix-fusion/fusion-move.hpp"
#include "opengm/graphicalmodel/graphicalmodel_manipulator.hxx"

#include "opengm/utilities/modelTrees.hxx"
#include "opengm/inference/dynamicprogramming.hxx"
#include "opengm/utilities/disjoint-set.hxx"

#include "opengm/functions/view.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"

namespace opengm {
  template<class GM>
  class ReducedInferenceHelper
  {
  public:
    typedef typename GM::ValueType ValueType;
    typedef typename GM::LabelType LabelType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::OperatorType OperatorType;
    typedef DiscreteSpace<IndexType, LabelType> SpaceType;

    typedef typename meta::TypeListGenerator< ViewFixVariablesFunction<GM>, 
					      ViewFunction<GM>, 
					      ConstantFunction<ValueType, IndexType, LabelType>,
					      ExplicitFunction<ValueType, IndexType, LabelType>
					      >::type FunctionTypeList;
    typedef GraphicalModel<ValueType, OperatorType, FunctionTypeList, SpaceType> InfGmType;
  };  

  //! [class reducedinference]
  /// Reduced Inference
  /// Implementation of the reduction techniques proposed in
  /// J.H. Kappes, M. Speth, G. Reinelt, and C. Schnörr: Towards Efficient and Exact MAP-Inference for Large Scale Discrete Computer Vision Problems via Combinatorial Optimization, CVPR 2013
  ///
  /// it provides:
  /// * modelreduction by partial optimality
  /// * seperate optimization of independent subparts of the objective
  /// * preoptimization of acyclic subproblems (only second order so far)
  ///
  /// additional to the CVPR-Paper
  /// * the complete code is refactort - parts of the code are moved to graphicalmodel_manipulator.hxx
  /// * higher order models are supported 
  ///
  /// it requires:
  /// * external-qpbo
  /// * Boost for order reduction (we hope to remove this dependence soon)
  ///
  /// Parts of the original code was implemented during the bachelor thesis of Jan Kuske
  ///
  /// Corresponding author: Jörg Hendrik Kappes
  ///
  ///\ingroup inference
  template<class GM, class ACC, class INF>
  class ReducedInference : public Inference<GM, ACC>
  {
  public:
    typedef ACC AccumulationType;
    typedef GM GmType;
    typedef GM GraphicalModelType;
    typedef INF InfType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef visitors::VerboseVisitor<ReducedInference<GM, ACC, INF> > VerboseVisitorType;
    typedef visitors::EmptyVisitor<ReducedInference<GM, ACC, INF> >   EmptyVisitorType;
    typedef visitors::TimingVisitor<ReducedInference<GM, ACC, INF> >  TimingVisitorType;


    class Parameter{
    public:
      typename INF::Parameter subParameter_;
      bool Persistency_;
      bool Tentacle_;
      bool ConnectedComponents_;
      Parameter(){
        Persistency_ = false;
        Tentacle_ = false;
        ConnectedComponents_ = false;
      };
    };

    ReducedInference(const GmType&, const Parameter & = Parameter() );
    std::string name() const;
    const GmType& graphicalModel() const;
    InferenceTermination infer();
    typename GM::ValueType bound() const; 
    template<class VisitorType>
    InferenceTermination infer(VisitorType&);
    virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;
    typename GM::ValueType value() const;
    
  private:
    //typedef typename ReducedInferenceHelper<GM>::InfGmType GM2;
    //typedef external::QPBO<GM>            QPBO;
    
    //// typedef Partition<IndexType> Set;
    //typedef disjoint_set<IndexType> Set;
    //typedef opengm::DynamicProgramming<GM2,AccumulationType>  dynP;
    //typedef modelTrees<GM2> MT;
    

    const GmType& gm_; 
  
    Parameter param_;  
    ValueType bound_;
    ValueType value_;

    std::vector<LabelType> state_;  

    void getPartialOptimalityByQPBO(std::vector<LabelType>&, std::vector<bool>&);
    void getPartialOptimalityByFixsHOQPBO(std::vector<LabelType>&, std::vector<bool>&);
    void getPartialOptimalityByKovtunsMethod(std::vector<LabelType>&, std::vector<bool>&);
    void getPartialOptimalityByMQPBO(std::vector<LabelType>&, std::vector<bool>&);
    void getPartialOptimalityByAutoSelection(std::vector<LabelType>&, std::vector<bool>&);
    void setPartialOptimality(std::vector<LabelType>&, std::vector<bool>&);

    void subinf(const typename ReducedInferenceHelper<GM>::InfGmType&,const bool,std::vector<LabelType>&, typename GM::ValueType&, typename GM::ValueType&);
  
    //std::vector<bool> variableOpt_;
    //std::vector<bool> factorOpt_;
    //ValueType const_;
    //std::vector<IndexType>  model2gm_;
    
    //void updateFactorOpt(std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >&);
    //void getConnectComp(std::vector< std::vector<IndexType> >&, std::vector<GM2>&, std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >&, bool );
    //void getTentacle(std::vector< std::vector<IndexType> >&, std::vector<IndexType>&, std::vector< std::vector<ValueType> >&, std::vector< std::vector<std::vector<LabelType> > >&, std::vector< std::vector<IndexType> >&, std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >& );
    //void getRoots(std::vector< std::vector<IndexType> >&, std::vector<IndexType>&  );
    //void getTentacleCC(std::vector< std::vector<IndexType> >&, std::vector<IndexType>&, std::vector< std::vector<ValueType> >&, std::vector< std::vector<std::vector<LabelType> > >&, std::vector< std::vector<IndexType> >&, std::vector<GM2>&, GM2&);

  };
  //! [class reducedinference]


  template<class GM, class ACC, class INF>
  ReducedInference<GM,ACC,INF>::ReducedInference
  (
  const GmType& gm,
  const Parameter& parameter
  )
  :  gm_( gm ),
     param_(parameter)
  {     
 
    ACC::ineutral(bound_);
    OperatorType::neutral(value_);
    state_.resize(gm.numberOfVariables(),0);

    //variableOpt_.resize(gm_.numberOfVariables(),false);
    //factorOpt_.resize(gm.numberOfFactors(),false);
    //const_ = 0;
  }

  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getPartialOptimalityByAutoSelection(std::vector<LabelType>& arg, std::vector<bool>& opt)
  {
    bool      binary  = true;
    bool      potts   = true;
    IndexType order   = 0;

    for(IndexType j = 0; j < gm_.numberOfVariables(); ++j) {
      if(gm_.numberOfLabels(j) != 2) {
        binary = false;
      }
    }
    
    for(IndexType j = 0; j < gm_.numberOfFactors(); ++j) {
      if(potts && gm_[j].numberOfVariables() >1 && (gm_[j].numberOfVariables() > 3 || !gm_[j].isPotts() ) )
	potts=false;
      if(gm_[j].numberOfVariables() > order) {
	order = gm_[j].numberOfVariables();
      }
    }
    
    if(binary){
      if(order<=2)
	getPartialOptimalityByQPBO(arg,opt);
      else
	getPartialOptimalityByFixsHOQPBO(arg,opt);
    }
    else{
      if(potts)
	getPartialOptimalityByKovtunsMethod(arg,opt);
      else if(order<=2)
	getPartialOptimalityByMQPBO(arg,opt);
      else
        throw RuntimeError("This implementation of Reduced Inference supports no higher order multi-label problems.");
    }
  }
  
  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getPartialOptimalityByQPBO(std::vector<LabelType>& arg, std::vector<bool>& opt)
  {   
    typedef external::QPBO<GM> QPBO;
    typename QPBO::Parameter paraQPBO;
    paraQPBO.strongPersistency_=false;         
    QPBO qpbo(gm_,paraQPBO);
    qpbo.infer();
    qpbo.arg(arg);
    qpbo.partialOptimality(opt); 
    bound_=qpbo.bound();
  }  
  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getPartialOptimalityByFixsHOQPBO(std::vector<LabelType>& arg, std::vector<bool>& opt)
  {   
    const size_t maxOrder = 10;
    ValueType constV = 0;
    HigherOrderEnergy<ValueType, maxOrder> hoe;
    hoe.AddVars(gm_.numberOfVariables());
    for(IndexType f=0; f<gm_.numberOfFactors(); ++f){
      IndexType size = gm_[f].numberOfVariables();
      const LabelType l0 = 0;
      const LabelType l1 = 1;
      if (size == 0) {
	constV += gm_[f](&l0);
	continue;
      } else if (size == 1) {
	IndexType var = gm_[f].variableIndex(0);
	const ValueType e0 = gm_[f](&l0);
	const ValueType e1 = gm_[f](&l1);
	hoe.AddUnaryTerm(var, e1 - e0);
      } else {
	unsigned int numAssignments = 1 << size;
	ValueType coeffs[numAssignments];
	for (unsigned int subset = 1; subset < numAssignments; ++subset) {
	  coeffs[subset] = 0;
	}
	// For each boolean assignment, get the clique energy at the 
	// corresponding labeling
	LabelType cliqueLabels[size];
	for(unsigned int assignment = 0;  assignment < numAssignments; ++assignment){
	  for (unsigned int i = 0; i < size; ++i) {
	    if (assignment & (1 << i)) { 
	      cliqueLabels[i] = l1;
	    } else {
	      cliqueLabels[i] = l0;
	    }
	  }
	  ValueType energy = gm_[f](cliqueLabels);
	  for (unsigned int subset = 1; subset < numAssignments; ++subset){
	    if (assignment & ~subset) {
	      continue;
	    } else {
	      int parity = 0;
	      for (unsigned int b = 0; b < size; ++b) {
		parity ^=  (((assignment ^ subset) & (1 << b)) != 0);
	      }
	      coeffs[subset] += parity ? -energy : energy;
	    }
	  }
	}
	typename HigherOrderEnergy<ValueType, maxOrder> ::VarId vars[maxOrder];
	for (unsigned int subset = 1; subset < numAssignments; ++subset) {
	  int degree = 0;
	  for (unsigned int b = 0; b < size; ++b) {
	    if (subset & (1 << b)) {
	      vars[degree++] = gm_[f].variableIndex(b);
	    }
	  }
	  std::sort(vars, vars+degree);
	  hoe.AddTerm(coeffs[subset], degree, vars);
	}
      }
    }  
    kolmogorov::qpbo::QPBO<ValueType>  qr(gm_.numberOfVariables(), 0); 
    hoe.ToQuadratic(qr);
    qr.Solve();
  
    for (IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
      int label = qr.GetLabel(i);
      if(label == 0 ){
	arg[i] = 0;
	opt[i] = true;
      }
      else if(label == 1){
	arg[i] = 1;
	opt[i] = true;
      } 
      else{
	arg[i] = 0;
	opt[i] = false;
      }
    }  
    bound_ = constV + 0.5 * qr.ComputeTwiceLowerBound();
  }
  
  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getPartialOptimalityByMQPBO(std::vector<LabelType>& arg, std::vector<bool>& opt)
  { 
    typedef opengm::MQPBO<GM,ACC> MQPBOType;
    typename MQPBOType::Parameter mqpboPara; 
    mqpboPara.useKovtunsMethod_  = false;
    mqpboPara.strongPersistency_ = true;
    mqpboPara.rounds_            = 10;
    mqpboPara.permutationType_   = MQPBOType::RANDOM; 
    MQPBOType mqpbo(gm_,mqpboPara);
    mqpbo.infer();
    arg.resize(gm_.numberOfVariables(),0);
    opt.resize(gm_.numberOfVariables(),false);
    for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
      opt[var] = mqpbo.partialOptimality(var,arg[var]);
    }
  }
  
  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getPartialOptimalityByKovtunsMethod(std::vector<LabelType>& arg, std::vector<bool>& opt)
  { 
    typedef opengm::MQPBO<GM,ACC> MQPBOType;
    typename MQPBOType::Parameter mqpboPara;
    mqpboPara.strongPersistency_ = true;   
    MQPBOType mqpbo(gm_,mqpboPara);
    mqpbo.infer(); 
    arg.resize(gm_.numberOfVariables(),0);
    opt.resize(gm_.numberOfVariables(),false);
    for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
      opt[var] = mqpbo.partialOptimality(var,arg[var]);
    }
  }


  template<class GM, class ACC, class INF>
  inline std::string
  ReducedInference<GM,ACC,INF>::name() const
  {
    return "ReducedInference";
  }

  template<class GM, class ACC, class INF>
  inline const typename ReducedInference<GM,ACC,INF>::GmType&
  ReducedInference<GM,ACC,INF>::graphicalModel() const
  {
    return gm_;
  }
  
  template<class GM, class ACC, class INF>
  inline InferenceTermination
  ReducedInference<GM,ACC,INF>::infer()
  {  
    EmptyVisitorType v;
    return infer(v);
  }

  
  template<class GM, class ACC, class INF>
  template<class VisitorType>
  InferenceTermination ReducedInference<GM,ACC,INF>::infer
  (
  VisitorType& visitor
  )
  { 
    visitor.begin(*this);
    
    GraphicalModelManipulator<GM> gmm(gm_);
 
    // Find persistency
    size_t numFixedVars = 0;
    if(param_.Persistency_ == true){
       std::vector<bool>      opt(gm_.numberOfVariables(),false);
       std::vector<LabelType> arg(gm_.numberOfVariables(),0);
       getPartialOptimalityByAutoSelection(arg,opt);
       for(IndexType i=0; i<gm_.numberOfVariables(); ++i){
          if(opt[i]){
             ++numFixedVars;
             gmm.fixVariable(i, arg[i]); 
          }
       }
    } 
    
    //std::cout << numFixedVars <<" of " <<gm_.numberOfVariables() << " are fixed."<<std::endl;
  
    if(numFixedVars == gm_.numberOfVariables()){
       gmm.lock();
       std::vector<LabelType> arg(0);
       gmm.modifiedState2OriginalState(arg, state_);
       bound_ = value();
       //visitor(*this);
       visitor.end(*this);
       return NORMAL;
    }
  
    if(param_.Tentacle_ == true){
       //std::cout << " Search for tentacles." <<std::endl;
       gmm.template lockAndTentacelElimination<ACC>();
    }
    else{
       gmm.lock();
    } 

    if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) {
       visitor.end(*this);
       return NORMAL;
    }


    //ValueType sv, v;
    ValueType sb, b, v;
    OperatorType::neutral(sb);
    //OperatorType::neutral(sv);   

    // CONNTECTED COMPONENTS INFERENCE
    if(param_.ConnectedComponents_ == true){
      gmm.buildModifiedSubModels();
      std::vector<std::vector<LabelType> > args(gmm.numberOfSubmodels(),std::vector<LabelType>() );
      for(size_t i=0; i<gmm.numberOfSubmodels(); ++i){
         args[i].resize(gmm.getModifiedSubModel(i).numberOfVariables());
      } 
      for(size_t i=0; i<gmm.numberOfSubmodels(); ++i){
         typename ReducedInferenceHelper<GM>::InfGmType agm = gmm.getModifiedSubModel(i);
         subinf(agm, param_.Tentacle_, args[i],v,b);
         //OperatorType::op(v,sv);
         OperatorType::op(b,sb);
         //gmm.modifiedSubStates2OriginalState(args, state_);
         //visitor(*this,value(),bound(),"numberOfComp",i);
         if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) {
            visitor.end(*this);
            return NORMAL;
         }
      }
      bound_= sb;
      gmm.modifiedSubStates2OriginalState(args, state_);
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) {
         visitor.end(*this);
         return NORMAL;
      }
      //gmm.modifiedSubStates2OriginalState(args, state_);
    
    }
    else{
       //size_t i=0;
      std::vector<LabelType> arg;
      gmm.buildModifiedModel();
      typename ReducedInferenceHelper<GM>::InfGmType agm =  gmm.getModifiedModel();
      subinf(agm, param_.Tentacle_, arg,v,b);
      gmm.modifiedState2OriginalState(arg, state_); 
      //visitor(*this,value(),bound(),"numberOfComp",i);
      //gmm.modifiedState2OriginalState(arg, state_); 
      bound_=b;
    }
    //value_=gm_.evaluate(state_);
    visitor.end(*this);
    return NORMAL;
  }


  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::subinf
  (
   const typename ReducedInferenceHelper<GM>::InfGmType& agm,
   const bool tentacleElimination,
   std::vector<LabelType>& arg,
   typename GM::ValueType& value,
   typename GM::ValueType& bound
   )
  {
     //std::cout << "solve model with "<<agm.numberOfVariables()<<" and "<<agm.numberOfFactors()<<" factors."<<std::endl; 
     InfType inf(agm, param_.subParameter_);
     inf.infer();
     arg.resize(agm.numberOfVariables());
     inf.arg(arg);   
     value = inf.value();
     bound = inf.bound();
  }


  template<class GM, class ACC, class INF>
  typename GM::ValueType ReducedInference<GM,ACC,INF>::bound() const {
    return bound_;
  }

  template<class GM, class ACC, class INF>
  typename GM::ValueType ReducedInference<GM,ACC,INF>::value() const { 
    return gm_.evaluate(state_);
  }

  template<class GM, class ACC, class INF>
  inline InferenceTermination
  ReducedInference<GM,ACC,INF>::arg
  (
  std::vector<LabelType>& x,
  const size_t N
  ) const
  {
    if(N==1){
      x.resize(gm_.numberOfVariables());
      for(size_t i=0;  i<x.size(); ++i){
        x[i] = state_[i];
      }
      return NORMAL;
    }
    else {
      return UNKNOWN;
    }
  }
} // namespace opengm

#endif // #ifndef OPENGM_ReducedInference_HXX

