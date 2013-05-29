#pragma once
#ifndef OPENGM_REDUCEDINFERENCE_HXX
#define OPENGM_REDUCEDINFERENCE_HXX

#include <vector>
#include <set>
#include <map>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/datastructures/partition.hxx"

#include "opengm/inference/external/qpbo.hxx"
#include "opengm/inference/mqpbo.hxx"

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

    typedef typename meta::TypeListGenerator<
    opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
       opengm::ViewFunction<GM>
    >::type FunctionTypeList;

    typedef opengm::GraphicalModel<
    ValueType,
    OperatorType,
    FunctionTypeList,
    SpaceType
    > InfGmType;
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
  /// it requires:
  /// * external-qpbo
  ///
  /// Parts of this code was implemented during the bachelor thesis of Jan Kuske
  /// TODO: Code is still not optimized and has to be refactored!
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
    typedef VerboseVisitor<ReducedInference<GM,ACC,INF> > VerboseVisitorType;
    typedef EmptyVisitor<ReducedInference<GM,ACC,INF> > EmptyVisitorType; 
    typedef TimingVisitor<ReducedInference<GM,ACC,INF> > TimingVisitorType; 
    typedef typename ReducedInferenceHelper<GM>::InfGmType GM2;
    typedef external::QPBO<GM>            QPBO;
    
    // typedef Partition<IndexType> Set;
    typedef disjoint_set<IndexType> Set;
    typedef opengm::DynamicProgramming<GM2,AccumulationType>  dynP;
    typedef modelTrees<GM2> MT;
    
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
    const GmType& gm_;
    Parameter param_;
    
    ValueType bound_;
    ValueType value_;
    std::vector<LabelType> states_;   
    std::vector<bool> variableOpt_;
    std::vector<bool> factorOpt_;
    bool binaryModel_;
    ValueType const_;
    std::vector<IndexType>  model2gm_;
    
    void updateFactorOpt(std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >&);
    void getConnectComp(std::vector< std::vector<IndexType> >&, std::vector<GM2>&, std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >&, bool );
    void getTentacle(std::vector< std::vector<IndexType> >&, std::vector<IndexType>&, std::vector< std::vector<ValueType> >&, std::vector< std::vector<std::vector<LabelType> > >&, std::vector< std::vector<IndexType> >&, std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >& );
    void getRoots(std::vector< std::vector<IndexType> >&, std::vector<IndexType>&  );
    void getTentacleCC(std::vector< std::vector<IndexType> >&, std::vector<IndexType>&, std::vector< std::vector<ValueType> >&, std::vector< std::vector<std::vector<LabelType> > >&, std::vector< std::vector<IndexType> >&,
    std::vector<GM2>&, GM2&);
    
  };
  //! [class reducedinference]


  template<class GM, class ACC, class INF>
  ReducedInference<GM,ACC,INF>::ReducedInference
  (
  const GmType& gm,
  const Parameter& parameter
  )
  :  gm_(gm),    
  param_(parameter),
  binaryModel_(true)
  {

    for(size_t j = 0; j < gm_.numberOfVariables(); ++j) {
      if(gm_.numberOfLabels(j) != 2) {
        binaryModel_=false;
        //throw RuntimeError("Reduced Inference supports only binary variables.");
      }
    }
    for(size_t j = 0; j < gm_.numberOfFactors(); ++j) {
      if(gm_[j].numberOfVariables() > 2) {
        throw RuntimeError("This implementation of Reduced Inference supports only factors of order <= 2.");
      }
    }
    
    ACC::ineutral(bound_);
    value_ = 0;
    states_.resize(gm.numberOfVariables(),0);
    variableOpt_.resize(gm_.numberOfVariables(),false);
    factorOpt_.resize(gm.numberOfFactors(),false);
    const_ = 0;
    
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
    
    IndexType numVarQPBO = gm_.numberOfVariables();
    std::vector<std::set<IndexType> > neighbors;
    std::vector<ExplicitFunction<ValueType,IndexType,LabelType> > unaryFunctions(gm_.numberOfVariables());
    for(IndexType i=0; i<gm_.numberOfVariables(); ++i){
      LabelType numLabels = gm_.numberOfLabels(i);
      unaryFunctions[i] = ExplicitFunction<ValueType,IndexType,LabelType>(&numLabels,&(numLabels)+1);
    }
    
    
    visitor.begin(*this);
    
    // QPBO
    if(param_.Persistency_ == true){
      
      if(binaryModel_){
        typename QPBO::Parameter paraQPBO;
        paraQPBO.strongPersistency_=false;         
        QPBO qpbo(gm_,paraQPBO);
        qpbo.infer();
        qpbo.arg(states_);
        qpbo.partialOptimality(variableOpt_); 
        bound_=qpbo.bound();
      }
      else{//Multilabel
        typedef opengm::MQPBO<GM,ACC> MQPBOType;
        typename MQPBOType::Parameter mqpboPara;   
        MQPBOType mqpbo(gm_,mqpboPara);
        mqpbo.infer();
        for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
          variableOpt_[var] = mqpbo.partialOptimality(var,states_[var]);
        }
      }
      for(IndexType i = 0 ; i < gm_.numberOfVariables() ; ++i){
        if(variableOpt_[i] == true){
          numVarQPBO -= 1;
        }
      }
    }
    
    updateFactorOpt(unaryFunctions);
    
    visitor(*this,value(),bound(),"partOpt",1.0 - (1.0*numVarQPBO)/gm_.numberOfVariables(),"remainVars",(double)(numVarQPBO));
    //visitor.visit(*this,"partOpt",1.0 - (1.0*numVarQPBO)/gm_.numberOfVariables());
    //visitor.visit(*this,"remainVars",numVarQPBO);
    
    // std::cout << "remainVars: " << numVarQPBO << std::endl;
    
    // CONNTECTED COMPONENTS
    std::vector<GM2> GraphModels;
    std::vector< std::vector<IndexType> > cc2gm;
    if(param_.ConnectedComponents_ == true){
      getConnectComp(cc2gm, GraphModels, unaryFunctions);
    }
    else if(param_.Tentacle_ == false){
      getConnectComp(cc2gm, GraphModels, unaryFunctions,true);
    }
    
    visitor(*this,value(),bound(),"numberOfComp",GraphModels.size());
    
    
    // TENTACLE
    std::vector< std::vector<IndexType> > tree2gm;
    std::vector<IndexType> roots;
    std::vector< std::vector<ValueType> > values;
    std::vector< std::vector<std::vector<LabelType> > > substates;
    std::vector< std::vector<IndexType> > nodes;
    
    if(param_.Tentacle_ == true && param_.ConnectedComponents_ == false){
      getRoots(tree2gm, roots);
      getTentacle(tree2gm, roots, values, substates, nodes, unaryFunctions);
      getConnectComp(cc2gm, GraphModels, unaryFunctions,true);
      visitor(*this,value(),bound(),"numberOfTrees",tree2gm.size());
    }
    else if(param_.Tentacle_ == true && param_.ConnectedComponents_ == true){
      getConnectComp(cc2gm, GraphModels, unaryFunctions);
    }
    
    
    // INFERECNE 
    
    for(IndexType graph = 0 ; graph < GraphModels.size() ; ++graph){
      
      if(param_.Tentacle_ == true && param_.ConnectedComponents_ == true){
     
        std::vector<GM2> model;  
        size_t ccelements =cc2gm[graph].size();

        getTentacleCC(tree2gm, roots, values, substates, nodes, model, GraphModels[graph]);
        

        visitor(*this,value(),bound(),"CCElements",ccelements,"numberOfTrees",tree2gm.size(),"withoutTrees",model2gm_.size());

        //visitor.visit(*this,"numberOfTrees",tree2gm.size());
        //visitor.visit(*this,"withoutTrees",model2gm_.size());
        
        InfType inf(model[0], param_.subParameter_);
        //std::cout << "Infer..." << std::endl;
        inf.infer();
        std::vector<LabelType > x(model[0].numberOfVariables());
        inf.arg(x);
        for(IndexType var = 0 ; var < x.size() ; ++var){
          IndexType varCC = model2gm_[var];
          states_[cc2gm[graph][varCC]] = x[var];
        }
        value_ += inf.value();
        
        for(IndexType r = 0 ; r < roots.size() ; ++r ){
          IndexType root = roots[r];
          LabelType rootLabel = states_[cc2gm[graph][root]];
          
          for(IndexType i = 0 ; i < substates[r].size() ; ++i){
            for(IndexType j = 0 ; j < substates[r][i].size() ; ++j){
            }
          }
          
          for(IndexType node = 0 ; node < nodes[r].size() ; ++node){
            IndexType treeNode = nodes[r][node];
            LabelType nodeState = substates[r][rootLabel][node];
            states_[ cc2gm[graph][ tree2gm[r][treeNode] ] ] = nodeState;
          }
        }
        
         
        
      }
      else{
        
        visitor(*this,value(),bound(),"CCElements",cc2gm[graph].size());
        
        InfType inf(GraphModels[graph],param_.subParameter_);
        //std::cout << "Infer..." << std::endl;
        inf.infer();
        std::vector<LabelType > x(GraphModels[graph].numberOfVariables());
        inf.arg(x);
        for(IndexType var = 0 ; var < x.size() ; ++var){
          states_[cc2gm[graph][var]] = x[var];
        }
        value_ += inf.value();
        
        for(IndexType r = 0 ; r < roots.size() ; ++r ){
          IndexType root = roots[r];
          LabelType rootLabel = states_[root];
          for(IndexType node = 0 ; node < nodes[r].size() ; ++node){
            IndexType treeNode = nodes[r][node];
            LabelType nodeState = substates[r][rootLabel][node];
            states_[tree2gm[r][treeNode]] = nodeState;
          }
        }
        
        
        
      }
      
    }
    
    
    value_ += const_;
    
    visitor.end(*this);
    
    // for(IndexType i = 0 ; i < states_.size() ; ++i){
    // std::cout << "var: " << i << " <-- " << states_[i] <<std::endl;
    // }
    // std::cout << std::endl;
    
    return NORMAL;

  }

  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::updateFactorOpt(std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >& unaryFunc){
    
    const LabelType l0 = 0;
    // std::cout << " Faktoren:  " << std::endl;
    for(IndexType factor=0 ; factor < gm_.numberOfFactors() ; ++factor){
      
      // if(factorOpt_[factor] == false){
      
      if(gm_[factor].numberOfVariables() == 0){
        const_ +=  gm_[factor](&l0);
        // factorOpt_[factor] == true;
      }
      else if(gm_[factor].numberOfVariables() == 1){
        
        IndexType var = gm_.variableOfFactor(factor,0);

        if(variableOpt_[var] == true){
          const_ += gm_[factor](&states_[var]);
        }
        else{
          LabelType labels = gm_.numberOfLabels(var);
          for(LabelType l = 0 ; l < labels ; ++l){
            unaryFunc[var](&l) += gm_[factor](&l);
          }
        }
      }
      else if(gm_[factor].numberOfVariables() == 2){
        IndexType var1 = gm_.variableOfFactor(factor,0);
        IndexType var2 = gm_.variableOfFactor(factor,1);
        if(variableOpt_[var1] == true && variableOpt_[var2] == true){
          LabelType Label[] = {states_[var1],states_[var2]};
          const_ += gm_[factor](Label);
          // factorOpt_[factor] == true;
        }
        else if(variableOpt_[var1] == true || variableOpt_[var2] == true){

          if(variableOpt_[var1] == true){
            LabelType Label[] = {states_[var1],0};
            LabelType Labels = gm_.numberOfLabels(var2);
            for(LabelType l = 0 ; l < Labels ; ++l){
              Label[1]=l;
              unaryFunc[var2](&l) += gm_[factor](Label);
            }
          }
          else{
            LabelType Label[] = {0,states_[var2]};
            LabelType Labels = gm_.numberOfLabels(var1);
            for(LabelType l = 0 ; l < Labels ; ++l){
              Label[0]=l;
              unaryFunc[var1](&l) += gm_[factor](Label);
            }
          }
          // factorOpt_[factor] == true;
        }
      }
      else{
        throw RuntimeError("This implementation of Reduced Inference supports only factors of order <= 2.");
      }
      // }
    }
    // std::cout <<  std::endl;
  }

  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getTentacleCC(
  std::vector< std::vector<IndexType> >& tree2gm, 
  std::vector<IndexType>& roots, 
  std::vector< std::vector<ValueType> >& values, 
  std::vector< std::vector<std::vector<LabelType> > >& substates, 
  std::vector< std::vector<IndexType> >& nodes,
  std::vector<GM2>& model, 
  GM2& gm
  )
  {
    
    roots.clear();
    tree2gm.clear();
    values.clear();
    substates.clear();
    nodes.clear();
    model.clear();
    
    MT getTrees(gm);
    getTrees.roots(roots);
    getTrees.nodes(tree2gm);
    std::vector<bool> opt(gm.numberOfVariables(),false);
    
    //FIND TENTACLE FACTORS
    std::vector< std::set<IndexType> > ttFactors(tree2gm.size());
    std::vector<IndexType> gm2treeIDX(gm.numberOfVariables(),0);
    // std::vector< IndexType > treeCount(tree2gm.size(),0);
    for(IndexType tt = 0 ; tt < tree2gm.size() ; ++tt){
      for(IndexType var = 0 ; var < tree2gm[tt].size() ; ++var){
        gm2treeIDX[tree2gm[tt][var]] = var;
        for(IndexType fkt = 0 ; fkt < gm.numberOfFactors(tree2gm[tt][var]) ; ++fkt){
          IndexType factor = gm.factorOfVariable(tree2gm[tt][var],fkt);
          if(gm[factor].numberOfVariables() == 1){
            ttFactors[tt].insert(factor);
          }
          else if(gm[factor].numberOfVariables() == 2 && tree2gm[tt][var] != roots[tt]){
            ttFactors[tt].insert(factor);
          }
        }
      }
    }
    
    
    //BUILD TENTACLE
    
    std::vector<GM2> Tentacle(tree2gm.size());
    typename std::set<typename GM2::IndexType>::iterator it;
    
    for(IndexType i=0;i<tree2gm.size();++i){
      LabelType StateSpace[tree2gm[i].size()];
      for(IndexType j=0;j<tree2gm[i].size();++j){
        LabelType label=gm.numberOfLabels(tree2gm[i][j]);
        StateSpace[j]=label;
      }
      
      GM2 gmV(typename GM2::SpaceType(StateSpace,StateSpace+tree2gm[i].size()));
      
      for(it=ttFactors[i].begin();it!=ttFactors[i].end();it++){
        // if(gm.numberOfVariables(*it) == 2){
        
        IndexType var[gm.numberOfVariables(*it)];
        std::vector<LabelType> shape;
        for(IndexType l=0;l<gm.numberOfVariables(*it);++l){
          IndexType idx=gm.variableOfFactor(*it,l);
          shape.push_back(gm.numberOfLabels(idx));
          var[l]=gm2treeIDX[idx];
        }
        // ViewFunction<GM2> func(gm[*it]);
        opengm::ExplicitFunction<ValueType,IndexType,LabelType> func(shape.begin(),  shape.end());
        
        if(gm.numberOfVariables(*it) == 1){
          LabelType labels = shape[0];
          for(LabelType label = 0 ; label < labels ; ++label ){
            func(&label) = gm[*it](&label);
          }
          IndexType v = gm.variableOfFactor(*it,0);
          if(v != roots[i] ){
            opt[v] = true;
            // std::cout << "opt TRUE: " << v << std::endl;
          }
          
        }
        else if(gm.numberOfVariables(*it) == 2){
          LabelType labels = shape[0];
          LabelType label[] = {0,0};
          for(label[0] = 0 ; label[0] < labels ; ++label[0]){
            for(label[1] = 0 ; label[1] < labels ; ++label[1]){
              func(label) = gm[*it](label);
            }
          }
        }        
        else{
          throw RuntimeError("This implementation of Reduced Inference supports only factors of order <= 2.");
        }
        
        gmV.addFactor(gmV.addFunction(func),var,var + gm.numberOfVariables(*it));
        // }
        // else{
        // IndexType idx=gm.variableOfFactor(*it,0);
        // IndexType var[]={gm2treeIDX[idx]};          
        // gmV.addFactor(gmV.addFunction(unaryFunc[idx]),var,var + 1);
        // }
      }
      Tentacle[i]=gmV;
    }
    //Dynamic Programming
    values.resize(tree2gm.size());
    substates.resize(tree2gm.size());
    nodes.resize(tree2gm.size());
    
    for(IndexType m = 0 ; m < Tentacle.size() ; ++m){
      typename dynP::Parameter para_dynp;
      std::vector<IndexType> r;
      r.push_back(gm2treeIDX[roots[m]]);
      para_dynp.roots_=r;
      dynP dynp(Tentacle[m],para_dynp);
      dynp.infer();
      
      dynp.getNodeInfo(gm2treeIDX[roots[m]], values[m] ,substates[m] ,nodes[m] );
      
    }
    
    // for(IndexType r = 0 ; r < roots.size() ; ++r){
    // IndexType var = roots[r];
    // IndexType Labels = gm.numberOfLabels(var);
    // for(LabelType l = 0 ; l < Labels ; ++l){
    // unaryFunc[var](&l) = values[r][l];
    // }
    // }
    
// std::cout << "numberOfVars: " << gm.numberOfVariables() << std::endl;

    
    //BUILD MODEL
    // std::vector<IndexType>  model2gm_;
    model2gm_.clear();
    std::vector<IndexType>  gm2model(gm.numberOfVariables());
    std::set<IndexType> setFactors;
    IndexType modelCount = 0;
    for(IndexType var = 0 ; var < gm.numberOfVariables() ; ++var){
      if(opt[var] == false){
        model2gm_.push_back(var);
        gm2model[var] = modelCount;
        // std::cout << var<< " -->  " << modelCount << std::endl;
        modelCount++;
        for(IndexType fkt = 0 ; fkt < gm.numberOfFactors(var) ; ++fkt){
          IndexType factor = gm.factorOfVariable(var,fkt);
          if(gm[factor].numberOfVariables() == 1){
            setFactors.insert(factor);
          }
          else if(gm[factor].numberOfVariables() == 2){
            IndexType var1 = gm.variableOfFactor(factor,0);
            IndexType var2 = gm.variableOfFactor(factor,1);
            if(opt[var1] == false && opt[var2] == false){
              // std::cout << "Factor: " << factor << std::endl;
              // std::cout << "var1: " << var1 << "  var2: " << var2 << std::endl;
              setFactors.insert(factor);
            }
          }
        }
      }
    }
    
    // std::cout << "numberOfVarsR: " << model2gm_.size() << std::endl;
    
    LabelType StateSpace[model2gm_.size()];
    for(IndexType j=0;j<model2gm_.size();++j){
      LabelType label=gm.numberOfLabels(model2gm_[j]);
      StateSpace[j]=label;
    }
    
    GM2 gmV(typename GM2::SpaceType(StateSpace,StateSpace+model2gm_.size()));
    
    for(it=setFactors.begin();it!=setFactors.end();it++){
      // if(gm.numberOfVariables(*it) == 2){
      
      IndexType var[gm.numberOfVariables(*it)];
      std::vector<LabelType> shape;
      for(IndexType l=0;l<gm.numberOfVariables(*it);++l){
        IndexType idx=gm.variableOfFactor(*it,l);
        shape.push_back(gm.numberOfLabels(idx));
        var[l]=gm2model[idx];
        // std::cout << var[l] << "   " << gm2model[idx] << "  " << *it << std::endl;
      }
      // ViewFunction<GM2> func(gm[*it]);
      opengm::ExplicitFunction<ValueType,IndexType,LabelType> func(shape.begin(),  shape.end());
      
      if(gm.numberOfVariables(*it) == 1){
        
        IndexType v = gm.variableOfFactor(*it,0);
        if(getTrees.treeOfVariable(v) == v){
          LabelType labels = shape[0];
          for(LabelType label = 0 ; label < labels ; ++label ){
            func(&label) = values[getTrees.treeOfRoot(v)][label];
          }
        }
        else{
          LabelType labels = shape[0];
          for(LabelType label = 0 ; label < labels ; ++label ){
            func(&label) = gm[*it](&label);
          }
        }
        
      }
      else if(gm.numberOfVariables(*it) == 2){
        LabelType labels = shape[0];
        LabelType label[] = {0,0};
        for(label[0] = 0 ; label[0] < labels ; ++label[0]){
          for(label[1] = 0 ; label[1] < labels ; ++label[1]){
            func(label) = gm[*it](label);
          }
        }
      }        
      else{
        throw RuntimeError("This implementation of Reduced Inference supports only factors of order <= 2.");
      }
 
      gmV.addFactor(gmV.addFunction(func),var,var + gm.numberOfVariables(*it));
    }
    
    model.push_back(gmV);
    
    
  }
  
  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getTentacle(
  std::vector< std::vector<IndexType> >& tree2gm,
  std::vector<IndexType>& roots,
  std::vector< std::vector<ValueType> >& values, 
  std::vector< std::vector<std::vector<LabelType> > >& substates, 
  std::vector< std::vector<IndexType> >& nodes,
  std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >& unaryFunc
  )
  {
    //FIND TENTACLE FACTORS
    std::vector< std::set<IndexType> > ttFactors(tree2gm.size());
    std::vector<IndexType> gm2treeIDX(gm_.numberOfVariables(),0);
    // std::vector< IndexType > treeCount(tree2gm.size(),0);
    for(IndexType tt = 0 ; tt < tree2gm.size() ; ++tt){
      for(IndexType var = 0 ; var < tree2gm[tt].size() ; ++var){
        gm2treeIDX[tree2gm[tt][var]] = var;
        for(IndexType fkt = 0 ; fkt < gm_.numberOfFactors(tree2gm[tt][var]) ; ++fkt){
          IndexType factor = gm_.factorOfVariable(tree2gm[tt][var],fkt);
          if(gm_[factor].numberOfVariables() == 1){
            ttFactors[tt].insert(factor);
          }
          else if(gm_[factor].numberOfVariables() == 2 && tree2gm[tt][var] != roots[tt]){
            IndexType var1 = gm_[factor].variableIndex(0);
            IndexType var2 = gm_[factor].variableIndex(1);
            if(variableOpt_[var1] == false && variableOpt_[var2] == false){
              ttFactors[tt].insert(factor);
            }
          }
        }
      }
    }
    //BUILD TENTACLE
    
    std::vector<GM2> Tentacle(tree2gm.size());
    typename std::set<typename GM2::IndexType>::iterator it;
    
    for(IndexType i=0;i<tree2gm.size();++i){
      LabelType StateSpace[tree2gm[i].size()];
      for(IndexType j=0;j<tree2gm[i].size();++j){
        LabelType label=gm_.numberOfLabels(tree2gm[i][j]);
        StateSpace[j]=label;
      }
      
      GM2 gmV(typename GM2::SpaceType(StateSpace,StateSpace+tree2gm[i].size()));
      
      for(it=ttFactors[i].begin();it!=ttFactors[i].end();it++){
        if(gm_.numberOfVariables(*it) == 2){
          
          IndexType var[gm_.numberOfVariables(*it)];
          for(IndexType l=0;l<gm_.numberOfVariables(*it);++l){
            IndexType idx=gm_.variableOfFactor(*it,l);
            var[l]=gm2treeIDX[idx];
            
          }
          ViewFunction<GM> func(gm_[*it]);
          gmV.addFactor(gmV.addFunction(func),var,var + gm_.numberOfVariables(*it));
        }
        else{
          IndexType idx=gm_.variableOfFactor(*it,0);
          if(idx != roots[i]){
            variableOpt_[idx] = true;
          }
          IndexType var[]={gm2treeIDX[idx]};          
          gmV.addFactor(gmV.addFunction(unaryFunc[idx]),var,var + 1);
        }
      }
      
      Tentacle[i]=gmV;
      
    }
    //Dynamic Programming
    values.resize(tree2gm.size());
    substates.resize(tree2gm.size());
    nodes.resize(tree2gm.size());
    
    for(IndexType m = 0 ; m < Tentacle.size() ; ++m){
      typename dynP::Parameter para_dynp;
      std::vector<IndexType> r;
      r.push_back(gm2treeIDX[roots[m]]);
      para_dynp.roots_=r;
      dynP dynp(Tentacle[m],para_dynp);
      dynp.infer();
      
      dynp.getNodeInfo(gm2treeIDX[roots[m]], values[m] ,substates[m] ,nodes[m] );
    }
    
    for(IndexType r = 0 ; r < roots.size() ; ++r){
      IndexType var = roots[r];
      IndexType Labels = gm_.numberOfLabels(var);
      for(LabelType l = 0 ; l < Labels ; ++l){
        unaryFunc[var](&l) = values[r][l];
      }
    }
    
  }

  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getRoots(
  std::vector< std::vector<IndexType> >& tree2gm,
  std::vector<IndexType>& roots
  )
  {
    //NEIGHBOURHOOD
    std::vector<std::set<IndexType> > neighbors(gm_.numberOfVariables());
    for(IndexType factor = 0 ; factor < gm_.numberOfFactors() ; ++factor){
      if(gm_[factor].numberOfVariables() == 2){
        IndexType var1 = gm_.variableOfFactor(factor,0);
        IndexType var2 = gm_.variableOfFactor(factor,1);
        
        if(variableOpt_[var1] == false && variableOpt_[var2] == false){
          neighbors[var1].insert(var2);          
          neighbors[var2].insert(var1);
        }
        
      }
    }
    //TREES
    std::map<IndexType, IndexType> representives;
    std::vector<IndexType> degree(gm_.numberOfVariables());
    std::vector<IndexType> leafs;
    std::vector<bool>isRoot(gm_.numberOfVariables());
    std::vector<IndexType> parents(gm_.numberOfVariables());
    typename std::set<typename GM::IndexType>::iterator it;
    typename std::set<typename GM::IndexType>::iterator fi;
    
    for(IndexType i=0;i<degree.size();++i){
      degree[i]=neighbors[i].size();
      parents[i]=gm_.numberOfVariables();
      if(degree[i]==1){
        leafs.push_back(i);
      }
    }
    while(!leafs.empty()){
      IndexType l=leafs.back();
      leafs.pop_back();
      if(degree[l]>0){
        it=neighbors[l].begin();
        isRoot[*it]=1;
        isRoot[l]=0;
        parents[l]=*it;
        parents[*it]=*it;
        degree[*it]=degree[*it]-1;
        fi=neighbors[*it].find(l);
        neighbors[*it].erase(fi);
        if(degree[*it]==1){
          leafs.push_back(*it);
        }
      }
    }
    
    IndexType numberOfRoots = 0;    
    for(IndexType i=0;i<gm_.numberOfVariables();++i){
      if(isRoot[i]==1){
        representives[i]=numberOfRoots;
        numberOfRoots++;
      }
    }
    //FILL ROOTS AND TREE2GM
    roots.resize(numberOfRoots);
    tree2gm.resize(numberOfRoots);
    // IndexType rootCount = 0;
    
    for(IndexType i=0;i<gm_.numberOfVariables();++i){
      if(parents[i] != gm_.numberOfVariables()){
        IndexType tree = i;
        while(parents[tree]!=tree){
          tree = parents[tree];
        }
        tree2gm[representives[tree]].push_back(i);
      }
      if(isRoot[i] == true){
        roots[representives[i]] = i;
        // rootCount++;
      }
    }
    
    
  }
  
  
  
  template<class GM, class ACC, class INF>
  void ReducedInference<GM,ACC,INF>::getConnectComp(
  std::vector< std::vector<IndexType> >& cc2gm, 
  std::vector<GM2>& models, std::vector<ExplicitFunction<ValueType,IndexType,LabelType> >& unaryFunc, 
  bool forceConnect = false
  )
  {
    
    models.clear();
    Set CC(gm_.numberOfVariables());
    std::map<IndexType, IndexType> representives;
    std::vector< std::vector<IndexType> > cc2gmINT;
    std::vector< IndexType > gm2ccIDX(gm_.numberOfVariables());
    
    if(forceConnect == false){    
      for(IndexType f=0 ; f < gm_.numberOfFactors() ; ++f){
        if(gm_[f].numberOfVariables() == 2){
          IndexType var1 = gm_[f].variableIndex(0);
          IndexType var2 = gm_[f].variableIndex(1);
          if(variableOpt_[var1] == false && variableOpt_[var2] == false){
            CC.join(var1,var2);
          }
        }
      }
    }
    else{
      IndexType trueVar = 0;
      while(variableOpt_[trueVar] == true){
        trueVar++;
      }
      for(IndexType v = trueVar+1 ; v < gm_.numberOfVariables() ; ++v){
        if(variableOpt_[v] == false){
          CC.join(trueVar,v);
        }
      }
    }
    
    CC.representativeLabeling(representives);
    
    std::vector<bool> isSet(CC.numberOfSets(),true);
    cc2gmINT.resize(CC.numberOfSets());
    std::vector<std::set<IndexType> > setFactors(CC.numberOfSets());
    IndexType numCC = CC.numberOfSets();
    std::vector<IndexType> IndexOfCC(CC.numberOfSets(),0);
    for(IndexType var = 0 ; var < gm_.numberOfVariables() ; ++var){
      
      IndexType n = CC.find(var);
      n = representives[n];
      if(variableOpt_[var] == false){
        cc2gmINT[n].push_back(var);
        gm2ccIDX[var]=IndexOfCC[n];
        IndexOfCC[n]++;
        
        for(IndexType i=0;i<gm_.numberOfFactors(var);++i){
          IndexType fkt = gm_.factorOfVariable(var,i);
          if(gm_[fkt].numberOfVariables() == 1){
            setFactors[n].insert(fkt);
          }
          else if(gm_[fkt].numberOfVariables() == 2){
            IndexType var1 = gm_[fkt].variableIndex(0);
            IndexType var2 = gm_[fkt].variableIndex(1);
            if(variableOpt_[var1] == false && variableOpt_[var2] == false){
              setFactors[n].insert(fkt);
            }
          }
        }  
        
      }
      else{
        if(isSet[n] == true && CC.size(var) == 1){
          
          isSet[n] = false;
          numCC -= 1;
        }
      }
    }
    models.resize(numCC);
    cc2gm.resize(numCC);
    IndexType countCC = 0;
    typename std::set<typename GM2::IndexType>::iterator it;
    
    for(IndexType i=0;i<CC.numberOfSets();++i){
      if(isSet[i] == true){
        LabelType StateSpace[cc2gmINT[i].size()];
        for(IndexType j=0;j<cc2gmINT[i].size();++j){
          LabelType label=gm_.numberOfLabels(cc2gmINT[i][j]);
          StateSpace[j]=label;
        }
        GM2 gmV(typename GM2::SpaceType(StateSpace,StateSpace+cc2gmINT[i].size()));
        
        for(it=setFactors[i].begin();it!=setFactors[i].end();it++){
          if(gm_.numberOfVariables(*it) == 2){
            
            IndexType var[gm_.numberOfVariables(*it)];
            for(IndexType l=0;l<gm_.numberOfVariables(*it);++l){
              IndexType idx=gm_.variableOfFactor(*it,l);
              var[l]=gm2ccIDX[idx];
              
            }
            ViewFunction<GM> func(gm_[*it]);
            gmV.addFactor(gmV.addFunction(func),var,var + gm_.numberOfVariables(*it));
          }
          else{
            IndexType idx=gm_.variableOfFactor(*it,0);
            IndexType var[]={gm2ccIDX[idx]};          
            gmV.addFactor(gmV.addFunction(unaryFunc[idx]),var,var + 1);
          }
        }
        
        models[countCC]=gmV;
        cc2gm[countCC]=cc2gmINT[i];
        countCC++;
        
      }
    }
    
  }

  template<class GM, class ACC, class INF>
  typename GM::ValueType ReducedInference<GM,ACC,INF>::bound() const {
    return bound_;
  }

  template<class GM, class ACC, class INF>
  typename GM::ValueType ReducedInference<GM,ACC,INF>::value() const {
    return value_;
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
        x[i] = states_[i];
      }
      return NORMAL;
    }
    else {
      return UNKNOWN;
    }
  }
} // namespace opengm

#endif // #ifndef OPENGM_ReducedInference_HXX

