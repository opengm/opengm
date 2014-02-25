#pragma once
#ifndef OPENGM_ASTAR_HXX
#define OPENGM_ASTAR_HXX

#include <cmath>
#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <functional>

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include <opengm/graphicalmodel/graphicalmodel_manipulator.hxx>
#include "opengm/inference/inference.hxx"
#include "opengm/inference/messagepassing/messagepassing.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

   // node of the search tree for the a-star search
   template<class FactorType> struct AStarNode {
      typename std::vector<typename FactorType::LabelType>    conf;
      typename FactorType::ValueType     value;
   };
/*
   template<class AStar, bool Verbose=false>
   class AStarVisitor {
   public:
      typedef AStar astar_type;
      typedef typename astar_type::ValueType ValueType;
      AStarVisitor();
      void operator()(const astar_type&, const std::vector<size_t>& conf, const size_t heapsize, const ValueType& bound1, const ValueType& bound2, const double& runtime);
   private:
      size_t step_;
   };
*/

/// \endcond 

   /// \brief A star search algorithm
   ///
   /// Kappes, J. H. "Inference on Highly-Connected Discrete Graphical Models with Applications to Visual Object Recognition". Ph.D. Thesis 2011
   /// Bergtholdt, M. & Kappes, J. H. & Schnoerr, C.: "Learning of Graphical Models and Efficient Inference for Object Class Recognition", DAGM 2006
   /// Bergtholdt, M. & Kappes, J. H. & Schmidt, S. & Schnoerr, C.: "A Study of Parts-Based Object Class Detection Using Complete Graphs" IJCV 2010
   /// 
   /// Within this implementation of the AStar-Algo the user has to set the
   /// the node order and a acyclic graph for the calculation of the heuristic.  
   /// A good choice for both is critical for good performance!
   /// A heuristic which select these parameters automatically will be added in the next version
   /// 
   /// The AStar-Algo transform the problem into a shortest path problem in an exponentially large graph.
   /// Due to the problem structure, this graph can be represented implicitly! 
   /// To find the shortest path we perform a best first search and use a admissable tree-based heuristic 
   /// to underestimate the cost to a goal node. This lower bound allows us to reduce the search to an
   /// manageable subspace of the exponentially large search-space.
   ///
   /// \ingroup inference
   template<class GM,class ACC>
   class AStar : public Inference<GM,ACC>
   {
   public:
      ///graphical model type
      typedef GM                                          GraphicalModelType;
      // -- obsolet --  typedef typename GraphicalModelType::template Rebind<true>::RebindType EditableGraphicalModelType;
      ///accumulation type
      typedef ACC                                         AccumulationType;
      OPENGM_GM_TYPE_TYPEDEFS;
      /// configuration vector type
      typedef typename std::vector<LabelType>             ConfVec ;
      /// configuration iterator
      typedef typename ConfVec::iterator                  ConfVecIt;
      /// visitor 
      typedef opengm::visitors::VerboseVisitor<AStar<GM, ACC> > VerboseVisitorType;
      typedef opengm::visitors::TimingVisitor<AStar<GM, ACC> > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<AStar<GM, ACC> > EmptyVisitorType;
      
      enum Heuristic{
         DEFAULT_HEURISTIC = 0,
         FAST_HEURISTIC = 1,
         STANDARD_HEURISTIC = 2
      };
      struct Parameter {
         Parameter()
            {
               maxHeapSize_    = 3000000;
               numberOfOpt_    = 1;
               objectiveBound_ = AccumulationType::template neutral<ValueType>();
               heuristic_      = Parameter::DEFAULTHEURISTIC;
            };
            /// constuctor

         /// \brief add tree factor id
         /// \param id factor id
         void addTreeFactorId(size_t id)
            { treeFactorIds_.push_back(id); }
         /// DEFAULTHEURISTIC ;
         static const size_t DEFAULTHEURISTIC = 0;
         /// FASTHEURISTIC
         static const size_t FASTHEURISTIC = 1;
         /// STANDARDHEURISTIC
         static const size_t STANDARDHEURISTIC = 2;
         /// maxHeapSize_ maximum size of the heap
         size_t maxHeapSize_;
         /// number od N-best solutions that should be found
         size_t              numberOfOpt_;
         /// objective bound
         ValueType          objectiveBound_;
         /// heuritstic
         ///
         /// DEFAULTHEURISTIC = 0;
         /// FASTHEURISTIC = 1
         /// STANDARDHEURISTIC = 2
         size_t heuristic_;  
         std::vector<IndexType> nodeOrder_;
         std::vector<size_t> treeFactorIds_;
       
      };
      AStar(const GM& gm, Parameter para = Parameter());
      virtual std::string name() const {return "AStar";}
      const GraphicalModelType& graphicalModel() const;
      virtual InferenceTermination infer();
      virtual void reset();
      template<class VisitorType> InferenceTermination infer(VisitorType& vistitor);
      ValueType bound()const {return belowBound_;}
      ValueType value()const;
      virtual InferenceTermination marginal(const size_t,IndependentFactorType& out)const        {return UNKNOWN;}
      virtual InferenceTermination factorMarginal(const size_t, IndependentFactorType& out)const {return UNKNOWN;}
      virtual InferenceTermination arg(std::vector<LabelType>& v, const size_t = 1)const;
      virtual InferenceTermination args(std::vector< std::vector<LabelType> >& v)const;

   private:
      const GM&                                   gm_;
      Parameter                                   parameter_;
      // remeber passed parameter in  parameterInitial_
      // to reset astar
      Parameter                                   parameterInitial_;
      std::vector<AStarNode<IndependentFactorType> >  array_;
      std::vector<size_t>                         numStates_;
      size_t                                      numNodes_;
      std::vector<IndependentFactorType>          treeFactor_;
      std::vector<IndependentFactorType>          optimizedFactor_;
      std::vector<ConfVec >                       optConf_;
      std::vector<bool>                           isTreeFactor_;
      ValueType                                   aboveBound_;
      ValueType                                   belowBound_;
      template<class VisitorType> void  expand(VisitorType& vistitor);
      std::vector<ValueType>           fastHeuristic(ConfVec conf);
      inline static bool                comp1(const AStarNode<IndependentFactorType>& a, const AStarNode<IndependentFactorType>& b)
         {return  AccumulationType::ibop(a.value,b.value);};
      inline static bool                comp2(const AStarNode<IndependentFactorType>& a, const AStarNode<IndependentFactorType>& b)
         { return  AccumulationType::bop(a.value,b.value);};
      inline static ValueType          better(ValueType a, ValueType b)   {return AccumulationType::op(a,b);};
      inline static ValueType          wrose(ValueType a,  ValueType b)   {return AccumulationType::iop(a,b);};
   };


//*******************
//** Impelentation **
//*******************

/// \brief constructor
/// \param gm graphical model
/// \param para AStar parameter
   template<class GM, class ACC >
   AStar<GM,ACC>
   ::AStar
   (
      const GM& gm,
      Parameter para
   ):gm_(gm)
   {
      parameterInitial_=para;
      parameter_ = para;
      if( parameter_.heuristic_ == Parameter::DEFAULTHEURISTIC) {
         if(gm_.factorOrder()<=2)
            parameter_.heuristic_ = Parameter::FASTHEURISTIC;
         else
            parameter_.heuristic_ = Parameter::STANDARDHEURISTIC;
      }
      OPENGM_ASSERT(parameter_.heuristic_ == Parameter::FASTHEURISTIC || parameter_.heuristic_ == Parameter::STANDARDHEURISTIC);
      ACC::ineutral(belowBound_);
      ACC::neutral(aboveBound_);
      //Set variables
      isTreeFactor_.resize(gm_.numberOfFactors());
      numStates_.resize(gm_.numberOfVariables());
      numNodes_ = gm_.numberOfVariables();
      for(size_t i=0; i<numNodes_;++i)
         numStates_[i] = gm_.numberOfLabels(i);
      //Check nodeOrder
      if(parameter_.nodeOrder_.size()==0) {
         parameter_.nodeOrder_.resize(numNodes_);
         for(size_t i=0; i<numNodes_; ++i)
            parameter_.nodeOrder_[i]=i;
      }
      if(parameter_.nodeOrder_.size()!=numNodes_)
         throw RuntimeError("The node order does not fit to the model.");
      OPENGM_ASSERT(std::set<size_t>(parameter_.nodeOrder_.begin(), parameter_.nodeOrder_.end()).size()==numNodes_);
      for(size_t i=0;i<numNodes_; ++i) {
         OPENGM_ASSERT(parameter_.nodeOrder_[i] < numNodes_);
         OPENGM_ASSERT(parameter_.nodeOrder_[i] >= 0);
      }
      //Check FactorIds
      if(parameter_.treeFactorIds_.size()==0) {
         //Select tree factors
         for(size_t i=0; i<gm_.numberOfFactors(); ++i) {
            if((gm_[i].numberOfVariables()==2) &&
               (gm_[i].variableIndex(0)==parameter_.nodeOrder_.back() || gm_[i].variableIndex(1)==parameter_.nodeOrder_.back())
               )
               parameter_.addTreeFactorId(i);
         }
      }
      for(size_t i=0; i<parameter_.treeFactorIds_.size(); ++i)
         OPENGM_ASSERT(gm_.numberOfFactors() > parameter_.treeFactorIds_[i]);
      //compute optimized factors
      optimizedFactor_.resize(gm_.numberOfFactors());
      for(size_t i=0; i<gm_.numberOfFactors(); ++i) {
         if(gm_[i].numberOfVariables()<=1) continue;
         std::vector<size_t> index(gm_[i].numberOfVariables());
         gm_[i].variableIndices(index.begin());
         optimizedFactor_[i].assign(gm_ ,index.end()-1, index.end());
         opengm::accumulate<ACC>(gm[i],index.begin()+1,index.end(),optimizedFactor_[i]);
         //gm_[i].template accumulate<ACC>(index.begin()+1,index.end(),optimizedFactor_[i]);
         OPENGM_ASSERT(optimizedFactor_[i].numberOfVariables() == 1);
         OPENGM_ASSERT(optimizedFactor_[i].variableIndex(0) == index[0]);
      }
      //PUSH EMPTY CONFIGURATION TO HEAP
      AStarNode<IndependentFactorType> a;
      a.conf.resize(0);
      a.value = 0;
      array_.push_back(a);
      make_heap(array_.begin(), array_.end(), comp1);
      //Check if maximal order is smaller equal 2, otherwise fall back to naive computation of heuristic
      if(parameter_.heuristic_ == parameter_.FASTHEURISTIC) {
         for(size_t i=0; i<parameter_.treeFactorIds_.size(); ++i) {
            if(gm_[parameter_.treeFactorIds_[i]].numberOfVariables() > 2) {
               throw RuntimeError("The heuristic includes factor of order > 2.");
            }
         }
      }
      //Init treefactor structure
      treeFactor_.clear();
      for(size_t i=0; i<gm_.numberOfFactors(); ++i)
         isTreeFactor_[i] = false;
      for(size_t i=0; i<parameter_.treeFactorIds_.size(); ++i) {
         int factorId = parameter_.treeFactorIds_[i];
         isTreeFactor_[factorId] = true;
         treeFactor_.push_back(gm_[factorId]);
      }
   }
  
   /// \brief reset
   ///
   /// \warning  reset assumes that the structure of
   /// the graphical model has not changed
   ///
   /// TODO
   template<class GM, class ACC >
   void
   AStar<GM,ACC>::reset()
   {
      ///todo
   }

   template <class GM, class ACC>
   InferenceTermination
   AStar<GM,ACC>::infer()
   { 
      EmptyVisitorType v;
      return infer(v);
   }

/// \brief inference with visitor
/// \param visitor visitor
   template<class GM, class ACC>
   template<class VisitorType>
   InferenceTermination AStar<GM,ACC>::infer(VisitorType& visitor)
   { 
      size_t exitFlag=0;
      optConf_.resize(0);
      visitor.begin(*this);    
      while(array_.size()>0 && exitFlag==0) {
         if(parameter_.numberOfOpt_ == optConf_.size()) {
            visitor.end(*this);
            return NORMAL;
         }
         while(array_.front().conf.size() < numNodes_ && exitFlag==0) {
            expand(visitor);
            belowBound_ = array_.front().value;
            exitFlag = visitor(*this); 
            //visitor(*this, array_.front().conf, array_.size(), array_.front().value, globalBound_, time);
         }
         if(array_.front().conf.size()>=numNodes_){
            ValueType  value = array_.front().value;
            belowBound_ = value;
            std::vector<LabelType> conf(numNodes_);
            for(size_t n=0; n<numNodes_; ++n) {
               conf[parameter_.nodeOrder_[n]] = array_.front().conf[n];
            } 
            optConf_.push_back(conf);
            visitor(*this);
            if(ACC::bop(parameter_.objectiveBound_, value)) {
               visitor.end(*this);
               return NORMAL;
            }
         }
         pop_heap(array_.begin(), array_.end(),  comp1); //greater<FactorType,Accumulation>);
         array_.pop_back();
      }
      visitor.end(*this);     
      return UNKNOWN;
   } 

   template<class GM, class ACC>
   typename GM::ValueType AStar<GM, ACC>::value() const
   {
      if(optConf_.size()>=1){
         return gm_.evaluate(optConf_[0]);
      }
      else{
         return ACC::template neutral<ValueType>();
      }
   }

   template<class GM, class ACC>
   InferenceTermination AStar<GM, ACC>
   ::arg(ConfVec& conf, const size_t n)const
   {
      if(n>optConf_.size()) {
         conf.resize(gm_.numberOfVariables(),0);
         return UNKNOWN;
      }
      //conf.resize(opt_conf[n-1].size());
      conf=optConf_[n-1];
      return NORMAL;
   }

/// \brief args
/// \param[out]conf state vectors
///
///get the inference solutions
   template<class GM, class ACC>
   InferenceTermination AStar<GM,ACC>
   ::args(std::vector<std::vector<typename AStar<GM,ACC>::LabelType> >& conf)const
   {
      conf=optConf_;
      return NORMAL;
   }

   template<class GM, class ACC>
   template<class VisitorType>
   void AStar<GM, ACC>::expand(VisitorType& visitor)
   {
      //CHECK HEAP SIZE
      if(array_.size()>parameter_.maxHeapSize_*0.99) {
         partial_sort(array_.begin(), array_.begin()+(int)(parameter_.maxHeapSize_/2), array_.end(),  comp2);
         array_.resize((int)(parameter_.maxHeapSize_/2));
      }
      //GET HEAP HEAD
      AStarNode<IndependentFactorType> a           = array_.front();
      size_t            subconfsize = a.conf.size();
      //REMOVE HEAD FROM HEAP
      OPENGM_ASSERT(array_.size() > 0);
      pop_heap(array_.begin(), array_.end(),  comp1); //greater<FactorType,Accumulation>);
      array_.pop_back();
      if( parameter_.heuristic_ == parameter_.STANDARDHEURISTIC) { 

         //BUILD GRAPHICAL MODEL FOR HEURISTC CALCULATION
          
         typedef typename opengm::DiscreteSpace<IndexType, LabelType> MSpaceType;
         typedef typename meta::TypeListGenerator< ExplicitFunction<ValueType,IndexType,LabelType>, ViewFixVariablesFunction<GM>, ViewFunction<GM>, ConstantFunction<ValueType, IndexType, LabelType> >::type MFunctionTypeList;
         typedef GraphicalModel<ValueType, typename GM::OperatorType, MFunctionTypeList, MSpaceType> MGM;

         IndexType numberOfVariables = 0;
         std::vector<IndexType> varMap(gm_.numberOfVariables(),0);
         std::vector<LabelType> fixVariableLabel(gm_.numberOfVariables(),0);
         std::vector<bool> fixVariable(gm_.numberOfVariables(),false);
         for(size_t i =0; i<subconfsize ; ++i) {
            fixVariableLabel[parameter_.nodeOrder_[i]] = a.conf[i];
            fixVariable[parameter_.nodeOrder_[i]] = true;
         }

         for(IndexType var=0; var<gm_.numberOfVariables();++var){
            if(fixVariable[var]==false){
               varMap[var] = numberOfVariables++;
            }
         }
         std::vector<LabelType> shape(numberOfVariables,0);
         for(IndexType var=0; var<gm_.numberOfVariables();++var){
            if(fixVariable[var]==false){
               shape[varMap[var]] = gm_.numberOfLabels(var);
            }
         }
         MSpaceType space(shape.begin(),shape.end());
         MGM mgm(space);
 
         std::vector<PositionAndLabel<IndexType,LabelType> > fixedVars;
         std::vector<IndexType> MVars;
         ValueType constant;
         GM::OperatorType::neutral(constant);

         for(IndexType f=0; f<gm_.numberOfFactors();++f){
            fixedVars.resize(0); 
            MVars.resize(0);
            for(IndexType i=0; i<gm_[f].numberOfVariables(); ++i){
               const IndexType var = gm_[f].variableIndex(i);
               if(fixVariable[var]){
                  fixedVars.push_back(PositionAndLabel<IndexType,LabelType>(i,fixVariableLabel[var]));
               }else{
                  MVars.push_back(varMap[var]);
               }
            }
            if(fixedVars.size()==gm_[f].numberOfVariables()){//all fixed
               std::vector<LabelType> fixedStates(gm_[f].numberOfVariables(),0);
               for(IndexType i=0; i<gm_[f].numberOfVariables(); ++i){
                  fixedStates[i]=fixVariableLabel[ gm_[f].variableIndex(i)];
               }     
               GM::OperatorType::op(gm_[f](fixedStates.begin()),constant);       
            }else{
               if(MVars.size()<2 || isTreeFactor_[f]){
                  const ViewFixVariablesFunction<GM> func(gm_[f], fixedVars);
                  mgm.addFactor(mgm.addFunction(func),MVars.begin(), MVars.end());
               }else{
                  std::vector<IndexType> variablesIndices(optimizedFactor_[f].numberOfVariables());
                  for(size_t i=0; i<variablesIndices.size(); ++i)
                     variablesIndices[i] = varMap[optimizedFactor_[f].variableIndex(i)];
                  LabelType numberOfLabels = optimizedFactor_[f].numberOfLabels(0);
                  opengm::ExplicitFunction<ValueType,IndexType,LabelType> func(&numberOfLabels,&numberOfLabels+1,0);
                  for(LabelType i=0; i<numberOfLabels; ++i)
                     func(i) = optimizedFactor_[f](i);
                  mgm.addFactor(mgm.addFunction(func),variablesIndices.begin(),variablesIndices.end() );
                  OPENGM_ASSERT(mgm[mgm.numberOfFactors()-1].numberOfVariables()==1);
               }
            }
         }
         {
            LabelType temp;
            ConstantFunction<ValueType, IndexType, LabelType> func(&temp, &temp, constant);
            mgm.addFactor(mgm.addFunction(func),MVars.begin(), MVars.begin());
         } 
         typedef typename opengm::BeliefPropagationUpdateRules<MGM,ACC> UpdateRules;
         typename MessagePassing<MGM, ACC, UpdateRules, opengm::MaxDistance>::Parameter bpPara;
         bpPara.isAcyclic_ = opengm::Tribool::False;
         bpPara.maximumNumberOfSteps_ = mgm.numberOfVariables();
         OPENGM_ASSERT(mgm.isAcyclic());
         MessagePassing<MGM, ACC, UpdateRules, opengm::MaxDistance> bp(mgm,bpPara);  
         try{
            bp.infer();//Asynchronous();
         }
         catch(...) {
            throw RuntimeError("bp failed in astar");
         }
         ACC::op(bp.value(),aboveBound_,aboveBound_);
         std::vector<LabelType> conf(mgm.numberOfVariables()); 
 
         std::vector<IndexType> theVar(1, varMap[parameter_.nodeOrder_[subconfsize]]); 
 
         std::vector<LabelType> theLabel(1,0);
         a.conf.resize(subconfsize+1);
         for(size_t i=0; i<numStates_[subconfsize]; ++i) {
            a.conf[subconfsize] = i;
            theLabel[0] =i;
            bp.constrainedOptimum(theVar,theLabel,conf);
            a.value   = mgm.evaluate(conf);
            array_.push_back(a);
            push_heap(array_.begin(), array_.end(),  comp1); //greater<FactorType,Accumulation>) ;
         }
      }
      if( parameter_.heuristic_ == parameter_.FASTHEURISTIC) {
         std::vector<LabelType> conf(subconfsize);
         for(size_t i=0;i<subconfsize;++i)
            conf[i] = a.conf[i];
         std::vector<ValueType> bound = fastHeuristic(conf);
         a.conf.resize(subconfsize+1);
         for(size_t i=0; i<numStates_[parameter_.nodeOrder_[subconfsize]]; ++i) {
            a.conf[subconfsize] = i;
            a.value             = bound[i];
            //if(bound[i]<10) {
            array_.push_back(a);
            push_heap(array_.begin(), array_.end(),  comp1); //greater<FactorType,Accumulation>) ;
            //}
         }
      }
   }

   template<class GM, class ACC>
   std::vector<typename AStar<GM, ACC>::ValueType>
   AStar<GM, ACC>::fastHeuristic(typename AStar<GM, ACC>::ConfVec conf)
   {
      std::list<size_t>                 factorList;
      std::vector<size_t>               nodeDegree(numNodes_,0);
      std::vector<int>                  nodeLabel(numNodes_,-1);
      std::vector<std::vector<ValueType > > nodeEnergy(numNodes_);
      size_t                            nextNode = parameter_.nodeOrder_[conf.size()];
      for(size_t i=0; i<numNodes_; ++i) {
         nodeEnergy[i].resize(numStates_[i]); //the energy passed to node i
         for(size_t j=0;j<numStates_[i];++j)
            OperatorType::neutral(nodeEnergy[i][j]);
      }
      for(size_t i=0;i<conf.size();++i) {
         nodeLabel[parameter_.nodeOrder_[i]] = conf[i];
      }
      //First run:
      // * add unarry function
      // * add pairwise functions with at least one observed node
      // * add the approximation for pairwise none-tree edges
      for(size_t i=0; i<gm_.numberOfFactors(); ++i) {
         const FactorType & f    = gm_[i];
         size_t nvar = f.numberOfVariables();
         //factors depending from 1 variable can be include
         if(nvar==1) {
            int index = f.variableIndex(0);
            if(nodeLabel[index]>=0) {
               nodeEnergy[index].resize(1);
               //nodeEnergy[index][0] = operatipon(f(nodeLabel[index]), nodeEnergy[index][0]);
               LabelType coordinates[]={static_cast<LabelType>(nodeLabel[index])};
               OperatorType::op(f(coordinates), nodeEnergy[index][0]);
            }
            else{
               OPENGM_ASSERT(numStates_[index] == nodeEnergy[index].size());
               for(size_t j=0;j<numStates_[index];++j) {
                  //nodeEnergy[index][j] = operation(f(j),nodeEnergy[index][j]);
                  LabelType coordinates[]={j};
                  OperatorType::op(f(coordinates),nodeEnergy[index][j]);
               }
            }
         }
         if(nvar==2) {
            size_t index1 = f.variableIndex(0);
            size_t index2 = f.variableIndex(1);
            if(nodeLabel[index1]>=0) {
               if(nodeLabel[index2]>=0) {
                  nodeEnergy[index1].resize(1);
                  //nodeEnergy[index1][0] = operation(f(nodeLabel[index1],nodeLabel[index2]),nodeEnergy[index1][0]);
                  LabelType coordinates[]={
                     static_cast<LabelType>(nodeLabel[index1]),
                     static_cast<LabelType>(nodeLabel[index2])
                  };
                  OperatorType::op(f(coordinates),nodeEnergy[index1][0]);
               }
               else{
                  OPENGM_ASSERT(numStates_[index2] == nodeEnergy[index2].size());
                  for(size_t j=0;j<numStates_[index2];++j) {
                     //nodeEnergy[index2][j] = operation(f(nodeLabel[index1],j), nodeEnergy[index2][j]);
                     LabelType coordinates[]={
                        static_cast<LabelType>(nodeLabel[index1]),
                        static_cast<LabelType>(j)
                     };
                     OperatorType::op(f(coordinates), nodeEnergy[index2][j]);
                  }
               }
            }
            else if(nodeLabel[index2]>=0) {
               OPENGM_ASSERT(numStates_[index1] == nodeEnergy[index1].size());
               for(size_t j=0;j<numStates_[index1];++j) {
                  //nodeEnergy[index1][j] = operation(f(j,nodeLabel[index2]),nodeEnergy[index1][j]);
                  LabelType coordinates[]={
                     static_cast<LabelType>(j),
                     static_cast<LabelType>(nodeLabel[index2])
                  };
                  OperatorType::op(f(coordinates),nodeEnergy[index1][j]);
               }
            }
            else if(isTreeFactor_[i]) {
               factorList.push_front(i);
               ++nodeDegree[index1];
               ++nodeDegree[index2];
               continue;
            }
            else{
               for(size_t j=0;j<numStates_[index1];++j) {
                  //nodeEnergy[index1][j] = operation(optimizedFactor_[i](j), nodeEnergy[index1][j]);
                  LabelType coordinates[]={j};
                  OperatorType::op(optimizedFactor_[i](coordinates), nodeEnergy[index1][j]);
               }
            }
         }
         if(nvar>2) {
            bool covered = true;
            std::vector<size_t> state(nvar);
            for(size_t j=0; j<nvar; ++j) {
               if(nodeLabel[f.variableIndex(j)]<0) {
                  state[j] = nodeLabel[f.variableIndex(j)];
                  covered = false;
               }
            }
            if(covered)
               nodeEnergy[f.variableIndex(0)][0] = f(state.begin());
            else{
               for(size_t j=0;j<numStates_[f.variableIndex(0)];++j) {
                  //nodeEnergy[f.variableIndex(0)][j] = operation(optimizedFactor_[i](j), nodeEnergy[f.variableIndex(0)][j]);
                  LabelType coordinates[]={j};
                  OperatorType::op(optimizedFactor_[i](coordinates), nodeEnergy[f.variableIndex(0)][j]);
               }
            }
         }
      }
      nodeDegree[nextNode] += numNodes_;
      // Start dynamic programming to solve the treestructured problem.
      while(factorList.size()>0) {
         size_t    id  = factorList.front();
         factorList.pop_front();
         const FactorType &  f      = gm_[id];
         size_t    index1 = f.variableIndex(0);
         size_t    index2 = f.variableIndex(1);
         typename FactorType::ValueType temp;
         OPENGM_ASSERT(index1 < numNodes_);
         OPENGM_ASSERT(index2 < numNodes_);
         OPENGM_ASSERT(gm_.numberOfLabels(index1) == numStates_[index1]);
         OPENGM_ASSERT(gm_.numberOfLabels(index2) == numStates_[index2]);
         if(nodeDegree[index1]==1) {
            typename FactorType::ValueType min;
            OPENGM_ASSERT(numStates_[index2] == nodeEnergy[index2].size());
            for(size_t j2=0;j2<numStates_[index2];++j2) {
               ACC::neutral(min);
               OPENGM_ASSERT(numStates_[index1] == nodeEnergy[index1].size());
               for(size_t j1=0;j1<numStates_[index1];++j1) {
                  LabelType coordinates[]={j1,j2};
                  OperatorType::op(f(coordinates),nodeEnergy[index1][j1],temp);
                  ACC::op(min,temp,min);
               }
               //nodeEnergy[index2][j2] = operation(min,nodeEnergy[index2][j2]);
               OperatorType::op(min,nodeEnergy[index2][j2]);
            }
            --nodeDegree[index1];
            --nodeDegree[index2];
            nodeEnergy[index1].resize(1);
            OperatorType::neutral(nodeEnergy[index1][0]);
         }
         else if(nodeDegree[index2]==1) {
            typename FactorType::ValueType min;
            OPENGM_ASSERT(numStates_[index1] == nodeEnergy[index1].size());
            for(size_t j1=0;j1<numStates_[index1];++j1) {
               ACC::neutral(min);
               OPENGM_ASSERT(numStates_[index2] == nodeEnergy[index2].size());
               for(size_t j2=0;j2<numStates_[index2];++j2) {
                  LabelType coordinates[]={j1,j2};
                  OperatorType::op(f(coordinates),nodeEnergy[index2][j2],temp);
                  ACC::op(min,temp,min);
                  //if(min>f(j1,j2)*node_energy[index2][j2]) min=f(j1,j2)*node_energy[index2][j2];
               }
               //nodeEnergy[index1][j1] = operation(min,nodeEnergy[index1][j1]);
               OperatorType::op(min,nodeEnergy[index1][j1]);
            }
            --nodeDegree[index1];
            --nodeDegree[index2];
            nodeEnergy[index2].resize(1);
            OperatorType::neutral(nodeEnergy[index2][0]);
         }
         else{
            factorList.push_back(id);
         }
      }
      //Evaluate
      ValueType result;
      ValueType min;
      OperatorType::neutral(result);
      std::vector<ValueType > bound;
      for(size_t i=0;i<numNodes_;++i) {
         if(i==nextNode) continue;
         ACC::neutral(min);
         for(size_t j=0; j<nodeEnergy[i].size();++j)
            ACC::op(min,nodeEnergy[i][j],min);
         //result = operation(result,min);
         OperatorType::op(min,result);
      }
      bound.resize(nodeEnergy[nextNode].size());
      for(size_t j=0; j<nodeEnergy[nextNode].size();++j) {
         //bound[j] = operation(nodeEnergy[nextNode][j],result);
         OperatorType::op(nodeEnergy[nextNode][j],result,bound[j]);
      }
      return bound;
   }

   template<class GM, class ACC>
   inline const typename AStar<GM, ACC>::GraphicalModelType&
   AStar<GM, ACC>::graphicalModel() const
   {
      return gm_;
   }

} // namespace opengm

#endif // #ifndef OPENGM_ASTAR_HXX

