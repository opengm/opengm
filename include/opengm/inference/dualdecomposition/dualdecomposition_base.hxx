#pragma once
#ifndef OPENGM_DUALDECOMPOSITION_BASE_HXX
#define OPENGM_DUALDECOMPOSITION_BASE_HXX

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <limits>

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/decomposition/graphicalmodeldecomposition.hxx"
#include "opengm/graphicalmodel/decomposition/graphicalmodeldecomposer.hxx"
#include "opengm/functions/modelviewfunction.hxx"
#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/utilities/tribool.hxx"
#include "opengm/inference/dualdecomposition/dddualvariableblock.hxx"
#include <opengm/utilities/timer.hxx>

namespace opengm { 

   class DualDecompositionBaseParameter{
   public:
      enum DecompositionId {MANUAL, TREE, SPANNINGTREES, BLOCKS};
      enum DualUpdateId {ADAPTIVE, STEPSIZE, STEPLENGTH, KIEWIL};
      
      /// type of decomposition that should be used (independent of model structure)
      DecompositionId decompositionId_;
      /// decomposition of the model (needs to fit to the model structure)
      GraphicalModelDecomposition decomposition_;
      /// vectors of factors of the subproblems - used form manual decomposition only.
      std::vector<std::vector<size_t> > subFactors_;
      /// maximum order of dual variables (order of the corresponding factor)
      size_t maximalDualOrder_;
      /// number of blocks for block decomposition
      size_t numberOfBlocks_;
      /// maximum number of dual iterations
      size_t maximalNumberOfIterations_;
      /// the absolut accuracy that has to be guaranteed to stop with an approximate solution (set 0 for optimality)
      double minimalAbsAccuracy_; 
      /// the relative accuracy that has to be guaranteed to stop with an approximate solution (set 0 for optimality)
      double minimalRelAccuracy_;
      /// number of threads for primal problems
      size_t numberOfThreads_;
      /// use filling to generate full labelings from non-spanning subproblems. If one labeling is generated for all non-spanning subproblems
      bool fillSubLabelings_;

      // Update parameters
      double stepsizeStride_;    //updateStride_;
      double stepsizeScale_;     //updateScale_;
      double stepsizeExponent_;  //updateExponent_;
      double stepsizeMin_;       //updateMin_;
      double stepsizeMax_;       //updateMax_;
      bool   stepsizePrimalDualGapStride_; //obsolete
      bool   stepsizeNormalizedSubgradient_; //obsolete
      // DualUpdateId dualUpdateId_ = ADAPTIVE;
    
      DualDecompositionBaseParameter() : 
         decompositionId_(SPANNINGTREES),  
         maximalDualOrder_(std::numeric_limits<size_t>::max()),
         numberOfBlocks_(2),
         maximalNumberOfIterations_(100),
         minimalAbsAccuracy_(0.0), 
         minimalRelAccuracy_(0.0),
         numberOfThreads_(1),
         fillSubLabelings_(false),
         stepsizeStride_(1),
         stepsizeScale_(1),
         stepsizeExponent_(0.5),
         stepsizeMin_(0),
         stepsizeMax_(std::numeric_limits<double>::infinity()),
         stepsizePrimalDualGapStride_(false),
         stepsizeNormalizedSubgradient_(false)
         
         {};

      double getStepsize(size_t iteration, double primalDualGap, double subgradientNorm){
         OPENGM_ASSERT(iteration>=0);
         double stepsize = stepsizeStride_;
         if(stepsizePrimalDualGapStride_){ 
            stepsize *= fabs(primalDualGap) / subgradientNorm / subgradientNorm;
         }
         else{
            stepsize /= (1+std::pow( stepsizeScale_*(double)(iteration),stepsizeExponent_));
            if(stepsizeNormalizedSubgradient_) 
               stepsize /= subgradientNorm; 
         }
         //stepsize /= (1+std::pow( stepsizeScale_*(double)(iteration),stepsizeExponent_));
         //stepsize = std::max(std::min(stepsizeMax_,stepsize),stepsizeMin_); 
         //if(stepsizeNormalizedSubgradient_) 
         //   stepsize /= subgradientNorm; 
         return stepsize;
      }
   };  

   /// Visitor
   template<class DD>
   class DualDecompositionEmptyVisitor {
   public:
      typedef DD DDType;
      typedef typename DDType::ValueType ValueType;

      void operator()(
         const DDType& dd, 
         const ValueType bound, const ValueType bestBound, 
         const ValueType value, const ValueType bestValue,
         double primalTime, double dualTime
         )
         {;}
      void startInference(){;}
    };

   /// Visitor
   template<class DD>
   class DualDecompositionVisitor {
   public:
      typedef DD DDType;
      typedef typename DDType::ValueType ValueType;

      void operator()(
         const DDType& dd, 
         const ValueType bound, const ValueType bestBound, 
         const ValueType value, const ValueType bestValue,
         const double primalTime, const double dualTime
         )
         {
            totalTimer_.toc(); 
            if(times_.size()==0)
               times_.push_back(totalTimer_.elapsedTime());
            else
               times_.push_back(times_.back() + totalTimer_.elapsedTime()); 
            values_.push_back(value);
            bounds_.push_back(bound);
            primalTimes_.push_back(primalTime);
            dualTimes_.push_back(dualTime);
/*
            std::cout << "(" << values_.size() << " / "<< times_.back() <<"  )  " 
                      << bound << " <= " << bestBound 
                      << " <= " << "E" 
                      << " <= " << bestValue << " <= " 
                      << value << std::endl;
*/
            totalTimer_.tic();
         }
      void startInference(){totalTimer_.tic();}     
      const std::vector<ValueType>& values(){return values_;}
      const std::vector<ValueType>& bounds(){return bounds_;}
      const std::vector<double>& times(){return times_;}
      const std::vector<double>& primalTimes(){return primalTimes_;}
      const std::vector<double>& dualTimes(){return dualTimes_;}

   private:
      std::vector<ValueType> values_;
      std::vector<ValueType> bounds_;
      std::vector<double>    times_;    
      std::vector<double>    primalTimes_;
      std::vector<double>    dualTimes_;
      opengm::Timer totalTimer_;
    };

   /// A framework for inference algorithms based on Lagrangian decomposition 
   template<class GM, class DUALBLOCK>
   class DualDecompositionBase
   {
   public:
      typedef GM                                                        GmType;
      typedef GM                                                        GraphicalModelType;
      typedef DUALBLOCK                                                 DualBlockType; 
      typedef typename DualBlockType::DualVariableType                  DualVariableType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef ModelViewFunction<GmType, DualVariableType>               ViewFunctionType;
      typedef GraphicalModel<ValueType, OperatorType,  typename meta::TypeListGenerator<ViewFunctionType>::type, opengm::DiscreteSpace<IndexType,LabelType> > SubGmType;
     

      typedef GraphicalModelDecomposition                               DecompositionType;
      typedef typename DecompositionType::SubVariable                   SubVariableType;
      typedef typename DecompositionType::SubVariableListType           SubVariableListType;
      typedef typename DecompositionType::SubFactor                     SubFactorType;
      typedef typename DecompositionType::SubFactorListType             SubFactorListType; 

     
      DualDecompositionBase(const GmType&);
      void init(DualDecompositionBaseParameter&);
      const SubGmType& subModel(size_t subModelId) const {return subGm_[subModelId];};
  
   protected:
      template<class ITERATOR> void addDualBlock(const SubFactorListType&,ITERATOR,ITERATOR);
      std::vector<DualVariableType*> getDualPointers(size_t);
      template<class ACC> void getBounds(const std::vector<std::vector<LabelType> >&, const std::vector<SubVariableListType>&, ValueType&, ValueType&, std::vector<LabelType>&);
      double subGradientNorm(double L=1) const;
      virtual DualDecompositionBaseParameter& parameter() = 0;
      virtual void allocate() = 0;
     
      const GmType&              gm_;
      std::vector<SubGmType>     subGm_;
      std::vector<DualBlockType> dualBlocks_;
      size_t                     numDualsOvercomplete_;
      size_t                     numDualsMinimal_;
      std::vector<Tribool>       modelWithSameVariables_;
   };
 
   /// \cond HIDDEN_SYMBOLS
   template<class DUALVAR> struct DDIsView                              {static bool isView(){return false;}};
   template<>              struct DDIsView<marray::View<double,false> > {static bool isView(){return true;}}; 
   template<>              struct DDIsView<marray::View<double,true> >  {static bool isView(){return true;}};  
   template<>              struct DDIsView<marray::View<float,false> >  {static bool isView(){return true;}};   
   template<>              struct DDIsView<marray::View<float,true> >   {static bool isView(){return true;}};     
   /// \endcond

///////////////////////////////////////////////
    
   template<class DUALVAR, class T>
   void DualVarAssign(DUALVAR& dv,T* t)
   {  
   }
   
   template <class T>
   void DualVarAssign(marray::View<T,false>& dv, T* t)
   { 
      dv.assign( dv.shapeBegin(),dv.shapeEnd(),t);
   }
 
   template<class GM, class DUALBLOCK>
   DualDecompositionBase<GM, DUALBLOCK>::DualDecompositionBase(const GmType& gm):gm_(gm)
   {}
  
   template<class GM, class DUALBLOCK>
   void DualDecompositionBase<GM, DUALBLOCK>::init(DualDecompositionBaseParameter& para) 
   {
      if(para.decompositionId_ == DualDecompositionBaseParameter::TREE){
         opengm::GraphicalModelDecomposer<GmType> decomposer;
         para.decomposition_ = decomposer.decomposeIntoTree(gm_);
         para.decomposition_.reorder();
         para.decomposition_.complete();
         para.maximalDualOrder_ = 1;
      }
      if(para.decompositionId_ == DualDecompositionBaseParameter::SPANNINGTREES){
         opengm::GraphicalModelDecomposer<GmType> decomposer;
         para.decomposition_ = decomposer.decomposeIntoSpanningTrees(gm_);
         para.decomposition_.reorder();
         para.decomposition_.complete();
         para.maximalDualOrder_ = 1;
      } 
      if(para.decompositionId_ == DualDecompositionBaseParameter::BLOCKS){
         opengm::GraphicalModelDecomposer<GmType> decomposer;
         para.decomposition_ = decomposer.decomposeIntoClosedBlocks(gm_,para.numberOfBlocks_);
         para.decomposition_.reorder();
         para.decomposition_.complete();
         para.maximalDualOrder_ = 1;
      }
      if(para.decompositionId_ == DualDecompositionBaseParameter::MANUAL){
         opengm::GraphicalModelDecomposer<GmType> decomposer;
         para.decomposition_ = decomposer.decomposeManual(gm_,para.subFactors_);
         para.decomposition_.reorder();
         para.decomposition_.complete();
         para.maximalDualOrder_ = 1;
      }

      OPENGM_ASSERT(para.decomposition_.isValid(gm_));
    
      //Build SubModels
      const std::vector<SubVariableListType>&                subVariableLists    =  para.decomposition_.getVariableLists();
      const std::vector<SubFactorListType>&                  subFactorLists      =  para.decomposition_.getFactorLists();
      const std::map<std::vector<size_t>,SubFactorListType>& emptySubFactorLists =  para.decomposition_.getEmptyFactorLists();
      const size_t                                           numberOfSubModels   =  para.decomposition_.numberOfSubModels();
      
      modelWithSameVariables_.resize(numberOfSubModels,Tribool::Maybe);

      typename SubVariableListType::const_iterator it;
      typename SubFactorListType::const_iterator it2; 
      typename SubFactorListType::const_iterator it3;
      typename std::map<std::vector<size_t>,SubFactorListType>::const_iterator it4;

 
      std::vector<std::vector<size_t> > numStates(numberOfSubModels);
      for(size_t subModelId=0; subModelId<numberOfSubModels; ++subModelId){
         numStates[subModelId].resize(para.decomposition_.numberOfSubVariables(subModelId),0);
      }
 
      for(size_t varId=0; varId<gm_.numberOfVariables(); ++varId){
         for(it = subVariableLists[varId].begin(); it!=subVariableLists[varId].end();++it){
            numStates[(*it).subModelId_][(*it).subVariableId_] = gm_.numberOfLabels(varId);        
         }
      }

      subGm_.resize(numberOfSubModels);
      for(size_t subModelId=0; subModelId<numberOfSubModels; ++subModelId){
         subGm_[subModelId] = SubGmType(opengm::DiscreteSpace<IndexType,LabelType>(numStates[subModelId].begin(),numStates[subModelId].end()));
      }
   
      // Add Duals 
      numDualsOvercomplete_ = 0;
      numDualsMinimal_ = 0;
     
      for(size_t factorId=0; factorId<gm_.numberOfFactors(); ++factorId){
         if(subFactorLists[factorId].size()>1 && (gm_[factorId].numberOfVariables() <= para.maximalDualOrder_)){
            addDualBlock(subFactorLists[factorId], gm_[factorId].shapeBegin(), gm_[factorId].shapeEnd());
            numDualsOvercomplete_ += subFactorLists[factorId].size()     *  gm_[factorId].size();
            numDualsMinimal_      += (subFactorLists[factorId].size()-1) *  gm_[factorId].size();
          }
      } 
  
      for(it4=emptySubFactorLists.begin() ; it4 != emptySubFactorLists.end(); it4++ ){
         if((*it4).second.size()>1 && ((*it4).first.size() <= para.maximalDualOrder_)){
            std::vector<size_t> shape((*it4).first.size());
            size_t temp = 1;
            for(size_t i=0; i<(*it4).first.size(); ++i){
               shape[i] = gm_.numberOfLabels((*it4).first[i]);
               temp *= gm_.numberOfLabels((*it4).first[i]);
            }  
            numDualsOvercomplete_ += (*it4).second.size()    * temp;
            numDualsMinimal_      += ((*it4).second.size()-1) * temp;
            addDualBlock((*it4).second,shape.begin(),shape.end());
         }
      }

      // Allocate memmory if this has not been done yet 
      this->allocate();

      // Add Factors
      size_t dualCounter = 0;
      for(size_t factorId=0; factorId<gm_.numberOfFactors(); ++factorId){
         if(subFactorLists[factorId].size()>1 && (gm_[factorId].numberOfVariables() <= para.maximalDualOrder_)){
            std::vector<DualVariableType*> offsets = getDualPointers(dualCounter++);
            size_t i=0;
            for(it2 = subFactorLists[factorId].begin(); it2!=subFactorLists[factorId].end();++it2){
               OPENGM_ASSERT(offsets[i]->dimension() == gm_[factorId].numberOfVariables());
               for(size_t j=0; j<offsets[i]->dimension();++j)
                  OPENGM_ASSERT(offsets[i]->shape(j) == gm_[factorId].shape(j));

               const size_t subModelId = (*it2).subModelId_;
               const ViewFunctionType function(gm_,factorId, 1.0/subFactorLists[factorId].size(),offsets[i++]); 
               const typename SubGmType::FunctionIdentifier funcId = subGm_[subModelId].addFunction(function); 
               subGm_[subModelId].addFactor(funcId,(*it2).subIndices_.begin(),(*it2).subIndices_.end());
            }
         }
         else{
            for(it2 = subFactorLists[factorId].begin(); it2!=subFactorLists[factorId].end();++it2){
               const size_t subModelId = (*it2).subModelId_;
               const ViewFunctionType function(gm_,factorId, 1.0/subFactorLists[factorId].size()); 
               const typename SubGmType::FunctionIdentifier funcId = subGm_[subModelId].addFunction(function); 
               subGm_[subModelId].addFactor(funcId,(*it2).subIndices_.begin(),(*it2).subIndices_.end());
            }
         }
      } 
      //Add EmptyFactors 
      for(it4=emptySubFactorLists.begin() ; it4 != emptySubFactorLists.end(); it4++ ){ 
         if((*it4).second.size()>1 && ((*it4).first.size() <= para.maximalDualOrder_)){
            size_t i=0;
            std::vector<DualVariableType*> offsets = getDualPointers(dualCounter++);
            for(it3 = (*it4).second.begin(); it3!=(*it4).second.end();++it3){
               const size_t subModelId = (*it3).subModelId_;
               const ViewFunctionType function(offsets[i++]); 
               const typename SubGmType::FunctionIdentifier funcId = subGm_[subModelId].addFunction(function); 
               subGm_[subModelId].addFactor(funcId,(*it3).subIndices_.begin(),(*it3).subIndices_.end());
            }
         }
      } 
   }



   template<class GM, class DUALBLOCK>
   template <class ITERATOR> 
   inline void DualDecompositionBase<GM,DUALBLOCK>::addDualBlock(const SubFactorListType& c,ITERATOR shapeBegin, ITERATOR shapeEnd)
   {  
      dualBlocks_.push_back(DualBlockType(c,shapeBegin,shapeEnd));
      return;
   } 



   template<class GM, class DUALBLOCK>
   inline std::vector<typename DUALBLOCK::DualVariableType*> DualDecompositionBase<GM,DUALBLOCK>::getDualPointers(size_t dualBlockId)
   {  
      return dualBlocks_[dualBlockId].getPointers();
   }

   template<class GM, class DUALBLOCK>
   template <class ACC>
   void DualDecompositionBase<GM,DUALBLOCK>::getBounds
   (
      const std::vector<std::vector<LabelType> >& subStates,
      const std::vector<SubVariableListType>& subVariableLists,
      ValueType& lowerBound, 
      ValueType& upperBound,
      std::vector<LabelType> & upperState
      )
   {
      bool useFilling = parameter().fillSubLabelings_;

      // Calculate lower-bound 
      lowerBound=0;
      for(size_t subModelId=0; subModelId<subGm_.size(); ++subModelId){ 
         lowerBound += subGm_[subModelId].evaluate(subStates[subModelId]); 
      }
      
      // Calculate upper-bound 
      Accumulation<ValueType,LabelType,ACC> ac;
     
      // Set modelWithSameVariables_
      if(modelWithSameVariables_[0] == Tribool::Maybe){
         for(size_t varId=0; varId<gm_.numberOfVariables(); ++varId){
            for(typename SubVariableListType::const_iterator its = subVariableLists[varId].begin();
                its!=subVariableLists[varId].end();++its){
               const size_t& subModelId    = (*its).subModelId_;
               const size_t& subVariableId = (*its).subVariableId_;
               if(subVariableId != varId){
                  modelWithSameVariables_[subModelId] = Tribool::False;
               }
            }
         } 
         for(size_t subModelId=0; subModelId<subGm_.size(); ++subModelId){ 
            if(gm_.numberOfVariables() != subGm_[subModelId].numberOfVariables()){
               modelWithSameVariables_[subModelId] = Tribool::False;
            }
            if(modelWithSameVariables_[subModelId] == Tribool::Maybe){
               modelWithSameVariables_[subModelId] = Tribool::True;
            }
         }
      }

      // Build Primal-Candidates

      // -- Build/Evaluate default canidate
      std::vector<std::vector<IndexType> > subVariable2TrueVariable(subGm_.size());
      if(useFilling){
         for(size_t s=0; s<subGm_.size();++s){
            subVariable2TrueVariable[s].resize(subGm_[s].numberOfVariables());
         }
      }
      std::vector<LabelType> defaultLabel(gm_.numberOfVariables());
      for(size_t varId=0; varId<gm_.numberOfVariables(); ++varId){
         std::map<LabelType,size_t> labelCount;
         for(typename SubVariableListType::const_iterator its = subVariableLists[varId].begin(); its!=subVariableLists[varId].end();++its){
            const size_t& subModelId    = (*its).subModelId_;
            const size_t& subVariableId = (*its).subVariableId_;
            if(useFilling){
               subVariable2TrueVariable[subModelId][subVariableId] = varId;
            }
            ++labelCount[subStates[subModelId][subVariableId]]; 
         } 
         size_t c=0;
         for(typename std::map<LabelType,size_t>::iterator it=labelCount.begin(); it!=labelCount.end(); ++it){
            if( it->second > c ){
               c = it->second;
               defaultLabel[varId] = it->first;
            }
         }
      }
      ac(gm_.evaluate(defaultLabel),defaultLabel);
   

      // -- Build/Evaluate subproblem canidates 
      size_t a;
      for(size_t subModelId=0; subModelId<subGm_.size(); ++subModelId){ 
         if(modelWithSameVariables_[subModelId] == Tribool::False){ 
            if(useFilling){
               std::vector<LabelType> label(defaultLabel);
               for(size_t i=0; i<subStates[subModelId].size(); ++i){
                  label[subVariable2TrueVariable[subModelId][i]]=subStates[subModelId][i];
               }
               ac(gm_.evaluate(label),label);
            }
         }
         else{
            ac(gm_.evaluate(subStates[subModelId]),subStates[subModelId]);
         }
      } 
      upperBound = ac.value();
      ac.state(upperState);
   }

   template <class GM, class DUALBLOCK> 
   double DualDecompositionBase<GM,DUALBLOCK>::subGradientNorm(double L) const
   { 
      double norm = 0;
      typename std::vector<DualBlockType>::const_iterator it;
      for(it = dualBlocks_.begin(); it != dualBlocks_.end(); ++it){
         norm  += (*it).duals_.size();
      }
      norm = pow(norm,1.0/L);
      return norm;
   } 

} // namespace opengm

#endif
