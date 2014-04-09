#pragma once
#ifndef OPENGM_MQPBO_HXX
#define OPENGM_MQPBO_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/inference/inference.hxx"
#include <opengm/utilities/metaprogramming.hxx>
#include "opengm/utilities/tribool.hxx"
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/functions/view_fix_variables_function.hxx>

//#define MQPBOHotFixOutPutPartialOPtimalityMap
#ifdef MQPBOHotFixOutPutPartialOPtimalityMap
#include <opengm/datastructures/marray/marray_hdf5.hxx>
#endif

#include "opengm/inference/external/qpbo.hxx"

namespace opengm {
   
   //! [class mqpbo]
   /// Multilabel QPBO (MQPBO)
   /// Implements the algorithms described in
   /// i) Ivan Kovtun: Partial Optimal Labeling Search for a NP-Hard Subclass of (max, +) Problems. DAGM-Symposium 2003                         (part. opt. for potts)
   /// ii) P. Kohli, A. Shekhovtsov, C. Rother, V. Kolmogorov, and P. Torr: On partial optimality in multi-label MRFs, ICML 2008                (MQPBO)
   /// iii) P. Swoboda, B.  Savchynskyy, J.H.  Kappes, and C. Schnörr : Partial Optimality via Iterative Pruning for the Potts Model, SSVM 2013 (MQPBO with permutation sampling)
   ///
   /// Corresponding author: Joerg Hendrik Kappes
   ///
   ///\ingroup inference
   template<class GM, class ACC>
   class MQPBO : public Inference<GM, ACC>
   {
   public:
      typedef ACC AccumulationType;
      typedef GM GmType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef visitors::VerboseVisitor<MQPBO<GM, ACC> > VerboseVisitorType;
      typedef visitors::EmptyVisitor<MQPBO<GM, ACC> >   EmptyVisitorType; 
      typedef visitors::TimingVisitor<MQPBO<GM, ACC> >  TimingVisitorType; 
      typedef ValueType                       GraphValueType;
      
      enum PermutationType {NONE, RANDOM, MINMARG};
      
      class Parameter{
      public:
         Parameter(): useKovtunsMethod_(true), probing_(false),  strongPersistency_(false), rounds_(0), permutationType_(NONE) {};
         std::vector<LabelType> label_;
         bool useKovtunsMethod_;
         const bool probing_; //do not use this!
         bool strongPersistency_;
         size_t rounds_;
         PermutationType permutationType_;
      };

      MQPBO(const GmType&, const Parameter& = Parameter());
      ~MQPBO();
      std::string name() const;
      const GmType& graphicalModel() const;
      InferenceTermination infer();
      void reset();
      typename GM::ValueType bound() const;
      typename GM::ValueType value() const;  
      template<class VisitorType>
         InferenceTermination infer(VisitorType&);
      InferenceTermination testQuess(std::vector<LabelType> &guess);
      InferenceTermination testPermutation(PermutationType permutationType);
      void setStartingPoint(typename std::vector<typename GM::LabelType>::const_iterator);
      virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;

      const std::vector<opengm::Tribool>& partialOptimality(IndexType var) const {return partialOptimality_[var];}
      bool partialOptimality(IndexType var, LabelType& l) const                  {l=label_[var]; return optimal_[var];}
    
      double optimalityV() const;
      double optimality() const;
   private: 
      InferenceTermination testQuess(LabelType guess);
      void AddUnaryTerm(int var, ValueType v0, ValueType v1);
      void AddPairwiseTerm(int var0, int var1,ValueType v00,ValueType v01,ValueType v10,ValueType v11);

      const GmType& gm_;
      Parameter param_;

      kolmogorov::qpbo::QPBO<GraphValueType>* qpbo_;
      ValueType constTerm_;
      ValueType bound_;
      //int* label_;
      //int* defaultLabel_;
   
      std::vector<std::vector<LabelType> >         permutation_;        // org -> new
      std::vector<std::vector<LabelType> >         inversePermutation_;  // new -> org
      std::vector<std::vector<opengm::Tribool> >   partialOptimality_;
      std::vector<bool>                            optimal_;
      std::vector<LabelType>                       label_;
      std::vector<size_t>                          variableOffset_; 

      size_t numNodes_;
      size_t numEdges_;

      GraphValueType scale;

   };
//! [class mqpbo]
      
  
   template<class GM, class ACC>
   MQPBO<GM,ACC>::MQPBO
   (
       const GmType& gm,
       const Parameter& parameter
   )
   :  gm_(gm),    
      param_(parameter),
      scale(1)
   {
      for(size_t j = 0; j < gm_.numberOfFactors(); ++j) {
         if(gm_[j].numberOfVariables() > 2) {
            throw RuntimeError("This implementation of MQPBO supports only factors of order <= 2.");
         }
      }
      
      //Allocate Memory for Permutations
      permutation_.resize(gm_.numberOfVariables());
      inversePermutation_.resize(gm_.numberOfVariables());
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         permutation_[var].resize(gm_.numberOfLabels(var));
         inversePermutation_[var].resize(gm_.numberOfLabels(var));
      }

      //Set Default Optimality
      partialOptimality_.resize(gm_.numberOfVariables());
      optimal_.resize(gm_.numberOfVariables(),false);
      label_.resize(gm_.numberOfVariables());
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
        partialOptimality_[var].resize(gm_.numberOfLabels(var),opengm::Tribool::Maybe); 
      }

      //Calculated number of nodes and edges
      numNodes_=0;
      numEdges_=0;
      size_t numSOF=0;
      variableOffset_.resize(gm_.numberOfVariables(),0);
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         variableOffset_[var] = numNodes_;
         numNodes_ += gm_.numberOfLabels(var)-1;
      }
      for(IndexType var=1; var<gm_.numberOfVariables(); ++var){
         OPENGM_ASSERT( variableOffset_[var-1]< variableOffset_[var]);
      }

      for(IndexType f=0; f<gm_.numberOfFactors(); ++f){
         if(gm_[f].numberOfVariables()==1)
            numEdges_ += gm_[f].numberOfLabels(0)-2;
         if(gm_[f].numberOfVariables()==2){
            numEdges_ += (gm_[f].numberOfLabels(0)-1);//*(gm_[f].numberOfLabels(1)-1);
            ++numSOF;
         }
      }

      if(param_.rounds_>0){
         std::cout << "Large" <<std::endl;
         qpbo_ = new kolmogorov::qpbo::QPBO<GraphValueType > (numNodes_, numEdges_); // max number of nodes & edges
         qpbo_->AddNode(numNodes_);
      }
      else{
         std::cout << "Small" <<std::endl;      
         qpbo_ = new kolmogorov::qpbo::QPBO<GraphValueType > (gm_.numberOfVariables(), numSOF); // max number of nodes & edges
         qpbo_->AddNode(gm_.numberOfVariables());
      }
   } 

   template<class GM, class ACC>
   MQPBO<GM,ACC>::~MQPBO
   (
      )
   {
      delete qpbo_;
   }
      
   /// reset assumes that the structure of
   /// the graphical model has not changed
   template<class GM, class ACC>
   inline void
   MQPBO<GM,ACC>::reset()
   {
      ///TODO
   }
   
   /// set starting point
   template<class GM, class ACC>
   inline void 
   MQPBO<GM,ACC>::setStartingPoint
   (
      typename std::vector<typename GM::LabelType>::const_iterator begin
   )
   { 
      ///TODO
   }
   
   template<class GM, class ACC>
   inline std::string
   MQPBO<GM,ACC>::name() const
   {
      return "MQPBO";
   }

   template<class GM, class ACC>
   inline const typename MQPBO<GM,ACC>::GmType&
   MQPBO<GM,ACC>::graphicalModel() const
   {
      return gm_;
   } 


   template<class GM, class ACC>
   inline void
   MQPBO<GM,ACC>::AddUnaryTerm(int var, ValueType v0, ValueType v1){
      qpbo_->AddUnaryTerm(var, scale*v0, scale*v1);
   }
   
   template<class GM, class ACC>
   inline void
   MQPBO<GM,ACC>::AddPairwiseTerm(int var0, int var1,ValueType v00,ValueType v01,ValueType v10,ValueType v11){
      qpbo_->AddPairwiseTerm(var0, var1,scale*v00,scale*v01,scale*v10,scale*v11);
   } 
       
   template<class GM, class ACC>
   inline InferenceTermination
   MQPBO<GM,ACC>::testQuess(LabelType guess)
   {
      qpbo_->Reset();
      qpbo_->AddNode(gm_.numberOfVariables());
      for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {
         if(gm_[f].numberOfVariables() == 0) {
            ;
         }
         else if(gm_[f].numberOfVariables() == 1) {
            const LabelType numLabels =  gm_[f].numberOfLabels(0);
            const IndexType var = gm_[f].variableIndex(0);
            
            ValueType v0 = gm_[f](&guess);
            ValueType v1; ACC::neutral(v1);
            for(LabelType i=0; i<guess; ++i)
               ACC::op(gm_[f](&i),v1);
            for(LabelType i=guess+1; i<numLabels; ++i)
               ACC::op(gm_[f](&i),v1);
            AddUnaryTerm(var, v0, v1);
         }
         else if(gm_[f].numberOfVariables() == 2) {
            const IndexType var0 = gm_[f].variableIndex(0);
            const IndexType var1 = gm_[f].variableIndex(1); 
            
            LabelType c[2] = {guess,guess};
            LabelType c2[2] = {0,1};

            ValueType v00 = gm_[f](c);
            ValueType v01 = gm_[f](c2);
            ValueType v10 = v01; 
            ValueType v11 = std::min(v00,v01);
            AddPairwiseTerm(var0, var1,v00,v01,v10,v11);
         }
      }
      qpbo_->MergeParallelEdges();
      qpbo_->Solve();
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         if(qpbo_->GetLabel(var)==0){
            for(LabelType l=0; l<gm_.numberOfLabels(var); ++l){
               partialOptimality_[var][l] =opengm::Tribool::False;   
            } 
            partialOptimality_[var][guess] =opengm::Tribool::True; 
            optimal_[var]=true;
            label_[var]=guess;
         }
      }
      return NORMAL;
   }


   template<class GM, class ACC>
   inline InferenceTermination
   MQPBO<GM,ACC>::testQuess(std::vector<LabelType> &guess)
   {
      qpbo_->Reset();
      qpbo_->AddNode(gm_.numberOfVariables());
      
      for(size_t var=0; var<gm_.numberOfVariables(); ++var){
         std::vector<ValueType> v(gm_.numberOfLabels(var),0);
         for(size_t i=0; i<gm_.numberOfFactors(var); ++i){
            size_t f =  gm_.factorOfVariable(var, i);
            if(gm_[f].numberOfVariables()==1){
               for(size_t j=0; j<v.size(); ++j){
                  v[j] += gm_[f](&j);
               }
               
            }
            else if(gm_[f].numberOfVariables() == 2) {
               LabelType c[] = {guess[gm_[f].variableIndex(0)],guess[gm_[f].variableIndex(1)]};
               if(gm_[f].variableIndex(0)==var){
                  for(c[0]=0; c[0]<guess[var]; ++c[0]){
                     v[c[0]] += gm_[f](c);
                  }
                  for(c[0]=guess[var]+1; c[0]<v.size(); ++c[0]){
                     v[c[0]] += gm_[f](c);
                  } 
               }
               else if(gm_[f].variableIndex(1)==var){
                  for(c[1]=0; c[1]<guess[var]; ++c[1]){
                     v[c[1]] += gm_[f](c);
                  }
                  for(c[1]=guess[var]+1; c[1]<v.size(); ++c[1]){
                     v[c[1]] += gm_[f](c);
                  } 
               }
               else{
                  OPENGM_ASSERT(false);
               }
            }
         } 
         ValueType v0 = v[guess[var]];
         ValueType v1; ACC::neutral(v1);
         for(size_t j=0; j<guess[var]; ++j){
            ACC::op(v[j],v1);
         }
         for(size_t j=guess[var]+1; j<v.size(); ++j){
            ACC::op(v[j],v1);
         } 
         AddUnaryTerm(var, v0, v1);
      }


      for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {    
         if(gm_[f].numberOfVariables() < 2) {
            continue;
         }
         else if(gm_[f].numberOfVariables() == 2) {
            const IndexType var0 = gm_[f].variableIndex(0);
            const IndexType var1 = gm_[f].variableIndex(1); 
            
            LabelType c[2] = {guess[var0],guess[var1]};
            LabelType c0[2] = {guess[var0],guess[var1]};
            LabelType c1[2] = {guess[var0],guess[var1]};
            ValueType v00 = gm_[f](c);
            ValueType v01 = 0;
            ValueType v10 = 0;
            ValueType v11; ACC::neutral(v11);

            for(c[0]=0; c[0]<gm_[f].numberOfLabels(0); ++c[0]){
               for(c[1]=0; c[1]<gm_[f].numberOfLabels(1); ++c[1]){
                  if(c[0]==guess[var0] || c[1]==guess[var1]){
                     continue;
                  }
                  else{
                     c0[0]=c[0];
                     c1[1]=c[1];
                     ValueType v = gm_[f](c) - gm_[f](c0) - gm_[f](c1);
                     ACC::op(v,v11);
                  }
               }
            }
            AddPairwiseTerm(var0, var1,v00,v01,v10,v11);
         }
      }
      qpbo_->MergeParallelEdges();
      qpbo_->Solve();
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         if(qpbo_->GetLabel(var)==0){
            for(LabelType l=0; l<gm_.numberOfLabels(var); ++l){
               partialOptimality_[var][l] =opengm::Tribool::False;   
            } 
            partialOptimality_[var][guess[var]] =opengm::Tribool::True; 
            optimal_[var]=true;
            label_[var]=guess[var];
         }
      }
      return NORMAL;
   }

   template<class GM, class ACC>
   inline InferenceTermination
   MQPBO<GM,ACC>::testPermutation(PermutationType permutationType)
   {
      //Set up MQPBO for current partial optimality
      std::vector<IndexType> var2VarR(gm_.numberOfVariables());
      std::vector<IndexType> varR2Var;
      std::vector<size_t>    varROffset;
      size_t numBVar=0;
      for(size_t var = 0; var < gm_.numberOfVariables(); ++var) {
         if(optimal_[var]){
            ;//do nothing
         }
         else{
            varROffset.push_back(numBVar);
            numBVar = numBVar + gm_.numberOfLabels(var)-1;
            var2VarR[var]=varR2Var.size();
            varR2Var.push_back(var);
         }
      }
      std::cout <<  varR2Var.size() <<" / "<<gm_.numberOfVariables()<<std::endl;

      //Find Permutation
      if(permutationType==NONE){ 
         for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
            for(LabelType l=0; l<gm_.numberOfLabels(var); ++l){
               permutation_[var][l]=l;
            }
         }
      }
      else if(permutationType==RANDOM){ 
         srand ( unsigned ( time (NULL) ) );
         for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
            LabelType numStates = gm_.numberOfLabels(var);
            //IDENTYTY PERMUTATION
            for(LabelType i=0; i<numStates;++i){
               permutation_[var][i]=i;
            }
            //SHUFFEL PERMUTATION  
            std::random_shuffle(permutation_[var].begin(),permutation_[var].end());
         }
      }
      else if(permutationType==MINMARG){
         typedef typename opengm::GraphicalModel<ValueType, OperatorType, opengm::ViewFixVariablesFunction<GM>, typename GM::SpaceType> SUBGM;

         std::vector<LabelType> numberOfLabels(varR2Var.size());
         for(size_t i=0; i<varR2Var.size(); ++i)
            numberOfLabels[i] = gm_.numberOfLabels(varR2Var[i]);
         typename GM::SpaceType subspace(numberOfLabels.begin(),numberOfLabels.end());
         SUBGM gm(subspace);
         for(IndexType f=0; f<gm_.numberOfFactors();++f){
            std::vector<PositionAndLabel<IndexType, LabelType> > fixed;
            std::vector<IndexType> vars;
            for(IndexType i=0; i<gm_[f].numberOfVariables();++i){
               const IndexType var = gm_[f].variableIndex(i);
               if(optimal_[var]){
                  fixed.push_back(PositionAndLabel<IndexType, LabelType>(i,label_[var]));
               }
               else{
                  vars.push_back(var2VarR[var]);
               }
            }
            opengm::ViewFixVariablesFunction<GM> func(gm_[f], fixed);
            gm.addFactor(gm.addFunction(func),vars.begin(),vars.end());
         }
      
         typedef typename opengm::MessagePassing<SUBGM, ACC,opengm::BeliefPropagationUpdateRules<SUBGM,ACC>, opengm::MaxDistance> LBP;  
         typename LBP::Parameter para;
         para.maximumNumberOfSteps_ = 100;
         para.damping_ = 0.5;
         LBP bp(gm,para);
         bp.infer();
         
         //for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         for(IndexType varR=0; varR<gm.numberOfVariables(); ++varR){
            IndexType var = varR2Var[varR];
            LabelType numStates = gm_.numberOfLabels(var);
            typename GM::IndependentFactorType marg;
            bp.marginal(varR, marg);
         
            //SHUFFEL PERMUTATION 
            std::vector<LabelType> list(numStates);
            for(LabelType i=0; i<numStates;++i){
               list[i]=i;
            }
            LabelType t;
            for(LabelType i=0; i<numStates;++i){
               for(LabelType j=i+1; i<numStates;++i){
                  if(marg(&list[j])<marg(&list[i])){
                     t = list[i];
                     list[i]=list[j];
                     list[j]=t;
                  }
               }
            }
            for(LabelType i=0; i<numStates;++i){
               permutation_[var][i] = list[i];
            }
         }
      }
      else{
         throw RuntimeError("Error: Unknown Permutation!");
      }
      //CALCULATE INVERSE PERMUTATION
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){   
         for(LabelType l=0; l<gm_.numberOfLabels(var); ++l){
            inversePermutation_[var][permutation_[var][l]]=l;
         }
      }
      
      
      //Build Graph
      ValueType constValue = 0;
      qpbo_->Reset();
      qpbo_->AddNode(numBVar);
      //qpbo_->AddNode(numNodes_);
      
      for(IndexType varR = 0; varR < varR2Var.size(); ++varR) {
         IndexType var = varR2Var[varR];
         for(size_t l = 0; l+1<gm_.numberOfLabels(var); ++l){
            AddUnaryTerm((int) (varROffset[varR]+l), 0.0, 0.0);
         } 
         for(LabelType l=1; l+1<gm_.numberOfLabels(var); ++l){
            AddPairwiseTerm((int) (varROffset[varR]+l-1), (int) (varROffset[varR]+l), 0.0,  1e30, 0.0, 0.0);
         }
      }
      /*      
      for(size_t var = 0; var < gm_.numberOfVariables(); ++var) {
         for(size_t l = 0; l+1<gm_.numberOfLabels(var); ++l){
            AddUnaryTerm((int) (variableOffset_[var]+l), 0.0, 0.0);
         } 
         for(LabelType l=1; l+1<gm_.numberOfLabels(var); ++l){
            AddPairwiseTerm((int) (variableOffset_[var]+l-1), (int) (variableOffset_[var]+l), 0.0,  1000000.0, 0.0, 0.0);
         }
      }
       */

      for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {
         if(gm_[f].numberOfVariables() == 0) {
            const LabelType l = 0;
            constValue += gm_[f](&l);
         }
         else if(gm_[f].numberOfVariables() == 1) {
            const LabelType numLabels =  gm_[f].numberOfLabels(0);
            const IndexType var = gm_[f].variableIndex(0);
            if(optimal_[var]){
               constValue += gm_[f](&(label_[var]));  
            }
            else{
               LabelType l0 = inversePermutation_[var][0];
               LabelType l1;
               constValue += gm_[f](&l0);
               const IndexType varR = var2VarR[var];
               for(LabelType i=1 ; i<numLabels; ++i){
                  l0 = inversePermutation_[var][i-1];
                  l1 = inversePermutation_[var][i];
                  AddUnaryTerm((int) (varROffset[varR]+i-1),  0.0, gm_[f](&l1)-gm_[f](&l0));       
                  //AddUnaryTerm((int) (variableOffset_[var]+i-1),  0.0, gm_[f](&l1)-gm_[f](&l0));
               }      
            }
         }
         else if(gm_[f].numberOfVariables() == 2) {
            const IndexType var0 = gm_[f].variableIndex(0);
            const IndexType var1 = gm_[f].variableIndex(1); 
            const IndexType varR0 = var2VarR[var0];
            const IndexType varR1 = var2VarR[var1]; 
            
            if(optimal_[var0]&&optimal_[var1]){
               LabelType l[2] = { label_[var0], label_[var1]};
               constValue += gm_[f](l);   
            }
            else if(optimal_[var0]){
               const LabelType numLabels =  gm_[f].numberOfLabels(1);
               LabelType l0[2] = { label_[var0], inversePermutation_[var1][0]};
               LabelType l1[2] = { label_[var0], 0};
               constValue += gm_[f](l0);
               for(LabelType i=1 ; i<numLabels; ++i){
                  l0[1] = inversePermutation_[var1][i-1];
                  l1[1] = inversePermutation_[var1][i];
                  //AddUnaryTerm((int) (variableOffset_[var1]+i-1),  0.0, gm_[f](l1)-gm_[f](l0)); 
                  AddUnaryTerm((int) (varROffset[varR1]+i-1),  0.0, gm_[f](l1)-gm_[f](l0));
               }
            }
            else if(optimal_[var1]){
               const LabelType numLabels =  gm_[f].numberOfLabels(0);  
               LabelType l0[2] = { inversePermutation_[var0][0], label_[var1]};
               LabelType l1[2] = { 0, label_[var1]};
               constValue += gm_[f](l0);
               for(LabelType i=1 ; i<numLabels; ++i){
                  l0[0] = inversePermutation_[var0][i-1];
                  l1[0] = inversePermutation_[var0][i];
                  AddUnaryTerm((int) (varROffset[varR0]+i-1),  0.0, gm_[f](l1)-gm_[f](l0));
                  //AddUnaryTerm((int) (variableOffset_[var0]+i-1),  0.0, gm_[f](l1)-gm_[f](l0));
               }
            }
            else{
               {
                  const LabelType l[2]={inversePermutation_[var0][0],inversePermutation_[var1][0]}; 
                  constValue += gm_[f](l);
               } 
               for(size_t i=1; i<gm_[f].numberOfLabels(0);++i){
                  const LabelType l1[2]={inversePermutation_[var0][i]  ,inversePermutation_[var1][0]};
                  const LabelType l2[2]={inversePermutation_[var0][i-1],inversePermutation_[var1][0]};
                  AddUnaryTerm((int) (varROffset[varR0]+i-1), 0.0, gm_[f](l1)-gm_[f](l2));
                  //AddUnaryTerm((int) (variableOffset_[var0]+i-1), 0.0, gm_[f](l1)-gm_[f](l2));
               } 
               for(size_t i=1; i<gm_[f].numberOfLabels(1);++i){
                  const LabelType l1[2]={inversePermutation_[var0][0],inversePermutation_[var1][i]};
                  const LabelType l2[2]={inversePermutation_[var0][0],inversePermutation_[var1][i-1]};
                  AddUnaryTerm((int) (varROffset[varR1]+i-1), 0.0, gm_[f](l1)-gm_[f](l2));
                  //AddUnaryTerm((int) (variableOffset_[var1]+i-1), 0.0, gm_[f](l1)-gm_[f](l2));
               }
               for(size_t i=1; i<gm_[f].numberOfLabels(0);++i){
                  for(size_t j=1; j<gm_[f].numberOfLabels(1);++j){
                     const int node0 = varROffset[varR0]+i-1;
                     const int node1 = varROffset[varR1]+j-1;
                     //const int node0 = variableOffset_[var0]+i-1;
                     //const int node1 = variableOffset_[var1]+j-1;
                     ValueType v = 0;
                     int l[2] = {(int)inversePermutation_[var0][i],(int)inversePermutation_[var1][j]};  v += gm_[f](l);
                     l[0]=inversePermutation_[var0][i-1];                                    v -= gm_[f](l);
                     l[1]=inversePermutation_[var1][j-1];                                    v += gm_[f](l);
                     l[0]=inversePermutation_[var0][i];                                      v -= gm_[f](l);
                     if(v!=0.0)
                        AddPairwiseTerm(node0, node1,0.0,0.0,0.0,v);
                  }
               }
            }
         }
      }
      qpbo_->MergeParallelEdges();
         
      //Optimize
      
      qpbo_->Solve();
      if(!param_.strongPersistency_)
         qpbo_->ComputeWeakPersistencies();
      //   if(!parameter_.strongPersistency_) {
      //      qpbo_->ComputeWeakPersistencies();
      //   } 

      bound_ = constValue + 0.5 * qpbo_->ComputeTwiceLowerBound();
      
      /*PROBEING*/
      if(param_.probing_) { 
         std::cout << "Start Probing ..."<<std::endl;
         // Initialize mapping for probe
         int *mapping = new int[numBVar];
         //int *mapping = new int[numNodes_];
         for(int i = 0; i < static_cast<int>(numBVar); ++i) {
            //for(int i = 0; i < static_cast<int>(numNodes_); ++i) {
            qpbo_->SetLabel(i, qpbo_->GetLabel(i));
            mapping[i] = i * 2;
         }
         typename kolmogorov::qpbo::QPBO<GraphValueType>::ProbeOptions options;
         options.C = 1000000000;
         if(!param_.strongPersistency_)
            options.weak_persistencies = 1;
         else
            options.weak_persistencies = 0;
         qpbo_->Probe(mapping, options);
         if(!param_.strongPersistency_)
            qpbo_->ComputeWeakPersistencies();      
         
         for(IndexType var=0; var<gm_.numberOfVariables();++var){
            if(optimal_[var]) continue;
            IndexType varR = var2VarR[var];
            //Lable==0
            {
               int l = qpbo_->GetLabel(mapping[varROffset[varR]]/2);
               if(l>=0) l = (l + mapping[varROffset[varR]]) % 2;
               //int l = qpbo_->GetLabel(mapping[variableOffset_[var]]/2);
               //if(l>=0) l = (l + mapping[variableOffset_[var]]) % 2;
               if(l==0)     {partialOptimality_[var][inversePermutation_[var][0]]&=opengm::Tribool::True;}
               else if(l==1){partialOptimality_[var][inversePermutation_[var][0]]&=opengm::Tribool::False;}
               else         {partialOptimality_[var][inversePermutation_[var][0]]&=opengm::Tribool::Maybe;}
            }
            //Label==max
            {
               int l = qpbo_->GetLabel(mapping[varROffset[varR]+gm_.numberOfLabels(var)-2]/2);
               if(l>=0) l = (l + mapping[varROffset[varR]+gm_.numberOfLabels(var)-2]) % 2;      
               //int l = qpbo_->GetLabel(mapping[variableOffset_[var]+gm_.numberOfLabels(var)-2]/2);
               //if(l>=0) l = (l + mapping[variableOffset_[var]+gm_.numberOfLabels(var)-2]) % 2;      
               if(l==0)     {partialOptimality_[var][inversePermutation_[var][gm_.numberOfLabels(var)-1]]&=opengm::Tribool::False;}
               else if(l==1){partialOptimality_[var][inversePermutation_[var][gm_.numberOfLabels(var)-1]]&=opengm::Tribool::True;}
               else         {partialOptimality_[var][inversePermutation_[var][gm_.numberOfLabels(var)-1]]&=opengm::Tribool::Maybe;}
            }
            //ELSE
            
            for(LabelType l=1; l+1<gm_.numberOfLabels(var);++l)
            {
               int l1 = qpbo_->GetLabel(mapping[varROffset[varR]+l-1]/2);
               int l2 = qpbo_->GetLabel(mapping[varROffset[varR]+l]/2);
               if(l1>=0) l1 = (l1 + mapping[varROffset[varR]+l-1]) % 2;
               if(l2>=0) l2 = (l2 + mapping[varROffset[varR]+l]) % 2;
               //int l1 = qpbo_->GetLabel(mapping[variableOffset_[var]+l-1]/2);
               //int l2 = qpbo_->GetLabel(mapping[variableOffset_[var]+l]/2);
               //if(l1>=0) l1 = (l1 + mapping[variableOffset_[var]+l-1]) % 2;
               //if(l2>=0) l2 = (l2 + mapping[variableOffset_[var]+l]) % 2;
               
               OPENGM_ASSERT(!(l1==0 && l2==1));
               if(l1==1 && l2==0) {partialOptimality_[var][inversePermutation_[var][l]]&=opengm::Tribool::True;}
               else if(l2==1)     {partialOptimality_[var][inversePermutation_[var][l]]&=opengm::Tribool::False;}
               else if(l1==0)     {partialOptimality_[var][inversePermutation_[var][l]]&=opengm::Tribool::False;}
               //else               {partialOptimality_[var][inversePermutation_[var][l]]&=opengm::Tribool::Maybe;}
            }  
         }
         delete mapping;
      }
      else{
         for(IndexType var=0; var<gm_.numberOfVariables();++var){
            if(optimal_[var]) continue;
            IndexType varR = var2VarR[var];
            //Lable==0
            {
               int l = qpbo_->GetLabel(varROffset[varR]);
               //int l = qpbo_->GetLabel(variableOffset_[var]);
               if(l==0){
                  OPENGM_ASSERT(!(partialOptimality_[var][inversePermutation_[var][0]]==opengm::Tribool::False));
                  partialOptimality_[var][inversePermutation_[var][0]]&=opengm::Tribool::True;
               }
               else if(l==1){
                  OPENGM_ASSERT(!(partialOptimality_[var][inversePermutation_[var][0]]==opengm::Tribool::True));
                  partialOptimality_[var][inversePermutation_[var][0]]&=opengm::Tribool::False;
               }
               //  else         {partialOptimality_[var][permutation_[var][0]]&=opengm::Tribool::Maybe;}
            }
            //Label==max
            {
               int l = qpbo_->GetLabel(varROffset[varR]+gm_.numberOfLabels(var)-2);
               //int l = qpbo_->GetLabel(variableOffset_[var]+gm_.numberOfLabels(var)-2);
               if(l==0){
                  OPENGM_ASSERT(!(partialOptimality_[var][inversePermutation_[var][gm_.numberOfLabels(var)-1]]==opengm::Tribool::True));
                  partialOptimality_[var][inversePermutation_[var][gm_.numberOfLabels(var)-1]]&=opengm::Tribool::False;
               }
               else if(l==1){
                  OPENGM_ASSERT(!(partialOptimality_[var][inversePermutation_[var][gm_.numberOfLabels(var)-1]]==opengm::Tribool::False));        
                  partialOptimality_[var][inversePermutation_[var][gm_.numberOfLabels(var)-1]]&=opengm::Tribool::True;
               }
               //else         {partialOptimality_[var][permutation_[var][gm_.numberOfLabels(var)-1]]&=opengm::Tribool::Maybe;}
            }
            //ELSE
            
            for(LabelType l=1; l+1<gm_.numberOfLabels(var);++l)
            {
               int l1 = qpbo_->GetLabel(varROffset[varR]+l-1);
               int l2 = qpbo_->GetLabel(varROffset[varR]+l);
               //int l1 = qpbo_->GetLabel(variableOffset_[var]+l-1);
               //int l2 = qpbo_->GetLabel(variableOffset_[var]+l);
               OPENGM_ASSERT(!(l1==0 && l2==1));
               if(l1==1 && l2==0) {
                  OPENGM_ASSERT(!(partialOptimality_[var][inversePermutation_[var][l]]==opengm::Tribool::False));
                  partialOptimality_[var][inversePermutation_[var][l]]&=opengm::Tribool::True;
               }
               else if(l2==1){
                  OPENGM_ASSERT(!(partialOptimality_[var][inversePermutation_[var][l]]==opengm::Tribool::True));
                  partialOptimality_[var][inversePermutation_[var][l]]&=opengm::Tribool::False;
               }
               else if(l1==0){
                  OPENGM_ASSERT(!(partialOptimality_[var][inversePermutation_[var][l]]==opengm::Tribool::True));
                  partialOptimality_[var][inversePermutation_[var][l]]&=opengm::Tribool::False;
               }
               //else{  
               //   partialOptimality_[var][permutation_[var][l]]&=opengm::Tribool::Maybe;
               //}
            }  
         }
      }
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         if(optimal_[var]) continue;
         LabelType countTRUE = 0;
         LabelType countFALSE = 0;
         for(LabelType l=1; l+1<gm_.numberOfLabels(var);++l){
            if(partialOptimality_[var][l]==opengm::Tribool::True) 
               ++countTRUE;
            if(partialOptimality_[var][l]==opengm::Tribool::False) 
               ++countFALSE;
         }
         if(countTRUE==1){
            optimal_[var]=true;
            for(LabelType l=1; l+1<gm_.numberOfLabels(var);++l){
               if(partialOptimality_[var][l]==opengm::Tribool::True)
                  label_[var]=l;
               else
                  partialOptimality_[var][l]=opengm::Tribool::False;
            }
         }
         if(countFALSE+1==gm_.numberOfLabels(var)){
            optimal_[var]=true;
            for(LabelType l=1; l+1<gm_.numberOfLabels(var);++l){
               if(partialOptimality_[var][l]==opengm::Tribool::Maybe){
                  label_[var]=l;
                  partialOptimality_[var][l]=opengm::Tribool::True;
               }
            }
         }
      }
      return NORMAL; 
   }

   template<class GM, class ACC>
   InferenceTermination MQPBO<GM,ACC>::infer
   ()
   {
      EmptyVisitorType visitor;
      return infer(visitor);
   }
     
   template<class GM, class ACC>
   template<class VisitorType>
   InferenceTermination MQPBO<GM,ACC>::infer
   (
      VisitorType& visitor
   )
   { 
      visitor.addLog("optimality");
      visitor.addLog("optimalityV");
      if(param_.rounds_>1 && param_.strongPersistency_==false)
         std::cout << "WARNING: Using weak persistency and several rounds may lead to wrong results if solution is not unique!"<<std::endl;

      LabelType maxNumberOfLabels = 0;
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         maxNumberOfLabels = std::max(maxNumberOfLabels, gm_.numberOfLabels(var));
      }
      bool isPotts = true;

      for(IndexType f=0; f< gm_.numberOfFactors(); ++f){
         if(gm_[f].numberOfVariables()<2) continue;
         isPotts &= gm_[f].isPotts();
         if(!isPotts) break;
      }

      visitor.begin(*this);

      if(param_.useKovtunsMethod_){
         if(isPotts){
            std::cout << "Use Kovtuns method for potts"<<std::endl;
            for(LabelType l=0; l<maxNumberOfLabels; ++l) {
               testQuess(l);
               double xoptimality = optimality(); 
               double xoptimalityV = optimalityV();
               visitor(*this);
               visitor.log("optimality",xoptimality);
               visitor.log("optimalityV",xoptimalityV);

               //std::cout << "partialOptimality  : " << optimality() << std::endl; 
            }
         }
         else{
            std::cout << "Use Kovtuns method for non-potts is not supported yet"<<std::endl;
            /*
            for(LabelType l=0; l<maxNumberOfLabels; ++l){
               std::vector<LabelType> guess(gm_.numberOfVariables(),l);
               for(IndexType var=0; var<gm_.numberOfVariables();++var){
                  if(l>=gm_.numberOfLabels(var)){
                     guess[var]=l-1;
                  }
               }
               testQuess(guess);
               double xoptimality = optimality();
               visitor(*this,this->value(),bound(),"partialOptimality",xoptimality);
               //std::cout << "partialOptimality  : " << optimality() << std::endl;
            }
            */
         }
      }

      if(param_.rounds_>0){
         std::cout << "Start "<<param_.rounds_ << " of multilabel QPBO for different permutations" <<std::endl;
         for(size_t rr=0; rr<param_.rounds_;++rr){
            testPermutation(param_.permutationType_);
            double xoptimality = optimality();
            double xoptimalityV = optimalityV();
            visitor(*this);
            visitor.log("optimality",xoptimality);
            visitor.log("optimalityV",xoptimalityV);

            //std::cout << "partialOptimality  : " << optimality() << std::endl;
         }
      }

#ifdef MQPBOHotFixOutPutPartialOPtimalityMap
      hid_t fid = marray::hdf5::createFile("mqpbotmp.h5");
      std::vector<double> optimal;
      for(size_t i=0; i<optimal_.size();++i)
         optimal.push_back((double)(optimal_[i]));
      marray::hdf5::save(fid, "popt", optimal);
      marray::hdf5::closeFile(fid);
#endif  
   
      visitor.end(*this);
     
      return NORMAL;
   }
   
   template<class GM, class ACC>
   double 
   MQPBO<GM,ACC>::optimality
   () const
   { 
      size_t labeled   = 0;
      size_t unlabeled = 0; 
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         for(LabelType l=0; l<gm_.numberOfLabels(var);++l){
            if(partialOptimality_[var][l]==opengm::Tribool::Maybe)
               ++unlabeled;
            else
               ++labeled;
         }
      }
      return labeled*1.0/(labeled+unlabeled);
   }  

   template<class GM, class ACC>
   double 
   MQPBO<GM,ACC>::optimalityV
   () const
   { 
      size_t labeled   = 0; 
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         for(LabelType l=0; l<gm_.numberOfLabels(var);++l){
            if(partialOptimality_[var][l]==opengm::Tribool::True){
               ++labeled;
               continue;
            }
         }
      }
      return labeled*1.0/gm_.numberOfVariables();
   } 

   template<class GM, class ACC>
   typename GM::ValueType 
   MQPBO<GM,ACC>::bound
   () const
   {
      return bound_;
   } 
   
   template<class GM, class ACC>
   typename GM::ValueType  MQPBO<GM,ACC>::value() const { 
      std::vector<LabelType> states;
      arg(states);
      return gm_.evaluate(states);
   }

   template<class GM, class ACC>
   inline InferenceTermination
   MQPBO<GM,ACC>::arg
   (
      std::vector<LabelType>& x,
      const size_t N
   ) const
   {
      if(N==1){
         x.resize(gm_.numberOfVariables(),0); 

         for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
            size_t countTrue  = 0;
            size_t countFalse = 0;
            size_t countMaybe = 0;
            x[var]=0;
            for(LabelType l=0; l<gm_.numberOfLabels(var);++l){
               if(partialOptimality_[var][l]==opengm::Tribool::Maybe){ 
                  x[var] = l;
                  ++countMaybe;
               }
               if(partialOptimality_[var][l]==opengm::Tribool::True){ 
                  x[var] = l;
                  ++countTrue;
               } 
               if(partialOptimality_[var][l]==opengm::Tribool::False){ 
                  ++countFalse;
               }
            }
            OPENGM_ASSERT(countTrue+countFalse+countMaybe == gm_.numberOfLabels(var));
            OPENGM_ASSERT(countTrue<2); 
            OPENGM_ASSERT(countFalse<gm_.numberOfLabels(var));
         }
         return NORMAL;
      }
      else {
         return UNKNOWN;
      }
   }
} // namespace opengm

#endif // #ifndef OPENGM_MQPBO_HXX
