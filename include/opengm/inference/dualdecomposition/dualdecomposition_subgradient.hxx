#pragma once
#ifndef OPENGM_DUALDDECOMPOSITION_SUBGRADIENT_HXX
#define OPENGM_DUALDDECOMPOSITION_SUBGRADIENT_HXX

#include <vector>
#include <list>
#include <typeinfo>

#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/inference/dualdecomposition/dualdecomposition_base.hxx"

namespace opengm {

   /// \brief Dual-Decomposition-Subgradient\n\n
   /// Inference based on dual decomposition using sub-gradient descent\n
   /// Reference:\n
   /// Kappes, J. H. and Savchynskyy, B. and Schnoerr, C.:
   /// "A Bundle Approach To Efficient MAP-Inference by Lagrangian Relaxation".
   /// In CVPR 2012, 2012. 
   /// \ingroup inference 
   template<class GM, class INF, class DUALBLOCK >
   class DualDecompositionSubGradient 
      : public Inference<GM,typename INF::AccumulationType>,  public DualDecompositionBase<GM, DUALBLOCK > 
   {
   public:
      typedef GM                                                 GmType;
      typedef GM                                                 GraphicalModelType;
      typedef typename INF::AccumulationType                     AccumulationType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef visitors::VerboseVisitor<DualDecompositionSubGradient<GM, INF,DUALBLOCK> > VerboseVisitorType;
      typedef visitors::TimingVisitor<DualDecompositionSubGradient<GM, INF,DUALBLOCK> >  TimingVisitorType;
      typedef  visitors::EmptyVisitor<DualDecompositionSubGradient<GM, INF,DUALBLOCK> >   EmptyVisitorType;

      typedef INF                                                InfType;
      typedef DUALBLOCK                                          DualBlockType;
      typedef DualDecompositionBase<GmType, DualBlockType>       DDBaseType;
     
      typedef typename DualBlockType::DualVariableType           DualVariableType;
      typedef typename DDBaseType::SubGmType                     SubGmType;
      typedef typename DualBlockType::SubFactorType              SubFactorType;
      typedef typename DualBlockType::SubFactorListType          SubFactorListType; 
      typedef typename DDBaseType::SubVariableType               SubVariableType;
      typedef typename DDBaseType::SubVariableListType           SubVariableListType; 

      class Parameter : public DualDecompositionBaseParameter{
      public:
         /// Parameter for Subproblems
         typename InfType::Parameter subPara_;
         bool useAdaptiveStepsize_;
         bool useProjectedAdaptiveStepsize_;
         Parameter() : useAdaptiveStepsize_(false), useProjectedAdaptiveStepsize_(false){};
      };

      using  DualDecompositionBase<GmType, DualBlockType >::gm_;
      using  DualDecompositionBase<GmType, DualBlockType >::subGm_;
      using  DualDecompositionBase<GmType, DualBlockType >::dualBlocks_;
      using  DualDecompositionBase<GmType, DualBlockType >::numDualsOvercomplete_;
      using  DualDecompositionBase<GmType, DualBlockType >::numDualsMinimal_;

      DualDecompositionSubGradient(const GmType&);
      DualDecompositionSubGradient(const GmType&, const Parameter&);
      virtual std::string name() const {return "DualDecompositionSubGradient";};
      virtual const GmType& graphicalModel() const {return gm_;};
      virtual InferenceTermination infer();
      template <class VISITOR> InferenceTermination infer(VISITOR&);
      virtual ValueType bound() const;
      virtual ValueType value() const;
      virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1)const;
     
   private: 
      virtual void allocate();
      virtual DualDecompositionBaseParameter& parameter();
      void dualStep(const size_t);
      template <class T_IndexType, class T_LabelType>
      void getPartialSubGradient(const size_t, const std::vector<T_IndexType>&, std::vector<T_LabelType>&)const;
      double euclideanProjectedSubGradientNorm();

      // Members
      std::vector<std::vector<LabelType> >  subStates_;
   
      Accumulation<ValueType,LabelType,AccumulationType> acUpperBound_;
      Accumulation<ValueType,LabelType,AccumulationType> acNegLowerBound_;
      ValueType upperBound_;
      ValueType lowerBound_;
      std::vector<ValueType> values_;

      Parameter              para_;
      std::vector<ValueType> mem_;
  
      opengm::Timer primalTimer_;
      opengm::Timer dualTimer_;
      double primalTime_;
      double dualTime_;

   };  
      
//**********************************************************************************
   template<class GM, class INF, class DUALBLOCK>
   DualDecompositionSubGradient<GM,INF,DUALBLOCK>::DualDecompositionSubGradient(const GmType& gm)
      : DualDecompositionBase<GmType, DualBlockType >(gm)
   {
      this->init(para_);
      subStates_.resize(subGm_.size());
      for(size_t i=0; i<subGm_.size(); ++i)
         subStates_[i].resize(subGm_[i].numberOfVariables()); 
   }
   
   template<class GM, class INF, class DUALBLOCK>
   DualDecompositionSubGradient<GM,INF,DUALBLOCK>::DualDecompositionSubGradient(const GmType& gm, const Parameter& para)
      :   DualDecompositionBase<GmType, DualBlockType >(gm),para_(para)
   {  
      this->init(para_);  
      subStates_.resize(subGm_.size());
      for(size_t i=0; i<subGm_.size(); ++i)
         subStates_[i].resize(subGm_[i].numberOfVariables());
   }


////////////////////////////////////////////////////////////////////

   template <class GM, class INF, class DUALBLOCK> 
   void DualDecompositionSubGradient<GM,INF,DUALBLOCK>::allocate()  
   { 
      if(DDIsView<DualVariableType>::isView()){
         mem_.resize(numDualsOvercomplete_,0.0);
      }
      else 
         mem_.resize(1,0.0);
      //std::cout << mem_.size() <<std::flush;
      ValueType *data = &mem_[0];
      for(typename std::vector<DualBlockType>::iterator it=dualBlocks_.begin(); it!=dualBlocks_.end(); ++it){
         for(size_t i=0; i<(*it).duals_.size(); ++i){
            DualVariableType& dv = (*it).duals_[i];
            DualVarAssign(dv, data);
            if(DDIsView<DualVariableType>::isView()) 
               data += dv.size(); 
         }
      }
   }   
   template <class GM, class INF, class DUALBLOCK> 
   DualDecompositionBaseParameter& DualDecompositionSubGradient<GM,INF,DUALBLOCK>::parameter()
   {
      return para_;
   }

/////////////////////////
   template<class GM, class INF, class DUALBLOCK>
   InferenceTermination DualDecompositionSubGradient<GM,INF,DUALBLOCK>::
   infer() 
   {
      EmptyVisitorType visitor;
      return infer(visitor);
   }
   template<class GM, class INF, class DUALBLOCK> 
   template<class VISITOR>
   InferenceTermination DualDecompositionSubGradient<GM,INF,DUALBLOCK>::
   infer(VISITOR& visitor) 
   {
      std::cout.precision(15);
      //visitor.startInference();
      visitor.begin(*this);
          
      for(size_t iteration=0; iteration<para_.maximalNumberOfIterations_; ++iteration){  
       
	 // Solve Subproblems
	 ////primalTime_=0;
	 ////primalTimer_.tic();
         //omp_set_num_threads(para_.numberOfThreads_);
 
         //#pragma omp parallel for
         for(size_t subModelId=0; subModelId<subGm_.size(); ++subModelId){ 
            InfType inf(subGm_[subModelId],para_.subPara_);
            inf.infer(); 
            inf.arg(subStates_[subModelId]); 
         } 
         ////primalTimer_.toc(); 

         ////dualTimer_.tic();
         // Calculate lower-bound 
         std::vector<LabelType> temp;  
         std::vector<LabelType> temp2; 
         const std::vector<SubVariableListType>& subVariableLists = para_.decomposition_.getVariableLists();
         (*this).template getBounds<AccumulationType>(subStates_, subVariableLists, lowerBound_, upperBound_, temp);
         acNegLowerBound_(-lowerBound_,temp2);
         acUpperBound_(upperBound_, temp);
         
          //dualStep
         double stepsize;
         if(para_.useAdaptiveStepsize_){
            stepsize = para_.stepsizeStride_ * fabs(acUpperBound_.value() - lowerBound_) 
                       /(*this).subGradientNorm(1);
         }
         else if(para_.useProjectedAdaptiveStepsize_){
            double subgradientNorm = euclideanProjectedSubGradientNorm();
            stepsize = para_.stepsizeStride_ * fabs(acUpperBound_.value() - lowerBound_) 
                       /subgradientNorm/subgradientNorm;
         }
         else{
            double primalDualGap = fabs(acUpperBound_.value() + acNegLowerBound_.value());
            double subgradientNorm = euclideanProjectedSubGradientNorm();//(*this).subGradientNorm(1); 
            stepsize = para_.getStepsize(iteration,primalDualGap,subgradientNorm); 
//            std::cout << stepsize << std::endl;
         }
     
         if(typeid(AccumulationType) == typeid(opengm::Minimizer))
            stepsize *=1;
         else 
            stepsize *= -1;
                  
         std::vector<size_t> s;
         for(typename std::vector<DualBlockType>::const_iterator it = dualBlocks_.begin(); it != dualBlocks_.end(); ++it){
            const size_t numDuals = (*it).duals_.size();
            typename SubFactorListType::const_iterator lit = (*((*it).subFactorList_)).begin();
            s.resize((*lit).subIndices_.size());
            for(size_t i=0; i<numDuals; ++i){
	      getPartialSubGradient<size_t>((*lit).subModelId_, (*lit).subIndices_, s); 
               ++lit;     
               (*it).duals_[i](s.begin()) += stepsize;
               for(size_t j=0; j<numDuals; ++j){ 
                  (*it).duals_[j](s.begin()) -= stepsize/numDuals;
               }
            }
            //(*it).test();
         }          
         ////dualTimer_.toc();

         ////primalTime_ = primalTimer_.elapsedTime();
         ////dualTime_   = dualTimer_.elapsedTime();  
         if(visitor(*this)!= 0){
	   break;
	 }; 
         //visitor((*this), lowerBound_, -acNegLowerBound_.value(), upperBound_, acUpperBound_.value(), primalTime_, dualTime_); 

     
         // Test for Convergence
         ValueType o;
         AccumulationType::iop(0.0001,-0.0001,o);
         OPENGM_ASSERT(AccumulationType::bop(lowerBound_, upperBound_+o));
         OPENGM_ASSERT(AccumulationType::bop(-acNegLowerBound_.value(), acUpperBound_.value()+o));
         
         if(   fabs(acUpperBound_.value() + acNegLowerBound_.value())                       <= para_.minimalAbsAccuracy_
            || fabs((acUpperBound_.value()+ acNegLowerBound_.value())/acUpperBound_.value()) <= para_.minimalRelAccuracy_){
            //std::cout << "bound reached ..." <<std::endl;
	   visitor.end(*this);
	    //visitor.end((*this), acUpperBound_.value(), -acNegLowerBound_.value()); 
            return NORMAL;
         } 
      } 
      //std::cout << "maximal number of dual steps ..." <<std::endl;
      visitor.end(*this); 
           
      return NORMAL;
   }
   

   template<class GM, class INF, class DUALBLOCK>
   InferenceTermination DualDecompositionSubGradient<GM,INF,DUALBLOCK>::
   arg(std::vector<LabelType>& conf, const size_t n)const 
   {
      if(n!=1){
         return UNKNOWN;
      }
      else{ 
         acUpperBound_.state(conf);
         return NORMAL;
      }
   }

   template<class GM, class INF, class DUALBLOCK>
   typename GM::ValueType DualDecompositionSubGradient<GM,INF,DUALBLOCK>::value() const 
   {
      return acUpperBound_.value();
   }

   template<class GM, class INF, class DUALBLOCK>
   typename GM::ValueType DualDecompositionSubGradient<GM,INF,DUALBLOCK>::bound() const 
   {
      return -acNegLowerBound_.value();
   }


///////////////////////////////////////////////////////////////
 
   template <class GM, class INF, class DUALBLOCK> 
   void DualDecompositionSubGradient<GM,INF,DUALBLOCK>::dualStep(const size_t iteration)
   { 
    
   } 

   template <class GM, class INF, class DUALBLOCK> 
   template <class T_IndexType, class T_LabelType>
   inline void DualDecompositionSubGradient<GM,INF,DUALBLOCK>::getPartialSubGradient 
   ( 
      const size_t                             subModelId,
      const std::vector<T_IndexType>&    subIndices, 
      std::vector<T_LabelType> &                 s
   )const 
   {
      OPENGM_ASSERT(subIndices.size() == s.size());
      for(size_t n=0; n<s.size(); ++n){
         s[n] = subStates_[subModelId][subIndices[n]];
      }
   } 

   template <class GM, class INF, class DUALBLOCK> 
   double DualDecompositionSubGradient<GM,INF,DUALBLOCK>::euclideanProjectedSubGradientNorm()
   { 
      double norm = 0;
      std::vector<LabelType> s;
      marray::Marray<double> M;
      typename std::vector<DUALBLOCK>::const_iterator it;
      typename SubFactorListType::const_iterator                  lit;
      for(it = dualBlocks_.begin(); it != dualBlocks_.end(); ++it){
         const size_t numDuals = (*it).duals_.size(); 
         marray::Marray<double> M( (*it).duals_[0].shapeBegin(), (*it).duals_[0].shapeEnd() ,0.0);
         lit = (*((*it).subFactorList_)).begin();
         s.resize((*lit).subIndices_.size());
         for(size_t i=0; i<numDuals; ++i){
            getPartialSubGradient((*lit).subModelId_, (*lit).subIndices_, s); 
            ++lit; 
            M(s.begin()) += 1;
         }
         for(size_t i=0; i<M.size(); ++i){
            norm  += M(i)            * pow(1.0 - M(i)/numDuals,2);
            norm  += (numDuals-M(i)) * pow(      M(i)/numDuals,2);
         }
      }
      return sqrt(norm);
   } 


}
   
#endif

