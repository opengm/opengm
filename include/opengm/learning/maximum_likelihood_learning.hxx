#pragma once
#ifndef OPENGM_MAXIMUM_LIKELIHOOD_LEARNER_HXX
#define OPENGM_MAXIMUM_LIKELIHOOD_LEARNER_HXX

#include <vector>
#include <fstream>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/functions/view_convert_function.hxx>

namespace opengm {
   namespace learning {

      template<class DATASET>
      class MaximumLikelihoodLearner
      {
      public:
         typedef DATASET                     DatasetType;
         typedef typename DATASET::GMType    GMType;
         typedef typename GMType::ValueType  ValueType;
         typedef typename GMType::IndexType  IndexType;
         typedef typename GMType::LabelType  LabelType;
         typedef typename GMType::FactorType FactorType;
         typedef Weights<ValueType>          WeightType;  

         class Parameter{
         public:
	     IndexType maximumNumberOfIterations_;
	     ValueType gradientStep_;
	     ValueType weightAccuracy_;
             ValueType gradientStoppingCriteria_;
             bool infoFlag_;
             bool infoEveryStep_;

  	     size_t maxNumSteps_;
	     double reg_;
	     double temperature_;
	     Parameter():
	         maximumNumberOfIterations_(123),
	         gradientStep_(0.123),
		 weightAccuracy_(0.0000123),
		 gradientStoppingCriteria_(0.0000000123),
		 infoFlag_(true),
		 infoEveryStep_(false),
		 maxNumSteps_(10), 
		 reg_(1.0), 
		 temperature_(0.3)
	   {;}
         };

         class WeightGradientFunctor{
         public:
            WeightGradientFunctor(DatasetType& ds) : dataset_(ds) { gradient_.resize(ds.getNumberOfWeights(),0.0);}
            void setModel(size_t m) { modelID_ = m; } 
            void setMarg(typename GMType::IndependentFactorType* marg){marg_= marg;}
            double getGradient(size_t i) {return gradient_[i];}
            
            template<class F>
            void operator()(const F & function ){
               std::vector<LabelType> labelVector(marg_->numberOfVariables());
               for(size_t i=0; i<marg_->numberOfVariables(); ++i)
                  labelVector[i] = dataset_.getGT(modelID_)[marg_->variableIndex(i)]; 
               for(size_t i=0; i<function.numberOfWeights();++i){
		  size_t wID = function.weightIndex(i);
                  gradient_[wID] -= function.weightGradient(i, labelVector.begin());
               } 
               
               opengm::ShapeWalker<typename F::FunctionShapeIteratorType> shapeWalker(function.functionShapeBegin(), function.dimension());
               for(size_t i=0;i<function.size();++i, ++shapeWalker) {                   
                  for(size_t i=0; i<function.numberOfWeights();++i){
                     size_t wID = function.weightIndex(i);
                     gradient_[wID] += (*marg_)(shapeWalker.coordinateTuple().begin()) * function.weightGradient(i, shapeWalker.coordinateTuple().begin() );
                  }
               }              
            }
            
         private:
            DatasetType&                            dataset_;
            size_t                                  modelID_;
            std::vector<double>                     gradient_;  
            typename GMType::IndependentFactorType* marg_;
         };
         
         MaximumLikelihoodLearner(DATASET&, const Parameter&);

	 template<class INF>
	 void learn(const typename INF::Parameter & infParametersBP);
         
         const opengm::learning::Weights<ValueType>& getModelWeights(){return weights_;}
         WeightType& getLerningWeights(){return weights_;}

      private:
         DATASET&     dataset_;
         WeightType   weights_;
         Parameter    param_;
      }; 

      template<class DATASET>
      MaximumLikelihoodLearner<DATASET>::MaximumLikelihoodLearner(DATASET& ds, const Parameter& param )
         : dataset_(ds), param_(param)
      {
          weights_ = opengm::learning::Weights<ValueType>(ds.getNumberOfWeights());
      }
      
      template<class DATASET>
      template<class INF>
      void MaximumLikelihoodLearner<DATASET>::learn(const typename INF::Parameter & infParametersBP){

         typedef typename opengm::ExplicitFunction<ValueType,IndexType,LabelType>                                                    FunctionType;
         typedef typename opengm::ViewConvertFunction<GMType,Minimizer,ValueType>                                                    ViewFunctionType;
         typedef typename GMType::FunctionIdentifier                                                                                 FunctionIdentifierType;
         typedef typename opengm::meta::TypeListGenerator<FunctionType,ViewFunctionType>::type                                       FunctionListType;
         typedef opengm::GraphicalModel<ValueType,opengm::Multiplier, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GmBpType;
         typedef BeliefPropagationUpdateRules<GmBpType, opengm::Integrator>                                                          UpdateRules;
         typedef MessagePassing<GmBpType, opengm::Integrator, UpdateRules, opengm::MaxDistance>                                      BeliefPropagation;
         
         bool search = true; 
         double invTemperature = 1.0/param_.temperature_;

         std::cout << std::endl;
	 if(param_.infoFlag_){
	     std::cout << "INFO: Maximum Likelihood Learner: Maximum Number Of Iterations "<< param_.maximumNumberOfIterations_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Gradient Step "<< param_.gradientStep_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Weight Accuracy "<< param_.weightAccuracy_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Gradient Stopping Criteria "<<param_. gradientStoppingCriteria_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Info Flag "<< param_.infoFlag_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Info Every Step "<< param_.infoEveryStep_ << std::endl;
	 }

         //Parameters for inference
	 const IndexType maxNumberOfBPIterations = infParametersBP.maximumNumberOfSteps_; //40
	 const double convergenceBound = infParametersBP.bound_; //1e-7;
	 const double damping = infParametersBP.damping_; //0.5;

	 if(param_.infoFlag_){
	     std::cout << "INFO: Belief Propagation: Maximum Number Of Belief Propagation Iterations "<< maxNumberOfBPIterations << std::endl;
	     std::cout << "INFO: Belief Propagation: Convergence Bound "<< convergenceBound << std::endl;
	     std::cout << "INFO: Belief Propagation: Damping "<< damping << std::endl;
	 }
         typename BeliefPropagation::Parameter infParam(maxNumberOfBPIterations, convergenceBound, damping);

         size_t iterationCount = 0;
         while(search){
            if(iterationCount>=param_.maximumNumberOfIterations_) break;
            ++iterationCount;
            std::cout << "\r Progress :  " << iterationCount << "/"<<param_.maximumNumberOfIterations_ <<" iteration     0/"<< dataset_.getNumberOfModels() << " models "<< std::flush;
            typename GMType::IndependentFactorType marg;
            WeightGradientFunctor wgf(dataset_); 

            for(IndexType m=0; m<dataset_.getNumberOfModels(); ++m){ 
               std::cout << "\r Progress :  " << iterationCount << "/"<<param_.maximumNumberOfIterations_ << " iteration     "<<m<<"/"<< dataset_.getNumberOfModels()<<" models "<< std::flush;
               dataset_.lockModel(m);
               wgf.setModel(m);

               //*********************************
               //** Build dummy model and infer
               //*********************************
               GmBpType bpModel(dataset_.getModel(m).space());
               for(IndexType f = 0; f<dataset_.getModel(m).numberOfFactors();++f){
                  const typename GMType::FactorType& factor=dataset_.getModel(m)[f];
                  typedef typename opengm::ViewConvertFunction<GMType,Minimizer,ValueType> ViewFunctionType;
                  typedef typename GMType::FunctionIdentifier FunctionIdentifierType;
                  FunctionIdentifierType fid = bpModel.addFunction(ViewFunctionType(factor,invTemperature));
                  bpModel.addFactor(fid, factor.variableIndicesBegin(), factor.variableIndicesEnd());
               } 

               BeliefPropagation bp(bpModel, infParam);
               bp.infer();
               for(IndexType f=0; f<dataset_.getModel(m).numberOfFactors();++f){
                  bp.factorMarginal(f, marg);
                  
                  
                  wgf.setMarg(&marg);
                  dataset_.getModel(m)[f].callFunctor(wgf);
               }
               dataset_.unlockModel(m);

            }

            //*****************************
            //** Gradient Step
            //************************
	    /*
	    if(param_.infoFlag_)
	        std::cout << " Best weights: ";
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
	        dataset_.getWeights().setWeight(p, weights_.getWeight(p) + param_.gradientStep_ * wgf.getGradient(p));
                weights_.setWeight(p, weights_.getWeight(p) + param_.gradientStep_ * wgf.getGradient(p));
		if(param_.infoFlag_)
		  std::cout << weights_.getWeight(p) << " ";
            }  
	    if(param_.infoFlag_)
	      std::cout << std::endl;
	    */

            //*****************************
            double norm = 0;
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
               norm += (wgf.getGradient(p)-2*param_.reg_*weights_.getWeight(p)) * (wgf.getGradient(p)-2*param_.reg_*weights_.getWeight(p));
            }
            norm = std::sqrt(norm);

	    if(param_.infoFlag_)
	        std::cout << "gradient = ( ";  
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
	        if(param_.infoFlag_)
                    std::cout << (wgf.getGradient(p)-2*param_.reg_*weights_.getWeight(p))/norm << " ";
                dataset_.getWeights().setWeight(p, weights_.getWeight(p) + param_.gradientStep_/iterationCount * (wgf.getGradient(p)-2*param_.reg_*weights_.getWeight(p))/norm);
                weights_.setWeight(p, weights_.getWeight(p) + param_.gradientStep_/iterationCount * (wgf.getGradient(p)-2*param_.reg_*weights_.getWeight(p))/norm); 
            } 
	    if(param_.infoFlag_){
                std::cout << ") ";
                std::cout << " weight = ( ";
                for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p)
                    std::cout <<  weights_.getWeight(p) << " ";
                std::cout << ")"<<std::endl;
	    }
         }
         std::cout << "\r Stoped after "<< iterationCount  << "/" << param_.maximumNumberOfIterations_<< " iterations.                             " <<std::endl;
      }
   }
}
#endif
