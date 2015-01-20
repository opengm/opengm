#pragma once
#ifndef OPENGM_MAXIMUM_LIKELIHOOD_LEARNER_HXX
#define OPENGM_MAXIMUM_LIKELIHOOD_LEARNER_HXX

#include <vector>
#include <fstream>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/functions/view_convert_function.hxx>
#include <iomanip>

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
	     size_t maximumNumberOfIterations_;
	     double gradientStepSize_;
	     double weightStoppingCriteria_;
             double gradientStoppingCriteria_;
             bool infoFlag_;
             bool infoEveryStep_; 
             double weightRegularizer_;
	     size_t beliefPropagationMaximumNumberOfIterations_;
	     double beliefPropagationConvergenceBound_;
	     double beliefPropagationDamping_;
	     double beliefPropagationTemperature_;
	     opengm::Tribool beliefPropagationIsAcyclic_;
	     Parameter():
	         maximumNumberOfIterations_(100),
	         gradientStepSize_(0.1),
		 weightStoppingCriteria_(0.0000000000000001),
		 gradientStoppingCriteria_(0.0000000000000001),
		 infoFlag_(true),
		 infoEveryStep_(false),
		 weightRegularizer_(1.0),
		 beliefPropagationMaximumNumberOfIterations_(40),
		 beliefPropagationConvergenceBound_(0.0000001),
		 beliefPropagationDamping_(0.5),
		 beliefPropagationTemperature_(0.3),
		 beliefPropagationIsAcyclic_(opengm::Tribool::Maybe)

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

	 void learn();
         
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
      void MaximumLikelihoodLearner<DATASET>::learn(){

         typedef typename opengm::ExplicitFunction<ValueType,IndexType,LabelType>                                                    FunctionType;
         typedef typename opengm::ViewConvertFunction<GMType,Minimizer,ValueType>                                                    ViewFunctionType;
         typedef typename GMType::FunctionIdentifier                                                                                 FunctionIdentifierType;
         typedef typename opengm::meta::TypeListGenerator<FunctionType,ViewFunctionType>::type                                       FunctionListType;
         typedef opengm::GraphicalModel<ValueType,opengm::Multiplier, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GmBpType;
         typedef BeliefPropagationUpdateRules<GmBpType, opengm::Integrator>                                                          UpdateRules;
         typedef MessagePassing<GmBpType, opengm::Integrator, UpdateRules, opengm::MaxDistance>                                      BeliefPropagation;
         
         bool search = true; 
         double invTemperature = 1.0/param_.beliefPropagationTemperature_;

         if(param_.infoFlag_){
	     std::cout << "INFO: Maximum Likelihood Learner: Maximum Number Of Iterations "<< param_.maximumNumberOfIterations_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Gradient Step Size "<< param_.gradientStepSize_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Gradient Stopping Criteria "<<param_. gradientStoppingCriteria_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Weight Stopping Criteria "<< param_.weightStoppingCriteria_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Info Flag "<< param_.infoFlag_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Info Every Step "<< param_.infoEveryStep_ << std::endl;
	     std::cout << "INFO: Maximum Likelihood Learner: Strength of regularizer for the Weight "<< param_.weightRegularizer_ << std::endl;
	     std::cout << "INFO: Belief Propagation: Maximum Number Of Belief Propagation Iterations "<< param_.beliefPropagationMaximumNumberOfIterations_ << std::endl;
	     std::cout << "INFO: Belief Propagation: Convergence Bound "<< param_.beliefPropagationConvergenceBound_ << std::endl;
	     std::cout << "INFO: Belief Propagation: Damping "<< param_.beliefPropagationDamping_ << std::endl;
	     std::cout << "INFO: Belief Propagation: Temperature "<< param_.beliefPropagationTemperature_ << std::endl;
	     std::cout << "INFO: Belief Propagation: Acyclic Model "<< param_.beliefPropagationIsAcyclic_ << std::endl;
	 }

	 typename UpdateRules::SpecialParameterType specialParameter;//=UpdateRules::SpecialParameterType();
         typename BeliefPropagation::Parameter infParam(
	     param_.beliefPropagationMaximumNumberOfIterations_, 
	     param_.beliefPropagationConvergenceBound_, 
	     param_.beliefPropagationDamping_,
	     specialParameter,
	     param_.beliefPropagationIsAcyclic_
	 );

         size_t iterationCount = 0;
         while(search){
            if(iterationCount>=param_.maximumNumberOfIterations_) break;
            ++iterationCount;
	    if(param_.infoFlag_)
	        std::cout << "\r Progress :  " << iterationCount << "/"<<param_.maximumNumberOfIterations_ <<" iteration     0/"<< dataset_.getNumberOfModels() << " models ";

            typename GMType::IndependentFactorType marg;
            WeightGradientFunctor wgf(dataset_); 

            for(IndexType m=0; m<dataset_.getNumberOfModels(); ++m){ 
	       if(param_.infoFlag_)
                  std::cout << "\r Progress :  " << iterationCount << "/"<<param_.maximumNumberOfIterations_ << " iteration     "<<m<<"/"<< dataset_.getNumberOfModels()<<" models ";

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
            double gradientNorm = 0;
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
               gradientNorm += (wgf.getGradient(p)-2*param_.weightRegularizer_*weights_.getWeight(p)) * (wgf.getGradient(p)-2*param_.weightRegularizer_*weights_.getWeight(p));
            }
            gradientNorm = std::sqrt(gradientNorm);

	    if(gradientNorm < param_.gradientStoppingCriteria_)
	        search = false;

	    if(param_.infoFlag_ and param_.infoEveryStep_)
	        std::cout << "\r" << std::flush << " Iteration " << iterationCount <<" Gradient = ( ";

	    double normGradientDelta = 0;
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
	        if(param_.infoFlag_ and param_.infoEveryStep_)
		    std::cout << std::left << std::setfill(' ') << std::setw(10) << (wgf.getGradient(p)-2*param_.weightRegularizer_*weights_.getWeight(p))/gradientNorm << " ";

		double gradientDelta;
		gradientDelta=param_.gradientStepSize_/iterationCount * (wgf.getGradient(p)-2*param_.weightRegularizer_*weights_.getWeight(p))/gradientNorm;

		normGradientDelta +=gradientDelta*gradientDelta;
                dataset_.getWeights().setWeight(p, weights_.getWeight(p) + gradientDelta);
                weights_.setWeight(p, weights_.getWeight(p) + gradientDelta); 
            }
	    normGradientDelta=std::sqrt(normGradientDelta);
	    if( normGradientDelta < param_.weightStoppingCriteria_)
	        search = false;

	    if(param_.infoFlag_ and param_.infoEveryStep_){
                std::cout << ") ";
                std::cout << " Weight = ( ";
                for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p)
                    std::cout << std::left << std::setfill(' ') << std::setw(10) <<  weights_.getWeight(p) << " ";
                std::cout << ") "<< "GradientNorm " << std::left << std::setfill(' ') << std::setw(10) << gradientNorm << " GradientDeltaNorm "<< std::setw(10) << normGradientDelta << "             " << std::endl;
	    }
	    else if (param_.infoFlag_)
	      std::cout << "GradientNorm " << std::left << std::setfill(' ') << std::setw(10) << gradientNorm << " GradientDeltaNorm "<< std::setw(10) << normGradientDelta << "             " << std::flush;
         }
	 std::cout << "\r                                                                                                                                                                                                                                                                                                                                                                                                            " << std::flush;
         std::cout << "\r Stoped after "<< iterationCount  << "/" << param_.maximumNumberOfIterations_<< " iterations. " <<std::endl;
      }
   }
}
#endif
