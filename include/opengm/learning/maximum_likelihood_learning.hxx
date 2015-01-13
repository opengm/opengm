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
            size_t maxNumSteps_;
            Parameter() : maxNumSteps_(100) {;}
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
                  gradient_[wID] -= function.weightGradient(wID, labelVector.begin());
               } 
               
               opengm::ShapeWalker<typename F::FunctionShapeIteratorType> shapeWalker(function.functionShapeBegin(), function.dimension());
               for(size_t i=0;i<function.size();++i, ++shapeWalker) {                   
                  for(size_t i=0; i<function.numberOfWeights();++i){
                     size_t wID = function.weightIndex(i);
                     gradient_[wID] += (*marg_)(shapeWalker.coordinateTuple().begin()) * function.weightGradient(wID, shapeWalker.coordinateTuple().begin() );
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

         //Parameters for inference
         const IndexType maxNumberOfIterations = 40;
         const double convergenceBound = 1e-7;
         const double damping = 0.5;
         typename BeliefPropagation::Parameter infParam(maxNumberOfIterations, convergenceBound, damping);

         std::cout << std::endl;
         double eta   = 0.001;
         size_t count = 0;
         while(search){
            if(count>=param_.maxNumSteps_) break;
            ++count;
            std::cout << "\r Progress :  " << count << "/"<<param_.maxNumSteps_ <<" iteration     0/"<< dataset_.getNumberOfModels() << " models "<< std::flush;
            typename GMType::IndependentFactorType marg;
            WeightGradientFunctor wgf(dataset_); 

            for(IndexType m=0; m<dataset_.getNumberOfModels(); ++m){ 
               std::cout << "\r Progress :  " << count << "/"<<param_.maxNumSteps_ << " iteration     "<<m<<"/"<< dataset_.getNumberOfModels()<<" models "<< std::flush;
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
                  FunctionIdentifierType fid = bpModel.addFunction(ViewFunctionType(factor));
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
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
               dataset_.getWeights().setWeight(p, weights_.getWeight(p) + eta * wgf.getGradient(p));
               weights_.setWeight(p, weights_.getWeight(p) + eta * wgf.getGradient(p));
            }  
         }
         std::cout << "\r Stoped after "<< count  << "/"<<param_.maxNumSteps_<< " iterations.                             " <<std::endl;
      }
   }
}
#endif
