#pragma once
#ifndef OPENGM_GRIDSEARCH_LEARNER_HXX
#define OPENGM_GRIDSEARCH_LEARNER_HXX

#include <vector>

namespace opengm {
   namespace learning {

      
      template<class DATASET>
      class GridSearchLearner
      {
      public: 
         typedef DATASET DatasetType;
         typedef typename DATASET::GMType   GMType; 
         typedef typename DATASET::LossType LossType;
         typedef typename GMType::ValueType ValueType;
         typedef typename GMType::IndexType IndexType;
         typedef typename GMType::LabelType LabelType; 

         class Parameter{
         public:
            std::vector<double> parameterUpperbound_; 
            std::vector<double> parameterLowerbound_;
            std::vector<size_t> testingPoints_;
            Parameter(){;}
         };


         GridSearchLearner(DATASET&, const Parameter& );

         template<class INF>
         void learn(const typename INF::Parameter& para); 
         //template<class INF, class VISITOR>
         //void learn(typename INF::Parameter para, VITITOR vis);

         const opengm::learning::Weights<double>& getWeights(){return weights_;}
         Parameter& getLerningParameters(){return para_;}

      private:
         DATASET& dataset_;
         opengm::learning::Weights<double> weights_;
         Parameter para_;
      }; 

      template<class DATASET>
      GridSearchLearner<DATASET>::GridSearchLearner(DATASET& ds, const Parameter& p )
         : dataset_(ds), para_(p)
      {
         weights_ = opengm::learning::Weights<double>(ds.getNumberOfWeights());
         if(para_.parameterUpperbound_.size() != ds.getNumberOfWeights())
            para_.parameterUpperbound_.resize(ds.getNumberOfWeights(),10.0);
         if(para_.parameterLowerbound_.size() != ds.getNumberOfWeights())
            para_.parameterLowerbound_.resize(ds.getNumberOfWeights(),0.0);
         if(para_.testingPoints_.size() != ds.getNumberOfWeights())
            para_.testingPoints_.resize(ds.getNumberOfWeights(),10);
      }


      template<class DATASET>
      template<class INF>
      void GridSearchLearner<DATASET>::learn(const typename INF::Parameter& para){
         // generate model Parameters
         opengm::learning::Weights<double> modelPara( dataset_.getNumberOfWeights() );
         opengm::learning::Weights<double> bestModelPara( dataset_.getNumberOfWeights() );
         double bestLoss = std::numeric_limits<double>::infinity();
         std::vector<size_t> itC(dataset_.getNumberOfWeights(),0);
         
         bool search=true;
         while(search){
            // Get Parameter
            for(size_t p=0; p<dataset_.getNumberOfWeights(); ++p){
               modelPara.setWeight(p, para_.parameterLowerbound_[p] + double(itC[p])/double(para_.testingPoints_[p]-1)*(para_.parameterUpperbound_[p]-para_.parameterLowerbound_[p]) );
            }
            // Evaluate Loss
            opengm::learning::Weights<double>& mp =  dataset_.getWeights();
            mp = modelPara;
            const double loss = dataset_. template getTotalLoss<INF>(para);
           

            // **************

            if(loss<bestLoss){
                 // *call visitor*
                for(size_t p=0; p<dataset_.getNumberOfWeights(); ++p){
                   std::cout << modelPara[p] <<" ";
                }
                std::cout << " ==> ";
                std::cout << loss << std::endl;

                bestLoss=loss;
                bestModelPara=modelPara;
                if(loss<=0.000000001){
                    search = false;
                }
            }
            //Increment Parameter
            for(size_t p=0; p<dataset_.getNumberOfWeights(); ++p){
               if(itC[p]<para_.testingPoints_[p]-1){
                  ++itC[p];
                  break;
               }
               else{
                  itC[p]=0;
                  if (p==dataset_.getNumberOfWeights()-1)
                     search = false; 
               }             
            }
         }
         std::cout << "Best"<<std::endl;
         for(size_t p=0; p<dataset_.getNumberOfWeights(); ++p){
            std::cout << bestModelPara[p] <<" ";
         }
         std::cout << " ==> ";
         std::cout << bestLoss << std::endl;
         weights_ = bestModelPara;

         // save best weights in dataset
         for(size_t p=0; p<dataset_.getNumberOfWeights(); ++p){
            dataset_.getWeights().setWeight(p, weights_[p]);
         }
      };
   }
}
#endif
