#pragma once
#ifndef OPENGM_GRIDSEARCH_LEARNER_HXX
#define OPENGM_GRIDSEARCH_LEARNER_HXX

#include <vector>
#include <opengm/functions/learnablefunction.hxx>

namespace opengm {
   namespace learning {

      template<class DATASET, class LOSS>
      class GridSearchLearner
      {
      public: 
         typedef typename DATASET::GMType   GMType; 
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


         GridSearchLearner(DATASET&, Parameter& );

         template<class INF>
         void learn(typename INF::Parameter& para); 
         //template<class INF, class VISITOR>
         //void learn(typename INF::Parameter para, VITITOR vis);

         const opengm::Parameters<double,size_t>& getModelParameters(){return modelParameters_;} 
         Parameter& getLerningParameters(){return para_;}

      private:
         DATASET& dataset_;
         opengm::Parameters<double,size_t> modelParameters_;
         Parameter para_;
      }; 

      template<class DATASET, class LOSS>
      GridSearchLearner<DATASET, LOSS>::GridSearchLearner(DATASET& ds, Parameter& p )
         : dataset_(ds), para_(p)
      {
         modelParameters_ = opengm::Parameters<double,size_t>(ds.getNumberOfParameters());
         if(para_.parameterUpperbound_.size() != ds.getNumberOfParameters())
            para_.parameterUpperbound_.resize(ds.getNumberOfParameters(),10.0);  
         if(para_.parameterLowerbound_.size() != ds.getNumberOfParameters())
            para_.parameterLowerbound_.resize(ds.getNumberOfParameters(),0.0); 
         if(para_.testingPoints_.size() != ds.getNumberOfParameters())
            para_.testingPoints_.resize(ds.getNumberOfParameters(),10); 
      }


      template<class DATASET, class LOSS>
      template<class INF>
      void GridSearchLearner<DATASET, LOSS>::learn(typename INF::Parameter& para){
         // generate model Parameters
         opengm::Parameters<double,size_t> modelPara( dataset_.getNumberOfParameters() );
         opengm::Parameters<double,size_t> bestModelPara( dataset_.getNumberOfParameters() );
         double                            bestLoss = 100000000.0; 
         std::vector<size_t> itC(dataset_.getNumberOfParameters(),0);

         LOSS lossFunction;
         bool search=true;
         while(search){
            // Get Parameter
            for(size_t p=0; p<dataset_.getNumberOfParameters(); ++p){
               modelPara.setParameter(p, para_.parameterLowerbound_[p] + double(itC[p])/double(para_.testingPoints_[p]-1)*(para_.parameterUpperbound_[p]-para_.parameterLowerbound_[p]) );
            }
            // Evaluate Loss
            opengm::Parameters<double,size_t>& mp =  dataset_.getModelParameters();
            mp = modelPara;
            std::vector< std::vector<typename INF::LabelType> > confs( dataset_.getNumberOfModels() );
            double loss = 0;
            for(size_t m=0; m<dataset_.getNumberOfModels(); ++m){
               INF inf( dataset_.getModel(m),para);
               inf.infer();
               inf.arg(confs[m]);
               const std::vector<typename INF::LabelType>& gt =  dataset_.getGT(m);
               loss += lossFunction.loss(confs[m].begin(), confs[m].end(), gt.begin(), gt.end());
            }
            
            // *call visitor*
            for(size_t p=0; p<dataset_.getNumberOfParameters(); ++p){
               std::cout << modelPara[p] <<" ";
            }
            std::cout << " ==> ";
            std::cout << loss << std::endl;
            // **************

            if(loss<bestLoss){
               bestLoss=loss;
               bestModelPara=modelPara;
            }
            //Increment Parameter
            for(size_t p=0; p<dataset_.getNumberOfParameters(); ++p){
               if(itC[p]<para_.testingPoints_[p]-1){
                  ++itC[p];
                  break;
               }
               else{
                  itC[p]=0;
                  if (p==dataset_.getNumberOfParameters()-1)
                     search = false; 
               }             
            }

         }
         std::cout << "Best"<<std::endl;
         for(size_t p=0; p<dataset_.getNumberOfParameters(); ++p){
            std::cout << bestModelPara[p] <<" ";
         }
         std::cout << " ==> ";
         std::cout << bestLoss << std::endl;
         modelParameters_ = bestModelPara;
      };
   }
}
#endif
