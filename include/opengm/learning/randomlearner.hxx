#pragma once
#ifndef OPENGM_RANDOM_LEARNER_HXX
#define OPENGM_RANDOM_LEARNER_HXX

#include <vector>
#include <opengm/functions/learnablefunction.hxx>

namespace opengm {
   namespace learning {
      template<class DATASET, class LOSS>
      class RandomLearner<DATASET, LOSS>
      {
      public:
         typedef GMType; // This will be constructed as a combination of DATASET and LOSS (especially the functiontypelist


         class Parameter{
         public:
            std::vector<double> parameterUpperbound_; 
            std::vector<double> parameterLowerbound_;
            size_t iterations_;
            Parameter():iterations_(10){;}
         };


         RandomLearner(DATASET&, Parameter& );

         template<class INF>
         void learn(typename INF::Parameter para); 
         //template<class INF, class VISITOR>
         //void learn(typename INF::Parameter para, VITITOR vis);

         const opengm::Parameters<ValueType,IndexType>& getModelParameters(){return modelParameters_;} 
         Parameter& getLerningParameters(){return para_;}

      private:
         DATASET& dataset_;
         opengm::Parameters<double,size_t> modelParameters_;
         Parameter para_;
      }; 

      template<class DATASET, class LOSS>
      RandomLearner<DATASET, LOSS>::RandomLearner(DATASET& ds, Parameter& p )
         : dataset_(ds), para_(p)
      {
         modelParameters_ = opengm::Parameters<double,size_t>(ds.numberOfParameters());
      }


      template<class DATASET, class LOSS>
      template<class INF>
      void RandomLearner<DATASET, LOSS>::learn(typename INF::Parameter& para){
         //todo
      };
   }
}
#endif
