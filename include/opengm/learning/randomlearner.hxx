#pragma once
#ifndef OPENGM_RANDOM_LEARNER_HXX
#define OPENGM_RANDOM_LEARNER_HXX

#include <vector>
#include <opengm/functions/learnablefunction.hxx>

namespace opengm {
   namespace learning {

      template<class DATASET, class LOSS>
      class RandomLearner
      {
      public: 
         typedef opengm::GraphicalModel<double,opengm::Adder,typename opengm::meta::TypeListGenerator<opengm::ExplicitFunction<double>, opengm::functions::learnable::LPotts<double> >::type, opengm::DiscreteSpace<size_t, size_t> >GMType; // This will be constructed as a combination of DATASET and LOSS (especially the functiontypelist


         class Parameter{
         public:
            std::vector<double> parameterUpperbound_; 
            std::vector<double> parameterLowerbound_;
            size_t iterations_;
            Parameter():iterations_(10){;}
         };


         RandomLearner(DATASET&, Parameter& );

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
      RandomLearner<DATASET, LOSS>::RandomLearner(DATASET& ds, Parameter& p )
         : dataset_(ds), para_(p)
      {
         modelParameters_ = opengm::Parameters<double,size_t>(ds.numberOfParameters());
         if(para_.parameterUpperbound_ != ds.numberOfParameters())
            para_.parameterUpperbound_.resize(ds.numberOfParameters(),1000.0);  
         if(para_.parameterLowerbound_ != ds.numberOfParameters())
            para_.parameterLowerbound_.resize(ds.numberOfParameters(),-1000.0); 
      }


      template<class DATASET, class LOSS>
      template<class INF>
      void RandomLearner<DATASET, LOSS>::learn(typename INF::Parameter& para){
         // generate model Parameters
         std::vector< opengm::Parameters<double,size_t> > paras(para_.iterations_, opengm::Parameters<double,size_t>( dataset_.numberOfParameters()));
         std::vector< double >                            loss(para_.iterations_,0);

         for(size_t i=0;i<para_.iterations_;++i){
            // following is a very stupid parameter selection and not usefull with more than 1 parameter
            for(size_t p=0; p< dataset_.numberOfParameters(); ++p){
               paras[i][p] = para_.parameterLowerbound_[p] + double(i)/double(para_.iterations_)*(para_.parameterUpperbound_[p]-para_.parameterLowerbound_[p]);
            }
         }
         LOSS lossFunction;
         size_t best = 0;
         for(size_t i=0;i<para_.iterations_;++i){
            opengm::Parameters<double,size_t> mp =  dataset_.getModelParameter();
            mp = paras[i];
            std::vector< std::vector<typename INF::LabelType> > confs( dataset_.numberOfModels() );
            for(size_t m=0; m<dataset_.numberOfModels(); ++m){
               INF inf( dataset_.getModel(m),para);
               inf.infer();
               inf.arg(confs[m]);
               const std::vector<typename INF::LabelType>& gt =  dataset_.getGT(m);
               loss[i] += lossFunction.loss(confs[m].begin(), confs[m].end(), gt.begin(), gt.end());
            }
            if(loss[i]<loss[best])
               best=i;
         }
         modelParameters_ = para[best];
      };
   }
}
#endif
