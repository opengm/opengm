#pragma once
#ifndef OPENGM_GRIDSEARCH_LEARNER_HXX
#define OPENGM_GRIDSEARCH_LEARNER_HXX

#include <vector>

namespace opengm {
    namespace learning {

    
    // map a global labeling 
    // to a factor labeling iterator



    template<class GM, class LABEL_ITER>
    struct FeatureAccumulator{

        typedef typename GM::LabelType LabelType;
        typedef typename GM::IndexType IndexType;
        typedef typename GM::ValueType ValueType;


        FeatureAccumulator(const size_t nW)
        :   accFeaturesGt_(nW),
            accFeaturesMap_(nW),
            gtLabel_(),
            mapLabel_(),
            factor_(NULL){
        }

        void setLabels(const LABEL_ITER gtLabel, const LABEL_ITER mapLabel){
            gtLabel_ = gtLabel;
            mapLabel_  = mapLabel;
        }

        void resetWeights(){
            for(size_t i=0; i<accFeaturesGt_.size(); ++i){
                accFeaturesGt_[i] = 0.0;
                accFeaturesMap_[i] = 0.0;
            }
        }
        double fDiff(const size_t wi)const{
            return accFeaturesMap_[wi] - accFeaturesGt_[wi];
        }
        void setFactor(const typename GM::FactorType & factor){
            factor_ = &factor;
        }
        template<class F>
        void operator()(const F & f){

            // get the number of weights
            const size_t nWeights = f.numberOfWeights();
            if(nWeights>0){
                // loop over all weights
                for(size_t wi=0; wi<nWeights; ++wi){
                    // accumulate features for both labeling
                    const size_t gwi = f.weightIndex(wi);

                    // for gt label
                    accFeaturesGt_[gwi] += f.weightGradient(wi, factor_->gmToFactorLabelsBegin(gtLabel_));

                    // for test label
                    accFeaturesMap_[gwi] += f.weightGradient(wi, factor_->gmToFactorLabelsBegin(mapLabel_));
                }
            }
        }


        std::vector<double>  accFeaturesGt_;
        std::vector<double>  accFeaturesMap_;
        LABEL_ITER gtLabel_;
        LABEL_ITER mapLabel_;
        const typename  GM::FactorType * factor_;
    };



      
    template<class DATASET>
    class StructuredPerceptron
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
            Parameter(){
                eps_ = 0.00001;
                maxIterations_ = 0;
                stopLoss_ = 0.0;
                kappa_ = 0.1;
            }       

            double eps_;
            size_t maxIterations_;
            double stopLoss_;
            double kappa_;
        };


        StructuredPerceptron(DATASET&, const Parameter& );

        template<class INF>
        void learn(const typename INF::Parameter& para); 
        //template<class INF, class VISITOR>
        //void learn(typename INF::Parameter para, VITITOR vis);

        const opengm::learning::Weights<double>& getWeights(){return weights_;}
        Parameter& getLerningParameters(){return para_;}

        private:

        template<class INF, class FEATURE_ACCUMULATOR>
        double accumulateFeatures(const typename INF::Parameter& para, FEATURE_ACCUMULATOR & featureAcc); 

        DATASET& dataset_;
        opengm::learning::Weights<double> weights_;
        Parameter para_;
        }; 

        template<class DATASET>
        StructuredPerceptron<DATASET>::StructuredPerceptron(DATASET& ds, const Parameter& p )
        : dataset_(ds), para_(p)
        {
            weights_ = opengm::learning::Weights<double>(ds.getNumberOfWeights());
      
        }


    template<class DATASET>
    template<class INF>
    void StructuredPerceptron<DATASET>::learn(const typename INF::Parameter& para){


        typedef typename std::vector<LabelType>::const_iterator LabelIterator;
        typedef FeatureAccumulator<GMType, LabelIterator> FeatureAcc;


        const size_t nModels = dataset_.getNumberOfModels();
        const size_t nWegihts = dataset_.getNumberOfWeights();

        FeatureAcc featureAcc(nWegihts);


        size_t iteration = 0 ;
        while(true){
            if(para_.maxIterations_!=0 && iteration>para_.maxIterations_){
                std::cout<<"reached max iteration"<<"\n";
                break;
            }

            // accumulate features
            double currentLoss = this-> template accumulateFeatures<INF, FeatureAcc>(para, featureAcc);
            

            if(currentLoss < para_.stopLoss_){
                std::cout<<"reached stopLoss"<<"\n";
                break;
            }

            //if(currentLoss==0){
            //    doLearning = false;
            //    break;
            //}

            double wChange = 0.0;
            // update weights
            for(size_t wi=0; wi<nWegihts; ++wi){
                const double learningRate = 1.0 /((1.0/para_.kappa_)*std::sqrt(1.0 + iteration));
                const double wOld = dataset_.getWeights().getWeight(wi);
                const double wNew = wOld + learningRate*featureAcc.fDiff(wi);
                wChange += std::pow(wOld-wNew,2);
                dataset_.getWeights().setWeight(wi, wNew);
            }
            ++iteration;
            if(iteration % 25 ==0)
                std::cout<<iteration<<" loss "<<currentLoss<<" dw "<<wChange<<"\n";

            if(wChange <= para_.eps_ ){
                std::cout<<"converged"<<"\n";
                break;
            }
        }
        weights_ = dataset_.getWeights();
    }

    template<class DATASET>
    template<class INF, class FEATURE_ACCUMULATOR>
    double StructuredPerceptron<DATASET>::accumulateFeatures(
        const typename INF::Parameter& para,
        FEATURE_ACCUMULATOR & featureAcc
    ){


        typedef typename std::vector<LabelType>::const_iterator LabelIterator;
        typedef FeatureAccumulator<GMType, LabelIterator> FeatureAcc;
        const size_t nModels = dataset_.getNumberOfModels();

        double totalLoss=0.0;

        // reset the accumulated features
        featureAcc.resetWeights();

        // iterate over all models
        for(size_t gmi=0; gmi<nModels; ++gmi){

            // lock the model
            dataset_.lockModel(gmi);

            // get model
            const GMType & gm = dataset_.getModel(gmi);

            // do inference
            INF inf(gm, para);
            std::vector<LabelType> arg;
            inf.infer();
            inf.arg(arg);

            LossType lossFunction(dataset_.getLossParameters(gmi));

            totalLoss +=lossFunction.loss(gm, arg.begin(), arg.end(),
                dataset_.getGT(gmi).begin(), dataset_.getGT(gmi).end());

            // pass arg and gt to featureAccumulator
            featureAcc.setLabels(dataset_.getGT(gmi).begin(), arg.begin());

            
            // iterate over all factors
            // and accumulate features
            for(size_t fi=0; fi<gm.numberOfFactors(); ++fi){
                featureAcc.setFactor(gm[fi]);
                gm[fi].callFunctor(featureAcc);
            }
            // unlock model
            dataset_.unlockModel(gmi);
        }

        return totalLoss;
    }

}
}
#endif
