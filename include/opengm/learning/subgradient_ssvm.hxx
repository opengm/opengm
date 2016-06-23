#pragma once
#ifndef OPENGM_SUBGRADIENT_SSVM_LEARNER_HXX
#define OPENGM_SUBGRADIENT_SSVM_LEARNER_HXX

#include <iomanip>
#include <vector>
#include <opengm/inference/inference.hxx>
#include <opengm/graphicalmodel/weights.hxx>
#include <opengm/utilities/random.hxx>
#include <opengm/learning/gradient-accumulator.hxx>
#include <opengm/learning/weight_averaging.hxx>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <boost/circular_buffer.hpp>



namespace opengm {
    namespace learning {



           
    template<class DATASET>
    class SubgradientSSVM
    {
    public: 
        typedef DATASET DatasetType;
        typedef typename DATASET::GMType   GMType; 
        typedef typename DATASET::GMWITHLOSS GMWITHLOSS;
        typedef typename DATASET::LossType LossType;
        typedef typename GMType::ValueType ValueType;
        typedef typename GMType::IndexType IndexType;
        typedef typename GMType::LabelType LabelType; 
        typedef opengm::learning::Weights<double> WeightsType;
        typedef typename std::vector<LabelType>::const_iterator LabelIterator;
        typedef FeatureAccumulator<GMType, LabelIterator> FeatureAcc;

        typedef std::vector<LabelType> ConfType;
        typedef boost::circular_buffer<ConfType> ConfBuffer;
        typedef std::vector<ConfBuffer> ConfBufferVec;

        class Parameter{
        public:

            enum LearningMode{
                Online = 0,
                Batch = 1
            };


            Parameter(){
                eps_ = 0.00001;
                maxIterations_ = 10000;
                stopLoss_ = 0.0;
                learningRate_ = 1.0;
                C_ = 1.0;
                learningMode_ = Batch;
                averaging_ = -1;
                nConf_ = 0;
            }       

            double eps_;
            size_t maxIterations_;
            double stopLoss_;
            double learningRate_;
            double C_;
            LearningMode learningMode_;
            int averaging_;
            int nConf_;
        };


        SubgradientSSVM(DATASET&, const Parameter& );

        template<class INF>
        void learn(const typename INF::Parameter& para); 
        //template<class INF, class VISITOR>
        //void learn(typename INF::Parameter para, VITITOR vis);

        const opengm::learning::Weights<double>& getWeights(){return weights_;}
        Parameter& getLerningParameters(){return para_;}


        double getLearningRate( )const{
            if(para_.decayExponent_<=0.000000001 && para_.decayExponent_>=-0.000000001 ){
                return 1.0;
            }
            else{
                return std::pow(para_.decayT0_ + static_cast<double>(iteration_),para_.decayExponent_);
            }
        }

        double getLoss(const GMType & gm ,const GMWITHLOSS  & gmWithLoss, std::vector<LabelType> & labels){

            double loss = 0 ;
            std::vector<LabelType> subConf(20,0);

            for(size_t fi=gm.numberOfFactors(); fi<gmWithLoss.numberOfFactors(); ++fi){
                for(size_t v=0; v<gmWithLoss[fi].numberOfVariables(); ++v){
                    subConf[v] = labels[ gmWithLoss[fi].variableIndex(v)];
                }
                loss +=  gmWithLoss[fi](subConf.begin());
            }
            return loss;
        }

    private:

        double updateWeights();

        DATASET& dataset_;
        WeightsType  weights_;
        Parameter para_;
        size_t iteration_;
        FeatureAcc featureAcc_;
        WeightRegularizer<ValueType> wReg_;
        WeightAveraging<double> weightAveraging_;
    }; 

    template<class DATASET>
    SubgradientSSVM<DATASET>::SubgradientSSVM(DATASET& ds, const Parameter& p )
    :   dataset_(ds), 
        para_(p),
        iteration_(0),
        featureAcc_(ds.getNumberOfWeights()),
        wReg_(2, 1.0/p.C_),
        weightAveraging_(ds.getWeights(),p.averaging_)
    {
        featureAcc_.resetWeights();
        weights_ = opengm::learning::Weights<double>(ds.getNumberOfWeights());
    }


    template<class DATASET>
    template<class INF>
    void SubgradientSSVM<DATASET>::learn(const typename INF::Parameter& para){


        typedef typename INF:: template RebindGm<GMWITHLOSS>::type InfLossGm;
        typedef typename InfLossGm::Parameter InfLossGmParam;
        InfLossGmParam infLossGmParam(para);


        const size_t nModels = dataset_.getNumberOfModels();
        const size_t nWegihts = dataset_.getNumberOfWeights();

        
        for(size_t wi=0; wi<nWegihts; ++wi){
            dataset_.getWeights().setWeight(wi, 0.0);
        }
        std::cout<<"PARAM nConf_"<<para_.nConf_<<"\n";
        const bool useWorkingSets = para_.nConf_>0;

        ConfBufferVec buffer(useWorkingSets? nModels : 0, ConfBuffer(para_.nConf_));

        std::vector<bool> isViolated(para_.nConf_);

        if(para_.learningMode_ == Parameter::Online){
            RandomUniform<size_t> randModel(0, nModels);
            //std::cout<<"online mode\n";
            for(iteration_=0 ; iteration_<para_.maxIterations_; ++iteration_){




                // get random model
                const size_t gmi = randModel();
                // lock the model
                dataset_.lockModel(gmi);
                const GMWITHLOSS & gmWithLoss = dataset_.getModelWithLoss(gmi);

                // do inference
                std::vector<LabelType> arg;
                opengm::infer<InfLossGm>(gmWithLoss, infLossGmParam, arg);
                featureAcc_.resetWeights();
                featureAcc_.accumulateModelFeatures(dataset_.getModel(gmi), dataset_.getGT(gmi).begin(), arg.begin());
                dataset_.unlockModel(gmi);

                // update weights
                const double wChange =updateWeights();

                if(iteration_%nModels*2 == 0 ){
                    std::cout << '\r'
                              << std::setw(6) << std::setfill(' ') << iteration_ << ':'
                              << std::setw(8) << dataset_. template getTotalLossParallel<INF>(para) <<"  "<< std::flush;

                }

            }
        }
        else if(para_.learningMode_ == Parameter::Batch){
            //std::cout<<"batch mode\n";
            for(iteration_=0 ; iteration_<para_.maxIterations_; ++iteration_){
                // this 
                

                // reset the weights
                featureAcc_.resetWeights();
                double totalLoss = 0;

                #ifdef WITH_OPENMP
                omp_lock_t modelLockUnlock;
                omp_init_lock(&modelLockUnlock);
                omp_lock_t featureAccLock;
                omp_init_lock(&featureAccLock);
                #pragma omp parallel for reduction(+:totalLoss)  
                #endif
                for(long long llgmi=0; llgmi<(long long)nModels; ++llgmi){
                    size_t gmi=(size_t)llgmi;
                    // lock the model
                    #ifdef WITH_OPENMP
                    omp_set_lock(&modelLockUnlock);
                    dataset_.lockModel(gmi);     
                    omp_unset_lock(&modelLockUnlock);
                    #else
                    dataset_.lockModel(gmi);     
                    #endif
                        
                    

                    const GMWITHLOSS & gmWithLoss = dataset_.getModelWithLoss(gmi);
                    const GMType     & gm = dataset_.getModel(gmi);
                    //run inference
                    std::vector<LabelType> arg;
                    opengm::infer<InfLossGm>(gmWithLoss, infLossGmParam, arg);

                    totalLoss = totalLoss + getLoss(gm, gmWithLoss, arg);

             
                    if(useWorkingSets){
                        // append current solution
                        buffer[gmi].push_back(arg);

                        size_t vCount=0;
                        // check which violates
                        for(size_t cc=0; cc<buffer[gmi].size(); ++cc){
                            const double mLoss = dataset_.getLoss(buffer[gmi][cc], gmi);
                            const double argVal = gm.evaluate(buffer[gmi][cc]);
                            const double gtVal =  gm.evaluate(dataset_.getGT(gmi));
                            const double ll = (argVal - mLoss) - gtVal;
                            //std::cout<<" argVal "<<argVal<<" gtVal "<<gtVal<<" mLoss "<<mLoss<<"   VV "<<ll<<"\n";
                            if(ll<0){
                                isViolated[cc] = true;
                                ++vCount;
                            }
                        }
                        FeatureAcc featureAcc(nWegihts);
                        for(size_t cc=0; cc<buffer[gmi].size(); ++cc){
                            if(isViolated[cc]){

                                featureAcc.accumulateModelFeatures(gm, dataset_.getGT(gmi).begin(), buffer[gmi][cc].begin(),1.0/double(vCount));

                            }
                        }
                        #ifdef WITH_OPENMP
                        omp_set_lock(&featureAccLock);
                        featureAcc_.accumulateFromOther(featureAcc);
                        omp_unset_lock(&featureAccLock);
                        #else
                        featureAcc_.accumulateFromOther(featureAcc);
                        #endif
                    }
                    else{
                        FeatureAcc featureAcc(nWegihts);
                        featureAcc.accumulateModelFeatures(gm, dataset_.getGT(gmi).begin(), arg.begin());
                        #ifdef WITH_OPENMP
                        omp_set_lock(&featureAccLock);
                        featureAcc_.accumulateFromOther(featureAcc);
                        omp_unset_lock(&featureAccLock);
                        #else
                        featureAcc_.accumulateFromOther(featureAcc);
                        #endif
                    }



                    // acc features
                    //omp_set_lock(&featureAccLock);
                    //featureAcc_.accumulateFromOther(featureAcc);
                    //omp_unset_lock(&featureAccLock);

                    // unlock the model
                    #ifdef WITH_OPENMP
                    omp_set_lock(&modelLockUnlock);
                    dataset_.unlockModel(gmi);     
                    omp_unset_lock(&modelLockUnlock);
                    #else
                    dataset_.unlockModel(gmi);     
                    #endif


                }

                //const double wRegVal = wReg_(dataset_.getWeights());
                //const double tObj = std::abs(totalLoss) + wRegVal;
                if(iteration_%1==0){
                    std::cout << '\r'
                              << std::setw(6) << std::setfill(' ') << iteration_ << ':'
                              << std::setw(8) << -1.0*totalLoss <<"  "<< std::flush;
                }
                // update the weights
                const double wChange =updateWeights();
                
            }
        }
        weights_ = dataset_.getWeights();
    }


    template<class DATASET>
    double SubgradientSSVM<DATASET>::updateWeights(){

        const size_t nWegihts = dataset_.getNumberOfWeights();

        WeightsType p(nWegihts);
        WeightsType newWeights(nWegihts);

        if(para_.learningMode_ == Parameter::Batch){
            for(size_t wi=0; wi<nWegihts; ++wi){
                p[wi] =  dataset_.getWeights().getWeight(wi);
                p[wi] += para_.C_ * featureAcc_.getWeight(wi)/double(dataset_.getNumberOfModels());
            }
        }
        else{
            for(size_t wi=0; wi<nWegihts; ++wi){
                p[wi] =  dataset_.getWeights().getWeight(wi);
                p[wi] += para_.C_ * featureAcc_.getWeight(wi);
            }
        }


        double wChange = 0.0;
        
        for(size_t wi=0; wi<nWegihts; ++wi){
            const double wOld = dataset_.getWeights().getWeight(wi);
            const double wNew = wOld - (para_.learningRate_/double(iteration_+1))*p[wi];
            newWeights[wi] = wNew;
        }

        weightAveraging_(newWeights);



        weights_ = dataset_.getWeights();
        return wChange;
    }
}
}
#endif
