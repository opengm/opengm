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
#include <boost/math/distributions/normal.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>


namespace opengm {
    namespace learning {



    template<class T>
    double gen_normal_3(T &generator)
    {
      return generator();
    }

    // Version that fills a vector
    template<class T>
    void gen_normal_3(T &generator,
                  std::vector<double> &res)
    {
      for(size_t i=0; i<res.size(); ++i)
        res[i]=generator();
    }


           
    template<class DATASET>
    class Rws
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



            Parameter(){
                eps_ = 0.00001;
                maxIterations_ = 10000;
                stopLoss_ = 0.0;
                learningRate_ = 1.0;
                C_ = 1.0;
                averaging_ = -1;
                p_ = 10;
                sigma_ = 1.0;
            }       

            double eps_;
            size_t maxIterations_;
            double stopLoss_;
            double learningRate_;
            double C_;
            int averaging_;
            size_t p_;
            double sigma_;
        };


        Rws(DATASET&, const Parameter& );

        template<class INF>
        void learn(const typename INF::Parameter& para); 
        //template<class INF, class VISITOR>
        //void learn(typename INF::Parameter para, VITITOR vis);

        const opengm::learning::Weights<double>& getWeights(){return weights_;}
        Parameter& getLerningParameters(){return para_;}



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
    Rws<DATASET>::Rws(DATASET& ds, const Parameter& p )
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
    void Rws<DATASET>::learn(const typename INF::Parameter& para){


        const size_t nModels = dataset_.getNumberOfModels();
        const size_t nWegihts = dataset_.getNumberOfWeights();

        
        //for(size_t wi=0; wi<nWegihts; ++wi){
        //    dataset_.getWeights().setWeight(wi, 0.0);
        //}



        RandomUniform<size_t> randModel(0, nModels);
        boost::math::normal_distribution<ValueType> nDist(0.0, para_.sigma_);
        std::vector< std::vector<ValueType> > noiseVecs(para_.p_, std::vector<ValueType>(nWegihts));
        std::vector<ValueType> lossVec(para_.p_);

        std::vector<ValueType> gradient(nWegihts);

        boost::variate_generator<boost::mt19937, boost::normal_distribution<> >
        generator(boost::mt19937(time(0)),boost::normal_distribution<>(0.0, para_.sigma_));

        std::cout<<"online mode "<<nWegihts<<"\n";

        std::cout <<"start loss"<< std::setw(6) << std::setfill(' ') << iteration_ << ':'
                          << std::setw(8) << dataset_. template getTotalLossParallel<INF>(para) <<"  \n\n\n\n";


        for(iteration_=0 ; iteration_<para_.maxIterations_; ++iteration_){




            // get random model
            const size_t gmi = randModel();

            // save the current weights
            WeightsType currentWeights  = dataset_.getWeights();


            featureAcc_.resetWeights();

            // lock the model
            dataset_.lockModel(gmi);

            for(size_t p=0; p<para_.p_; ++p){


                // fill noise 
                gen_normal_3(generator, noiseVecs[p]);

                // add noise to the weights
                for(size_t wi=0; wi<nWegihts; ++wi){
                    const ValueType cw = currentWeights[wi];
                    const ValueType nw = cw + noiseVecs[p][wi];
                    dataset_.getWeights().setWeight(wi, nw);
                }


                const GMType & gm = dataset_.getModel(gmi);
                // do inference
                std::vector<LabelType> arg;
                opengm::infer<INF>(gm, para, arg);
                lossVec[p] = dataset_.getLoss(arg, gmi);
                
                //featureAcc_.accumulateModelFeatures(gm, dataset_.getGT(gmi).begin(), arg.begin());
                // update weights
                //const double wChange =updateWeights();      
            }

            //for(size_t wi=0; wi<nWegihts; ++wi){
            //    gradient[wi] = featureAcc_.getWeight(wi);
            //}
            std::fill(gradient.begin(), gradient.end(),0.0);
            for(size_t p=0; p<para_.p_; ++p){
                for(size_t wi=0; wi<nWegihts; ++wi){
                    gradient[wi] += (1.0/para_.p_)*(noiseVecs[p][wi])*lossVec[p];
                }
            }

            const ValueType actualLearningRate = para_.learningRate_/(1.0 + iteration_);
            //const ValueType actualLearningRate = para_.learningRate_;///(1.0 + iteration_);
            // do update
            for(size_t wi=0; wi<nWegihts; ++wi){
                const ValueType oldWeight = currentWeights[wi];
                const ValueType newWeights = (oldWeight - actualLearningRate*gradient[wi])*para_.C_;
                //std::cout<<"wi "<<newWeights<<"\n";
                dataset_.getWeights().setWeight(wi, newWeights);
            }
            std::cout<<"\n";
            dataset_.unlockModel(gmi);

            if(iteration_%10==0){
            //if(iteration_%nModels*2 == 0 ){
                std::cout << '\n'
                          << std::setw(6) << std::setfill(' ') << iteration_ << ':'
                          << std::setw(8) << dataset_. template getTotalLossParallel<INF>(para) <<"  "<< std::flush;

            }

        }
  
        weights_ = dataset_.getWeights();
    }


    template<class DATASET>
    double Rws<DATASET>::updateWeights(){

        const size_t nWegihts = dataset_.getNumberOfWeights();

        WeightsType p(nWegihts);
        WeightsType newWeights(nWegihts);


        for(size_t wi=0; wi<nWegihts; ++wi){
            p[wi] =  dataset_.getWeights().getWeight(wi);
            p[wi] += para_.C_ * featureAcc_.getWeight(wi);
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
