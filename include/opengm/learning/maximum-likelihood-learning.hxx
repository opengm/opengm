#pragma once
#ifndef OPENGM_MAXIMUM_LIKELIHOOD_LEARNER_HXX
#define OPENGM_MAXIMUM_LIKELIHOOD_LEARNER_HXX

#include <vector>
#include <opengm/functions/learnablefunction.hxx>
#include <fstream>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/view_convert_function.hxx>
#include <opengm/functions/learnable/lpotts.hxx>
#include <opengm/functions/learnable/sum_of_experts.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/icm.hxx>

typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType;
typedef opengm::meta::TypeListGenerator<
    opengm::ExplicitFunction<ValueType,IndexType,LabelType>,
    opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>,
    opengm::functions::learnable::SumOfExperts<ValueType,IndexType,LabelType>
>::type FunctionListType;

typedef opengm::GraphicalModel<
    ValueType,opengm::Adder,
    FunctionListType,
    opengm::DiscreteSpace<IndexType,LabelType>
> GM;

typedef opengm::ICM<GM,opengm::Minimizer> INF;
typedef opengm::learning::Weights<ValueType> WeightType;

struct WeightGradientFunctor{
    WeightGradientFunctor(IndexType weight, std::vector<LabelType>::iterator labelVectorBegin)
        : weight_(weight),
          labelVectorBegin_(labelVectorBegin){
    }

    template<class F>
    void operator()(const F & function ){
        IndexType index;
        for(size_t i=0; i<function.numberOfWeights();++i)
            if(function.weightIndex(i)==weight_)
                index=i;
        result_ =  function.weightGradient(index, labelVectorBegin_);
    }

    IndexType weight_;
    std::vector<LabelType>::iterator labelVectorBegin_;
    ValueType result_;
};

namespace opengm {
namespace learning {

template<class DATASET, class LOSS>
class MaximumLikelihoodLearner
{
public:
    typedef typename DATASET::GMType   GMType;
    typedef typename GMType::ValueType ValueType;
    typedef typename GMType::IndexType IndexType;
    typedef typename GMType::LabelType LabelType;
    typedef typename GMType::FactorType FactorType;

    class Weight{
    public:
        std::vector<double> weightUpperbound_;
        std::vector<double> weightLowerbound_;
        std::vector<IndexType> testingPoints_;
        Weight(){;}
    };


    MaximumLikelihoodLearner(DATASET&, Weight& );

    template<class INF>
    void learn(typename INF::Parameter& weight);

    const opengm::learning::Weights<ValueType>& getModelWeights(){return modelWeights_;}
    Weight& getLerningWeights(){return weight_;}

private:
    DATASET& dataset_;
    opengm::learning::Weights<ValueType> modelWeights_;
    Weight weight_;
};

template<class DATASET, class LOSS>
MaximumLikelihoodLearner<DATASET, LOSS>::MaximumLikelihoodLearner(DATASET& ds, Weight& w )
    : dataset_(ds), weight_(w)
{
    modelWeights_ = opengm::learning::Weights<ValueType>(ds.getNumberOfWeights());
    if(weight_.weightUpperbound_.size() != ds.getNumberOfWeights())
        weight_.weightUpperbound_.resize(ds.getNumberOfWeights(),10.0);
    if(weight_.weightLowerbound_.size() != ds.getNumberOfWeights())
        weight_.weightLowerbound_.resize(ds.getNumberOfWeights(),0.0);
    if(weight_.testingPoints_.size() != ds.getNumberOfWeights())
        weight_.testingPoints_.resize(ds.getNumberOfWeights(),10);
}


template<class DATASET, class LOSS>
template<class INF>
void MaximumLikelihoodLearner<DATASET, LOSS>::learn(typename INF::Parameter& weight){
    // generate model Weights

    opengm::learning::Weights<ValueType> modelWeight( dataset_.getNumberOfWeights() );
    opengm::learning::Weights<ValueType> bestModelWeight( dataset_.getNumberOfWeights() );
    double                            bestLoss = 100000000.0;
    //std::vector<IndexType> itC(dataset_.getNumberOfWeights(),0);


    std::vector<ValueType> point(dataset_.getNumberOfWeights(),0);
    std::vector<ValueType> gradient(dataset_.getNumberOfWeights(),0);
    std::vector<ValueType> Delta(dataset_.getNumberOfWeights(),0);
    for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
        point[p] = ValueType((weight_.weightUpperbound_[p]-weight_.weightLowerbound_[p])/2);
    }
/*
    // test only
    point[0]=0.25;
    point[1]=0.5;
    point[2]=0.0;
    // end test only
*/
    LOSS lossFunction;
    bool search=true;
    int count=0;

    std::vector< std::vector<ValueType> > w( dataset_.getNumberOfModels(), std::vector<ValueType> ( dataset_.getModel(0).numberOfVariables()) );
    std::vector<ValueType> wBar( dataset_.getNumberOfModels() );

    /***********************************************************************************************************/
    // construct Ground Truth dependent weights
    /***********************************************************************************************************/

    for(IndexType m=0; m<dataset_.getNumberOfModels(); ++m){ // for each model
        const GMType &model = dataset_.getModel(m);
        const std::vector<typename INF::LabelType>& gt =  dataset_.getGT(m);

        for(IndexType v=0; v<dataset_.getModel(m).numberOfVariables();++v){
            w[m][v]=gt[v];
            wBar[m] += w [m][v];
        }
        // normalize w
        for(IndexType v=0; v<dataset_.getModel(m).numberOfVariables();++v)
            w[m][v] = (ValueType)w[m][v] / wBar[m];
    }

    ValueType eta = 0.1111111;
    ValueType delta = 0.25; // 0 <= delta <= 0.5
    ValueType D_a = 1.0; // distance treshold
    while(search){
        ++count;
        //if (count % 1000 == 0)
        std::cout << "---count--->" << count << "     ";

        // Get Weights
        for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
            modelWeight.setWeight(p, point[p]);
        }

        /***********************************************************************************************************/
        // calculate current loss
        /***********************************************************************************************************/
        opengm::learning::Weights<ValueType>& mp =  dataset_.getWeights();
        mp = modelWeight;
        std::vector< std::vector<typename INF::LabelType> > confs( dataset_.getNumberOfModels() );
        double loss = 0;
        for(size_t m=0; m<dataset_.getNumberOfModels(); ++m){
           INF inf( dataset_.getModel(m),weight);
           inf.infer();
           inf.arg(confs[m]);
           const std::vector<typename INF::LabelType>& gt =  dataset_.getGT(m);
           loss += lossFunction.loss(confs[m].begin(), confs[m].end(), gt.begin(), gt.end());
        }

        std::cout << " eta = " << eta << "   weights  ";//<< std::endl;
        for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
            std::cout << modelWeight[p] << " " ;
        }
        std::cout << "   loss-->" << loss << std::endl;

        /***********************************************************************************************************/
        // Loopy Belief Propagation setup
        /***********************************************************************************************************/
        typedef typename opengm::ExplicitFunction<ValueType,IndexType,LabelType> FunctionType;
        typedef typename opengm::ViewConvertFunction<GMType,Minimizer,ValueType> ViewFunctionType;
        typedef typename GMType::FunctionIdentifier FunctionIdentifierType;
        typedef typename opengm::meta::TypeListGenerator<FunctionType,ViewFunctionType>::type FunctionListType;
        typedef opengm::GraphicalModel<ValueType,opengm::Multiplier, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GmBpType;
        typedef BeliefPropagationUpdateRules<GmBpType, opengm::Integrator> UpdateRules;
        typedef MessagePassing<GmBpType, opengm::Integrator, UpdateRules, opengm::MaxDistance> BeliefPropagation;

        const IndexType maxNumberOfIterations = 40;
        const double convergenceBound = 1e-7;
        const double damping = 0.5;
        typename BeliefPropagation::Parameter weight(maxNumberOfIterations, convergenceBound, damping);

        std::vector< std::vector<LabelType> > labels(dataset_.getNumberOfModels(), std::vector<LabelType> (dataset_.getModel(0).numberOfVariables()) );
        std::vector< std::vector<ValueType> > b  ( dataset_.getNumberOfModels(), std::vector<ValueType> ( dataset_.getModel(0).numberOfFactors()) );

        for(IndexType m=0; m<dataset_.getNumberOfModels(); ++m){
            GmBpType bpModel(dataset_.getModel(m).space());

            for(IndexType f = 0; f<dataset_.getModel(m).numberOfFactors();++f){
                const typename GMType::FactorType& factor=dataset_.getModel(m)[f];
                typedef typename opengm::ViewConvertFunction<GMType,Minimizer,ValueType> ViewFunctionType;
                typedef typename GMType::FunctionIdentifier FunctionIdentifierType;
                FunctionIdentifierType fid = bpModel.addFunction(ViewFunctionType(factor));
                bpModel.addFactor(fid, factor.variableIndicesBegin(), factor.variableIndicesEnd());
            }
            /***********************************************************************************************************/
            // run: Loopy Belief Propagation
            /***********************************************************************************************************/
            BeliefPropagation bp(bpModel, weight);
            const std::vector<typename INF::LabelType>& gt =  dataset_.getGT(m);
            bp.infer();
            typename GMType::IndependentFactorType marg;
            for(IndexType f = 0; f<dataset_.getModel(m).numberOfFactors();++f){
                bp.factorMarginal(f, marg);
                std::vector<IndexType> indexVector( marg.variableIndicesBegin(), marg.variableIndicesEnd() );
                std::vector<LabelType> labelVector( marg.numberOfVariables());
                for(IndexType v=0; v<marg.numberOfVariables();++v)
                    labelVector[v] = gt[indexVector[v]];
                b[m][f] = marg(labelVector.begin());
            }
        }

        /***********************************************************************************************************/
        // Calculate Gradient
        /***********************************************************************************************************/
        std::vector<ValueType> sum(dataset_.getNumberOfWeights());

        for(IndexType p=0; p<dataset_.getNumberOfWeights();++p){
            std::vector< std::vector<ValueType> >
                piW(dataset_.getNumberOfModels(),
                    std::vector<ValueType> ( dataset_.getModel(0).numberOfFactors()));

            for(IndexType m=0; m<dataset_.getNumberOfModels(); ++m){
                const GMType &model = dataset_.getModel(m);
                const std::vector<typename INF::LabelType>& gt =  dataset_.getGT(m);
                ValueType f_x=0.0; // f^{d}_{C;k} ( x^d_C ) J. Kappes p. 64

                for(IndexType f=0; f<dataset_.getModel(m).numberOfFactors();++f){
                    const FactorType &factor = dataset_.getModel(m)[f];
                    std::vector<IndexType> indexVector( factor.variableIndicesBegin(), factor.variableIndicesEnd() );
                    std::vector<LabelType> labelVector( factor.numberOfVariables());

                    for(IndexType v=0; v<factor.numberOfVariables();++v){
                        labelVector[v] = gt[indexVector[v]];
                        piW[m][f] *=w[m][v];
                    }
                    WeightGradientFunctor weightGradientFunctor(p, labelVector.begin());
                    factor.callFunctor(weightGradientFunctor);
                    f_x =weightGradientFunctor.result_;
                    // ( ground truth - marginals ) * factorWeightGradient
                    sum[p] += (piW[m][f] - b[m][f]) * f_x;
                }
            }
        }

        if(loss<bestLoss){
            bestLoss=loss;
            bestModelWeight=modelWeight;
        }
        else{
            eta /= 2;
            //for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p)
                //std::cout << " sum[p] ---->" << sum[p] << std::endl;
        }


        if (count>=20 ){
            search = false;
            //for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p)
                //std::cout << " sum[p] ---->" << sum[p] << std::endl;
        }else{
            // Calculate the next point
            ValueType norm2=0.0;
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){

                // maximum likelihood gradient (J.K. p. 64)
                gradient[p] = sum[p];
                norm2 += gradient[p]*gradient[p];
                //std::cout << " grad[p] ---->" << gradient[p] << std::endl;

            }
            norm2 = std::sqrt(norm2);
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p)
                point[p] += eta * gradient[p]/norm2;
            eta *= (ValueType)count/(count+1);
        }
    } // end while search

    std::cout <<std::endl<< "Best weights: ";
    for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
        std::cout << bestModelWeight[p] <<" ";
    }
    std::cout << " ==> ";
    std::cout << bestLoss << std::endl;

    modelWeights_ = bestModelWeight;
};
}
}
#endif


