#pragma once
#ifndef OPENGM_MAXIMUM_LIKELIHOOD_LEARNER_HXX
#define OPENGM_MAXIMUM_LIKELIHOOD_LEARNER_HXX

#include <vector>
#include <fstream>
#include <opengm/inference/messagepassing/messagepassing.hxx>
//#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/view_convert_function.hxx>
//#include <opengm/functions/learnable/lpotts.hxx>
//#include <opengm/functions/learnable/lsum_of_experts.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
//#include <opengm/inference/icm.hxx>
//
//typedef double ValueType;
//typedef size_t IndexType;
//typedef size_t LabelType;
//typedef opengm::meta::TypeListGenerator<
//    opengm::ExplicitFunction<ValueType,IndexType,LabelType>,
//    opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>,
//    opengm::functions::learnable::LSumOfExperts<ValueType,IndexType,LabelType>
//>::type FunctionListType;
//
//typedef opengm::GraphicalModel<
//    ValueType,opengm::Adder,
//    FunctionListType,
//    opengm::DiscreteSpace<IndexType,LabelType>
//> GM;
//
//typedef opengm::ICM<GM,opengm::Minimizer> INF;
//typedef opengm::learning::Weights<ValueType> WeightType;



namespace opengm {
namespace learning {

template<class IT> 
class WeightGradientFunctor{
public:
   WeightGradientFunctor(size_t weightIndex, IT labelVectorBegin) //std::vector<size_t>::iterator labelVectorBegin)
        : weightIndex_(weightIndex),
          labelVectorBegin_(labelVectorBegin){
    }

    template<class F>
    void operator()(const F & function ){
        size_t index=-1;
        for(size_t i=0; i<function.numberOfWeights();++i)
            if(function.weightIndex(i)==weightIndex_)
                index=i;
        if(index!=-1)
            result_ = function.weightGradient(index, labelVectorBegin_);
        else
            result_ = 0;
    }

    size_t weightIndex_;
    IT  labelVectorBegin_;
    double result_;
};

template<class DATASET>
class MaximumLikelihoodLearner
{
public:
    typedef DATASET DatasetType;
    typedef typename DATASET::GMType   GMType;
    typedef typename GMType::ValueType ValueType;
    typedef typename GMType::IndexType IndexType;
    typedef typename GMType::LabelType LabelType;
    typedef typename GMType::FactorType FactorType;
    typedef opengm::learning::Weights<ValueType> WeightType;  

    typedef typename opengm::ExplicitFunction<ValueType,IndexType,LabelType> FunctionType;
    typedef typename opengm::ViewConvertFunction<GMType,Minimizer,ValueType> ViewFunctionType;
    typedef typename GMType::FunctionIdentifier FunctionIdentifierType;
    typedef typename opengm::meta::TypeListGenerator<FunctionType,ViewFunctionType>::type FunctionListType;
    typedef opengm::GraphicalModel<ValueType,opengm::Multiplier, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GmBpType;
    typedef BeliefPropagationUpdateRules<GmBpType, opengm::Integrator> UpdateRules;
    typedef MessagePassing<GmBpType, opengm::Integrator, UpdateRules, opengm::MaxDistance> BeliefPropagation;
   
    class Parameter{
    public:
       size_t maxNumSteps_;
       Parameter() :
          maxNumSteps_(100)
          {;}
    };
   

    MaximumLikelihoodLearner(DATASET&, const Parameter & w= Parameter() );

   //  template<class INF>
   void learn();//const typename INF::Parameter&);

    const opengm::learning::Weights<ValueType>& getModelWeights(){return modelWeights_;}
    Parameter& getLerningWeights(){return param_;}

private:
    DATASET& dataset_;
    opengm::learning::Weights<ValueType> modelWeights_;
    Parameter param_;
};

template<class DATASET>
MaximumLikelihoodLearner<DATASET>::MaximumLikelihoodLearner(DATASET& ds, const Parameter& w)
    : dataset_(ds), param_(w)
{
    modelWeights_ = opengm::learning::Weights<ValueType>(ds.getNumberOfWeights());
}


template<class DATASET>
//template<class INF>
void MaximumLikelihoodLearner<DATASET>::learn(){//const typename INF::Parameter &infParam){

    opengm::learning::Weights<ValueType> modelWeight( dataset_.getNumberOfWeights() );
    opengm::learning::Weights<ValueType> bestModelWeight( dataset_.getNumberOfWeights() );
    //double bestLoss = 100000000.0;
    std::vector<ValueType> point(dataset_.getNumberOfWeights(),0);
    std::vector<ValueType> gradient(dataset_.getNumberOfWeights(),0);
    std::vector<ValueType> Delta(dataset_.getNumberOfWeights(),0);
    for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p)
        point[p] = ValueType((0));


    typename DATASET::LossType lossFunction;
    bool search=true;
    int count=0;

    std::vector< std::vector<ValueType> > w( dataset_.getNumberOfModels(), std::vector<ValueType> ( dataset_.getModel(0).numberOfVariables()) );

    /***********************************************************************************************************/
    // construct Ground Truth dependent weights
    /***********************************************************************************************************/

    for(IndexType m=0; m<dataset_.getNumberOfModels(); ++m){ // for each model
        const GMType &model = dataset_.getModel(m);
        const std::vector<LabelType>& gt =  dataset_.getGT(m);

        for(IndexType v=0; v<model.numberOfVariables();++v)
            w[m][v]=(ValueType)gt[v];
    }

    ValueType eta = 0.1;
    ValueType delta = 0.25; // 0 <= delta <= 0.5
    ValueType D_a = 1.0; // distance treshold
    ValueType optFun, bestOptFun=0.0;

    while(search){
        ++count;
        //if (count % 1000 == 0)
        std::cout << "---count--->" << count << "     ";

        // Get Weights
        for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
            modelWeight.setWeight(p, point[p]);
        }

        // /***********************************************************************************************************/
        // // calculate current loss - not needed
        // /***********************************************************************************************************/
        // opengm::learning::Weights<ValueType>& mp =  dataset_.getWeights();
        // mp = modelWeight;
        // std::vector< std::vector<typename INF::LabelType> > confs( dataset_.getNumberOfModels() );
        // double loss = 0;
        // for(size_t m=0; m<dataset_.getNumberOfModels(); ++m){
        //    INF inf( dataset_.getModel(m),infParam);
        //    inf.infer();
        //    inf.arg(confs[m]);
        //    const std::vector<typename INF::LabelType>& gt =  dataset_.getGT(m);
        //    loss += lossFunction.loss(dataset_.getModel(m), confs[m].begin(), confs[m].end(), gt.begin(), gt.end());
        // }

        // std::cout << " eta = " << eta << "   weights  ";//<< std::endl;
        // for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
        //     std::cout << modelWeight[p] << " " ;
        // }

        // optFun=0.0;

        /***********************************************************************************************************/
        // Loopy Belief Propagation setup
        /***********************************************************************************************************/
     

        const IndexType maxNumberOfIterations = 40;
        const double convergenceBound = 1e-7;
        const double damping = 0.5;
        typename BeliefPropagation::Parameter weight(maxNumberOfIterations, convergenceBound, damping);

        std::vector< std::vector<ValueType> > b  ( dataset_.getNumberOfModels(), std::vector<ValueType> ( dataset_.getModel(0).numberOfFactors()) );

        for(IndexType m=0; m<dataset_.getNumberOfModels(); ++m){

           //****************************************
           // Build dummy model
           //***************************************
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
            const std::vector<LabelType>& gt =  dataset_.getGT(m);
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
                const std::vector<LabelType>& gt =  dataset_.getGT(m);
                ValueType f_p;

                for(IndexType f=0; f<dataset_.getModel(m).numberOfFactors();++f){
                    const FactorType &factor = dataset_.getModel(m)[f];
                    std::vector<IndexType> indexVector( factor.variableIndicesBegin(), factor.variableIndicesEnd() );
                    std::vector<LabelType> labelVector( factor.numberOfVariables());
                    piW[m][f]=1.0;

                    for(IndexType v=0; v<factor.numberOfVariables();++v){
                        labelVector[v] = gt[indexVector[v]];
                        piW[m][f] *=w[m][indexVector[v]];
                    }
                    WeightGradientFunctor<typename std::vector<LabelType>::iterator> weightGradientFunctor(p, labelVector.begin());
                    factor.callFunctor(weightGradientFunctor);
                    f_p =weightGradientFunctor.result_;

                    // gradient
                    // ( marginals - ground_truth ) * factor_gradient_p
                    sum[p] += (b[m][f] - piW[m][f]) * f_p;

                    // likelihood function
                    // marginals - ground_truth * factor
                    optFun += b[m][f] - piW[m][f] * factor(labelVector.begin());
                }
            }
        }
        //std::cout << " loss = " << loss << " optFun = " << optFun << " optFunTmp = " << optFunTmp << std::endl;
        //std::cout << " loss = " << loss << " optFun = " << optFun << std::endl; 
        std::cout << " optFun = " << optFun << std::endl;

        if(optFun>=bestOptFun){
            bestOptFun=optFun;
            bestModelWeight=modelWeight;
            bestOptFun=optFun;
            //bestLoss=loss;
        }

        if (count>=param_.maxNumSteps_){
            search = false;
        }else{
            // Calculate the next point
            ValueType norm2=0.0;
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
                gradient[p] = sum[p];
                norm2 += gradient[p]*gradient[p];
            }
            norm2 = std::sqrt(norm2);
            for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
                gradient[p] /= norm2;
                std::cout << " gradient [" << p << "] = " << gradient[p] << std::endl;
                point[p] += eta * gradient[p];

            }
            eta *= (ValueType)count/(count+1);
        }
    } // end while search

    std::cout <<std::endl<< "Best weights: ";
    for(IndexType p=0; p<dataset_.getNumberOfWeights(); ++p){
        std::cout << bestModelWeight[p] <<" ";
    }
    std::cout << " ==> ";
    //std::cout << " loss = " << bestLoss << " bestOptFun = " << bestOptFun << " gradient [" << 0 << "] = " << gradient[0] << std::endl;
    std::cout << " bestOptFun = " << bestOptFun << " gradient [" << 0 << "] = " << gradient[0] << std::endl;

    modelWeights_ = bestModelWeight;
};
}
}
#endif


