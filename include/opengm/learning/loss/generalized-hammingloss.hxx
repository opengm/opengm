#pragma once
#ifndef OPENGM_GENERALIZED_HAMMING_LOSS_HXX
#define OPENGM_GENERALIZED_HAMMING_LOSS_HXX

#include "opengm/functions/explicit_function.hxx"
namespace opengm {
namespace learning {

/**
 * The generalized Hamming Loss incurs a penalty of nodeLossMultiplier[n] * labelLossMultiplier[l]
 * for node n taking label l, only if l is the same label as in the ground truth this amounts to zero.
 * One can imagine the overall cost matrix as outer product nodeLossMultiplier * labelLossMultiplier,
 * with zeros where the node label equals the ground truth.
 **/
class GeneralizedHammingLoss{
public:
    class Parameter{
    public:
        std::vector<double> nodeLossMultiplier_;
        std::vector<double> labelLossMultiplier_;

        bool operator==(const GeneralizedHammingLoss & other) const{
                return nodeLossMultiplier_ == labelLossMultiplier_;
        }
        bool operator<(const GeneralizedHammingLoss & other) const{
                return nodeLossMultiplier_ < labelLossMultiplier_;
        }
        bool operator>(const GeneralizedHammingLoss & other) const{
                nodeLossMultiplier_ > labelLossMultiplier_;
        }
    };


public:
    GeneralizedHammingLoss(const Parameter& param = Parameter()) : param_(param){}

    template<class IT1, class IT2>
            double loss(IT1 labelBegin, IT1 labelEnd, IT2 GTBegin,IT2 GTEnd) const;

    template<class GM, class IT>
    void addLoss(GM& gm, IT GTBegin) const;

private:
    Parameter param_;
};

template<class IT1, class IT2>
double GeneralizedHammingLoss::loss(IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
{
    double loss = 0.0;
    size_t nodeIndex = 0;

    for(; labelBegin!= labelEnd; ++labelBegin, ++GTBegin, ++nodeIndex){
        if(*labelBegin != *GTBegin){
            loss += param_.nodeLossMultiplier_[nodeIndex] * param_.labelLossMultiplier_[*labelBegin];
        }
    }
    return loss;
}

template<class GM, class IT>
void GeneralizedHammingLoss::addLoss(GM& gm, IT gt) const
{

    for(typename GM::IndexType i=0; i<gm.numberOfVariables(); ++i){
        typename GM::LabelType numL = gm.numberOfLabels(i);
        opengm::ExplicitFunction<typename GM::ValueType,typename GM::IndexType, typename GM::LabelType> f(&numL, &(numL)+1, 0);

        for(typename GM::LabelType l = 0; l < numL; ++l){
            f(l) = - param_.nodeLossMultiplier_[i] * param_.labelLossMultiplier_[l];
        }

        f(*gt) = 0;
        ++gt;
        gm.addFactor(gm.addFunction(f), &i, &(i)+1);
    }
}

} // namespace learning
} // namespace opengm

#endif 
