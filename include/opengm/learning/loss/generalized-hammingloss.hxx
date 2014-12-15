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
    template<class IT1, class IT2>
    GeneralizedHammingLoss(IT1 nodeLossMultiplierBegin,
                           IT1 nodeLossMultiplierEnd,
                           IT2 labelLossMultiplierBegin,
                           IT2 labelLossMultiplierEnd);

    template<class IT1, class IT2>
            double loss(IT1 labelBegin, IT1 labelEnd, IT2 GTBegin,IT2 GTEnd) const;

    template<class GM, class IT>
    void addLoss(GM& gm, IT GTBegin) const;

private:
    std::vector<double> nodeLossMultiplier_;
    std::vector<double> labelLossMultiplier_;
};

template<class IT1, class IT2>
GeneralizedHammingLoss::GeneralizedHammingLoss(IT1 nodeLossMultiplierBegin,
                                               IT1 nodeLossMultiplierEnd,
                                               IT2 labelLossMultiplierBegin,
                                               IT2 labelLossMultiplierEnd):
    nodeLossMultiplier_(nodeLossMultiplierBegin, nodeLossMultiplierEnd),
    labelLossMultiplier_(labelLossMultiplierBegin, labelLossMultiplierEnd)
{
}

template<class IT1, class IT2>
double GeneralizedHammingLoss::loss(IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
{
    double loss = 0.0;
    size_t nodeIndex = 0;

    for(; labelBegin!= labelEnd; ++labelBegin, ++GTBegin, ++nodeIndex){
        if(*labelBegin != *GTBegin){
            loss += nodeLossMultiplier_[nodeIndex] * labelLossMultiplier_[*labelBegin];
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
            f(l) = nodeLossMultiplier_[i] * labelLossMultiplier_[l];
        }

        f(*gt) = 0;
        ++gt;
        gm.addFactor(gm.addFunction(f), &i, &(i)+1);
    }
}

} // namespace learning
} // namespace opengm

#endif 
