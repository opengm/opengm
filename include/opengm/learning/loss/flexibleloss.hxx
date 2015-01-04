#pragma once
#ifndef OPENGM_FLEXIBLE_LOSS_HXX
#define OPENGM_FLEXIBLE_LOSS_HXX

#include "opengm/functions/explicit_function.hxx"
#include "opengm/graphicalmodel/graphicalmodel_hdf5.hxx"
#include "hdf5.h"

namespace opengm {
namespace learning {

/**
 * The generalized Hamming Loss incurs a penalty of nodeLossMultiplier[n] * labelLossMultiplier[l]
 * for node n taking label l, only if l is the same label as in the ground truth this amounts to zero.
 * One can imagine the overall cost matrix as outer product nodeLossMultiplier * labelLossMultiplier,
 * with zeros where the node label equals the ground truth.
 **/
class FlexibleLoss{
public:
    class Parameter{
    public:

        Parameter{
            lambdaWeight = 1.0;
        }
        enum LossType{
            Hamming = 0 ,
            L1 = 1,
            L2 = 2,
            Partition = 3,
            ConfMat = 4
        };

        bool operator==(const FlexibleLoss & other) const{
            throw opengm::RuntimeError("do not call me");
        }
        bool operator<(const FlexibleLoss & other) const{
            throw opengm::RuntimeError("do not call me");    
        }
        bool operator>(const FlexibleLoss & other) const{
            throw opengm::RuntimeError("do not call me");
        }
        double getNodeLossMultiplier(const size_t i) const;
        double getLabelLossMultiplier(const size_t i) const;

        double getLabelConfMatMultiplier(const size_t l, const size_t lgt)const;
        /**
         * serializes the parameter object to the given hdf5 group handle;
         * the group must contain a dataset "lossType" containing the
         * loss type as a string
         **/
        void save(hid_t& groupHandle) const;
        void load(const hid_t& groupHandle);
        static std::size_t getLossId() { return lossId_; }


        std::vector<double>     nodeLossMultiplier_;
        std::vector<double>     labelLossMultiplier_;
        std::vector<double>     factorMultipier_;
        marray::Marray<double>  confMat_;
        LossType lossType_;
        double lambdaWeight;


    private:
        static const std::size_t lossId_ = 16002;

    };


public:
    FlexibleLoss(const Parameter& param = Parameter()) : param_(param){}

    template<class GM, class IT1, class IT2>
            double loss(const GM & gm, IT1 labelBegin, IT1 labelEnd, IT2 GTBegin,IT2 GTEnd) const;

    template<class GM, class IT>
    void addLoss(GM& gm, IT GTBegin) const;

private:
    Parameter param_;
};

inline double FlexibleLoss::Parameter::getNodeLossMultiplier(const size_t i) const {
    if(i >= this->nodeLossMultiplier_.size()) {
        return 1.;
    }
    return this->nodeLossMultiplier_[i];
}

inline double FlexibleLoss::Parameter::getLabelLossMultiplier(const size_t i) const {
    if(i >= this->labelLossMultiplier_.size()) {
        return 1.;
    }
    return this->labelLossMultiplier_[i];
}

double FlexibleLoss::Parameter::getLabelConfMatMultiplier(const size_t l, const size_t lgt)const{
    if(l<confMat_.shape(0) && lgt<confMat_.shape(1)){
        return confMat_(l, lgt);
    }
    return 1.0;
}

inline void FlexibleLoss::Parameter::save(hid_t& groupHandle) const {
    std::vector<std::size_t> name;
    name.push_back(this->getLossId());
    marray::hdf5::save(groupHandle,"lossId",name);

    if (this->nodeLossMultiplier_.size() > 0) {
        marray::hdf5::save(groupHandle,"nodeLossMultiplier",this->nodeLossMultiplier_);
    }
    if (this->labelLossMultiplier_.size() > 0) {
        marray::hdf5::save(groupHandle,"labelLossMultiplier",this->labelLossMultiplier_);
    }
}

inline void FlexibleLoss::Parameter::load(const hid_t& groupHandle) {
    if (H5Dopen(groupHandle, "nodeLossMultiplier", H5P_DEFAULT) >= 0) {
        marray::hdf5::loadVec(groupHandle, "nodeLossMultiplier", this->nodeLossMultiplier_);
    } else {
        std::cout << "nodeLossMultiplier of FlexibleLoss not found, setting default values" << std::endl;
    }

    if (H5Dopen(groupHandle, "labelLossMultiplier", H5P_DEFAULT) >= 0) {
        marray::hdf5::loadVec(groupHandle, "labelLossMultiplier", this->labelLossMultiplier_);
    } else {
        std::cout << "labelLossMultiplier of FlexibleLoss not found, setting default values" << std::endl;
    }
}

template<class GM, class IT1, class IT2>
double FlexibleLoss::loss(const GM & gm, IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
{


    double loss = 0.0;
    size_t nodeIndex = 0;
    if(param_.lossType_ == Parameter::Hamming){
        for(; labelBegin!= labelEnd; ++labelBegin, ++GTBegin, ++nodeIndex){
            if(*labelBegin != *GTBegin){            
                loss += param_.getNodeLossMultiplier(nodeIndex) * param_.getLabelLossMultiplier(*labelBegin);
            }
        }
    }
    return loss;
}

template<class GM, class IT>
void FlexibleLoss::addLoss(GM& gm, IT gt) const
{
    typedef typename  GM::LabelType LabelType;
    typedef typename  GM::IndexType IndexType;
    typedef typename  GM::ValueType ValueType;
    typedef opengm::ExplicitFunction<ValueType, IndexType,  LabelType>  ExplicitFunction;


    if(param_.lossType_ == Parameter::Hamming){
        for(IndexType i=0; i<gm.numberOfVariables(); ++i){
            LabelType numL = gm.numberOfLabels(i);
            ExplicitFunction f(&numL, &numL+1, 0);
            for(LabelType l = 0; l < numL; ++l){
                f(l) = - param_.getNodeLossMultiplier(i) * param_.getLabelLossMultiplier(l);
            }
            f(*gt) = 0;
            ++gt;
            gm.addFactor(gm.addFunction(f), &i, &i+1);     
        }
    }
    else if(param_.lossType_ == Parameter::L1 || param_.lossType_ == Parameter::L2){
        const size_t norm == aram_.lossType_ == Parameter::L1 ? 1 : 2;
        for(IndexType i=0; i<gm.numberOfVariables(); ++i){
            LabelType numL = gm.numberOfLabels(i);
            ExplicitFunction f(&numL, &numL+1, 0);
            const LabelType gtL = *gt;
            for(LabelType l = 0; l < numL; ++l){
                f(l) = - param_.getNodeLossMultiplier(i) * std::pow(std::abs(gtL - l), norm) * param_.lambdaWeight;
            }
            f(*gt) = 0;
            ++gt;
            gm.addFactor(gm.addFunction(f), &i, &i+1);     
        }
    }
    else if(param_.lossType_ == Parameter::L1 || param_.lossType_ == Parameter::L2){
        const size_t norm == aram_.lossType_ == Parameter::L1 ? 1 : 2;
        for(IndexType i=0; i<gm.numberOfVariables(); ++i){
            LabelType numL = gm.numberOfLabels(i);
            ExplicitFunction f(&numL, &numL+1, 0);
            const LabelType gtL = *gt;
            for(LabelType l = 0; l < numL; ++l){
                f(l) = - param_.getNodeLossMultiplier(i) * param_.getLabelConfMatMultiplier(l, gtL);
            }
            f(*gt) = 0;
            ++gt;
            gm.addFactor(gm.addFunction(f), &i, &i+1);     
        }
    }
    else if(param_.lossType_ == Parameter::Partition){
        throw opengm::RuntimeError("Partition / Multicut Loss is not yet implemented");
    }
    else{
        throw opengm::RuntimeError("INTERNAL ERROR: unknown Loss Type");
    }
}

} // namespace learning
} // namespace opengm

#endif 
