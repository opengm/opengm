#pragma once
#ifndef OPENGM_GENERALIZED_HAMMING_LOSS_HXX
#define OPENGM_GENERALIZED_HAMMING_LOSS_HXX

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
class GeneralizedHammingLoss{
public:
    class Parameter{
    public:
        double getNodeLossMultiplier(const size_t i) const;
        double getLabelLossMultiplier(const size_t i) const;


        bool operator==(const GeneralizedHammingLoss & other) const{
                return nodeLossMultiplier_ == labelLossMultiplier_;
        }
        bool operator<(const GeneralizedHammingLoss & other) const{
                return nodeLossMultiplier_ < labelLossMultiplier_;
        }
        bool operator>(const GeneralizedHammingLoss & other) const{
                return nodeLossMultiplier_ > labelLossMultiplier_;
        }

        /**
         * serializes the parameter object to the given hdf5 group handle;
         * the group must contain a dataset "lossType" containing the
         * loss type as a string
         **/
        void save(hid_t& groupHandle) const;
        void load(const hid_t& groupHandle);
        static std::size_t getLossId() { return lossId_; }


        std::vector<double> nodeLossMultiplier_;
        std::vector<double> labelLossMultiplier_;


    private:
        static const std::size_t lossId_ = 16001;

    };


public:
    GeneralizedHammingLoss(const Parameter& param = Parameter()) : param_(param){}

    template<class GM, class IT1, class IT2>
            double loss(const GM & gm, IT1 labelBegin, IT1 labelEnd, IT2 GTBegin,IT2 GTEnd) const;

    template<class GM, class IT>
    void addLoss(GM& gm, IT GTBegin) const;

private:
    Parameter param_;
};

inline double GeneralizedHammingLoss::Parameter::getNodeLossMultiplier(const size_t i) const {
    if(i >= this->nodeLossMultiplier_.size()) {
        return 1.;
    }
    return this->nodeLossMultiplier_[i];
}

inline double GeneralizedHammingLoss::Parameter::getLabelLossMultiplier(const size_t i) const {
    if(i >= this->labelLossMultiplier_.size()) {
        return 1.;
    }
    return this->labelLossMultiplier_[i];
}

inline void GeneralizedHammingLoss::Parameter::save(hid_t& groupHandle) const {
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

inline void GeneralizedHammingLoss::Parameter::load(const hid_t& groupHandle) {
    if (H5Dopen(groupHandle, "nodeLossMultiplier", H5P_DEFAULT) >= 0) {
        marray::hdf5::loadVec(groupHandle, "nodeLossMultiplier", this->nodeLossMultiplier_);
    } else {
        std::cout << "nodeLossMultiplier of GeneralizedHammingLoss not found, setting default values" << std::endl;
    }

    if (H5Dopen(groupHandle, "labelLossMultiplier", H5P_DEFAULT) >= 0) {
        marray::hdf5::loadVec(groupHandle, "labelLossMultiplier", this->labelLossMultiplier_);
    } else {
        std::cout << "labelLossMultiplier of GeneralizedHammingLoss not found, setting default values" << std::endl;
    }
}

template<class GM, class IT1, class IT2>
double GeneralizedHammingLoss::loss(const GM & gm, IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
{
    double loss = 0.0;
    size_t nodeIndex = 0;

    for(; labelBegin!= labelEnd; ++labelBegin, ++GTBegin, ++nodeIndex){
        if(*labelBegin != *GTBegin){            
            loss += param_.getNodeLossMultiplier(nodeIndex) * param_.getLabelLossMultiplier(*labelBegin);
        }
    }
    return loss;
}

template<class GM, class IT>
void GeneralizedHammingLoss::addLoss(GM& gm, IT gt) const
{
    //std::cout<<"start to add loss\n";
    for(typename GM::IndexType i=0; i<gm.numberOfVariables(); ++i){
        //std::cout<<"   vi"<<i<<"\n";
        typename GM::LabelType numL = gm.numberOfLabels(i);
        //std::cout<<"   vi numL"<<numL<<"\n";
        opengm::ExplicitFunction<typename GM::ValueType,typename GM::IndexType, typename GM::LabelType> f(&numL, &numL+1, 0);

        //std::cout<<"   apply multiplier\n";
        for(typename GM::LabelType l = 0; l < numL; ++l){
            f(l) = - param_.getNodeLossMultiplier(i) * param_.getLabelLossMultiplier(l);
        }

        f(*gt) = 0;
        //std::cout<<"   increment\n";
        ++gt;
        //std::cout<<"   add\n";
        gm.addFactor(gm.addFunction(f), &i, &i+1);
        //std::cout<<"   next\n";
    }
    //std::cout<<"end add loss\n";
}

} // namespace learning
} // namespace opengm

#endif 
