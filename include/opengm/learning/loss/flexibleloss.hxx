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

        enum LossType{
            Hamming = 0 ,
            L1 = 1,
            L2 = 2,
            Partition = 3,
            ConfMat = 4
        };

        Parameter(){
            lossType_ = Hamming;
        }


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
        double getFactorLossMultiplier(const size_t i) const;
        double getLabelConfMatMultiplier(const size_t l, const size_t lgt)const;
        /**
         * serializes the parameter object to the given hdf5 group handle;
         * the group must contain a dataset "lossType" containing the
         * loss type as a string
         **/
        void save(hid_t& groupHandle) const;
        void load(const hid_t& groupHandle);
        static std::size_t getLossId() { return lossId_; }

        LossType lossType_;
        std::vector<double>     nodeLossMultiplier_;
        std::vector<double>     labelLossMultiplier_;
        std::vector<double>     factorMultipier_;
        marray::Marray<double>  confMat_;
        


    private:
        static const std::size_t lossId_ = 16006;

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

inline double FlexibleLoss::Parameter::getFactorLossMultiplier(const size_t i) const {
    if(i >= this->factorMultipier_.size()) {
        return 1.;
    }
    return this->factorMultipier_[i];
}

inline double FlexibleLoss::Parameter::getLabelLossMultiplier(const size_t i) const {
    if(i >= this->labelLossMultiplier_.size()) {
        return 1.;
    }
    return this->labelLossMultiplier_[i];
}

inline double FlexibleLoss::Parameter::getLabelConfMatMultiplier(const size_t l, const size_t lgt)const{
    if(l<confMat_.shape(0) && lgt<confMat_.shape(1)){
        return confMat_(l, lgt);
    }
    return 1.0;
}

inline void FlexibleLoss::Parameter::save(hid_t& groupHandle) const {
    std::vector<std::size_t> name;
    name.push_back(this->getLossId());
    marray::hdf5::save(groupHandle,"lossId",name);


    std::vector<size_t> lossType(1, size_t(lossType_));
    marray::hdf5::save(groupHandle,"lossType",lossType);

    if (this->factorMultipier_.size() > 0) {
        marray::hdf5::save(groupHandle,"factorLossMultiplier",this->factorMultipier_);
    }
    if (this->nodeLossMultiplier_.size() > 0) {
        marray::hdf5::save(groupHandle,"nodeLossMultiplier",this->nodeLossMultiplier_);
    }
    if (this->labelLossMultiplier_.size() > 0) {
        marray::hdf5::save(groupHandle,"labelLossMultiplier",this->labelLossMultiplier_);
    }
}

inline void FlexibleLoss::Parameter::load(const hid_t& groupHandle) {

    std::cout<<"load loss type \n";
    std::vector<size_t> lossType;
    marray::hdf5::loadVec(groupHandle, "lossType", lossType);
    if(lossType[0] == size_t(Hamming)){
        lossType_ = Hamming;
    }
    else if(lossType[0] == size_t(L1)){
        lossType_ = L1;
    }
    else if(lossType[0] == size_t(L1)){
        lossType_ = L1;
    }
    else if(lossType[0] == size_t(L2)){
        lossType_ = L2;
    }
    else if(lossType[0] == size_t(Partition)){
        lossType_ = Partition;
    }
    else if(lossType[0] == size_t(ConfMat)){
        lossType_ = ConfMat;
    }

    
    if (H5Lexists(groupHandle, "nodeLossMultiplier", H5P_DEFAULT)) {
        marray::hdf5::loadVec(groupHandle, "nodeLossMultiplier", this->nodeLossMultiplier_);
    } 
    else {
        //std::cout << "nodeLossMultiplier of FlexibleLoss not found, setting default values" << std::endl;
    }

    //std::cout<<"load factorLossMultiplier \n";
    if (H5Lexists(groupHandle, "factorLossMultiplier", H5P_DEFAULT)  ) {
        marray::hdf5::loadVec(groupHandle, "factorLossMultiplier", this->factorMultipier_);
    } 
    else {
        //std::cout << "factorLossMultiplier of FlexibleLoss not found, setting default values" << std::endl;
    }

    //std::cout<<"load labelLossMultiplier \n";
    if (H5Lexists(groupHandle, "labelLossMultiplier", H5P_DEFAULT) ) {
        marray::hdf5::loadVec(groupHandle, "labelLossMultiplier", this->labelLossMultiplier_);
    } 
    else {
        //std::cout << "labelLossMultiplier of FlexibleLoss not found, setting default values" << std::endl;
    }
}

template<class GM, class IT1, class IT2>
double FlexibleLoss::loss(const GM & gm, IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
{
    typedef typename  GM::LabelType LabelType;
    typedef typename  GM::IndexType IndexType;
    typedef typename  GM::ValueType ValueType;

    double loss = 0.0;
    size_t nodeIndex = 0;
    if(param_.lossType_ == Parameter::Hamming){
        for(; labelBegin!= labelEnd; ++labelBegin, ++GTBegin, ++nodeIndex){
            if(*labelBegin != *GTBegin){            
                loss += param_.getNodeLossMultiplier(nodeIndex) * param_.getLabelLossMultiplier(*labelBegin);
            }
        }
    }
    else if(param_.lossType_ == Parameter::L1 || param_.lossType_ == Parameter::L2){
        const size_t norm = param_.lossType_ == Parameter::L1 ? 1 : 2;
        for(; labelBegin!= labelEnd; ++labelBegin, ++GTBegin, ++nodeIndex){
            if(*labelBegin != *GTBegin){            
                loss += param_.getNodeLossMultiplier(nodeIndex) * std::pow(std::abs(*GTBegin - *labelBegin), norm);
            }
        }
    }
    else if(param_.lossType_ == Parameter::ConfMat){
        throw opengm::RuntimeError("ConfMat Loss is not yet implemented");
    }
    else if(param_.lossType_ == Parameter::Partition){

        const size_t nFac = gm.numberOfFactors();

        for(size_t fi=0; fi<nFac; ++fi){
            const size_t nVar = gm[fi].numberOfVariables();
            OPENGM_CHECK_OP(nVar,==,2,"Partition / Multicut Loss  is only allowed if the graphical model has only"
                                      " second order factors (this might be changed in the future");
            const IndexType vis[2] = { gm[fi].variableIndex(0), gm[fi].variableIndex(1)};
            const LabelType nl[2]  = { gm.numberOfLabels(vis[0]), gm.numberOfLabels(vis[1])};
            const double facVal = param_.getFactorLossMultiplier(fi);
            // in the gt they are in the same cluster
            if( (GTBegin[vis[0]] == GTBegin[vis[1]]) !=
                (labelBegin[vis[0]] == labelBegin[vis[1]])  ){
                loss +=facVal;
            }
        }
    }
    else{
        throw opengm::RuntimeError("INTERNAL ERROR: unknown Loss Type");
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
    typedef opengm::PottsFunction<ValueType, IndexType,  LabelType>  Potts;

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
        const size_t norm = param_.lossType_ == Parameter::L1 ? 1 : 2;
        for(IndexType i=0; i<gm.numberOfVariables(); ++i){
            LabelType numL = gm.numberOfLabels(i);
            ExplicitFunction f(&numL, &numL+1, 0);
            const LabelType gtL = *gt;
            for(LabelType l = 0; l < numL; ++l){
                f(l) = - param_.getNodeLossMultiplier(i) * std::pow(std::abs(gtL - l), norm);
            }
            f(*gt) = 0;
            ++gt;
            gm.addFactor(gm.addFunction(f), &i, &i+1);     
        }
    }
    else if(param_.lossType_ == Parameter::ConfMat){
        throw opengm::RuntimeError("ConfMat Loss is not yet implemented");
    }
    else if(param_.lossType_ == Parameter::Partition){

        const size_t nFactorsInit = gm.numberOfFactors();

        for(size_t fi=0; fi<nFactorsInit; ++fi){
            const size_t nVar = gm[fi].numberOfVariables();
            OPENGM_CHECK_OP(nVar,==,2,"Partition / Multicut Loss  is only allowed if the graphical model has only"
                                      " second order factors (this might be changed in the future");

            const IndexType vis[2] = { gm[fi].variableIndex(0), gm[fi].variableIndex(1)};
            const LabelType nl[2]  = { gm.numberOfLabels(vis[0]), gm.numberOfLabels(vis[1])};

            const double facVal = param_.getFactorLossMultiplier(fi);

            // in the gt they are in the same cluster
            if(gt[vis[0]] == gt[vis[1]]){
                Potts pf(nl[0],nl[1], 0.0, -1.0*facVal);
                gm.addFactor(gm.addFunction(pf), vis,vis+2);
            }
            // in the gt they are in different clusters
            else{
                Potts pf(nl[0],nl[1], -1.0*facVal, 0.0);
                gm.addFactor(gm.addFunction(pf), vis,vis+2);
            }
        }
    }
    else{
        throw opengm::RuntimeError("INTERNAL ERROR: unknown Loss Type");
    }
}

} // namespace learning
} // namespace opengm

#endif 
