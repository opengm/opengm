#pragma once
#ifndef OPENGM_EDITABLEDATASET_HXX
#define OPENGM_EDITABLEDATASET_HXX

#include <vector>
#include <cstdlib>

#include <opengm/learning/dataset/dataset.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include "../../graphicalmodel/weights.hxx"
#include "../loss/noloss.hxx"

namespace opengm {
   namespace datasets{

     template<class GM, class LOSS>
      class EditableDataset : public Dataset<GM, LOSS>{
      public:
         typedef GM                     GMType;
         typedef GM                     GMWITHLOSS;
         typedef LOSS                   LossType;
         typedef typename LOSS::Parameter LossParameterType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef opengm::learning::Weights<ValueType> Weights;
         typedef std::vector<LabelType> GTVector;

         EditableDataset(size_t numInstances=0) : Dataset<GM, LOSS>(numInstances) {}
         EditableDataset(std::vector<GM>& gms, std::vector<GTVector >& gts, std::vector<LossParameterType>& lossParams);

         void setInstance(const size_t i, const GM& gm, const GTVector& gt, const LossParameterType& p=LossParameterType());
         void pushBackInstance(const GM& gm, const GTVector& gt, const LossParameterType& p=LossParameterType());
         void setWeights(Weights& w);
      };

    template<class GM, class LOSS>
    EditableDataset<GM, LOSS>::EditableDataset(std::vector<GM>& gms,
                                               std::vector<GTVector >& gts,
                                               std::vector<LossParameterType>& lossParams)
        : Dataset<GM, LOSS>(gms.size())
    {
        for(size_t i=0; i<gms.size(); ++i){
            setInstance(i, gms[i], gts[i], lossParams[i]);
            this->buildModelWithLoss(i);
        }
    }

    template<class GM, class LOSS>
    void EditableDataset<GM, LOSS>::setInstance(const size_t i, const GM& gm, const GTVector& gt, const LossParameterType& p) {
        OPENGM_CHECK_OP(i, <, this->gms_.size(),"");
        OPENGM_CHECK_OP(i, <, this->gts_.size(),"");
        OPENGM_CHECK_OP(i, <, this->lossParams_.size(),"");
        OPENGM_CHECK_OP(i, <, this->gmsWithLoss_.size(),"");
        this->gms_[i] = gm;
        this->gts_[i] = gt;
        this->lossParams_[i] = p;
        this->buildModelWithLoss(i);
    }

    template<class GM, class LOSS>
    void EditableDataset<GM, LOSS>::pushBackInstance(const GM& gm, const GTVector& gt, const LossParameterType& p) {
        this->gms_.push_back(gm);
        this->gts_.push_back(gt);
        this->lossParams_.push_back(p);
        this->gmsWithLoss_.resize(this->gts_.size());
        this->buildModelWithLoss(this->gts_.size()-1);
        this->count_.push_back(0);
        this->isCached_.push_back(bool());
        OPENGM_CHECK_OP(this->gms_.size(), ==, this->gts_.size(),"");
        OPENGM_CHECK_OP(this->gms_.size(), ==, this->lossParams_.size(),"");
        OPENGM_CHECK_OP(this->gms_.size(), ==, this->gmsWithLoss_.size(),"");
    }

    template<class GM, class LOSS>
    void EditableDataset<GM, LOSS>::setWeights(Weights& w) {
        this->weights_ = w;
    }

   } // namespace datasets
} // namespace opengm

#endif 
