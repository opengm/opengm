#pragma once
#ifndef OPENGM_EDITABLEDATASET_HXX
#define OPENGM_EDITABLEDATASET_HXX

#include <vector>
#include <cstdlib>

#include <opengm/learning/dataset/dataset.hxx>
#include "../../graphicalmodel/weights.hxx"
#include "../loss/noloss.hxx"

namespace opengm {
   namespace datasets{

    // template< typename Weights >
    // struct LinkWeights{

    //     Weights& w_;
    //     LinkWeights(const Weights& w):w_(w){}

    //     template<class FUNCTION>
    //     void operator()(const FUNCTION & function)
    //     {
    //         function.setWeights(w_);
    //     }
    // };

      template<class GM, class LOSS, class LOSS_GM = DefaultLossGm<GM> >
      class EditableDataset : public Dataset<GM, LOSS, LOSS_GM>{
      public:
         typedef GM                     GMType;
         typedef typename Dataset<GM, LOSS, LOSS_GM>::GMWITHLOSS   GMWITHLOSS;
         typedef LOSS                   LossType;
         typedef typename LOSS::Parameter LossParameterType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef opengm::learning::Weights<ValueType> Weights;
         typedef std::vector<LabelType> GTVector;

         EditableDataset(size_t numInstances=0) : Dataset<GM, LOSS,LOSS_GM>(numInstances) {}
         EditableDataset(std::vector<GM>& gms, std::vector<GTVector >& gts, std::vector<LossParameterType>& lossParams);

         void setInstance(const size_t i, const GM& gm, const GTVector& gt, const LossParameterType& p=LossParameterType());
         void setGT(const size_t i, const GTVector& gt);
         void pushBackInstance(const GM& gm, const GTVector& gt, const LossParameterType& p=LossParameterType());
         void setWeights(Weights& w);
      };

    template<class GM, class LOSS, class LOSS_GM>
    EditableDataset<GM, LOSS, LOSS_GM>::EditableDataset(std::vector<GM>& gms,
                                               std::vector<GTVector >& gts,
                                               std::vector<LossParameterType>& lossParams)
        : Dataset<GM, LOSS, LOSS_GM>(gms.size())
    {
        for(size_t i=0; i<gms.size(); ++i){
            setInstance(i, gms[i], gts[i], lossParams[i]);
            this->buildModelWithLoss(i);
        }
    }

    template<class GM, class LOSS, class LOSS_GM>
    void EditableDataset<GM, LOSS, LOSS_GM>::setInstance(const size_t i, const GM& gm, const GTVector& gt, const LossParameterType& p) {
        OPENGM_CHECK_OP(i, <, this->gms_.size(),"");
        OPENGM_CHECK_OP(i, <, this->gts_.size(),"");
        OPENGM_CHECK_OP(i, <, this->lossParams_.size(),"");
        OPENGM_CHECK_OP(i, <, this->gmsWithLoss_.size(),"");
        this->gms_[i] = gm;
        this->gts_[i] = gt;
        this->lossParams_[i] = p;
        //std::cout<<"build model with loss\n";
        this->buildModelWithLoss(i);
        //std::cout<<"build model with loss DONE\n";
    }

    template<class GM, class LOSS, class LOSS_GM>
    void EditableDataset<GM, LOSS, LOSS_GM>::setGT(const size_t i, const GTVector& gt) {
        OPENGM_CHECK_OP(i, <, this->gts_.size(),"");
        this->gts_[i] = gt;
        this->buildModelWithLoss(i);
    }

    template<class GM, class LOSS, class LOSS_GM>
    void EditableDataset<GM, LOSS, LOSS_GM>::pushBackInstance(const GM& gm, const GTVector& gt, const LossParameterType& p) {
        this->gms_.push_back(gm);
        this->gts_.push_back(gt);
        this->lossParams_.push_back(p);
        this->gmsWithLoss_.resize(this->gts_.size());
        this->isCached_.resize(this->gts_.size());
        this->count_.resize(this->gts_.size());
        this->buildModelWithLoss(this->gts_.size()-1);        
        OPENGM_CHECK_OP(this->gms_.size(), ==, this->gts_.size(),"");
        OPENGM_CHECK_OP(this->gms_.size(), ==, this->lossParams_.size(),"");
        OPENGM_CHECK_OP(this->gms_.size(), ==, this->gmsWithLoss_.size(),"");
    }

    template<class GM, class LOSS, class LOSS_GM>
    void EditableDataset<GM, LOSS, LOSS_GM>::setWeights(Weights& w) {
        this->weights_ = w;
        // LinkWeights<Weights> LinkFunctor(w);
        // for(size_t i=0; i<this->gms_.size(); ++i){
        //     (this->gms_[i])[0].callFunctor(LinkFunctor);
        // }
    }

   } // namespace datasets
} // namespace opengm

#endif 
