#pragma once
#ifndef OPENGM_DATASET_HXX
#define OPENGM_DATASET_HXX

#include <vector>
#include <cstdlib>

#include "../../graphicalmodel/weights.hxx"
#include "../../functions/unary_loss_function.hxx"
#include "../loss/noloss.hxx"

namespace opengm {
   namespace datasets{
     
    template<class GM>
    struct DefaultLossGm{

        // make the graphical model with loss
        typedef typename GM::SpaceType         SpaceType;
        typedef typename GM::ValueType         ValueType;
        typedef typename GM::IndexType         IndexType;
        typedef typename GM::LabelType         LabelType;
        typedef typename GM::OperatorType      OperatorType;
        typedef typename GM::FunctionTypeList  OrgFunctionTypeList;

        // extend the typelist
        typedef typename opengm::meta::TypeListGenerator<
            opengm::ExplicitFunction<ValueType,IndexType,LabelType>, 
            opengm::UnaryLossFunction<ValueType,IndexType,LabelType>
        >::type LossOnlyFunctionTypeList;

        typedef typename opengm::meta::MergeTypeListsNoDuplicates<
            OrgFunctionTypeList,LossOnlyFunctionTypeList
        >::type CombinedList;
        // loss graphical model type

        typedef GraphicalModel<ValueType, OperatorType, CombinedList, SpaceType> type;
    };

    template<class GM, class LOSS=opengm::learning::NoLoss, class LOSS_GM = DefaultLossGm<GM> >
    class Dataset{
    public:
        typedef GM                       GMType;

        // generate the gm with loss here atm (THIS IS WRONG)
        typedef typename opengm::meta::EvalIf<
        opengm::meta::Compare<LOSS_GM, DefaultLossGm<GM> >::value,
        DefaultLossGm<GM>,
        meta::Self<LOSS_GM>
        >::type GMWITHLOSS;

        //typedef GM                       GMWITHLOSS;
        typedef LOSS                     LossType;
        typedef typename LOSS::Parameter LossParameterType;
        typedef typename GM::ValueType   ValueType;
        typedef typename GM::IndexType   IndexType;
        typedef typename GM::LabelType   LabelType;


        typedef opengm::learning::Weights<ValueType> Weights;
        typedef opengm::learning::WeightConstraints<ValueType> WeightConstraintsType;


        bool                          lockModel(const size_t i)               { ++count_[i]; return count_[i]; }
        bool                          unlockModel(const size_t i)             { OPENGM_ASSERT(count_[i]>0); --count_[i]; return count_[i]; }
        const GM&                     getModel(const size_t i) const          { return gms_[i]; } 
        const GMWITHLOSS&             getModelWithLoss(const size_t i)const   { return gmsWithLoss_[i]; }
        const LossParameterType&      getLossParameters(const size_t i)const  { return lossParams_[i]; }
        const std::vector<LabelType>& getGT(const size_t i) const             { return gts_[i]; }
        Weights&                      getWeights()                            { return weights_; } 
        size_t                        getNumberOfWeights() const              { return weights_.numberOfWeights(); }
        size_t                        getNumberOfModels() const               { return gms_.size(); } 

        template<class INF>
        ValueType                     getTotalLoss(const typename INF::Parameter& para) const;

        template<class INF>
        ValueType                     getTotalLossParallel(const typename INF::Parameter& para) const;

        template<class INF>
        ValueType                     getLoss(const typename INF::Parameter& para, const size_t i) const;
        ValueType                     getLoss(std::vector<LabelType> conf , const size_t i) const;

        Dataset(size_t numInstances);

        Dataset(const Weights & weights = Weights(),const WeightConstraintsType & weightConstraints = WeightConstraintsType(),size_t numInstances=0);

        //void loadAll(std::string path,std::string prefix); 

        friend class DatasetSerialization;
        // friend void loadAll<Dataset<GM,LOSS> > (const std::string datasetpath, const std::string prefix, Dataset<GM,LOSS>& ds);

        //~Dataset(){
        //    std::cout<<"KILL DATASET\n";
        //}
    protected:	
        std::vector<size_t> count_;
        std::vector<bool> isCached_;
        std::vector<GM> gms_; 
        std::vector<GMWITHLOSS> gmsWithLoss_; 
        std::vector<LossParameterType> lossParams_;
        std::vector<std::vector<LabelType> > gts_;
        Weights weights_;
        WeightConstraintsType weightConstraints_;


        void buildModelWithLoss(size_t i);
    };
      

    template<class GM, class LOSS, class LOSS_GM>
    Dataset<GM, LOSS, LOSS_GM>::Dataset(size_t numInstances)
    : count_(std::vector<size_t>(numInstances)),
        isCached_(std::vector<bool>(numInstances)),
        gms_(std::vector<GM>(numInstances)),
        gmsWithLoss_(std::vector<GMWITHLOSS>(numInstances)),
        lossParams_(std::vector<LossParameterType>(numInstances)),
        gts_(std::vector<std::vector<LabelType> >(numInstances)),
        weights_(0),
        weightConstraints_()
    {
    }

    template<class GM, class LOSS, class LOSS_GM>
    Dataset<GM, LOSS, LOSS_GM>::Dataset(
        const Weights & weights, 
        const WeightConstraintsType & weightConstraints,
        size_t numInstances
    ):  count_(std::vector<size_t>(numInstances)),
        isCached_(std::vector<bool>(numInstances)),
        gms_(std::vector<GM>(numInstances)),
        gmsWithLoss_(std::vector<GMWITHLOSS>(numInstances)),
        lossParams_(std::vector<LossParameterType>(numInstances)),
        gts_(std::vector<std::vector<LabelType> >(numInstances)),
        weights_(weights),
        weightConstraints_(weightConstraints)
    {
    }


    template<class GM, class LOSS, class LOSS_GM>
    template<class INF>
    typename GM::ValueType Dataset<GM, LOSS, LOSS_GM>::getTotalLoss(const typename INF::Parameter& para) const {
        ValueType sum=0;
        for(size_t i=0; i<this->getNumberOfModels(); ++i) {
            sum += this->getLoss<INF>(para, i);
        }
        return sum;
    }
    template<class GM, class LOSS, class LOSS_GM>
    template<class INF>
    typename GM::ValueType Dataset<GM, LOSS, LOSS_GM>::getTotalLossParallel(const typename INF::Parameter& para) const {
        double totalLoss = 0;
        #pragma omp parallel for reduction(+:totalLoss)  
        for(size_t i=0; i<this->getNumberOfModels(); ++i) {
            totalLoss = totalLoss + this->getLoss<INF>(para, i);
        }
        return totalLoss;
    }

    template<class GM, class LOSS, class LOSS_GM>
    template<class INF>
    typename GM::ValueType Dataset<GM, LOSS, LOSS_GM>::getLoss(const typename INF::Parameter& para, const size_t i) const {
        LOSS lossFunction(lossParams_[i]);
        const GM& gm = this->getModel(i);
        const std::vector<typename INF::LabelType>& gt =  this->getGT(i);

        std::vector<typename INF::LabelType> conf;
        INF inf(gm,para);
        inf.infer();
        inf.arg(conf);

        return lossFunction.loss(gm, conf.begin(), conf.end(), gt.begin(), gt.end());

    }

    template<class GM, class LOSS, class LOSS_GM>
    typename GM::ValueType Dataset<GM, LOSS, LOSS_GM>::getLoss(std::vector<typename GM::LabelType> conf, const size_t i) const {
        LOSS lossFunction(lossParams_[i]);
        const GM& gm = this->getModel(i);
        const std::vector<LabelType>& gt =  this->getGT(i);
        return lossFunction.loss(gm, conf.begin(), conf.end(), gt.begin(), gt.end());
    }




    template<class GM, class LOSS, class LOSS_GM>
    void Dataset<GM, LOSS, LOSS_GM>::buildModelWithLoss(size_t i){
        OPENGM_ASSERT_OP(i, <, lossParams_.size());
        OPENGM_ASSERT_OP(i, <, gmsWithLoss_.size());
        OPENGM_ASSERT_OP(i, <, gms_.size());
        OPENGM_ASSERT_OP(i, <, gts_.size());
        //std::cout<<"copy gm\n";
        gmsWithLoss_[i] = gms_[i];    
        //std::cout<<"copy done\n";
        LOSS loss(lossParams_[i]);         
        OPENGM_CHECK_OP(gts_[i].size(),==, gmsWithLoss_[i].numberOfVariables(),"");
        loss.addLoss(gmsWithLoss_[i], gts_[i].begin());
    }

    // template<class GM, class LOSS, class LOSS_GM>
    // void Dataset<GM, LOSS, LOSS_GM>::loadAll(std::string datasetpath,std::string prefix){
    //     //Load Header 
    //     std::stringstream hss;
    //     hss << datasetpath << "/"<<prefix<<"info.h5";
    //     hid_t file =  marray::hdf5::openFile(hss.str());
    //     std::vector<size_t> temp(1);
    //     marray::hdf5::loadVec(file, "numberOfWeights", temp);
    //     size_t numWeights = temp[0];
    //     marray::hdf5::loadVec(file, "numberOfModels", temp);
    //     size_t numModel = temp[0];
    //     marray::hdf5::closeFile(file);

    //     gms_.resize(numModel); 
    //     gmsWithLoss_.resize(numModel);
    //     gt_.resize(numModel);
    //     weights_ = Weights(numWeights);
    //     //Load Models and ground truth
    //     for(size_t m=0; m<numModel; ++m){
    //         std::stringstream ss;
    //         ss  << datasetpath <<"/"<<prefix<<"gm_" << m <<".h5"; 
    //         hid_t file =  marray::hdf5::openFile(ss.str()); 
    //         marray::hdf5::loadVec(file, "gt", gt_[m]);
    //         marray::hdf5::closeFile(file);
    //         opengm::hdf5::load(gms_[m],ss.str(),"gm"); 
    //         buildModelWithLoss(m);
    //     }
    // };

}
} // namespace opengm

#endif 
