#pragma once
#ifndef OPENGM_DATASET_IO_HXX
#define OPENGM_DATASET_IO_HXX

#include <vector>
#include <cstdlib>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/opengm.hxx>
#include "opengm/learning/loss/generalized-hammingloss.hxx"
#include "opengm/learning/loss/hammingloss.hxx"
#include "opengm/learning/loss/noloss.hxx"
//#include <H5Cpp.h>

namespace opengm{
   namespace datasets{

      template <class W>
      struct WeightSetter {
         public:
           WeightSetter(W& w) : weights_(w) {}

           template<class F>
           void operator()(F& f) const { f.setWeights(weights_); }

         private:
           W& weights_;
      };

      class DatasetSerialization{
      public:
         template<class DATASET>
         static void save(const DATASET& dataset, const std::string datasetpath, const std::string prefix=""); 
         template<class DATASET>
         static void loadAll(const std::string datasetpath, const std::string prefix,  DATASET& dataset);  
      };

      template<class DATASET>
      void DatasetSerialization::save(const DATASET& dataset, const std::string datasetpath, const std::string prefix) {
         typedef typename DATASET::GMType   GMType;
         typedef typename DATASET::LossParameterType LossParameterType;
         typedef typename GMType::LabelType LabelType; 
         typedef typename GMType::ValueType ValueType;

         std::vector<size_t> numWeights(1,dataset.getNumberOfWeights());
         std::vector<size_t> numModels(1,dataset.getNumberOfModels());
  
         std::stringstream hss;
         hss << datasetpath << "/"<<prefix<<"info.h5";
         hid_t file = marray::hdf5::createFile(hss.str(), marray::hdf5::DEFAULT_HDF5_VERSION);
         marray::hdf5::save(file,"numberOfWeights",numWeights);
         marray::hdf5::save(file,"numberOfModels",numModels);
         marray::hdf5::closeFile(file); 

         for(size_t m=0; m<dataset.getNumberOfModels(); ++m){
            const GMType&                 gm = dataset.getModel(m); 
            const std::vector<LabelType>& gt = dataset.getGT(m);
            const LossParameterType&      lossParam = dataset.getLossParameters(m);
            std::stringstream ss;
            ss  << datasetpath <<"/"<<prefix<<"gm_" << m <<".h5"; 
            opengm::hdf5::save(gm, ss.str(), "gm");
            hid_t file = marray::hdf5::openFile(ss.str(), marray::hdf5::READ_WRITE);

            //marray::Vector<LabelType> mgt(gt.size());
            //std::copy(gt.begin(), gt.end(), mgt.begin());

            marray::hdf5::save(file,"gt",gt);
            hid_t lossGrp = marray::hdf5::createGroup(file,"loss");

            lossParam.save(lossGrp);
            marray::hdf5::closeGroup(lossGrp);
            marray::hdf5::closeFile(file);
         }

      }

      template<class DATASET>
      void DatasetSerialization::loadAll(const std::string datasetpath, const std::string prefix, DATASET& dataset) {  
         typedef typename DATASET::GMType   GMType;
         typedef typename GMType::LabelType LabelType; 
         typedef typename GMType::ValueType ValueType;
         typedef typename DATASET::LossParameterType LossParameterType;
         
         //Load Header 
         std::stringstream hss;
         hss << datasetpath << "/"<<prefix<<"info.h5";
         hid_t file =  marray::hdf5::openFile(hss.str());
         std::vector<size_t> temp(1);
         marray::hdf5::loadVec(file, "numberOfWeights", temp);
         size_t numWeights = temp[0];
         marray::hdf5::loadVec(file, "numberOfModels", temp);
         size_t numModel = temp[0];
         marray::hdf5::closeFile(file);
         
         dataset.gms_.resize(numModel); 
         dataset.gmsWithLoss_.resize(numModel);
         dataset.gts_.resize(numModel);
         dataset.lossParams_.resize(numModel);
         dataset.count_.resize(numModel);
         dataset.isCached_.resize(numModel);
         dataset.weights_ = opengm::learning::Weights<ValueType>(numWeights);
         OPENGM_ASSERT_OP(dataset.lossParams_.size(), ==, numModel);
         WeightSetter<opengm::learning::Weights<ValueType> > wSetter(dataset.weights_);

         //Load Models and ground truth
         for(size_t m=0; m<numModel; ++m){
            std::stringstream ss;
            ss  << datasetpath <<"/"<<prefix<<"gm_" << m <<".h5"; 
            hid_t file =  marray::hdf5::openFile(ss.str()); 
            marray::hdf5::loadVec(file, "gt", dataset.gts_[m]);
            opengm::hdf5::load(dataset.gms_[m],ss.str(),"gm");

            for(size_t fi = 0; fi < dataset.gms_[m].numberOfFactors(); ++fi) {
                dataset.gms_[m][fi].callFunctor(wSetter);
            }

            LossParameterType lossParam;
            hid_t lossGrp = marray::hdf5::openGroup(file, "loss");
            lossParam.load(lossGrp);
            std::vector<std::size_t> lossId;
            marray::hdf5::loadVec(lossGrp, "lossId", lossId);
            OPENGM_CHECK_OP(lossId.size(), ==, 1, "");
            OPENGM_CHECK_OP(lossParam.getLossId(), ==, lossId[0],"the dataset needs to be initialized with the same loss type as saved");
            dataset.lossParams_[m] = lossParam;

            OPENGM_CHECK_OP(dataset.gts_[m].size(), == ,dataset.gms_[m].numberOfVariables(), "");
            marray::hdf5::closeFile(file);            
            dataset.buildModelWithLoss(m);
         }
      }

   }
}

#endif
