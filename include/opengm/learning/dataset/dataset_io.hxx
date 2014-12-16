#pragma once
#ifndef OPENGM_DATASET_IO_HXX
#define OPENGM_DATASET_IO_HXX

#include <vector>
#include <cstdlib>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
//#include <H5Cpp.h>

namespace opengm{
   namespace datasets{

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
            std::stringstream ss, ss2;
            ss  << datasetpath <<"/"<<prefix<<"gm_" << m <<".h5"; 
            opengm::hdf5::save(gm,ss.str(),"gm");
            hid_t file = marray::hdf5::openFile(ss.str(), marray::hdf5::READ_WRITE);
            marray::hdf5::save(file,"gt",gt);
            marray::hdf5::closeFile(file);
         }

      };

      template<class DATASET>
      void DatasetSerialization::loadAll(const std::string datasetpath, const std::string prefix, DATASET& dataset) {  
         typedef typename DATASET::GMType   GMType;
         typedef typename GMType::LabelType LabelType; 
         typedef typename GMType::ValueType ValueType;
         
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
         dataset.weights_ = opengm::learning::Weights<ValueType>(numWeights);
         //Load Models and ground truth
         for(size_t m=0; m<numModel; ++m){
            std::stringstream ss;
            ss  << datasetpath <<"/"<<prefix<<"gm_" << m <<".h5"; 
            hid_t file =  marray::hdf5::openFile(ss.str()); 
            marray::hdf5::loadVec(file, "gt", dataset.gts_[m]);
            marray::hdf5::closeFile(file);
            opengm::hdf5::load(dataset.gms_[m],ss.str(),"gm"); 
	    dataset.buildModelWithLoss(m);
         }
      };

   }
}

#endif
