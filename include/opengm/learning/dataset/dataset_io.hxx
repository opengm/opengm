#pragma once
#ifndef OPENGM_DATASET_IO_HXX
#define OPENGM_DATASET_IO_HXX

#include <vector>
#include <cstdlib>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include "H5Cpp.h"

namespace opengm{
   template<class DATASET>
   void save(DATASET& dataset, std::string datasetpath, std::string prefix=""){

      typedef typename DATASET::GMType   GMType;
      typedef typename GMType::LabelType LabelType;
     
      std::vector<size_t> numPara(1,dataset.getNumberOfParameters());
      std::vector<size_t> numModels(1,dataset.getNumberOfModels());
  
      std::stringstream hss;
      hss << datasetpath << "/"<<prefix<<"info.h5";
      hid_t file = marray::hdf5::createFile(hss.str(), marray::hdf5::DEFAULT_HDF5_VERSION);
      marray::hdf5::save(file,"numberOfParameters",numPara);
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
}

#endif
