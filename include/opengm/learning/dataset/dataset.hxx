#pragma once
#ifndef OPENGM_DATASET_HXX
#define OPENGM_DATASET_HXX

#include <vector>
#include <cstdlib>


namespace opengm {
   namespace datasets{

      template<class GM>
      class Dataset{
      public:
         typedef GM                     GMType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType; 
         typedef opengm::Parameters<ValueType,IndexType> ModelParameters;

         GM&                           getModel(const size_t i)  { return gms_[i]; }
         const std::vector<LabelType>& getGT(const size_t i)     { return gt_; }
         ModelParameters&              getModelParameters()      { return modelParameters_; }
         size_t                        getNumberOfParameters()   { return 1; }
         size_t                        getNumberOfModels()       { return gms_.size(); } 
         
         Dataset();
         void load(std::string path,std::string prefix);

      private:
         std::vector<GM> gms_; 
         std::vector<std::vector<LabelType> > gt_; 
         ModelParameters modelParameters_;
      };
      


      template<class GM>
      Dataset<GM>::Dataset()
         :  gms_(std::vector<GM>(0)),
            gt_(std::vector<std::vector<LabelType> >(0)),
            modelParameters_(ModelParameters(0))
      {
      }; 

      template<class GM>
      void Dataset<GM>::load(std::string datasetpath,std::string prefix){

         //Load Header 
         std::stringstream hss;
         hss << datasetpath << "/"<<prefix<<"info.h5";
         hid_t file =  marray::hdf5::openFile(hss.str());
         std::vector<size_t> temp(1);
         marray::hdf5::loadVec(file, "numberOfParameters", temp);
         size_t numPara = temp[0];
         marray::hdf5::loadVec(file, "numberOfModels", temp);
         size_t numModel = temp[0];
         marray::hdf5::closeFile(file);
         
         gms_.resize(numModel);
         gt_.resize(numModel);
         modelParameters_ = ModelParameters(numPara);
         //Load Models and ground truth
         for(size_t m=0; m<numModel; ++m){
            std::stringstream ss;
            ss  << datasetpath <<"/"<<prefix<<"gm_" << m <<".h5"; 
            hid_t file =  marray::hdf5::openFile(ss.str()); 
            marray::hdf5::loadVec(file, "gt", gt_[m]);
            marray::hdf5::closeFile(file);
            opengm::hdf5::load(gms_[m],ss.str(),"gm");
         }

      };
   }
} // namespace opengm

#endif 
