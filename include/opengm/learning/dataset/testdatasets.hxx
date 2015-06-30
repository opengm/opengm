#pragma once
#ifndef OPENGM_TESTDATASETS_HXX
#define OPENGM_TESTDATASETS_HXX

#include <vector>
#include <cstdlib>

#include <opengm/learning/dataset/dataset.hxx>
#include <opengm/learning/dataset/editabledataset.hxx>
#include <opengm/functions/learnable/lpotts.hxx>
#include <opengm/functions/learnable/lsum_of_experts.hxx>

namespace opengm {
   namespace datasets{

      template<class GM, class LOSS>
      class TestDataset0 : public Dataset<GM,LOSS,GM>{ 
      public:
         typedef GM                     GMType;
         typedef GM                     GMWITHLOSS;
         typedef LOSS                   LossType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef opengm::learning::Weights<ValueType> Weights;

         TestDataset0(size_t numModels=5); 
      };

      template<class GM, class LOSS>
      class TestDataset1 : public Dataset<GM,LOSS,GM>{ 
      public:
         typedef GM                     GMType;
         typedef GM                     GMWITHLOSS;
         typedef LOSS                   LossType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef opengm::learning::Weights<ValueType> Weights;

         TestDataset1(size_t numModels=5); 
      };


      template<class GM, class LOSS>
      class TestDataset2 : public Dataset<GM,LOSS,GM>{ 
      public:
         typedef GM                     GMType;
         typedef GM                     GMWITHLOSS;
         typedef LOSS                   LossType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef opengm::learning::Weights<ValueType> Weights;

         TestDataset2(size_t numModels=4); 
      };

      template<class GM, class LOSS>
      class TestDatasetSimple : public Dataset<GM,LOSS,GM>{ 
      public:
         typedef GM                     GMType;
         typedef GM                     GMWITHLOSS;
         typedef LOSS                   LossType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef opengm::learning::Weights<ValueType> Weights;

         TestDatasetSimple(size_t numModels=1); 
      };

      template<class GM, class LOSS>
      class EditableTestDataset : public EditableDataset<GM,LOSS,GM>{ 
      public:
         typedef GM                     GMType;
         typedef GM                     GMWITHLOSS;
         typedef LOSS                   LossType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef opengm::learning::Weights<ValueType> Weights;

         EditableTestDataset(size_t numModels=5); 
      };

//***********************************
//** IMPL TestDataset 0
//***********************************
      template<class GM, class LOSS>
      TestDataset0<GM,LOSS>::TestDataset0(size_t numModels)
      { 
         this->lossParams_.resize(numModels);
         this->isCached_.resize(numModels);
         this->count_.resize(numModels,0);
         this->weights_ = Weights(1);
         LabelType numberOfLabels = 2;
         this->gts_.resize(numModels,std::vector<LabelType>(64,0));
         for(size_t m=0;m<numModels;++m){
            for(size_t i=16; i<48; ++i){
               this->gts_[m][i] = 1;
            }
         }
         this->gms_.resize(numModels);
         this->gmsWithLoss_.resize(numModels);
         for(size_t m=0; m<numModels; ++m){
            std::srand(m);
            for (int j = 0; j < 64; j++)
               this->gms_[m].addVariable(2);
            for(size_t y = 0; y < 64; ++y){ 
               // function
               const size_t shape[] = {numberOfLabels};
               ExplicitFunction<ValueType> f(shape, shape + 1);
               ValueType val = (double)(this->gts_[m][y]) + (double)(std::rand()) / (double) (RAND_MAX) * 1.5 - 0.75 ;
               f(0) = std::fabs(val-0);
               f(1) = std::fabs(val-1);
               typename GM::FunctionIdentifier fid =  this->gms_[m].addFunction(f);

               // factor
               size_t variableIndices[] = {y};
               this->gms_[m].addFactor(fid, variableIndices, variableIndices + 1);         
            }
          
            opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> f(this->weights_,2,std::vector<size_t>(1,0),std::vector<ValueType>(1,1));
            typename GM::FunctionIdentifier fid = this->gms_[m].addFunction(f);      
            for(size_t y = 0; y < 64; ++y){ 
               if(y + 1 < 64) { // (x, y) -- (x, y + 1)
                  size_t variableIndices[] = {y, y+1};
                  //sort(variableIndices, variableIndices + 2);
                  this->gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
               }
            }
            this->buildModelWithLoss(m);
         }      
      };

//***********************************
//** IMPL TestDataset 1
//***********************************
      template<class GM, class LOSS>
      TestDataset1<GM,LOSS>::TestDataset1(size_t numModels)
      { 
         this->lossParams_.resize(numModels);
         this->isCached_.resize(numModels);
         this->count_.resize(numModels,0);
         this->weights_ = Weights(1);
         LabelType numberOfLabels = 2;
         this->gts_.resize(numModels,std::vector<LabelType>(64*64,0));
         for(size_t m=0;m<numModels;++m){
            for(size_t i=32*64; i<64*64; ++i){
               this->gts_[m][i] = 1;
            }
         }
         this->gms_.resize(numModels);
         this->gmsWithLoss_.resize(numModels);
         for(size_t m=0; m<numModels; ++m){
            std::srand(m);
            for (int j = 0; j < 64*64; j++)
               this->gms_[m].addVariable(2);
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  // function
                  const size_t shape[] = {numberOfLabels};
                  ExplicitFunction<ValueType> f(shape, shape + 1);
                  ValueType val = (double)(this->gts_[m][y*64+x]) + (double)(std::rand()) / (double) (RAND_MAX) * 1.5 - 0.75 ;
                  f(0) = std::fabs(val-0);
                  f(1) = std::fabs(val-1);
                  typename GM::FunctionIdentifier fid =  this->gms_[m].addFunction(f);

                  // factor
                  size_t variableIndices[] = {y*64+x};
                  this->gms_[m].addFactor(fid, variableIndices, variableIndices + 1);
               }
            }
          
            opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> f(this->weights_,2,std::vector<size_t>(1,0),std::vector<ValueType>(1,1));
            typename GM::FunctionIdentifier fid = this->gms_[m].addFunction(f);      
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  if(x + 1 < 64) { // (x, y) -- (x + 1, y)
                     size_t variableIndices[] = {y*64+x, y*64+x+1};
                     //sort(variableIndices, variableIndices + 2);
                     this->gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
                  }
                  if(y + 1 < 64) { // (x, y) -- (x, y + 1)
                     size_t variableIndices[] = {y*64+x, (y+1)*64+x};
                     //sort(variableIndices, variableIndices + 2);
                     this->gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
                  }
               }    
            }
            this->buildModelWithLoss(m);
         }      
      };

//***********************************
//** IMPL TestDataset 2
//***********************************
      template<class GM, class LOSS>
      TestDataset2<GM,LOSS>::TestDataset2(size_t numModels)
      { 
         this->lossParams_.resize(numModels);
         this->isCached_.resize(numModels);
         this->count_.resize(numModels,0);
         this->weights_ = Weights(3);
         LabelType numberOfLabels = 2;
         this->gts_.resize(numModels,std::vector<size_t>(64*64,0));
         for(size_t m=0;m<numModels;++m){
            for(size_t i=32*64; i<64*64; ++i){
               this->gts_[m][i] = 1;
            }
         }
         this->gms_.resize(numModels);
         this->gmsWithLoss_.resize(numModels);
         for(size_t m=0; m<numModels; ++m){
            std::srand(m);
            for (int j = 0; j < 64*64; j++)
               this->gms_[m].addVariable(2);
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  // function
                  const size_t numExperts = 2;
                  const std::vector<size_t> shape(1,numberOfLabels);
                  std::vector<marray::Marray<ValueType> > feat(numExperts,marray::Marray<ValueType>(shape.begin(), shape.end()));
                  ValueType val0 = (double)(this->gts_[m][y*64+x]) + (double)(std::rand()) / (double) (RAND_MAX) * 1.0 - 0.5 ;
                  feat[0](0) = std::fabs(val0-0);
                  feat[0](1) = std::fabs(val0-1); 
                  ValueType val1 = (double)(this->gts_[m][y*64+x]) + (double)(std::rand()) / (double) (RAND_MAX) * 2.0 - 1.0 ;
                  feat[1](0) = std::fabs(val1-0);
                  feat[1](1) = std::fabs(val1-1);
                  std::vector<size_t> wID(2);
                  wID[0]=1;  wID[1]=2;
                  opengm::functions::learnable::LSumOfExperts<ValueType,IndexType,LabelType> f(shape,this->weights_, wID, feat);
                  typename GM::FunctionIdentifier fid =  this->gms_[m].addFunction(f);

                  // factor
                  size_t variableIndices[] = {y*64+x};
                  this->gms_[m].addFactor(fid, variableIndices, variableIndices + 1);
               }
            }
          
            opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> f(this->weights_,2,std::vector<size_t>(1,0),std::vector<ValueType>(1,1));
            typename GM::FunctionIdentifier fid = this->gms_[m].addFunction(f);      
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  if(x + 1 < 64) { // (x, y) -- (x + 1, y)
                     size_t variableIndices[] = {y*64+x, y*64+x+1};
                     //sort(variableIndices, variableIndices + 2);
                     this->gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
                  }
                  if(y + 1 < 64) { // (x, y) -- (x, y + 1)
                     size_t variableIndices[] = {y*64+x, (y+1)*64+x};
                     //sort(variableIndices, variableIndices + 2);
                     this->gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
                  }
               }    
            }
            this->buildModelWithLoss(m);
         }
      };

//***********************************
//** Embarrassingly simple dataset
//***********************************
      template<class GM, class LOSS>
      TestDatasetSimple<GM,LOSS>::TestDatasetSimple(size_t numModels)
      { 
         this->lossParams_.resize(numModels);
         this->isCached_.resize(numModels);
         this->count_.resize(numModels,0);
         this->weights_ = Weights(2);
         LabelType numberOfLabels = 2;
         this->gts_.resize(numModels,std::vector<size_t>(1,0));
         for(size_t m=0; m<numModels; ++m){
            this->gts_[m][0] = 0;
         }
         this->gms_.resize(numModels);
         this->gmsWithLoss_.resize(numModels);
         for(size_t m=0; m<numModels; ++m){
            std::srand(m);
            this->gms_[m].addVariable(2);

			// function
            const size_t numExperts = 2;
            const std::vector<size_t> shape(1,numberOfLabels);
            std::vector<marray::Marray<ValueType> > feat(numExperts,marray::Marray<ValueType>(shape.begin(), shape.end()));
            ValueType val0 = 0.5;
            feat[0](0) = val0;
            feat[0](1) = val0-1; 
            ValueType val1 = -0.25;
            feat[1](0) = val1;
            feat[1](1) = val1-1;
            std::vector<size_t> wID(2);
            wID[0]=0;  wID[1]=1;
            opengm::functions::learnable::LSumOfExperts<ValueType,IndexType,LabelType> f(shape,this->weights_, wID, feat);
            typename GM::FunctionIdentifier fid =  this->gms_[m].addFunction(f);

			// factor
            size_t variableIndices[] = {0};
            this->gms_[m].addFactor(fid, variableIndices, variableIndices + 1);

            this->buildModelWithLoss(m);
         }
      };
 
//***********************************
//** IMPL TestDataset 2 (editable)
//***********************************
      template<class GM, class LOSS>
      EditableTestDataset<GM,LOSS>::EditableTestDataset(size_t numModels)
      { 
         this->lossParams_.resize(numModels);
         this->count_.resize(numModels,0);
         this->weights_ = Weights(3);
         LabelType numberOfLabels = 2;
         this->gts_.resize(numModels,std::vector<size_t>(64*64,0));
         for(size_t m=0;m<numModels;++m){
            for(size_t i=32*64; i<64*64; ++i){
               this->gts_[m][i] = 1;
            }
         }
         this->gms_.resize(numModels);
         this->gmsWithLoss_.resize(numModels);
         for(size_t m=0; m<numModels; ++m){
            std::srand(m);
            for (int j = 0; j < 64*64; j++)
               this->gms_[m].addVariable(2);
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  // function
                  const size_t numExperts = 2;
                  const std::vector<size_t> shape(1,numberOfLabels);
                  std::vector<marray::Marray<ValueType> > feat(numExperts,marray::Marray<ValueType>(shape.begin(), shape.end()));
                  ValueType val0 = (double)(this->gts_[m][y*64+x]) + (double)(std::rand()) / (double) (RAND_MAX) * 1.0 - 0.5 ;
                  feat[0](0) = std::fabs(val0-0);
                  feat[0](1) = std::fabs(val0-1); 
                  ValueType val1 = (double)(this->gts_[m][y*64+x]) + (double)(std::rand()) / (double) (RAND_MAX) * 2.0 - 1.0 ;
                  feat[1](0) = std::fabs(val1-0);
                  feat[1](1) = std::fabs(val1-1);
                  std::vector<size_t> wID(2);
                  wID[0]=1;  wID[1]=2;
                  opengm::functions::learnable::LSumOfExperts<ValueType,IndexType,LabelType> f(shape,this->weights_, wID, feat);
                  typename GM::FunctionIdentifier fid =  this->gms_[m].addFunction(f);

                  // factor
                  size_t variableIndices[] = {y*64+x};
                  this->gms_[m].addFactor(fid, variableIndices, variableIndices + 1);
               }
            }
          
            opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> f(this->weights_,2,std::vector<size_t>(1,0),std::vector<ValueType>(1,1));
            typename GM::FunctionIdentifier fid = this->gms_[m].addFunction(f);      
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  if(x + 1 < 64) { // (x, y) -- (x + 1, y)
                     size_t variableIndices[] = {y*64+x, y*64+x+1};
                     //sort(variableIndices, variableIndices + 2);
                     this->gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
                  }
                  if(y + 1 < 64) { // (x, y) -- (x, y + 1)
                     size_t variableIndices[] = {y*64+x, (y+1)*64+x};
                     //sort(variableIndices, variableIndices + 2);
                     this->gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
                  }
               }    
            }
            this->buildModelWithLoss(m);
         }
      };


   }
} // namespace opengm

#endif 
