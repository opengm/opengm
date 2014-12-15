#pragma once
#ifndef OPENGM_TESTDATASET_HXX
#define OPENGM_TESTDATASET_HXX

#include <vector>
#include <cstdlib>

#include <opengm/functions/learnable/lpotts.hxx>

namespace opengm {
   namespace datasets{

      template<class GM>
      class TestDataset{
      public:
         typedef GM                     GMType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType; 
         typedef opengm::learning::Weights<ValueType> Weights;

         GM&                           getModel(const size_t i)  { return gms_[i]; }
         const std::vector<LabelType>& getGT(const size_t i)     { return gt_; }
         Weights&                      getWeights()              { return weights_; }
         size_t                        getNumberOfWeights()      { return 1; }
         size_t                        getNumberOfModels()       { return gms_.size(); } 
         
         TestDataset(size_t numModels=10); 

      private:
         std::vector<GM> gms_; 
         std::vector<LabelType> gt_; 
         Weights weights_;
      };
      


      template<class GM>
      TestDataset<GM>::TestDataset(size_t numModels)
         : weights_(Weights(1))
      {
         LabelType numberOfLabels = 2;
         gt_.resize(64*64,0);
         for(size_t i=32*64; i<64*64; ++i){
            gt_[i] = 1;
         }
         gms_.resize(numModels);
         for(size_t m=0; m<numModels; ++m){
            std::srand(m);
			for (int j = 0; j < 64*64; j++)
				gms_[m].addVariable(2);
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  // function
                  const size_t shape[] = {numberOfLabels};
                  ExplicitFunction<ValueType> f(shape, shape + 1);
                  ValueType val = (double)(gt_[y*64+x]) + (double)(std::rand()) / (double) (RAND_MAX) * 1.5 - 0.75 ;
                  f(0) = std::fabs(val-0);
                  f(1) = std::fabs(val-1);
                  typename GM::FunctionIdentifier fid =  gms_[m].addFunction(f);

                  // factor
                  size_t variableIndices[] = {y*64+x};
                  gms_[m].addFactor(fid, variableIndices, variableIndices + 1);
               }
            }
          
            opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> f(weights_,2,std::vector<size_t>(1,0),std::vector<ValueType>(1,1));
            typename GM::FunctionIdentifier fid = gms_[m].addFunction(f);      
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  if(x + 1 < 64) { // (x, y) -- (x + 1, y)
                     size_t variableIndices[] = {y*64+x, y*64+x+1};
                     //sort(variableIndices, variableIndices + 2);
                     gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
                  }
                  if(y + 1 < 64) { // (x, y) -- (x, y + 1)
                     size_t variableIndices[] = {y*64+x, (y+1)*64+x};
                     //sort(variableIndices, variableIndices + 2);
                     gms_[m].addFactor(fid, variableIndices, variableIndices + 2);
                  }
               }    
            }
         }
         
      };

   }
} // namespace opengm

#endif 
