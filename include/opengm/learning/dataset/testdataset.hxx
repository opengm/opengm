#pragma once
#ifndef OPENGM_TESTDATASET_HXX
#define OPENGM_TESTDATASET_HXX

#include <vector>
#include <cstdlib>


namespace opengm {
   namespace datasets{

      template<class GM>
      class TestDataset{
      public:
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType; 
         typedef opengm::Parameters<ValueType,IndexType> ModelParameters;

         const GM&                     getModel(const size_t i)  { return gms_[i]; }
         const std::vector<LabelType>& getGT(const size_t i)     { return gt_; }
         ModelParameters&              getModelParameters()      { return modelParameters_; }
         size_t                        getNumberOfParameters()   { return 1; }
         size_t                        getNumberOfModels()       { return gms_.size(); } 
         
         TestDataset(size_t numModels=3); 

      private:
         std::vector<GM> gms_; 
         std::vector<LabelType> gt_; 
         ModelParameters modelParameters_;
      };
      


      template<class GM>
      TestDataset<GM>::TestDataset(size_t numModels)
         : modelParameters_(ModelParameters(1))
      {
         LabelType numberOfLabels = 2;
         gt_.resize(64*64,0);
         for(size_t i=32*64; i<64*64; ++i){
            gt_[i] = 1;
         }
         gms_.resize(numModels);
         for(size_t m=0; m<numModels; ++m){
            std::srand(m);
            gms_[m].addVariables(64*64,2);
            for(size_t y = 0; y < 64; ++y){ 
               for(size_t x = 0; x < 64; ++x) {
                  // function
                  const size_t shape[] = {numberOfLabels};
                  ExplicitFunction<double> f(shape, shape + 1);
                  ValueType val = (double)(gt_[y*64+x]) + (double)(std::rand()) / (double) (RAND_MAX) * 0.75 ;
                  f(0) = std::fabs(val-0);
                  f(1) = std::fabs(val-1);
                  typename GM::FunctionIdentifier fid =  gms_[m].addFunction(f);

                  // factor
                  size_t variableIndices[] = {y*64+x};
                  gms_[m].addFactor(fid, variableIndices, variableIndices + 1);
               }
            }
          
            opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> f(modelParameters_,2,std::vector<size_t>(1,0),std::vector<double>(1,1));
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
