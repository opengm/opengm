//include matlab headers
#include "mex.h"

#include <opengm/utilities/metaprogramming.hxx>

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

/**
 * @brief This file implements an interface to retrieve a compact representation of a given Potts model.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 5).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * plhs[0] will contain the constant term.
 * plhs[1] will contain the unary terms. 
 * plhs[2] will contain the coupling strength of pairwise terms. 
 * plhs[3] will contain the indicator if pairwise terms exists. 
 * plhs[4] will contain the valid model flag.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 1)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab.
 * prhs[0] needs to contain the handle for the model.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   //check if data is in correct format
   if(nrhs != 1) {
      mexErrMsgTxt("Wrong number of input variables specified (one needed)\n");
   }
   if(nlhs != 5) {
      mexErrMsgTxt("Wrong number of output variables specified (four needed)\n");
   }

   // get model out of handle
   typedef opengm::interface::MatlabModelType::GmType GmType;
   GmType& gm = opengm::interface::handle<GmType>::getObject(prhs[0]);

   // get dimension
   mwSize nVar = static_cast<mwSize>(gm.numberOfVariables()); 
   mwSize nLab = static_cast<mwSize>(gm.numberOfLabels(0));
   mwSize* dimsU = new mwSize[2];
   mwSize* dimsP = new mwSize[2];
   mwSize one = 1;
   mwSize two = 2;
   dimsU[0]=nLab; dimsU[1] = nVar;
   dimsP[0]=nVar; dimsP[1] = nVar;
   
   plhs[0] = mxCreateNumericArray(one , &one , mxDOUBLE_CLASS, mxREAL);
   plhs[1] = mxCreateNumericArray(two , dimsU, mxDOUBLE_CLASS, mxREAL);
   plhs[2] = mxCreateNumericArray(two , dimsP, mxDOUBLE_CLASS, mxREAL);
   plhs[3] = mxCreateNumericArray(two , dimsP, mxDOUBLE_CLASS, mxREAL);
   plhs[4] = mxCreateNumericArray(one , &one , mxDOUBLE_CLASS, mxREAL);
   
   double* constTerm = reinterpret_cast<double*>(mxGetData(plhs[0]));
   double* unaryTerm = reinterpret_cast<double*>(mxGetData(plhs[1]));
   double* pairTerm  = reinterpret_cast<double*>(mxGetData(plhs[2]));
   double* pairAdj   = reinterpret_cast<double*>(mxGetData(plhs[3]));
   double* flag      = reinterpret_cast<double*>(mxGetData(plhs[4]));
  
   // cleanup
   delete[] dimsU;  
   delete[] dimsP;
   
   for(size_t i=0; i<nVar; ++i){
      if(gm.numberOfLabels(i)!=nLab){
         *flag = 0;
         return;
      }   
   } 
   constTerm[0]=0;
   flag[0]=1;
   for(size_t i=0; i<nVar; ++i){ 
      for(size_t l=0; l<nLab; ++l){
         unaryTerm[l+i*nLab] = 0;
      }
      for(size_t j=0; j<nVar; ++j){
         pairTerm[j+i*nLab] = 0;
      }
   }


   // copy values
   for(size_t f=0; f<gm.numberOfFactors(); ++f){
      if(gm[f].numberOfVariables()==0){ 
         const size_t l0[] = {0};
         constTerm[0] += gm[f](l0);
      }
      else if(gm[f].numberOfVariables()==1){
         size_t var = gm[f].variableIndex(0);
         for(size_t l=0; l<nLab; ++l){
            unaryTerm[l+var*nLab] += gm[f](&l);
         }
      }
      else if(gm[f].numberOfVariables()==2){
         if(nLab==2){
            size_t var0 = gm[f].variableIndex(0);
            size_t var1 = gm[f].variableIndex(1);
            const size_t l00[] = {0,0};
            const size_t l01[] = {0,1}; 
            const size_t l10[] = {1,0};
            const size_t l11[] = {1,1};
            const double v00   = gm[f](l00); 
            const double v01   = gm[f](l01);
            const double v10   = gm[f](l10);
            const double v11   = gm[f](l11);
            constTerm[0] += v00;
            const double D = 0.5*(v11-v00);
            const double M = 0.5*(v10-v01);
            unaryTerm[1+var0*nLab] += D+M;
            unaryTerm[1+var1*nLab] += D-M;
            pairTerm[var0+var1*nVar] += v10-v00-D-M;
            pairAdj[var0+var1*nVar] = 1;
         }
         else if(gm[f].isPotts()){
            size_t var0 = gm[f].variableIndex(0);
            size_t var1 = gm[f].variableIndex(1);
            const size_t l00[] = {0,0};
            const size_t l01[] = {0,1};
            constTerm[0]             += gm[f](l00);
            pairTerm[var0+nLab*var1] += gm[f](l01) - gm[f](l00);
            pairAdj[var0+nLab*var1] = 1;
         }
         else{
            flag[0]=0;
            return;
         }
      }
      else{
         flag[0]=0;
         return;
      }
   }
   return;
}
