//include matlab headers
#include "mex.h"

#include <opengm/utilities/metaprogramming.hxx>

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

/**
 * @brief This file implements an interface to retrieve the factor table of a factor from a given model.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 2).
 * @param[out] plhs pointer to the mxArrays containing the results. plhs[0] will contain the factor table.
 * plhs[1] will contain the corresponding variables.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 2)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the handle for the model. prhs[1] needs to contain the index of the factor.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   //check if data is in correct format
   if(nrhs != 2) {
      mexErrMsgTxt("Wrong number of input variables specified (two needed)\n");
   }
   if(nlhs != 2) {
      mexErrMsgTxt("Wrong number of output variables specified (two needed)\n");
   }

   // get model out of handle
   typedef opengm::interface::MatlabModelType::GmType GmType;
   GmType& gm = opengm::interface::handle<GmType>::getObject(prhs[0]);

   // get factor index
   GmType::IndexType factorIndex;

   typedef opengm::interface::helper::copyValue<GmType::IndexType> copyType;
   typedef opengm::interface::helper::forFirstValue<copyType> copyFirstElement;
   typedef opengm::interface::helper::getDataFromMXArray<copyFirstElement> getFirstElement;
   getFirstElement getter;
   copyType duplicator(&factorIndex);
   copyFirstElement functor(duplicator);
   getter(functor, prhs[1]);

   // get dimension
   mwSize nDim = static_cast<mwSize>(gm[factorIndex].numberOfVariables());
   mwSize* dims = new mwSize[nDim];
   for(GmType::IndexType i = 0; i < static_cast<GmType::IndexType>(nDim); i++) {
      dims[i] = static_cast<mwSize>(gm[factorIndex].shape(i));
   }

   // create table dependent on value_type
   if(opengm::meta::Compare<GmType::ValueType, float>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxSINGLE_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, double>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxDOUBLE_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, int8_T>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxINT8_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, int16_T>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxINT16_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, int32_T>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxINT32_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, int64_T>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxINT64_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, uint8_T>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxUINT8_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, uint16_T>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxUINT16_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, uint32_T>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxUINT32_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::ValueType, uint64_T>::value) {
      plhs[0] = mxCreateNumericArray(nDim, dims, mxUINT64_CLASS, mxREAL);
   } else {
      mexErrMsgTxt("can not create factor table, unsupported value type");
   }

   // cleanup
   delete[] dims;

   // copy values
   gm[factorIndex].copyValues(reinterpret_cast<GmType::ValueType*>(mxGetData(plhs[0])));

   // get corresponding variables

   // get num variables
   mwSize numVariables = static_cast<mwSize>(gm[factorIndex].numberOfVariables());

   // create variable list dependent on index_type
   if(opengm::meta::Compare<GmType::IndexType, float>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxSINGLE_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, double>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxDOUBLE_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, int8_T>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxINT8_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, int16_T>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxINT16_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, int32_T>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxINT32_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, int64_T>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxINT64_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, uint8_T>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxUINT8_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, uint16_T>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxUINT16_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, uint32_T>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxUINT32_CLASS, mxREAL);
   } else if(opengm::meta::Compare<GmType::IndexType, uint64_T>::value) {
      plhs[1] = mxCreateNumericMatrix(numVariables, 1, mxUINT64_CLASS, mxREAL);
   } else {
      mexErrMsgTxt("can not create variable list, unsupported index type");
   }

   // copy values
   gm[factorIndex].variableIndices(reinterpret_cast<GmType::IndexType*>(mxGetData(plhs[1])));
}
