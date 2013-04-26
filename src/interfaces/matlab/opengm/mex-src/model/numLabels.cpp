//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

/**
 * @brief This file implements an interface to retrieve the number of labels for a variable from a given model.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 1).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 2)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the handle for the model. prhs[1] needs to contain the index of the variable.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //check if data is in correct format
  if(nrhs != 2) {
     mexErrMsgTxt("Wrong number of input variables specified (two needed)\n");
  }
  if(nlhs != 1) {
     mexErrMsgTxt("Wrong number of output variables specified (one needed)\n");
  }

  if(!mxIsNumeric(prhs[1]) || (mxGetNumberOfElements(prhs[1]) != 1)) {
     mexErrMsgTxt("Variable index has to be a numeric scalar\n");
  }

  // get model out of handle
  typedef opengm::interface::MatlabModelType::GmType GmType;
  GmType& gm = opengm::interface::handle<GmType>::getObject(prhs[0]);

  // get variable index
  GmType::IndexType variable;

  typedef opengm::interface::helper::copyValue<GmType::IndexType> copyType;
  typedef opengm::interface::helper::forFirstValue<copyType> copyFirstElement;
  typedef opengm::interface::helper::getDataFromMXArray<copyFirstElement> getFirstElement;
  getFirstElement getter;
  copyType duplicator(&variable);
  copyFirstElement functor(duplicator);
  getter(functor, prhs[1]);


  // get number of labels
  const GmType::IndexType numLabels = gm.numberOfLabels(variable);

  // return number of labels
  const size_t size = sizeof(GmType::IndexType);
  if(size > 8) {
     mexErrMsgTxt("Unsupported size of opengm IndexType!");
  } else if(size > 4) {
     plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
     reinterpret_cast<uint64_T*>(mxGetData(plhs[0]))[0] = static_cast<uint64_T>(numLabels);
  } else if(size > 2) {
     plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL);
     reinterpret_cast<uint32_T*>(mxGetData(plhs[0]))[0] = static_cast<uint32_T>(numLabels);
  } else if(size > 1) {
     plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT16_CLASS, mxREAL);
     reinterpret_cast<uint16_T*>(mxGetData(plhs[0]))[0] = static_cast<uint16_T>(numLabels);
  } else {
     plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT8_CLASS, mxREAL);
     reinterpret_cast<uint8_T*>(mxGetData(plhs[0]))[0] = static_cast<uint8_T>(numLabels);
  }
}
