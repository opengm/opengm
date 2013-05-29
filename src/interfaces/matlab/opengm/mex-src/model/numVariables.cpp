//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

/**
 * @brief This file implements an interface to retrieve the number of variables from a given model.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 1).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 1)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the handle for the model.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //check if data is in correct format
  if(nrhs != 1) {
     mexErrMsgTxt("Wrong number of input variables specified (one needed)\n");
  }
  if(nlhs != 1) {
     mexErrMsgTxt("Wrong number of output variables specified (one needed)\n");
  }

  // get model out of handle
  typedef opengm::interface::MatlabModelType::GmType GmType;
  GmType& gm = opengm::interface::handle<GmType>::getObject(prhs[0]);

  // get number of variables
  const GmType::IndexType numVariables = gm.numberOfVariables();

  // return number of variables
  const size_t size = sizeof(GmType::IndexType);
  if(size > 8) {
     mexErrMsgTxt("Unsupported size of opengm IndexType!");
  } else if(size > 4) {
     plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
     reinterpret_cast<uint64_T*>(mxGetData(plhs[0]))[0] = static_cast<uint64_T>(numVariables);
  } else if(size > 2) {
     plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL);
     reinterpret_cast<uint32_T*>(mxGetData(plhs[0]))[0] = static_cast<uint32_T>(numVariables);
  } else if(size > 1) {
     plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT16_CLASS, mxREAL);
     reinterpret_cast<uint16_T*>(mxGetData(plhs[0]))[0] = static_cast<uint16_T>(numVariables);
  } else {
     plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT8_CLASS, mxREAL);
     reinterpret_cast<uint8_T*>(mxGetData(plhs[0]))[0] = static_cast<uint8_T>(numVariables);
  }
}
