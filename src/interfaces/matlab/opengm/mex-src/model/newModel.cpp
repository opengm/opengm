//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

/**
 * @brief This file implements an interface to create a new opengm model in MatLab.
 *
 * A new model is created and a handle to the model will be passed back to MatLab
 * for further usage.
 *
 * @param[in] nlhs number of output arguments expected from MatLab
 * (needs to be 1).
 * @param[out] plhs pointer to the mxArrays containing the results. If the model
 * can be created, plhs[0] contains the handle to the model.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 0)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab (non required).
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //check if data is in correct format
  if(nrhs != 0) {
     mexErrMsgTxt("Wrong number of input variables specified (non needed)\n");
  }
  if(nlhs != 1) {
     mexErrMsgTxt("Wrong number of output variables specified (one needed)\n");
  }

  // create model
  typedef opengm::interface::MatlabModelType::GmType GmType;
  GmType* gm = new GmType();
  // create handle to model
  plhs[0] = opengm::interface::handle<GmType>::createHandle(gm);
}
