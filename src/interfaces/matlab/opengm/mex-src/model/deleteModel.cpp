//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

/**
 * @brief This file implements an interface to delete an opengm model in MatLab to free the used memory.
 *
 * This routine accepts a handle to an opengm model and deletes it.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 0).
 * @param[out] plhs pointer to the mxArrays containing the results (non returned)
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
  if(nlhs != 0) {
     mexErrMsgTxt("Wrong number of output variables specified (zero needed)\n");
  }

  // get handle out of mxArray and delte handle
  typedef opengm::interface::MatlabModelType::GmType GmType;
  opengm::interface::handle<GmType>::deleteObject(prhs[0]);
}
