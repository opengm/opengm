//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

/**
 * @brief This file implements an interface to retrieve the information if a given model has at least one potts factor.
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

  // get number of factors
  const GmType::IndexType numFactors = gm.numberOfFactors();

  // check if model has potts factor
  bool hasPotts = false;
  for(GmType::IndexType i = 0; i < numFactors; i++) {
     if(gm[i].isPotts()) {
        hasPotts = true;
        break;
     }
  }

  // return result
  plhs[0] = mxCreateLogicalScalar(hasPotts);
}
