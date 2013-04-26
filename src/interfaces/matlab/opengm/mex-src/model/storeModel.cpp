//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

/**
 * @brief This file implements an interface to store an opengm model in MatLab.
 *
 * This routine accepts a string containing the filename where a given opengm
 * model shall be stored in hd5 format.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 0).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 3)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the desiredfile location of the opengm
 * model. prhs[1] needs to contain the desired dataset. prhs[2] needs
 * to contain the handle for the model.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //check if data is in correct format
  if(nrhs != 3) {
     mexErrMsgTxt("Wrong number of input variables specified (three needed)\n");
  }
  if(nlhs != 0) {
     mexErrMsgTxt("Wrong number of output variables specified (zero needed)\n");
  }

  // get model out of handle
  typedef opengm::interface::MatlabModelType::GmType GmType;
  GmType& gm = opengm::interface::handle<GmType>::getObject(prhs[2]);

  // get file name and corresponding dataset
  std::string modelFilename = mxArrayToString(prhs[0]);
  if(modelFilename.data()==NULL) {
     mexErrMsgTxt("load: could not convert input to string.");
  }
  std::string dataset = mxArrayToString(prhs[1]);
  if(dataset.data()==NULL) {
     mexErrMsgTxt("load: could not convert input to string.");
  }

  // store model
  opengm::hdf5::save(gm, modelFilename, dataset);
}
