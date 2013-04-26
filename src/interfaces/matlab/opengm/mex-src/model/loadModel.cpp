//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

/**
 * @brief This file implements an interface to load an opengm model in MatLab.
 *
 * This routine accepts a string containing the filename of an opengm model
 * stored in hd5 format. The model is loaded and a handle to the model will be
 * passed back to MatLab for further usage.
 *
 * @param[in] nlhs number of output arguments expected from MatLab
 * (needs to be 1).
 * @param[out] plhs pointer to the mxArrays containing the results. If the model
 * can be loaded, plhs[0] contains the handle to the model.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 2)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the file location of the opengm model stored
 * in hdf5 format. prhs[1] needs to contain the desired dataset.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //check if data is in correct format
  if(nrhs != 2) {
     mexErrMsgTxt("Wrong number of input variables specified (one needed)\n");
  }
  if(nlhs != 1) {
     mexErrMsgTxt("Wrong number of output variables specified (one needed)\n");
  }

  // get file name and corresponding dataset
  std::string modelFilename = mxArrayToString(prhs[0]);
  if(modelFilename.data()==NULL) {
     mexErrMsgTxt("load: could not convert input to string.");
  }
  std::string dataset = mxArrayToString(prhs[1]);
  if(dataset.data()==NULL) {
     mexErrMsgTxt("load: could not convert input to string.");
  }

  // load model
  typedef opengm::interface::MatlabModelType::GmType GmType;
  GmType* gm = new GmType();
  std::cout << "Loading model..." << std::endl;
  opengm::hdf5::load(*gm, modelFilename, dataset);
  std::cout << "Loading model done" << std::endl;
  // create handle to model
  plhs[0] = opengm::interface::handle<GmType>::createHandle(gm);
}
