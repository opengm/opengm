//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

class addVariablesFunctor {
protected:
   typedef opengm::interface::MatlabModelType::GmType GmType;
   GmType& gm_;
public:
   addVariablesFunctor(GmType& gmIn) : gm_(gmIn) {}
   template <class DATATYPE>
   void operator() (const DATATYPE& dataIn) {
      gm_.addVariable(static_cast<GmType::LabelType>(dataIn));
   }
};

/**
 * @brief This file implements an interface to add variables to an opengm model in MatLab.
 *
 * This routine accepts a vector containing the number of states for each variable.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 0).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 2)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the handle for the model. prhs[1] needs to contain the vector for the number of states.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //check if data is in correct format
  if(nrhs != 2) {
     mexErrMsgTxt("Wrong number of input variables specified (two needed)\n");
  }
  if(nlhs != 0) {
     mexErrMsgTxt("Wrong number of output variables specified (zero needed)\n");
  }

  // get model out of handle
  typedef opengm::interface::MatlabModelType::GmType GmType;
  GmType& gm = opengm::interface::handle<GmType>::getObject(prhs[0]);

  // get numbers of states
  if(mxIsNumeric(prhs[1])) {
     if(mxGetNumberOfDimensions(prhs[1]) != 2 || ((mxGetM(prhs[1]) != 1) && (mxGetN(prhs[1]) != 1))) {
        mexErrMsgTxt("numbers of states has to be a vector!");
     }
     //GmType::IndexType numVariables = static_cast<GmType::IndexType>(mxGetNumberOfElements(prhs[1]));

     typedef opengm::interface::helper::forAllValues<addVariablesFunctor> forAllValues;
     typedef opengm::interface::helper::getDataFromMXArray<forAllValues> addVariables;
     addVariablesFunctor variableAdder(gm);
     forAllValues functor(variableAdder);
     addVariables()(functor, prhs[1]);

  } else {
     mexErrMsgTxt("Unsupported dataformat for numbers of states!");
  }
}
