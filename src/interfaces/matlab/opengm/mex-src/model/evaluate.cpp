//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

class addFactorFunctor {
protected:
   typedef opengm::interface::MatlabModelType::GmType GmType;
   typedef GmType::FunctionIdentifier FunctionIdentifier;
   const FunctionIdentifier& functionID_;
   GmType& gm_;
public:
   addFactorFunctor(const FunctionIdentifier& functionIDIn, GmType& gmIn) : functionID_(functionIDIn), gm_(gmIn) {}
   template <class DATATYPE>
   void operator() (DATATYPE* dataIn, const size_t numElements) {
      gm_.addFactor(functionID_, dataIn, dataIn + numElements);
   }
};

/**
 * @brief This file implements an interface to add a Factor to an opengm model in MatLab.
 *
 * This routine accepts a function id and a subset of variables and connects them to a new factor of the model.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 1).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * plhs[0] objective value of the labeling
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 2)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. 
 * prhs[0] needs to contain the handle for the model.
 * prhs[1] labeling
 *  //// prhs[2] needs to contain the vector representing the subset of variables.
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

  // get model out of handle
  typedef opengm::interface::MatlabModelType::GmType GmType;
  GmType& gm = opengm::interface::handle<GmType>::getObject(prhs[0]);

  // get labeling
  if(mxIsNumeric(prhs[1])) {
     if( ((mxGetM(prhs[1]) * mxGetN(prhs[1]) != gm.numberOfVariables()))) {
        mexErrMsgTxt("labeling has wrong size!");
     }
     //std::vector<GmType::ValueType> labeling;
     double* m_labeling = reinterpret_cast<double*>(mxGetData(prhs[1]) );
     //for (size_t i=0; i<gm.numberOfVariables(); ++i)
        //   labeling[i] =  m_labeling[i];

     GmType::ValueType value = gm.evaluate(m_labeling); 
     plhs[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
     reinterpret_cast<double*>(mxGetData(plhs[0]))[0] = static_cast<double>(value);
  } else {
     mexErrMsgTxt("Unsupported dataformat for subset of variables!");
  }
}
