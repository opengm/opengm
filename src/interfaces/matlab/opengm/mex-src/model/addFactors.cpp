//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

class addFactorsFunctor {
protected:
   typedef opengm::interface::MatlabModelType::GmType GmType;
   typedef GmType::FunctionIdentifier FunctionIdentifier;
   const FunctionIdentifier& functionID_;
   GmType& gm_;
   const size_t numFactors_;
   const size_t numVariablesPerFactor_;
public:
   addFactorsFunctor(const FunctionIdentifier& functionIDIn, GmType& gmIn, const size_t numFactorsIn, const size_t numVariablesPerFactorIn)
   : functionID_(functionIDIn), gm_(gmIn), numFactors_(numFactorsIn), numVariablesPerFactor_(numVariablesPerFactorIn) { }
   template <class DATATYPE>
   void operator() (DATATYPE* dataIn, const size_t numElements) {
      for(size_t i = 0; i < numFactors_; i++) {
         gm_.addFactor(functionID_, dataIn + (i * numVariablesPerFactor_), dataIn + ((i + 1) * numVariablesPerFactor_));
      }
   }
};

/**
 * @brief This file implements an interface to add a Factor to an opengm model in MatLab.
 *
 * This routine accepts a function id and a subset of variables and connects them to a new factor of the model.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 0).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 3)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the handle for the model.
 * prhs[1] needs to contain the function id. prhs[2] needs to contain a matrix where every column represents a subset of variables.
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
  GmType& gm = opengm::interface::handle<GmType>::getObject(prhs[0]);

  // get function id
  GmType::FunctionIdentifier::FunctionIndexType functionIndex;
  GmType::FunctionIdentifier::FunctionTypeIndexType functionType;

  if((mxIsUint8(prhs[1]) || mxIsUint16(prhs[1]) || mxIsUint32(prhs[1]) || mxIsUint64(prhs[1])) && (mxGetM(prhs[1]) == 1) && (mxGetN(prhs[1]) == 2)) {
     if(mxIsUint8(prhs[1])) {
        const uint8_T* data = static_cast<const uint8_T*>(mxGetData(prhs[1]));
        functionIndex = static_cast<GmType::FunctionIdentifier::FunctionIndexType>(data[0]);
        functionType = static_cast<GmType::FunctionIdentifier::FunctionTypeIndexType>(data[1]);
     } else if(mxIsUint16(prhs[1])) {
        const uint16_T* data = static_cast<const uint16_T*>(mxGetData(prhs[1]));
        functionIndex = static_cast<GmType::FunctionIdentifier::FunctionIndexType>(data[0]);
        functionType = static_cast<GmType::FunctionIdentifier::FunctionTypeIndexType>(data[1]);
     } else if(mxIsUint32(prhs[1])) {
        const uint32_T* data = static_cast<const uint32_T*>(mxGetData(prhs[1]));
        functionIndex = static_cast<GmType::FunctionIdentifier::FunctionIndexType>(data[0]);
        functionType = static_cast<GmType::FunctionIdentifier::FunctionTypeIndexType>(data[1]);
     } else if(mxIsUint64(prhs[1])) {
        const uint64_T* data = static_cast<const uint64_T*>(mxGetData(prhs[1]));
        functionIndex = static_cast<GmType::FunctionIdentifier::FunctionIndexType>(data[0]);
        functionType = static_cast<GmType::FunctionIdentifier::FunctionTypeIndexType>(data[1]);
     }
  } else {
     mexErrMsgTxt("Unsupported dataformat for function id!");
  }
  GmType::FunctionIdentifier functionID(functionIndex, functionType);

  // get subset of variables and add factor
  if(mxIsNumeric(prhs[2])) {
     if(mxGetNumberOfDimensions(prhs[2]) != 2) {
        mexErrMsgTxt("variables has to be a matrix where every column represents a subset of variables!");
     }
     size_t numFactors = mxGetN(prhs[2]);
     size_t numVariablesPerFactor = mxGetM(prhs[2]);
     typedef opengm::interface::helper::getDataFromMXArray<addFactorsFunctor> factorAdder;
     addFactorsFunctor functor(functionID, gm, numFactors, numVariablesPerFactor);
     factorAdder()(functor, prhs[2]);
  } else {
     mexErrMsgTxt("Unsupported dataformat for subset of variables!");
  }
}
