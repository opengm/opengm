//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

/**
 * @brief This file implements an interface to add a function to an opengm model in MatLab.
 *
 * This routine accepts a matrix containing the data for the function and creates a explicit function from it.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 1).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 2)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the handle for the model. prhs[1] needs to contain the data for the explicit function.
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

  // get shape
  mxArray* shapeIn = mxGetProperty(prhs[1], 0, "shape");
  size_t numDimensions = mxGetNumberOfElements(shapeIn);
  std::vector<GmType::IndexType> shape;
  shape.resize(numDimensions);

  typedef opengm::interface::helper::copyValue<GmType::ValueType, std::vector<GmType::IndexType>::iterator> copyFunctor;
  typedef opengm::interface::helper::forAllValues<copyFunctor> copyValues;
  typedef opengm::interface::helper::getDataFromMXArray<copyValues> coppyShape;
  copyFunctor variableAdder(shape.begin());
  copyValues functor(variableAdder);
  coppyShape()(functor, shapeIn);
  
  // create function corresponding to function type
  GmType::FunctionIdentifier functionID;
  if(mxIsClass(prhs[1], "openGMExplicitFunction")) {
     // create function
     opengm::interface::MatlabModelType::ExplicitFunction function(shape.begin(), shape.end());

     mxArray* functionValues = mxGetProperty(prhs[1], 0, "functionValues");

     typedef opengm::interface::helper::copyValue<GmType::ValueType, opengm::ExplicitFunction<GmType::ValueType>::iterator> copyFunctor;
     typedef opengm::interface::helper::forAllValues<copyFunctor> copyValues;
     typedef opengm::interface::helper::getDataFromMXArray<copyValues> coppyExplicitFunctionValues;
     copyFunctor variableAdder(function.begin());
     copyValues functor(variableAdder);
     coppyExplicitFunctionValues()(functor, functionValues);

     // add function to model
     functionID = gm.addFunction(function);
  } else if(mxIsClass(prhs[1], "openGMPottsFunction")) {
     // get values for PottsFunction
     GmType::ValueType valueEqual;
     GmType::ValueType valueNotEqual;
     typedef opengm::interface::helper::copyValue<GmType::ValueType> copyFunctor;
     typedef opengm::interface::helper::forFirstValue<copyFunctor> copyValue;
     typedef opengm::interface::helper::getDataFromMXArray<copyValue> coppyPottsFunctionValues;
     {
        copyFunctor copy(&valueEqual);
        copyValue functor(copy);
        coppyPottsFunctionValues()(functor, mxGetProperty(prhs[1], 0, "valueEqual"));
     }
     {
        copyFunctor copy(&valueNotEqual);
        copyValue functor(copy);
        coppyPottsFunctionValues()(functor, mxGetProperty(prhs[1], 0, "valueNotEqual"));
     }

     // create function
     opengm::interface::MatlabModelType::PottsFunction function(shape[0], shape[1], valueEqual, valueNotEqual);

     // add function to model
     functionID = gm.addFunction(function);
  } else if(mxIsClass(prhs[1], "openGMPottsNFunction")) {
     // get values for PottsNFunction
     GmType::ValueType valueEqual;
     GmType::ValueType valueNotEqual;
     typedef opengm::interface::helper::copyValue<GmType::ValueType> copyFunctor;
     typedef opengm::interface::helper::forFirstValue<copyFunctor> copyValue;
     typedef opengm::interface::helper::getDataFromMXArray<copyValue> coppyPottsNFunctionValues;
     {
        copyFunctor copy(&valueEqual);
        copyValue functor(copy);
        coppyPottsNFunctionValues()(functor, mxGetProperty(prhs[1], 0, "valueEqual"));
     }
     {
        copyFunctor copy(&valueNotEqual);
        copyValue functor(copy);
        coppyPottsNFunctionValues()(functor, mxGetProperty(prhs[1], 0, "valueNotEqual"));
     }

     // create function
     opengm::interface::MatlabModelType::PottsNFunction function(shape.begin(), shape.end(), valueEqual, valueNotEqual);

     // add function to model
     functionID = gm.addFunction(function);
  } else if(mxIsClass(prhs[1], "openGMPottsGFunction")) {
     // copy values
     mxArray* functionValues = mxGetProperty(prhs[1], 0, "values");
     std::vector<GmType::ValueType> values(mxGetNumberOfElements(functionValues));

     typedef opengm::interface::helper::copyValue<GmType::ValueType, std::vector<GmType::ValueType>::iterator> copyFunctor;
     typedef opengm::interface::helper::forAllValues<copyFunctor> copyValues;
     typedef opengm::interface::helper::getDataFromMXArray<copyValues> coppyPottsGValues;
     copyFunctor valuesAdder(values.begin());
     copyValues functor(valuesAdder);
     coppyPottsGValues()(functor, functionValues);

     // create function
     opengm::interface::MatlabModelType::PottsGFunction function(shape.begin(), shape.end(), values.begin());

     // add function to model
     functionID = gm.addFunction(function);
  } else if(mxIsClass(prhs[1], "openGMTruncatedSquaredDifferenceFunction")) {
     // get values for TruncatedSquaredDifferenceFunction
     GmType::ValueType truncation;
     GmType::ValueType weight;
     typedef opengm::interface::helper::copyValue<GmType::ValueType> copyFunctor;
     typedef opengm::interface::helper::forFirstValue<copyFunctor> copyValue;
     typedef opengm::interface::helper::getDataFromMXArray<copyValue> coppyPottsFunctionValues;
     {
        copyFunctor copy(&truncation);
        copyValue functor(copy);
        coppyPottsFunctionValues()(functor, mxGetProperty(prhs[1], 0, "truncation"));
     }
     {
        copyFunctor copy(&weight);
        copyValue functor(copy);
        coppyPottsFunctionValues()(functor, mxGetProperty(prhs[1], 0, "weight"));
     }

     // create function
     opengm::interface::MatlabModelType::TruncatedSquaredDifferenceFunction function(shape[0], shape[1], truncation, weight);

     // add function to model
     functionID = gm.addFunction(function);
  } else if(mxIsClass(prhs[1], "openGMTruncatedAbsoluteDifferenceFunction")) {
     // get values for PottsFunction
     GmType::ValueType truncation;
     GmType::ValueType weight;
     typedef opengm::interface::helper::copyValue<GmType::ValueType> copyFunctor;
     typedef opengm::interface::helper::forFirstValue<copyFunctor> copyValue;
     typedef opengm::interface::helper::getDataFromMXArray<copyValue> coppyPottsFunctionValues;
     {
        copyFunctor copy(&truncation);
        copyValue functor(copy);
        coppyPottsFunctionValues()(functor, mxGetProperty(prhs[1], 0, "truncation"));
     }
     {
        copyFunctor copy(&weight);
        copyValue functor(copy);
        coppyPottsFunctionValues()(functor, mxGetProperty(prhs[1], 0, "weight"));
     }

     // create function
     opengm::interface::MatlabModelType::TruncatedAbsoluteDifferenceFunction function(shape[0], shape[1], truncation, weight);

     // add function to model
     functionID = gm.addFunction(function);
  } else {
     mexErrMsgTxt("Unsupported function class!");
  }


  // return function id
  const size_t maxSize = sizeof(GmType::FunctionIdentifier::FunctionIndexType) > sizeof(GmType::FunctionIdentifier::FunctionTypeIndexType) ? sizeof(GmType::FunctionIdentifier::FunctionIndexType) : sizeof(GmType::FunctionIdentifier::FunctionTypeIndexType);
  if(maxSize > 8) {
     mexErrMsgTxt("Unsupported size of opengm IndexType!");
  } else if(maxSize > 4) {
     plhs[0] = mxCreateNumericMatrix(1, 2, mxUINT64_CLASS, mxREAL);
     reinterpret_cast<uint64_T*>(mxGetData(plhs[0]))[0] = static_cast<uint64_T>(functionID.getFunctionIndex());
     reinterpret_cast<uint64_T*>(mxGetData(plhs[0]))[1] = static_cast<uint64_T>(functionID.getFunctionType());
  } else if(maxSize > 2) {
     plhs[0] = mxCreateNumericMatrix(1, 2, mxUINT32_CLASS, mxREAL);
     reinterpret_cast<uint32_T*>(mxGetData(plhs[0]))[0] = static_cast<uint32_T>(functionID.getFunctionIndex());
     reinterpret_cast<uint32_T*>(mxGetData(plhs[0]))[1] = static_cast<uint32_T>(functionID.getFunctionType());
  } else if(maxSize > 1) {
     plhs[0] = mxCreateNumericMatrix(1, 2, mxUINT16_CLASS, mxREAL);
     reinterpret_cast<uint16_T*>(mxGetData(plhs[0]))[0] = static_cast<uint16_T>(functionID.getFunctionIndex());
     reinterpret_cast<uint16_T*>(mxGetData(plhs[0]))[1] = static_cast<uint16_T>(functionID.getFunctionType());
  } else {
     plhs[0] = mxCreateNumericMatrix(1, 2, mxUINT8_CLASS, mxREAL);
     reinterpret_cast<uint8_T*>(mxGetData(plhs[0]))[0] = static_cast<uint8_T>(functionID.getFunctionIndex());
     reinterpret_cast<uint8_T*>(mxGetData(plhs[0]))[1] = static_cast<uint8_T>(functionID.getFunctionType());
  }
}
