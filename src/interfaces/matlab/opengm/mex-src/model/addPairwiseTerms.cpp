//include matlab headers
#include <mex.h>

#include "opengm/functions/explicit_function.hxx"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

#include "../helper/mexHelper.hxx"

class sanityCheckFunctor {
public:
   typedef opengm::interface::MatlabModelType::GmType GmType;

   sanityCheckFunctor(const GmType& gm, const mxArray* functionValues) :
      gm_(gm), functionValues_(functionValues), maxNumLables_(0) {

   }
   template <class DATATYPE>
   void operator() (DATATYPE* dataIn, const size_t numElements) {
      // check variable ids and maximum number of states
      for(size_t i = 0; i < numElements; i++) {
         if(static_cast<GmType::IndexType>(dataIn[i]) >= gm_.numberOfVariables()) {
            mexErrMsgTxt("Variable ID is to high");
         } 
         if(static_cast<GmType::IndexType>(dataIn[i]+numElements) >= gm_.numberOfVariables()) {
            mexErrMsgTxt("Variable ID is to high");
         }
         if(gm_.numberOfLabels(dataIn[i]) > maxNumLables_) {
            maxNumLables_ = gm_.numberOfLabels(dataIn[i]);
         } 
         if(gm_.numberOfLabels(dataIn[i]+numElements) > maxNumLables_) {
            maxNumLables_ = gm_.numberOfLabels(dataIn[i]);
         }
      }
      if(maxNumLables_ > mxGetM(functionValues_)) {
         mexErrMsgTxt("Dimension mismatch: The number of rows of functionValues has to be at least the maximum number of labels of the provided variables!");
      }
   }
protected:
   const GmType& gm_;
   const mxArray* functionValues_;
   size_t maxNumLables_;
};

class addPairwiseFunctor {
public:
   typedef opengm::interface::MatlabModelType::GmType GmType;

   addPairwiseFunctor(GmType& gm, const mxArray* functionValues) :
      gm_(gm), functionValues_(functionValues) {

   }

   template <class DATATYPE>
   void operator() (DATATYPE* dataIn, const size_t numElements) {
      typename GmType::LabelType shape[2];
      size_t maxFunctionElements = mxGetM(functionValues_); 
      size_t numberOfPairs = mxGetN(functionValues_);
      double* functionValuePointer = mxGetPr(functionValues_);
      for(size_t i = 0; i < numberOfPairs; i++) {
         // create function
         shape[0] = gm_.numberOfLabels(static_cast<GmType::LabelType>(*(dataIn + 2*i)));
         shape[1] = gm_.numberOfLabels(static_cast<GmType::LabelType>(*(dataIn + 2*i+1)));
         PairwiseFunctionType pairwiseFunction(shape, shape + 2);
         // copy values
         for(typename GmType::LabelType j = 0; j < shape[0]*shape[1]; j++) {
            pairwiseFunction(j) = static_cast<GmType::ValueType>(*(functionValuePointer + j));
         } 
     
         functionValuePointer += maxFunctionElements;
         // add function
         FunctionIdentifier fid = gm_.addFunction(pairwiseFunction);

         // add factor
         gm_.addFactor(fid, dataIn + 2*i, dataIn + 2*i + 2);
      }
   }

protected:

   typedef GmType::FunctionIdentifier FunctionIdentifier;
   typedef opengm::ExplicitFunction<GmType::ValueType, GmType::IndexType, GmType::LabelType> PairwiseFunctionType;
   GmType& gm_;
   const mxArray* functionValues_;
};

bool sanityCheck(const opengm::interface::MatlabModelType::GmType& gm, const mxArray* variableIDs, const mxArray* functionValues) {
   // check dimensions
   if((mxGetNumberOfDimensions(variableIDs) != 2) || ((mxGetM(variableIDs) != 1) && (mxGetN(variableIDs) != 1))) {
      mexErrMsgTxt("variableIDs has to be a Vector where every element represents a variables ID!");
   }

   if(mxGetN(functionValues) != mxGetNumberOfElements(variableIDs)) {
      mexErrMsgTxt("Dimension mismatch: The number of columns of functionValues has to match the number of provided variable ids!");
   }
   typedef opengm::interface::helper::getDataFromMXArray<sanityCheckFunctor> sanityChecker;
   sanityCheckFunctor functor(gm, functionValues);
   sanityChecker()(functor, variableIDs);
   return true;
}

/**
 * @brief This file implements an interface to add unary factors to an opengm model in MatLab.
 *
 * This routine accepts a vector of variable ids and a matrix, where each column
 * represents the function values for the corresponding variable. The number of
 * columns must match the number of variables and the number of rows must at
 * least match the maximum number of states of the stated variables.
 *
 * @param[in] nlhs number of output arguments expected from MatLab.
 * (needs to be 0).
 * @param[out] plhs pointer to the mxArrays containing the results.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 3)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. prhs[0] needs to contain the handle for the model.
 * prhs[1] needs to contain a vector of variable ids. prhs[2] needs to contain a
 * matrix where each column represents the function values for the corresponding
 * variable. The number of columns must match the number of variables and the
 * number of rows must at least match the maximum number of states of the stated
 * variables.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
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

  // perform sanity check
  OPENGM_ASSERT(sanityCheck(gm, prhs[1], prhs[2]));

  // add unary factors
  typedef opengm::interface::helper::getDataFromMXArray<addPairwiseFunctor> pairwiseAdder;
  addPairwiseFunctor functor(gm, prhs[2]);
  pairwiseAdder()(functor, prhs[1]);
}
