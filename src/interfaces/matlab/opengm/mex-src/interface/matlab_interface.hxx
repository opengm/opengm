#ifndef MATLAB_INTERFACE_HXX_
#define MATLAB_INTERFACE_HXX_

#include <opengm/operations/adder.hxx>

#include "../io/io_matlab.hxx"
#include <../src/interfaces/common/io/interface_base.hxx>

namespace opengm {

namespace interface {

/********************
 * class definition *
 ********************/

template <
   class GM,
   class INFERENCETYPES
>
class MatlabInterface : public InterfaceBase<GM, INFERENCETYPES, IOMatlab> {
public:
   MatlabInterface(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
   virtual ~MatlabInterface();
/*
   template <template <class> class FUNCTOR>
   void runWithSelectedModelType(const FUNCTOR& functor);*/
protected:
   typedef InterfaceBase<GM, INFERENCETYPES, IOMatlab> baseClass;
   IOMatlab matlabIO_;
   using baseClass::argumentContainer_;
   using baseClass::gm_;

   // MatLab output storage
   int nlhs_;
   mxArray **plhs_;

   // Storage for input variables
   const mxArray* selectedModel_;
   mxArray* selectedOutputfile_;

   mxArray* defaultOutput;

   // References to created Arguments
   mxArrayConstArgument<>* model_;
   mxArrayArgument<>* outputfile_;

   virtual void loadModel();

   virtual void evaluateCallback(const typename GM::ValueType& result);

   virtual const std::string& interfaceName();
   static const std::string interfaceName_;

   virtual void callInferenceAlgorithm();
};

template <
   class GM,
   class INFERENCETYPES
>
const std::string MatlabInterface<GM, INFERENCETYPES>::interfaceName_ = "MatLab interface";

template <
   class GM,
   class INFERENCETYPES
>
MatlabInterface<GM, INFERENCETYPES>::MatlabInterface(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
   : baseClass(matlabIO_), matlabIO_(nrhs, prhs), nlhs_(nlhs), plhs_(plhs), defaultOutput(NULL) {
   model_ = &argumentContainer_.addArgument(mxArrayConstArgument<>(selectedModel_, "m", "model", "used to specify the desired model. Usage: filename:dataset", true));
   if(nlhs_ > 1) {
      mexErrMsgTxt("To many output variables specified\n");
   } else if(nlhs_ == 1) {
      outputfile_ = &argumentContainer_.addArgument(mxArrayArgument<>(plhs_[0], "o", "outputfile", "used to specify the desired output for the computed results"));
   } else {
      defaultOutput = mxCreateString("opengm_result.h5");
      outputfile_ = &argumentContainer_.addArgument(mxArrayArgument<>(selectedOutputfile_, "o", "outputfile", "used to specify the desired output for the computed results", defaultOutput));
   }
}

template <
   class GM,
   class INFERENCETYPES
>
MatlabInterface<GM, INFERENCETYPES>::~MatlabInterface() {
   if(defaultOutput) {
      mxDestroyArray(defaultOutput);
      defaultOutput = NULL;
   }
}

template <
   class GM,
   class INFERENCETYPES
>
inline void MatlabInterface<GM, INFERENCETYPES>::loadModel() {
   // get model either from hdf5 file or from handle
   matlabIO_.read(*model_);

   if(mxIsChar(selectedModel_)) {
      // input is string ==> load model from file
      // get model filename and desired dataset
      std::string selectedModel(mxArrayToString(selectedModel_));
      baseClass::loadModelFromFile(selectedModel);
   } else {
      // input is handle to existing model
      if(std::string(mxGetClassName(selectedModel_)) != std::string("openGMModel")) {
         mexErrMsgTxt("Unsupported OpenGM model class passed by MatLab");
      }
      // get handle from selectedModel_
      mxArray* modelHandle = mxGetProperty(selectedModel_, 0, "modelHandle");
      if(modelHandle == NULL) {
         mexErrMsgTxt("Could not get object handle");
      }
      // get object from handle
      gm_ = &opengm::interface::handle<GM>::getObject(modelHandle);
   }
}

template <
   class GM,
   class INFERENCETYPES
>
inline void MatlabInterface<GM, INFERENCETYPES>::evaluateCallback(const typename GM::ValueType& result) {
   if(nlhs_ == 1) {
      // return value to MatLab
      const size_t size = sizeof(typename GM::ValueType);
      if(size > 8) {
         mexErrMsgTxt("Unsupported size of opengm ValueType!");
      } else if(size > 4) {
         plhs_[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
         reinterpret_cast<double*>(mxGetData(plhs_[0]))[0] = static_cast<double>(result);
      } else {
         plhs_[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
         reinterpret_cast<float*>(mxGetData(plhs_[0]))[0] = static_cast<float>(result);
      }
   } else {
      // print value on screen
      matlabIO_.standardStream() << std::scientific << "result: " << result << std::endl;
   }
}

template <
   class GM,
   class INFERENCETYPES
>
inline const std::string& MatlabInterface<GM, INFERENCETYPES>::interfaceName() {
   return interfaceName_;
}

template <
   class GM,
   class INFERENCETYPES
>
inline void MatlabInterface<GM, INFERENCETYPES>::callInferenceAlgorithm() {
   typedef callInferenceAlgorithmFunctor<IOMatlab, GM, mxArrayArgument<> > currentOperator;
   currentOperator::outputfile_ = outputfile_;

   this->template callInferenceAlgorithmImpl<currentOperator>();
}

} // namespace interface

} // namespace opengm

#endif /* MATLAB_INTERFACE_HXX_ */
