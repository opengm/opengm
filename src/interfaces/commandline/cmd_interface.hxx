#ifndef CMD_INTERFACE_HXX_
#define CMD_INTERFACE_HXX_

#include <opengm/operations/adder.hxx>

#include "io_cmd.hxx"
#include "../common/io/interface_base.hxx"

namespace opengm {

namespace interface {

/********************
 * class definition *
 ********************/

template <
   class GM,
   class INFERENCETYPES
>
class CMDInterface : public InterfaceBase<GM, INFERENCETYPES, IOCMD> {
public:
   CMDInterface(int argc, char** argv);
   virtual ~CMDInterface();

protected:
   typedef InterfaceBase<GM, INFERENCETYPES, IOCMD> baseClass;
   // Create commandline-IO
   IOCMD commandlineIO_;
   using baseClass::argumentContainer_;
   using baseClass::gm_;

   // Storage for input variables
   std::string selectedModel_;
   std::string selectedOutputfile_;

   // References to created Arguments
   StringArgument<>* model_;
   StringArgument<>* outputfile_;

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
const std::string CMDInterface<GM, INFERENCETYPES>::interfaceName_ = "command line interface";

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <
   class GM,
   class INFERENCETYPES
>
inline CMDInterface<GM, INFERENCETYPES>::CMDInterface(int argc, char** argv)
: baseClass(commandlineIO_), commandlineIO_(argc, argv) {
   model_ = &argumentContainer_.addArgument(StringArgument<>(selectedModel_, "m", "model", "used to specify the desired model. Usage: filename:dataset", true));

   // FIXME
   // outputfile_ = &argumentContainer.addArgument(StringArgument<>(selectedOutputfile, "o", "outputfile", "used to specify the desired outputfile for the computed results", "opengm_result.h5"));
   // will call wrong constructor from StringArgument<>. Must use the following:
   std::string defaultOutput = "opengm_result.h5";
   outputfile_ = &argumentContainer_.addArgument(StringArgument<>(selectedOutputfile_, "o", "outputfile", "used to specify the desired outputfile for the computed results", defaultOutput));
   // end FIXME
}

template <
   class GM,
   class INFERENCETYPES
>
inline CMDInterface<GM, INFERENCETYPES>::~CMDInterface() {

}

template <
   class GM,
   class INFERENCETYPES
>
inline void CMDInterface<GM, INFERENCETYPES>::loadModel() {
   commandlineIO_.read(*model_);
   baseClass::loadModelFromFile(selectedModel_);
}

template <
   class GM,
   class INFERENCETYPES
>
inline void CMDInterface<GM, INFERENCETYPES>::evaluateCallback(const typename GM::ValueType& result) {
   // print value on screen
   commandlineIO_.standardStream() << std::scientific << "result: " << result << std::endl;
}

template <
   class GM,
   class INFERENCETYPES
>
inline const std::string& CMDInterface<GM, INFERENCETYPES>::interfaceName() {
   return interfaceName_;
}

template <
   class GM,
   class INFERENCETYPES
>
inline void CMDInterface<GM, INFERENCETYPES>::callInferenceAlgorithm() {
   typedef callInferenceAlgorithmFunctor<IOCMD, GM, StringArgument<> > currentOperator;
   currentOperator::outputfile_ = outputfile_;

   this->template callInferenceAlgorithmImpl<currentOperator>();
}

} // namespace interface

} // namespace opengm

#endif /* CMD_INTERFACE_HXX_ */
