#ifndef INFERENCEPARSERBASE_HXX_
#define INFERENCEPARSERBASE_HXX_

#include <string>
#include <vector>
#include <iomanip>
#include <boost/variant.hpp>

#include "Argument.hxx"
#include "helper/helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class InferenceParserBase {
protected:
  std::string inferenceParserName_;
  IOType& io_;
  std::string inferenceParserDescription_;
  std::vector<ArgTypes> arguments_;

  void read();
  template <typename ArgType>
  void addArgument(const ArgType& argIn);

public:
  InferenceParserBase(const std::string& InferenceParserNameIn, IOType& ioIn, const std::string& inferenceParserDescriptionIn);
  const std::string& getInferenceParserName();
  void printHelp(bool verboseRequested);
  virtual void run(ModelType& model,const std::string& outputfile) = 0;
};

template <class IOType, class ModelType, class Accumulation>
InferenceParserBase<IOType, ModelType, Accumulation>::InferenceParserBase(const std::string& InferenceParserNameIn, IOType& ioIn, const std::string& inferenceParserDescriptionIn)
  : inferenceParserName_(InferenceParserNameIn), io_(ioIn), inferenceParserDescription_(inferenceParserDescriptionIn) {
  ;
}

template <class IOType, class ModelType, class Accumulation>
const std::string& InferenceParserBase<IOType, ModelType, Accumulation>::getInferenceParserName() {
  return inferenceParserName_;
}

template <class IOType, class ModelType, class Accumulation>
void InferenceParserBase<IOType, ModelType, Accumulation>::printHelp(bool verboseRequested) {
  std::cout << "printing help for inference parser " << this->inferenceParserName_ << std::endl;
  std::cout << "description:" << std::endl;
  std::cout << this->inferenceParserDescription_ << std::endl;
  //are there any arguments?
  if(this->arguments_.empty()) {
      std::cout << this->inferenceParserName_ << " has no arguments." << std::endl;
      return;
    }
  std::cout << "arguments:" << std::endl;
  std::cout << std::setw(12) << std::left << "  short name" << std::setw(29) << std::left << "  long name" << std::setw(8) << std::left << "needed" << "description" << std::endl;

  //
  if(verboseRequested) {
    opPrintVerbose op;
    applyAll(op, this->arguments_);
  } else {
    opPrint op;
    applyAll(op, this->arguments_);
  }
}

template <class IOType, class ModelType, class Accumulation>
template <typename ArgType>
void InferenceParserBase<IOType, ModelType, Accumulation>::addArgument(const ArgType& argIn) {
  this->arguments_.push_back(argIn);
}

template <class IOType, class ModelType, class Accumulation>
void InferenceParserBase<IOType, ModelType, Accumulation>::read() {
  opRead<IOType> op(io_);
  applyAll(op, this->arguments_);
}

template <class IOType, class ModelType, class Accumulation>
void InferenceParserBase<IOType, ModelType, Accumulation>::run(ModelType& model,const std::string& outputfile) {
  read();
}

} // namespace interface

} // namespace opengm

#endif /* INFERENCEPARSERBASE_HXX_ */
