#ifndef ICMPARSER_HXX_
#define ICMPARSER_HXX_

#include <opengm/inference/icm.hxx>
#include "InferenceParserBase.hxx"
#include "helper/helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class ICMParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  typename opengm::ICM<ModelType, Accumulation>::Parameter icmParameter_;
  std::string startingPointFileLocation_;
public:
  ICMParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
ICMParser<IOType, ModelType, Accumulation>::ICMParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("ICM", ioIn, "detailed description of ICM Parser...") {
  this->addArgument(StringArg(this->startingPointFileLocation_, "x0", "startingpoint", "location of the file containing the values for the starting point", false, ""));
}

template <class IOType, class ModelType, class Accumulation>
void ICMParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  this->read();
  if(!this->startingPointFileLocation_.empty()) {
    loadVector(this->startingPointFileLocation_, this->icmParameter_.startPoint_);
  }
  opengm::ICM<ModelType, Accumulation> icm(model, this->icmParameter_);

  std::vector<size_t> states;
  if(!(icm.infer() == opengm::NORMAL)) {
    std::cerr << "error: icm did not solve the problem" << std::endl;
    abort();
  }
  if(!(icm.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: icm could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

} //namespace opengm
} //namespace interface

#endif /* ICMPARSER_HXX_ */
