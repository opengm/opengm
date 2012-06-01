#ifndef LAZYFLIPPERPARSER_HXX_
#define LAZYFLIPPERPARSER_HXX_

#include <opengm/inference/lazyflipper.hxx>
#include "InferenceParserBase.hxx"
#include "helper/helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class LazyFlipperParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  typename opengm::LazyFlipper<ModelType, Accumulation>::Parameter lazyflipperParameter_;
  std::string startingPointFileLocation_;

public:
  LazyFlipperParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
LazyFlipperParser<IOType, ModelType, Accumulation>::LazyFlipperParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("LazyFlipper", ioIn, "detailed description of LazyFlipper Parser...") {
  this->addArgument(StringArg(this->startingPointFileLocation_, "x0", "startingpoint", "location of the file containing the values for the starting point", false, ""));
  this->addArgument(Size_TArg(this->lazyflipperParameter_.maxSubgraphSize, "", "maxsubgraphsize", "maximum size of a subgraph for lazy flipper algorithm", false, this->lazyflipperParameter_.maxSubgraphSize));
}

template <class IOType, class ModelType, class Accumulation>
void LazyFlipperParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  //starting point
  if(!this->startingPointFileLocation_.empty()) {
    loadVector(this->startingPointFileLocation_, this->lazyflipperParameter_.startingPoint);
  }
  //maxSubgraphSize
  //this->io_.read(this->size_tArguments_.at(0));

  opengm::LazyFlipper<ModelType, Accumulation> lazyflipper(model, this->lazyflipperParameter_);

  std::vector<size_t> states;
  if(!(lazyflipper.infer() == opengm::NORMAL)) {
    std::cerr << "error: lazyflipper did not solve the problem" << std::endl;
    abort();
  }
  if(!(lazyflipper.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: lazyflipper could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

} //namespace opengm
} //namespace interface

#endif /* LAZYFLIPPERPARSER_HXX_ */
