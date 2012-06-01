#ifndef EXPANDANDFLIPPARSER_HXX_
#define EXPANDANDFLIPPARSER_HXX_

#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/expand_and_flip.hxx>
#include "InferenceParserBase.hxx"
#include "helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class ExpandAndFlipParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  typedef opengm::GraphCut<ModelType, Accumulation > GraphCut;
  typedef opengm::ExpandAndFlip<ModelType, GraphCut> ExpandAndFlip;
  typename ExpandAndFlip::Parameter expandandflipParameter_;
public:
  ExpandAndFlipParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
ExpandAndFlipParser<IOType, ModelType, Accumulation>::ExpandAndFlipParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("ExpandAndFlip", ioIn, "detailed description of ExpandAndFlip Parser...") {
  std::cout << "warning: ExpandAndFlip currently only supports the use of the inference algorithm GraphCut.\nFurthermore it only supports default parameter for the GraphCut algorithm" << std::endl;
  this->addArgument(Size_TArg(this->expandandflipParameter_.maxNumberOfIterations_, "", "numiter", "maximum number of iterations", false, this->expandandflipParameter_.maxNumberOfIterations_));
  this->addArgument(Size_TArg(this->expandandflipParameter_.lazyFlipperMaxSubgraphSize_, "", "maxsubgraphsize", "maximum size of a subgraph for lazy flipper algorithm", false, this->expandandflipParameter_.lazyFlipperMaxSubgraphSize_));

}

template <class IOType, class ModelType, class Accumulation>
void ExpandAndFlipParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  ExpandAndFlip expandandflip(model, this->expandandflipParameter_);

  std::vector<size_t> states;
  if(!(expandandflip.infer() == opengm::NORMAL)) {
    std::cerr << "error: expandandflip did not solve the problem" << std::endl;
    abort();
  }
  if(!(expandandflip.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: expandandflip could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

} //namespace opengm
} //namespace interface

#endif /* EXPANDANDFLIPPARSER_HXX_ */
