#ifndef GRAPHCUTPARSER_HXX_
#define GRAPHCUTPARSER_HXX_

#include <opengm/inference/graphcut.hxx>
#include "InferenceParserBase.hxx"
#include "helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class GraphCutParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  typename opengm::GraphCut<ModelType, Accumulation>::Parameter graphcutParameter_;
  std::string desiredMaxFlowAlgorithm_;
public:
  GraphCutParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
GraphCutParser<IOType, ModelType, Accumulation>::GraphCutParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("GraphCut", ioIn, "detailed description of GraphCut Parser...") {
  std::vector<std::string> permittedMaxFlowAlgorithms;
  permittedMaxFlowAlgorithms.push_back("PUSH_RELABEL");
  permittedMaxFlowAlgorithms.push_back("EDMONDS_KARP");
  permittedMaxFlowAlgorithms.push_back("KOLMOGOROV");
  this->addArgument(StringArg(this->desiredMaxFlowAlgorithm_, "", "MaxFlowAlgorithm", "select the desired max flow algorithm for the graph cut algorithm", false, permittedMaxFlowAlgorithms.at(0), permittedMaxFlowAlgorithms));
}

template <class IOType, class ModelType, class Accumulation>
void GraphCutParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {

  //MaxFlowAlgorithm
  //this->io_.read(this->stringArguments_.at(0));
  if(this->desiredMaxFlowAlgorithm_ == "PUSH_RELABEL") {
    this->graphcutParameter_.maxFlowAlgorithm_ = GraphCut<ModelType, Accumulation>::PUSH_RELABEL;
  } else if(this->desiredMaxFlowAlgorithm_ == "EDMONDS_KARP") {
    this->graphcutParameter_.maxFlowAlgorithm_ = GraphCut<ModelType, Accumulation>::EDMONDS_KARP;
  } else if(this->desiredMaxFlowAlgorithm_ == "KOLMOGOROV") {
    this->graphcutParameter_.maxFlowAlgorithm_ = GraphCut<ModelType, Accumulation>::KOLMOGOROV;
  } else {
    std::cerr << "error: unknown MaxFlowAlgorithm: " << this->desiredMaxFlowAlgorithm_ << std::endl;
    abort();
  }

  opengm::GraphCut<ModelType, Accumulation> graphcut(model, this->graphcutParameter_);

  std::vector<size_t> states;
  if(!(graphcut.infer() == opengm::NORMAL)) {
    std::cerr << "error: graphcut did not solve the problem" << std::endl;
    abort();
  }
  if(!(graphcut.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: graphcut could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

} //namespace opengm
} //namespace interface

#endif /* GRAPHCUTPARSER_HXX_ */
