#ifndef ALPHAEXPANSIONPARSER_HXX_
#define ALPHAEXPANSIONPARSER_HXX_

#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/graphcut.hxx>
#include "InferenceParserBase.hxx"
#include "helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class AlphaExpansionParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  typedef opengm::GraphCut<ModelType, Accumulation > GraphCut;
  typedef opengm::AlphaExpansion<ModelType, GraphCut> AlphaExpansion;
  typename AlphaExpansion::Parameter alphaexpansionParameter_;
  std::string desiredMaxFlowAlgorithm_;
  std::string desiredLabelInitialType_;
  std::string desiredOrderType_;
public:
  AlphaExpansionParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
AlphaExpansionParser<IOType, ModelType, Accumulation>::AlphaExpansionParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("AlphaExpansion", ioIn, "detailed description of AlphaExpansion Parser...") {
  std::cout << "warning: AlphaExpansion currently only supports the use of the inference algorithm GraphCut.\nFurthermore it only supports default parameter for the GraphCut algorithm" << std::endl;
  this->addArgument(Size_TArg(this->alphaexpansionParameter_.maxNumberOfSteps_, "", "numiter", "maximum number of iterations", false, this->alphaexpansionParameter_.maxNumberOfSteps_));
  std::vector<std::string> permittedMaxFlowAlgorithms;
  permittedMaxFlowAlgorithms.push_back("PUSH_RELABEL");
  permittedMaxFlowAlgorithms.push_back("EDMONDS_KARP");
  permittedMaxFlowAlgorithms.push_back("KOLMOGOROV");
  this->addArgument(StringArg(this->desiredMaxFlowAlgorithm_, "", "MaxFlowAlgorithm", "select the desired max flow algorithm for the graph cut algorithm", false, permittedMaxFlowAlgorithms.at(0), permittedMaxFlowAlgorithms));
  std::vector<std::string> permittedLabelInitialTypes;
  permittedLabelInitialTypes.push_back("DEFAULT_LABEL");
  permittedLabelInitialTypes.push_back("RANDOM_LABEL");
  permittedLabelInitialTypes.push_back("LOCALOPT_LABEL");
  permittedLabelInitialTypes.push_back("EXPLICITE_LABEL");
  this->addArgument(StringArg(this->desiredLabelInitialType_, "", "labelInitialType", "select the desired initial label", false, permittedLabelInitialTypes.at(0), permittedLabelInitialTypes));
  std::vector<std::string> permittedOrderTypes;
  permittedOrderTypes.push_back("DEFAULT_ORDER");
  permittedOrderTypes.push_back("RANDOM_ORDER");
  permittedOrderTypes.push_back("EXPLICITE_ORDER");
  tthis->addArgument(StringArg(this->desiredOrderType_, "", "orderType", "select the desired order", false, permittedOrderTypes.at(0), permittedOrderTypes));

  //TODO what's with std::vector<size_t> labelOrder_; and std::vector<size_t> label_; ?
}

template <class IOType, class ModelType, class Accumulation>
void AlphaExpansionParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  //maximalNumberOfIterations
  this->io_.read(this->size_tArguments_.at(0));

  //MaxFlowAlgorithm
  this->io_.read(this->stringArguments_.at(0));
  if(this->desiredMaxFlowAlgorithm_ == "PUSH_RELABEL") {
    this->alphaexpansionParameter_.para_.maxFlowAlgorithm_ = GraphCut::PUSH_RELABEL;
  } else if(this->desiredMaxFlowAlgorithm_ == "EDMONDS_KARP") {
    this->alphaexpansionParameter_.para_.maxFlowAlgorithm_ = GraphCut::EDMONDS_KARP;
  } else if(this->desiredMaxFlowAlgorithm_ == "KOLMOGOROV") {
    this->alphaexpansionParameter_.para_.maxFlowAlgorithm_ = GraphCut::KOLMOGOROV;
  } else {
    std::cerr << "error: unknown MaxFlowAlgorithm: " << this->desiredMaxFlowAlgorithm_ << std::endl;
    abort();
  }

  //LabelInitialType
  this->io_.read(this->stringArguments_.at(1));
  if(this->desiredLabelInitialType_ == "DEFAULT_LABEL") {
    this->alphaexpansionParameter_.labelInitialType_ = AlphaExpansion::Parameter::DEFAULT_LABEL;
  } else if(this->desiredLabelInitialType_ == "RANDOM_LABEL") {
    this->alphaexpansionParameter_.labelInitialType_ = AlphaExpansion::Parameter::RANDOM_LABEL;
  } else if(this->desiredLabelInitialType_ == "LOCALOPT_LABEL") {
    this->alphaexpansionParameter_.labelInitialType_ = AlphaExpansion::Parameter::LOCALOPT_LABEL;
  } else if(this->desiredLabelInitialType_ == "EXPLICITE_LABEL") {
    this->alphaexpansionParameter_.labelInitialType_ = AlphaExpansion::Parameter::EXPLICITE_LABEL;
  } else {
    std::cerr << "error: unknown initial label: " << this->desiredLabelInitialType_ << std::endl;
    abort();
  }

  //orderType
  this->io_.read(this->stringArguments_.at(2));
  if(this->desiredOrderType_ == "DEFAULT_ORDER") {
    this->alphaexpansionParameter_.orderType_ = AlphaExpansion::Parameter::DEFAULT_ORDER;
  } else if(this->desiredOrderType_ == "RANDOM_ORDER") {
    this->alphaexpansionParameter_.orderType_ = AlphaExpansion::Parameter::RANDOM_ORDER;
  } else if(this->desiredOrderType_ == "EXPLICITE_ORDER") {
    this->alphaexpansionParameter_.orderType_ = AlphaExpansion::Parameter::EXPLICITE_ORDER;
  } else {
    std::cerr << "error: unknown order type: " << this->desiredOrderType_ << std::endl;
    abort();
  }

  opengm::AlphaExpansion<ModelType, GraphCut> alphaexpansion(model, this->alphaexpansionParameter_);

  std::vector<size_t> states;
  if(!(alphaexpansion.infer() == opengm::NORMAL)) {
    std::cerr << "error: alphabetaswap did not solve the problem" << std::endl;
    abort();
  }
  if(!(alphaexpansion.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: alphabetaswap could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

} //namespace opengm
} //namespace interface

#endif /* ALPHAEXPANSIONPARSER_HXX_ */
