#ifndef ALPHABETASWAPPARSER_HXX_
#define ALPHABETASWAPPARSER_HXX_

#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/graphcut.hxx>
#include "InferenceParserBase.hxx"
#include "helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class AlphaBetaSwapParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  typedef opengm::GraphCut<ModelType, Accumulation > GraphCut;
  typename opengm::AlphaBetaSwap<ModelType, GraphCut>::Parameter alphabetaswapParameter_;
  std::string desiredMaxFlowAlgorithm_;
public:
  AlphaBetaSwapParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
AlphaBetaSwapParser<IOType, ModelType, Accumulation>::AlphaBetaSwapParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("AlphaBetaSwap", ioIn, "detailed description of AlphaBetaSwap Parser...") {
  std::cout << "warning: AlphaBetaSwap currently only supports the use of the inference algorithm GraphCut.\nFurthermore it only supports default parameter for the GraphCut algorithm" << std::endl;
  this->addArgument(Size_TArg(this->alphabetaswapParameter_.maximalNumberOfIterations_, "", "numiter", "maximum number of iterations", false, this->alphabetaswapParameter_.maximalNumberOfIterations_));
  std::vector<std::string> permittedMaxFlowAlgorithms;
  permittedMaxFlowAlgorithms.push_back("PUSH_RELABEL");
  permittedMaxFlowAlgorithms.push_back("EDMONDS_KARP");
  permittedMaxFlowAlgorithms.push_back("KOLMOGOROV");
  this->addArgument(StringArg(this->desiredMaxFlowAlgorithm_, "", "MaxFlowAlgorithm", "select the desired max flow algorithm for the graph cut algorithm", false, permittedMaxFlowAlgorithms.at(0), permittedMaxFlowAlgorithms));
}

template <class IOType, class ModelType, class Accumulation>
void AlphaBetaSwapParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  if(this->desiredMaxFlowAlgorithm_ == "PUSH_RELABEL") {
    this->alphabetaswapParameter_.para_.maxFlowAlgorithm_ = GraphCut::PUSH_RELABEL;
  } else if(this->desiredMaxFlowAlgorithm_ == "EDMONDS_KARP") {
    this->alphabetaswapParameter_.para_.maxFlowAlgorithm_ = GraphCut::EDMONDS_KARP;
  } else if(this->desiredMaxFlowAlgorithm_ == "KOLMOGOROV") {
    this->alphabetaswapParameter_.para_.maxFlowAlgorithm_ = GraphCut::KOLMOGOROV;
  } else {
    std::cerr << "error: unknown MaxFlowAlgorithm: " << this->desiredMaxFlowAlgorithm_ << std::endl;
    abort();
  }
  opengm::AlphaBetaSwap<ModelType, GraphCut> alphabetaswap(model, this->alphabetaswapParameter_);

  std::vector<size_t> states;
  if(!(alphabetaswap.infer() == opengm::NORMAL)) {
    std::cerr << "error: alphabetaswap did not solve the problem" << std::endl;
    abort();
  }
  if(!(alphabetaswap.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: alphabetaswap could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

//Alpha Beta Swap doesn't support Accumulation Type Integrator
template <class IOType, class ModelType>
class AlphaBetaSwapParser<IOType, ModelType, Integrator> : public InferenceParserBase<IOType, ModelType, Integrator>{
public:
  AlphaBetaSwapParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Integrator>("AlphaBetaSwap", ioIn, "detailed description of AlphaBetaSwap Parser...") {
    std::cerr << "error: accumulation type INT not allowed for AlphaBetaSwap algorithm" << std::endl;
    abort();
  }
  void run(ModelType& model, const std::string& outputfile) {
    std::cerr << "error: accumulation type INT not allowed for AlphaBetaSwap algorithm" << std::endl;
    abort();
  }
};

} //namespace opengm
} //namespace interface

#endif /* ALPHABETASWAPPARSER_HXX_ */
