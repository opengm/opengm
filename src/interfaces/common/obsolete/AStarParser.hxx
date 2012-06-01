#ifndef ASTARPARSER_HXX_
#define ASTARPARSER_HXX_

#include <opengm/inference/astar.hxx>
#include "InferenceParserBase.hxx"
#include "helper/helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class AStarParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  typename opengm::AStar<ModelType, Accumulation>::Parameter astarParameter_;
  std::string selectedHeuristic_;
  std::string treeFactorIdsFileLocation_;
  std::string nodeOrderFileLocation_;
public:
  AStarParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
AStarParser<IOType, ModelType, Accumulation>::AStarParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("AStar", ioIn, "detailed description of AStar Parser...") {
  this->addArgument(Size_TArg(this->astarParameter_.maxHeapSize_, "", "maxheapsize", "maximum size of the heap", false, this->astarParameter_.maxHeapSize_));
  this->addArgument(DoubleArg(this->astarParameter_.maxTimeMs_, "", "maxtime", "maximum time in milliseconds", false, this->astarParameter_.maxTimeMs_));
  this->addArgument(Size_TArg(this->astarParameter_.numberOfOpt_, "", "numopt", "number of optimizations", false, this->astarParameter_.numberOfOpt_));
  this->addArgument(Argument<typename ModelType::ValueType>(this->astarParameter_.objectiveBound_, "", "objectivebound", "boundary for the objective", false, this->astarParameter_.objectiveBound_));
  std::vector<std::string> possibleHeuristics;
  possibleHeuristics.push_back("DEFAULT");
  possibleHeuristics.push_back("FAST");
  possibleHeuristics.push_back("STANDARD");
  this->addArgument(StringArg(this->selectedHeuristic_, "", "heuristic", "selects the desired cost heuristic", false, possibleHeuristics.at(0), possibleHeuristics));
  this->addArgument(StringArg(this->treeFactorIdsFileLocation_, "", "factorids", "location of the file containing a vector which specifies the desired tree factor ids", false, ""));
  this->addArgument(StringArg(this->nodeOrderFileLocation_, "", "nodeorder", "location of the file containing a vector which specifies the desired node order", false, ""));
  //TODO what's with addTreeFactorId(size_t id) (std::vector<size_t> treeFactorIds_) and std::vector<size_t> nodeOrder_;

}

template <class IOType, class ModelType, class Accumulation>
void AStarParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  if(this->selectedHeuristic_ == "DEFAULT") {
    this->astarParameter_.heuristic_= AStar<ModelType, Accumulation>::Parameter::DEFAULTHEURISTIC;
  } else if(this->selectedHeuristic_ == "FAST") {
    this->astarParameter_.heuristic_= AStar<ModelType, Accumulation>::Parameter::FASTHEURISTIC;
  } else if(this->selectedHeuristic_ == "STANDARD") {
    this->astarParameter_.heuristic_= AStar<ModelType, Accumulation>::Parameter::STANDARDHEURISTIC;
  } else {
    std::cerr << "error: unknown heuristic" << std::endl;
    abort();
  }
  //treeFactorIds
  if(!this->treeFactorIdsFileLocation_.empty()) {
    loadVector(this->treeFactorIdsFileLocation_, this->astarParameter_.treeFactorIds_);
  }
  //treeFactorIds
  if(!this->nodeOrderFileLocation_.empty()) {
    loadVector(this->nodeOrderFileLocation_, this->astarParameter_.nodeOrder_);
  }
  opengm::AStar<ModelType, Accumulation> astar(model, this->astarParameter_);

  std::vector<size_t> states;
  if(!(astar.infer() == opengm::NORMAL)) {
    std::cerr << "error: astar did not solve the problem" << std::endl;
    abort();
  }
  if(!(astar.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: astar could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

/*
// accumulation type integrator not allowed for AStar algorithm
template <class IOType, class ModelType>
void AStarParser<IOType, ModelType, Integrator>::run(ModelType& model, const std::string& outputfile) {
  std::cerr << "error: accumulation type INT not allowed for ASTAR algorithm" << std::endl;
  abort();
}
*/

template <class IOType, class ModelType>
class AStarParser<IOType, ModelType, Integrator> : public InferenceParserBase<IOType, ModelType, Integrator> {
public:
  AStarParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Integrator>("AStar", ioIn, "detailed description of AStar Parser...") {
    std::cerr << "error: accumulation type INT not allowed for ASTAR algorithm" << std::endl;
    abort();
  }
  void run(ModelType& model, const std::string& outputfile) {
    std::cerr << "error: accumulation type INT not allowed for ASTAR algorithm" << std::endl;
    abort();
  }
};

} //namespace opengm
} //namespace interface

#endif /* ASTARPARSER_HXX_ */
