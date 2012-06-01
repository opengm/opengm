#ifndef BELIEFPROPAGATIONPARSER_HXX_
#define BELIEFPROPAGATIONPARSER_HXX_

#include <opengm/inference/beliefpropagation.hxx>
#include <opengm/operations/maxdistance.hxx>
#include "InferenceParserBase.hxx"
#include "helper/helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class BeliefPropagationParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  typename opengm::BeliefPropagation<ModelType, Accumulation, MaxDistance>::Parameter beliefpropagationParameter_;
  std::string sortedNodeListFileLocation_;
  std::string isAcyclic_;
public:
  BeliefPropagationParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
BeliefPropagationParser<IOType, ModelType, Accumulation>::BeliefPropagationParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("BeliefPropagation", ioIn, "detailed description of BeliefPropagation Parser...") {
  //*********************
  //** ADD PARAMETERS  **
  //*********************

  this->addArgument(Size_TArg(this->beliefpropagationParameter_.maximumNumberOfSteps_, "", "numiter", "maximum number of iterations", false, this->beliefpropagationParameter_.maximumNumberOfSteps_));
  //
  this->addArgument(Argument<typename ModelType::ValueType>(this->beliefpropagationParameter_.bound_, "", "bound", "boundary for the objective", false, this->beliefpropagationParameter_.bound_));
  //
  this->addArgument(Argument<typename ModelType::ValueType>(this->beliefpropagationParameter_.damping_, "", "damping", "damping", false, this->beliefpropagationParameter_.damping_));
  //
  this->addArgument(VoidArg(this->beliefpropagationParameter_.inferSequential_, "", "inferSequential", "set this flag to use sequential inference", false));
  //
  this->addArgument(VoidArg(this->beliefpropagationParameter_.useNormalization_, "", "useNormalization", "set this flag to use normalization", false));

  //TODO is this necessary?
  this->addArgument(StringArg(this->sortedNodeListFileLocation_, "", "nodelist", "location of the file containing a vector which specifies the desired sorting of the nodes", false, ""));
  //
  std::vector<std::string> possibleTriState;
  possibleTriState.push_back("TRUE");
  possibleTriState.push_back("MAYBE");
  possibleTriState.push_back("FALSE");
  this->addArgument(StringArg(this->isAcyclic_, "", "isAcyclic", "is graphical model acyclic?", false, possibleTriState.at(1), possibleTriState));
}

template <class IOType, class ModelType, class Accumulation>
void BeliefPropagationParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  if(!this->sortedNodeListFileLocation_.empty()) {
    loadVector(this->sortedNodeListFileLocation_, this->beliefpropagationParameter_.sortedNodeList_);
  }
  opengm::Tribool isAcyclic_;
  //this->io_.read(this->stringArguments_.at(1));
  if(this->isAcyclic_ == "TRUE") {
    this->beliefpropagationParameter_.isAcyclic_= 1;//beier::Tribool::True;
  } else if(this->isAcyclic_ == "MAYBE") {
    this->beliefpropagationParameter_.isAcyclic_= -1;//beier::Tribool::Maybe;
  } else if(this->isAcyclic_ == "FALSE") {
    this->beliefpropagationParameter_.isAcyclic_= 0;//beier::Tribool::False;
  } else {
    std::cerr << "error: tri-state: " << this->isAcyclic_ << std::endl;
    abort();
  }

  //TODO is thee any other distance type than opengm::MaxDistance?
  //NO NOT YET....currently only opengm::MaxDistance
  opengm::BeliefPropagation<ModelType, Accumulation, opengm::MaxDistance> beliefpropagation(model, this->beliefpropagationParameter_);

  std::vector<size_t> states;
  if(!(beliefpropagation.infer() == opengm::NORMAL)) {
    std::cerr << "error: BeliefPropagation did not solve the problem" << std::endl;
    abort();
  }
  if(!(beliefpropagation.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: BeliefPropagation could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

} //namespace opengm
} //namespace interface

#endif /* BELIEFPROPAGATIONPARSER_HXX_ */
