#ifndef LBPPARSER_HXX_
#define LBPPARSER_HXX_

#include "InferenceParserBase.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class LBPParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  bool simulate_;
  bool fast_;
  int maxIter_;
  double eps_;
  std::string optimizationType_;
public:
  LBPParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
LBPParser<IOType, ModelType, Accumulation>::LBPParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("LBP", ioIn, "detailed description of LBP Parser...") {
  //TODO define required arguments
  this->addArgument(VoidArg(this->simulate_, "", "simulate", "set this flag to only simulate the optimization", false));
  this->addArgument(VoidArg(this->fast_, "f", "fast", "set this flag to use a faster but more inaccurate optimization", false));
  std::vector<std::string> possibleOptimizationTypes;
  possibleOptimizationTypes.push_back("COMPLEX");
  possibleOptimizationTypes.push_back("SIMPLE");
  this->addArgument(StringArg(this->optimizationType_, "", "optimizationtype", "select the desired type of optimization", false, "COMPLEX", possibleOptimizationTypes));
  this->addArgument(IntArg(this->maxIter_, "mi", "maxiter", "set the maximum number of iterations", false, 100));
  // next line does not make sense to me  (and warnings "statement has no effect)
  //this->floatArguments_;
  this->addArgument(DoubleArg(this->eps_, "eps", "epsilon", "set the desired precision", true));
}

template <class IOType, class ModelType, class Accumulation>
void LBPParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  std::cout << "not yet implemented" << std::endl;
  std::cout << "running evaluation with:" << std::endl;
  //this->io_.read(this->intArguments_.at(0));
  std::cout << "maxIter: " << this->maxIter_ << std::endl;
  //this->io_.read(this->doubleArguments_.at(0));
  std::cout << "eps_: " << this->eps_ << std::endl;
  //this->io_.read(this->stringArguments_.at(0));

  std::cout << "optimizationType_: " << this->optimizationType_ << std::endl;
  //this->simulate_ = this->io_.read(this->voidArguments_.at(0));
  if(this->simulate_) {
    std::cout << "simulation requested" << std::endl;
  }
  //this->fast_ = this->io_.read(this->voidArguments_.at(1));
  if(this->fast_) {
    std::cout << "fast optimization requested" << std::endl;
  }
  std::cout << "storing results in: " << outputfile << std::endl;
}

} //namespace opengm
} //namespace interface

#endif /* LBPPARSER_HXX_ */
