#ifndef TRBPPARSER_HXX_
#define TRBPPARSER_HXX_

#include "InferenceParserBase.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class TRBPParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:
  bool fast_;
  int maxIter_;
  std::string optimizationType_;
  float x0_;
  double xMin_;
  double xMax_;
public:
  TRBPParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
TRBPParser<IOType, ModelType, Accumulation>::TRBPParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("TRBP", ioIn, "detailed description of TRBP Parser...") {
  //TODO define required arguments
  this->addArgument(VoidArg(this->fast_, "f", "fast", "set this flag to use a faster but more inaccurate optimization", false));
  std::vector<std::string> possibleOptimizationTypes;
  possibleOptimizationTypes.push_back("SUPERSIMPLE");
  possibleOptimizationTypes.push_back("SIMPLE");
  this->addArgument(StringArg(this->optimizationType_, "", "optimizationtype", "select the desired type of optimization", false, "SIMPLE", possibleOptimizationTypes));
  this->addArgument(IntArg(this->maxIter_, "mi", "maxiter", "set the maximum number of iterations", false, 100));
  this->addArgument(FloatArg(this->x0_, "x0", "startingpoint", "Point at which the algorithm begins to optimize", false, 3.14159));
  this->addArgument(DoubleArg(this->xMin_, "", "xmin", "set the minimal value for x", true));
  this->addArgument(DoubleArg(this->xMax_, "", "xmax", "set the maximal value for x", true));
}

template <class IOType, class ModelType, class Accumulation>
void TRBPParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  std::cout << "not yet implemented" << std::endl;
  std::cout << "running evaluation with:" << std::endl;
  //this->io_.read(this->intArguments_.at(0));
  std::cout << "maxIter: " << this->maxIter_ << std::endl;
  //this->io_.read(this->stringArguments_.at(0));
  std::cout << "optimizationType_: " << this->optimizationType_ << std::endl;
  //this->io_.read(this->floatArguments_.at(0));
  std::cout << "x0: " << this->x0_ << std::endl;
  //this->io_.read(this->doubleArguments_.at(0));
  std::cout << "xMin: " << this->xMin_ << std::endl;
  //this->io_.read(this->doubleArguments_.at(1));
  std::cout << "xMax: " << this->xMax_ << std::endl;
  if(this->fast_) {
    std::cout << "fast optimization requested" << std::endl;
  }
  std::cout << "storing results in: " << outputfile << std::endl;


}

} //namespace opengm
} //namespace interface

#endif /* TRBPPARSER_HXX_ */
