#ifndef BRUTEFORCEPARSER_HXX_
#define BRUTEFORCEPARSER_HXX_

#include <opengm/inference/bruteforce.hxx>
#include "InferenceParserBase.hxx"
#include "helper/helper.hxx"

namespace opengm {

namespace interface {

template <class IOType, class ModelType, class Accumulation>
class BruteforceParser : public InferenceParserBase<IOType, ModelType, Accumulation> {
protected:

public:
  BruteforceParser(IOType& ioIn);
  void run(ModelType& model, const std::string& outputfile);
};

template <class IOType, class ModelType, class Accumulation>
BruteforceParser<IOType, ModelType, Accumulation>::BruteforceParser(IOType& ioIn) : InferenceParserBase<IOType, ModelType, Accumulation>("Bruteforce", ioIn, "detailed description of Bruteforce Parser...") {
}

template <class IOType, class ModelType, class Accumulation>
void BruteforceParser<IOType, ModelType, Accumulation>::run(ModelType& model, const std::string& outputfile) {
  opengm::Bruteforce<ModelType, Accumulation> bruteforce(model);

  std::vector<size_t> states;
  if(!(bruteforce.infer() == opengm::NORMAL)) {
    std::cerr << "error: Bruteforce did not solve the problem" << std::endl;
    abort();
  }
  if(!(bruteforce.arg(states) == opengm::NORMAL)) {
    std::cerr << "error: Bruteforce could not return optimal argument" << std::endl;
    abort();
  }

  storeVector(outputfile, states);
}

} //namespace opengm
} //namespace interface

#endif /* BRUTEFORCEPARSER_HXX_ */
