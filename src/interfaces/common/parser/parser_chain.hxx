#ifndef PARSER_CHAIN_HXX_
#define PARSER_CHAIN_HXX_

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include "../argument/argument.hxx"
#include "execute_with_type_from_string.hxx"

namespace opengm {

namespace interface {

/********************
 * class definition *
 ********************/


template <
   class IO,
   class GM,
   class INFERENCETYPES
>
struct ParserChainManageModel {
   static IO* io_;
   static BoolArgument* verbose_;
   static BoolArgument* modelinfo_;
   static StringArgument<>* model_;
   static StringArgument<>* algorithm_;
   static StringArgument<>* outputfile_;
   static VectorArgument< std::vector<size_t> >* evaluate_;
   static void execute();
};

template <class IO, class GM>
struct ParserChainCallInferenceAlgorithm {
   static IO* io_;
   static GM* gm_;
   static BoolArgument* verbose_;
   static StringArgument<>* outputfile_;
   template <class INFERENCECALLER>
   static void execute();
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <
   class IO,
   class GM,
   class INFERENCETYPES
>
inline void ParserChainManageModel<IO, GM, INFERENCETYPES>::execute() {
   io_->read(*model_);

   // get model filename and desired dataset
   std::string selectedModel(model_->getValue());
   std::string modelFilename;
   std::string dataset;
   io_->separateFilename(selectedModel, modelFilename, dataset);
   if(dataset.empty()) {
      io_->standardStream() << "warning: using default dataset \"gm\"" << std::endl;
      io_->logStream() << "warning: using default dataset \"gm\"" << std::endl;
      dataset = "gm";
   }
   std::cout << "loading model: " << modelFilename << std::endl;
   std::cout << "using dataset: " << dataset << std::endl;

   // load model
   GM gm;
   opengm::hdf5::load(gm, modelFilename, dataset);
   io_->read(*modelinfo_);
   if(modelinfo_->getValue()) {
      //std::cout << "parserChainLoadModel: print modelinfo" << std::endl;
      io_->modelInfo(gm, io_->standardStream());
   } else {
      io_->read(*evaluate_);
      if(evaluate_->isSet()) {
         //std::cout << "parserChainLoadModel: evaluate vector" << std::endl;
         typename GM::ValueType result = gm.evaluate(evaluate_->getValue());
         std::cout << std::scientific << "result: " << result << std::endl;
      } else {
         typedef ParserChainCallInferenceAlgorithm<IO, GM> currentOperator;
         currentOperator::io_ = io_;
         currentOperator::gm_ = &gm;
         currentOperator::verbose_ = verbose_;
         currentOperator::outputfile_ = outputfile_;
         const size_t length = opengm::meta::LengthOfTypeList<INFERENCETYPES>::value;
         io_->read(*algorithm_);
         if(algorithm_->isSet()) {
            executeWithTypeFromString<currentOperator, INFERENCETYPES, 0, length, opengm::meta::EqualNumber<length, 0>::value>::execute(algorithm_->getValue());
         } else {
            throw RuntimeError(algorithm_->getLongName() + "not set, but required");
         }
      }
   }
}
template <class IO, class GM, class INFERENCETYPES>
IO* ParserChainManageModel<IO, GM, INFERENCETYPES>::io_ = NULL;
template <class IO, class GM, class INFERENCETYPES>
BoolArgument* ParserChainManageModel<IO, GM, INFERENCETYPES>::verbose_ = NULL;
template <class IO, class GM, class INFERENCETYPES>
BoolArgument* ParserChainManageModel<IO, GM, INFERENCETYPES>::modelinfo_ = NULL;
template <class IO, class GM, class INFERENCETYPES>
StringArgument<>* ParserChainManageModel<IO, GM, INFERENCETYPES>::model_ = NULL;
template <class IO, class GM, class INFERENCETYPES>
StringArgument<>* ParserChainManageModel<IO, GM, INFERENCETYPES>::algorithm_ = NULL;
template <class IO, class GM, class INFERENCETYPES>
StringArgument<>* ParserChainManageModel<IO, GM, INFERENCETYPES>::outputfile_ = NULL;
template <class IO, class GM, class INFERENCETYPES>
VectorArgument< std::vector<size_t> >* ParserChainManageModel<IO, GM, INFERENCETYPES>::evaluate_ = NULL;

template <class IO, class GM>
template <class INFERENCECALLER>
inline void ParserChainCallInferenceAlgorithm<IO, GM>::execute() {
   INFERENCECALLER caller(*io_);
   io_->read(*verbose_);
   caller.run(*gm_, *outputfile_, verbose_->getValue());
}
template <class IO, class GM> IO* ParserChainCallInferenceAlgorithm<IO, GM>::io_ = NULL;
template <class IO, class GM> GM* ParserChainCallInferenceAlgorithm<IO, GM>::gm_ = NULL;
template <class IO, class GM> BoolArgument* ParserChainCallInferenceAlgorithm<IO, GM>::verbose_ = NULL;
template <class IO, class GM> StringArgument<>* ParserChainCallInferenceAlgorithm<IO, GM>::outputfile_ = NULL;
} // namespace interface

} // namespace opengm

#endif /* PARSER_CHAIN_HXX_ */
