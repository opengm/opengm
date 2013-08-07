#ifndef INTERFACE_BASE_HXX_
#define INTERFACE_BASE_HXX_

#include "../argument/argument.hxx"
#include "../argument/argument_executer.hxx"

#include "../parser/execute_with_type_from_string.hxx"

namespace opengm {

namespace interface {

/*********************
 * class definitions *
 *********************/

template<class TYPELIST, size_t IX, size_t DX, bool END>
struct getPossibleStringValues;

template<class TYPELIST, size_t IX, size_t DX>
struct getPossibleStringValues<TYPELIST, IX, DX, false> {
   static void fill(std::vector<std::string>& possibleValues);
};

template<class TYPELIST, size_t IX, size_t DX>
struct getPossibleStringValues<TYPELIST, IX, DX, true> {
   static void fill(std::vector<std::string>& possibleValues);
};

template <class IO>
struct printHelpInferenceAlgorithm {
   static IO* io_;
   static bool verbose_;
   template <class INFERENCECALLER>
   static void execute();
};

template <class IO, class GM, class OUTPUTTYPE>
struct callInferenceAlgorithmFunctor {
   static IO* io_;
   static GM* gm_;
   static BoolArgument* verbose_;
   static OUTPUTTYPE* outputfile_;
   template <class INFERENCECALLER>
   static void execute();
};

template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
class InterfaceBase {
public:
   InterfaceBase(IOTYPE& ioIn);
   virtual ~InterfaceBase();

   void parse();
protected:
   static const size_t maxNumArguments = 100;
   IOTYPE& io_;

   ArgumentExecuter<IOTYPE> argumentContainer_;

   // graphical model
   GM* gm_;
   bool ownsGM_;

   // Storage for input variables
   bool helpRequested_;
   bool verboseRequested_;
   bool modelinfoRequested_;
   std::string selectedAlgorithm_;
   std::vector<size_t> evaluateVec_;

   // Allowed values
   std::vector<std::string> possibleAlgorithmValues_;

   // References to created Arguments
   BoolArgument* help_;
   BoolArgument* verbose_;
   BoolArgument* modelinfo_;
   StringArgument<>* algorithm_;
   // TODO currently only std::vector<size_t> is supported
   VectorArgument< std::vector<size_t> >* evaluate_;

   template <class TYPELIST>
   static void fillPossibleValues(std::vector<std::string>& possibleValues);
   void printDefaultHelp();
   void printHelpAlgorithm();

   virtual void loadModel() = 0;
   void loadModelFromFile(const std::string& selectedModel);

   virtual void evaluateCallback(const typename GM::ValueType& result) = 0;

   virtual const std::string& interfaceName() = 0;

   virtual void callInferenceAlgorithm() = 0;

   template<class FUNCTOR>
   void callInferenceAlgorithmImpl();

};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
inline InterfaceBase<GM, INFERENCETYPES, IOTYPE>::InterfaceBase(IOTYPE& ioIn)
   : io_(ioIn), argumentContainer_(io_, maxNumArguments), gm_(NULL), ownsGM_(false) {
   fillPossibleValues<INFERENCETYPES>(possibleAlgorithmValues_);

   help_ = &argumentContainer_.addArgument(BoolArgument(helpRequested_, "h", "help", "used to activate help output"));
   verbose_ = &argumentContainer_.addArgument(BoolArgument(verboseRequested_, "v", "verbose", "used to activate verbose output"));
   modelinfo_ = &argumentContainer_.addArgument(BoolArgument(modelinfoRequested_, "", "modelinfo", "used to print detailed informations about the specified model"));
   algorithm_ = &argumentContainer_.addArgument(StringArgument<>(selectedAlgorithm_, "a", "algorithm", "used to specify the desired algorithm", false, possibleAlgorithmValues_));
   evaluate_ = &argumentContainer_.addArgument(VectorArgument< std::vector<size_t> >(evaluateVec_, "e", "evaluate", "used to evaluate the selected model with the specified input file", false));
}

template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
inline InterfaceBase<GM, INFERENCETYPES, IOTYPE>::~InterfaceBase() {
   if(ownsGM_) {
      if(gm_) {
         delete gm_;
      }
   }
}

template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
inline void InterfaceBase<GM, INFERENCETYPES, IOTYPE>::parse() {
   // Check command options and proceed with appropriate behavior
   io_.read(*help_);

   if(help_->getValue()) {
      io_.read(*algorithm_);
      io_.read(*verbose_);
      if(algorithm_->isSet()) {
         printHelpAlgorithm();
      } else {
         printDefaultHelp();
      }
   } else {
      // load model. This has to be done by the child class as loading the model is dependent of the specific interface.
      loadModel();

      io_.read(*modelinfo_);
      if(modelinfo_->getValue()) {
         //std::cout << "parserChainLoadModel: print modelinfo" << std::endl;
         io_.modelInfo(*gm_, io_.standardStream());
      } else {
         io_.read(*evaluate_);
         if(evaluate_->isSet()) {
            //std::cout << "parserChainLoadModel: evaluate vector" << std::endl;
            typename GM::ValueType result = gm_->evaluate(evaluate_->getValue());
            evaluateCallback(result);
         } else {
            callInferenceAlgorithm();
            /*typedef callInferenceAlgorithmFunctor<IOTYPE, GM> currentOperator;
            currentOperator::io_ = &io_;
            currentOperator::gm_ = gm_;
            currentOperator::verbose_ = verbose_;
            currentOperator::outputfile_ = outputfile_;
            const size_t length = opengm::meta::LengthOfTypeList<INFERENCETYPES>::value;
            io_.read(*algorithm_);
            if(algorithm_->isSet()) {
               executeWithTypeFromString<currentOperator, INFERENCETYPES, 0, length, opengm::meta::EqualNumber<length, 0>::value>::execute(algorithm_->getValue());
            } else {
               throw RuntimeError(algorithm_->getLongName() + "not set, but required");
            }*/
         }
      }
   }
}

//TODO update description
/**
 * @brief Prints default help of the commandline interface.
 * @details This function prints the default help of the commandline interface
 *          on screen. It shows the user the basic usage of the commandline
 *          interface and shows him the possible command options he can use. If
 *          verbose output is requested, more detailed informations will be
 *          printed.
 * @param[in] verboseRequested specifies if verbose output is requested.
 * @param[in] arguments A vector containing all possible commandline arguments.
 */
template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
inline void InterfaceBase<GM, INFERENCETYPES, IOTYPE>::printDefaultHelp() {
   std::cout << "This is the help for the " << interfaceName() << " of the opengm library" << std::endl;
   std::cout << "Usage: opengm [arguments]" << std::endl;
   std::cout << "arguments:" << std::endl;
   std::cout << std::setw(12) << std::left << "  short name" << std::setw(29) << std::left << "  long name" << std::setw(8) << std::left << "needed" << "description" << std::endl;
   argumentContainer_.printHelp(std::cout, true);
}

template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
inline void InterfaceBase<GM, INFERENCETYPES, IOTYPE>::printHelpAlgorithm() {
   // Print help of selected algorithm.

   // Pass through necessary arguments.
   printHelpInferenceAlgorithm<IOTYPE>::io_ = &io_;
   printHelpInferenceAlgorithm<IOTYPE>::verbose_ = true;

   // Determine length of type list.
   const size_t length = opengm::meta::LengthOfTypeList<INFERENCETYPES>::value;

   // Call print help of selected algorithm
   executeWithTypeFromString<printHelpInferenceAlgorithm<IOTYPE>, INFERENCETYPES, 0, length, opengm::meta::EqualNumber<length, 0>::value>::execute(selectedAlgorithm_);
}

template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
inline void InterfaceBase<GM, INFERENCETYPES, IOTYPE>::loadModelFromFile(const std::string& selectedModel) {
   std::string modelFilename;
   std::string dataset;
   io_.separateFilename(selectedModel, modelFilename, dataset);
   if(dataset.empty()) {
      io_.standardStream() << "warning: using default dataset \"gm\"" << std::endl;
      io_.logStream() << "warning: using default dataset \"gm\"" << std::endl;
      dataset = "gm";
   }
   io_.standardStream() << "loading model: " << modelFilename << std::endl;
   io_.standardStream() << "using dataset: " << dataset << std::endl;

   // load model
   gm_ = new GM();
   ownsGM_ = true;
   opengm::hdf5::load(*gm_, modelFilename, dataset);
}

template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
template<class FUNCTOR>
void InterfaceBase<GM, INFERENCETYPES, IOTYPE>::callInferenceAlgorithmImpl() {
   FUNCTOR::io_ = &io_;
   FUNCTOR::gm_ = gm_;
   FUNCTOR::verbose_ = verbose_;
   const size_t length = opengm::meta::LengthOfTypeList<INFERENCETYPES>::value;
   io_.read(*algorithm_);
   if(algorithm_->isSet()) {
      executeWithTypeFromString<FUNCTOR, INFERENCETYPES, 0, length, opengm::meta::EqualNumber<length, 0>::value>::execute(algorithm_->getValue());
   } else {
      throw RuntimeError(algorithm_->getLongName() + "not set, but required");
   }
}

/**
 * @brief Fills a given vector with the names of the possible values from a
 *        given type list.
 * @tparam TYPELIST Type list from which the allowed names come.
 * @param[out] possibleValues Vector which will be filled with the names of the
 *                            possible value types.
 */
template <
   class GM,
   class INFERENCETYPES,
   class IOTYPE
>
template <class TYPELIST>
inline void InterfaceBase<GM, INFERENCETYPES, IOTYPE>::fillPossibleValues(std::vector<std::string>& possibleValues) {
   const size_t length = opengm::meta::LengthOfTypeList<TYPELIST>::value;
   getPossibleStringValues<TYPELIST, 0, length, opengm::meta::EqualNumber<length, 0>::value>::fill(possibleValues);
}

template<class TYPELIST, size_t IX, size_t DX>
inline void getPossibleStringValues<TYPELIST, IX, DX, false>::fill(std::vector<std::string>& possibleValues) {
   typedef typename opengm::meta::TypeAtTypeList<TYPELIST, IX>::type currentType;
   possibleValues.push_back((std::string)currentType::name_);
   // proceed with next type
   typedef typename opengm::meta::Increment<IX>::type NewIX;
   getPossibleStringValues<TYPELIST, NewIX::value, DX, opengm::meta::EqualNumber<NewIX::value, DX>::value >::fill(possibleValues);
}

template<class TYPELIST, size_t IX, size_t DX>
inline void getPossibleStringValues<TYPELIST, IX, DX, true>::fill(std::vector<std::string>& possibleValues) { }

template <class IO>
template <class INFERENCECALLER>
inline void printHelpInferenceAlgorithm<IO>::execute() {
   INFERENCECALLER caller(*io_);
   caller.printHelp(verbose_);
}
template <class IO> IO* printHelpInferenceAlgorithm<IO>::io_ = NULL;
template <class IO> bool printHelpInferenceAlgorithm<IO>::verbose_ = false;

template <class IO, class GM, class OUTPUTTYPE>
template <class INFERENCECALLER>
inline void callInferenceAlgorithmFunctor<IO, GM, OUTPUTTYPE>::execute() {
   INFERENCECALLER caller(*io_);
   io_->read(*verbose_);
   caller.run(*gm_, *outputfile_, verbose_->getValue());
}
template <class IO, class GM, class OUTPUTTYPE> IO* callInferenceAlgorithmFunctor<IO, GM, OUTPUTTYPE>::io_ = NULL;
template <class IO, class GM, class OUTPUTTYPE> GM* callInferenceAlgorithmFunctor<IO, GM, OUTPUTTYPE>::gm_ = NULL;
template <class IO, class GM, class OUTPUTTYPE> BoolArgument* callInferenceAlgorithmFunctor<IO, GM, OUTPUTTYPE>::verbose_ = NULL;
template <class IO, class GM, class OUTPUTTYPE> OUTPUTTYPE* callInferenceAlgorithmFunctor<IO, GM, OUTPUTTYPE>::outputfile_ = NULL;

} // namespace interface

} // namespace opengm

#endif /* INTERFACE_BASE_HXX_ */
