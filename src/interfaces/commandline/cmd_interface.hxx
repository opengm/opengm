#ifndef CMD_INTERFACE_HXX_
#define CMD_INTERFACE_HXX_

#include <opengm/operations/adder.hxx>

#include "../common/argument/argument.hxx"
#include "../common/argument/argument_executer.hxx"

#include "io_cmd.hxx"
#include "../common/parser/parser_chain.hxx"

namespace opengm {

namespace interface {

/********************
 * class definition *
 ********************/

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

template <
   class GM,
   class INFERENCETYPES
>
class CMDInterface {
protected:
   // Create commandline-IO
   IOCMD commandlineIO_;

   ArgumentExecuter<IOCMD> argumentContainer_;

   // Storage for input variables
   bool helpRequested_;
   bool verboseRequested_;
   bool modelinfoRequested_;
   std::string selectedModel_;
   std::string selectedAlgorithm_;
   std::string selectedOutputfile_;
   std::vector<size_t> evaluateVec_;

   // Allowed values
   std::vector<std::string> possibleAlgorithmValues_;

   // References to created Arguments
   BoolArgument* help_;
   BoolArgument* verbose_;
   BoolArgument* modelinfo_;
   StringArgument<>* model_;
   StringArgument<>* algorithm_;
   StringArgument<>* outputfile_;
   // TODO currently only std::vector<size_t> is supported
   VectorArgument< std::vector<size_t> >* evaluate_;

   template <class TYPELIST>
   void fillPossibleValues(std::vector<std::string>& possibleValues);
   void printDefaultHelp(ArgumentExecuter<IOCMD>& argumentContainer, bool verboseRequested);
   void printHelpAlgorithm(IOCMD& io, const std::string& algorithm, bool verboseRequested);

public:
   CMDInterface(int argc, char** argv);
   void parse();
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
   class INFERENCETYPES
>
inline CMDInterface<GM, INFERENCETYPES>::CMDInterface(int argc, char** argv)
: commandlineIO_(argc, argv), argumentContainer_(commandlineIO_, 50) {
   fillPossibleValues<INFERENCETYPES>(possibleAlgorithmValues_);

   help_ = &argumentContainer_.addArgument(BoolArgument(helpRequested_, "h", "help", "used to activate help output"));
   verbose_ = &argumentContainer_.addArgument(BoolArgument(verboseRequested_, "v", "verbose", "used to activate verbose output"));
   modelinfo_ = &argumentContainer_.addArgument(BoolArgument(modelinfoRequested_, "", "modelinfo", "used to print detailed informations about the specified model"));
   model_ = &argumentContainer_.addArgument(StringArgument<>(selectedModel_, "m", "model", "used to specify the desired model. Usage: filename:dataset", true));
   algorithm_ = &argumentContainer_.addArgument(StringArgument<>(selectedAlgorithm_, "a", "algorithm", "used to specify the desired algorithm", false, possibleAlgorithmValues_));

   // FIXME
   // outputfile_ = &argumentContainer.addArgument(StringArgument<>(selectedOutputfile, "o", "outputfile", "used to specify the desired outputfile for the computed results", "PRINT ON SCREEN"));
   // will call wrong constructor from StringArgument<>. Must use the following:
   outputfile_ = &argumentContainer_.addArgument(StringArgument<>(selectedOutputfile_, "o", "outputfile", "used to specify the desired outputfile for the computed results", std::string("PRINT ON SCREEN")));
   // end FIXME

   evaluate_ = &argumentContainer_.addArgument(VectorArgument< std::vector<size_t> >(evaluateVec_, "e", "evaluate", "used to evaluate the selected model with the specified input file", false));
}

template <
   class GM,
   class INFERENCETYPES
>
inline void CMDInterface<GM, INFERENCETYPES>::parse() {
   // Check command options and proceed with appropriate behavior
   commandlineIO_.read(*help_);

   if(help_->getValue()) {
      commandlineIO_.read(*algorithm_);
      commandlineIO_.read(*verbose_);
      if(algorithm_->isSet()) {
         printHelpAlgorithm(commandlineIO_, algorithm_->getValue(), true);
      } else {
         printDefaultHelp(argumentContainer_, true);
      }
   } else {

      // Pass through necessary arguments.
      typedef ParserChainManageModel<IOCMD, GM, INFERENCETYPES> parserChainStart;
      parserChainStart::io_ = &commandlineIO_;
      parserChainStart::verbose_ = verbose_;
      parserChainStart::modelinfo_ = modelinfo_;
      parserChainStart::model_ = model_;
      parserChainStart::evaluate_ = evaluate_;
      parserChainStart::algorithm_ = algorithm_;
      parserChainStart::outputfile_ = outputfile_;

      // Start workflow chain with management of selected model type.
      parserChainStart::execute();
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
   class INFERENCETYPES
>
template <class TYPELIST>
inline void CMDInterface<GM, INFERENCETYPES>::fillPossibleValues(std::vector<std::string>& possibleValues) {
   const size_t length = opengm::meta::LengthOfTypeList<TYPELIST>::value;
   getPossibleStringValues<TYPELIST, 0, length, opengm::meta::EqualNumber<length, 0>::value>::fill(possibleValues);
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
   class INFERENCETYPES
>
inline void CMDInterface<GM, INFERENCETYPES>::printDefaultHelp(ArgumentExecuter<IOCMD>& argumentContainer, bool verboseRequested) {
   std::cout << "This is the help for the commandline interface of the opengm library" << std::endl;
   std::cout << "Usage: opengm [arguments]" << std::endl;
   std::cout << "arguments:" << std::endl;
   std::cout << std::setw(12) << std::left << "  short name" << std::setw(29) << std::left << "  long name" << std::setw(8) << std::left << "needed" << "description" << std::endl;
   argumentContainer.printHelp(std::cout, verboseRequested);
}

template <
   class GM,
   class INFERENCETYPES
>
inline void CMDInterface<GM, INFERENCETYPES>::printHelpAlgorithm(IOCMD& io, const std::string& algorithm, bool verboseRequested) {
   // Print help of selected algorithm.

   // Pass through necessary arguments.
   printHelpInferenceAlgorithm<IOCMD>::io_ = &io;
   printHelpInferenceAlgorithm<IOCMD>::verbose_ = verboseRequested;

   // Determine length of type list.
   const size_t length = opengm::meta::LengthOfTypeList<INFERENCETYPES>::value;

   // Call print help of selected algorithm
   executeWithTypeFromString<printHelpInferenceAlgorithm<IOCMD>, INFERENCETYPES, 0, length, opengm::meta::EqualNumber<length, 0>::value>::execute(algorithm);
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


} // namespace interface

} // namespace opengm

#endif /* CMD_INTERFACE_HXX_ */
