#ifndef IO_CMD_HXX_
#define IO_CMD_HXX_

#include <iostream>
#include <map>
#include <cstdlib>

#include "../common/io/io_base.hxx"

// set argument delimiter for commandline interface
// this has to be done befor "../common/argument/argument.hxx" is included
#include "../common/argument/argument_delimiter.hxx"
const std::string opengm::interface::ArgumentBaseDelimiter::delimiter_ = "-";

#include "../common/argument/argument.hxx"

namespace opengm {

namespace interface {

/********************
 * class definition *
 ********************/
class IOCMD : public IOBase {
protected:
   typedef std::map<std::string, std::string> CommandMap;
   CommandMap userInput_;
   typedef std::pair<std::string, std::string> CommandPair;
   template <class ARGUMENT>
   const std::string getCommandOption(const ARGUMENT& command);
public:
   IOCMD(int argc, char** argv);

   template <class ARGUMENT>
   bool read(const ARGUMENT& command) { return IOBase::read<ARGUMENT>(command); }
   template <class VECTORTYPE, class CONTAINER>
   bool read(const VectorArgument<VECTORTYPE, CONTAINER>& command);
   template <class MARRAY, class CONTAINER>
   bool read(const MArrayArgument<MARRAY, CONTAINER>& command);

   template <class ARGUMENT>
   bool write(const ARGUMENT& command) { return IOBase::write<ARGUMENT>(command); }

   template <class ARGUMENT>
   bool info(const ARGUMENT& command) { return IOBase::info<ARGUMENT>(command); }
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

inline IOCMD::IOCMD(int argc, char** argv) : IOBase(std::cout, std::cerr, std::clog) {
   for(int i = 1; i < argc; i++){
      const std::string currentArgument(argv[i]);
      if(currentArgument.at(0) == '-') {
         if(i == argc - 1) {
            userInput_.insert(CommandPair(currentArgument, (std::string)" "));
         } else {
            userInput_.insert(CommandPair(currentArgument, (std::string)argv[i + 1]));
         }
      }
   }
}

template <class ARGUMENT>
inline const std::string IOCMD::getCommandOption(const ARGUMENT& command) {
   std::string searchString = ArgumentBaseDelimiter::delimiter_;
   searchString += command.getShortName();
   if(searchString != ArgumentBaseDelimiter::delimiter_) {
      CommandMap::iterator iter = userInput_.find(searchString);
      if(iter != userInput_.end()) {
         return iter->second;
      }
   }
   searchString = ArgumentBaseDelimiter::delimiter_ + ArgumentBaseDelimiter::delimiter_;
   searchString += command.getLongName();
   if(searchString != ArgumentBaseDelimiter::delimiter_ + ArgumentBaseDelimiter::delimiter_) {
      CommandMap::iterator iter = userInput_.find(searchString);
      if(iter != userInput_.end()) {
         return iter->second;
      }
   }
   return "";
}

template <>
inline bool IOCMD::read(const BoolArgument& command) {
   bool isSet = false;
   if(!getCommandOption(command).empty()) {
      isSet = true;
   }
   command(isSet, isSet);
   return isSet;
}

template <>
inline bool IOCMD::read(const ArgumentBase<std::string>& command) {
   const std::string commandOption(getCommandOption(command));
   if(!commandOption.empty() && !(commandOption == " ")) {
      command(commandOption, true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOCMD::read(const StringArgument<>& command) {
   return read(static_cast<const ArgumentBase<std::string>& >(command));
}

template <>
inline bool IOCMD::read(const ArgumentBase<int>& command) {
   const std::string commandOption(getCommandOption(command));
   if(!commandOption.empty() && !(commandOption == " ")) {
      command(atoi(commandOption.c_str()), true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOCMD::read(const IntArgument<>& command) {
   return read(static_cast<const ArgumentBase<int>& >(command));
}

template <>
inline bool IOCMD::read(const ArgumentBase<opengm::UInt32Type>& command) {
   const std::string commandOption(getCommandOption(command));
   if(!commandOption.empty() && !(commandOption == " ")) {
      command(atoll(commandOption.c_str()), true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOCMD::read(const ArgumentBase<opengm::UInt64Type>& command) {
   const std::string commandOption(getCommandOption(command));
   if(!commandOption.empty() && !(commandOption == " ")) {
      command(atoll(commandOption.c_str()), true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOCMD::read(const Size_TArgument<>& command) {
   return read(static_cast<const ArgumentBase<size_t>& >(command));
}


template <>
inline bool IOCMD::read(const ArgumentBase<double>& command) {
   const std::string commandOption(getCommandOption(command));
   if(!commandOption.empty() && !(commandOption == " ")) {
      command(atof(commandOption.c_str()), true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOCMD::read(const DoubleArgument<>& command) {
   return read(static_cast<const ArgumentBase<double>& >(command));
}

template <>
inline bool IOCMD::read(const ArgumentBase<float>& command) {
   const std::string commandOption(getCommandOption(command));
   if(!commandOption.empty() && !(commandOption == " ")) {
      command(atof(commandOption.c_str()), true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOCMD::read(const FloatArgument<>& command) {
   return read(static_cast<const ArgumentBase<float>& >(command));
}

template <class VECTORTYPE, class CONTAINER>
inline bool IOCMD::read(const VectorArgument<VECTORTYPE, CONTAINER>& command) {
   const std::string commandOption(getCommandOption(command));
   if(!commandOption.empty() && !(commandOption == " ")) {
      loadVector(commandOption, command.getReference());
      command.markAsSet();
      return true;
   } else {
      //return sanityCheck(command);
      return false;
   }
}

template <class MARRAY, class CONTAINER>
inline bool IOCMD::read(const MArrayArgument<MARRAY, CONTAINER>& command) {
   const std::string commandOption(getCommandOption(command));
   if(!commandOption.empty() && !(commandOption == " ")) {
      loadMArray(commandOption, command.getReference());
      command.markAsSet();
      return true;
   } else {
      //return sanityCheck(command);
      return false;
   }
}
} // namespace interface

} // namespace opengm

#endif /* IO_CMD_HXX_ */
