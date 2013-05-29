#ifndef IO_MATLAB_HXX_
#define IO_MATLAB_HXX_

#include <../src/interfaces/common/io/io_base.hxx>

// set argument delimiter for commandline interface
// this has to be done befor "../common/argument/argument.hxx" is included
#include <../src/interfaces/common/argument/argument_delimiter.hxx>
const std::string opengm::interface::ArgumentBaseDelimiter::delimiter_ = "";

#include <../src/interfaces/common/argument/argument.hxx>

#include "../helper/mexHelper.hxx"

namespace opengm {

namespace interface {

/********************
 * class definition *
 ********************/

class IOMatlab : public IOBase {
public:
   IOMatlab(int nrhs, const mxArray *prhs[]);

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

protected:
   typedef std::map<std::string, const mxArray*> CommandMap;
   CommandMap userInput_;
   typedef std::pair<std::string, const mxArray*> CommandPair;
   template <class ARGUMENT>
   const mxArray* getCommandOption(const ARGUMENT& command);
   template <class VALUETYPE>
   static void getFirstValue(VALUETYPE& storage, const mxArray*& input);
   template <class VECTORTYPE>
   static void copyVector(VECTORTYPE& storage, const size_t numVariables, const mxArray*& input);
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

IOMatlab::IOMatlab(int nrhs, const mxArray *prhs[]) : IOBase(std::cout, std::cerr, std::clog)  {
   for(int i = 0; i < nrhs; i++){
      if(mxIsChar(prhs[i])) {
         std::string currentArgument = mxArrayToString(prhs[i]);
         if(currentArgument.data() == NULL) {
            mexErrMsgTxt("load: could not convert input to string.");
         }
         if(i == nrhs - 1) {
            userInput_.insert(CommandPair(currentArgument, mxCreateString(" ")));
         } else {
            userInput_.insert(CommandPair(currentArgument, prhs[i + 1]));
         }
      }
   }
}

template <class ARGUMENT>
const mxArray* IOMatlab::getCommandOption(const ARGUMENT& command) {
   if(command.getShortName() != "") {
      CommandMap::iterator iter = userInput_.find(command.getShortName());
      if(iter != userInput_.end()) {
         return iter->second;
      }
   }

   if(command.getLongName() != "") {
      CommandMap::iterator iter = userInput_.find(command.getLongName());
      if(iter != userInput_.end()) {
         return iter->second;
      }
   }
   return NULL;
}

template <class VALUETYPE>
void IOMatlab::getFirstValue(VALUETYPE& storage, const mxArray*& input) {
   typedef helper::copyValue<VALUETYPE> copyType;
   typedef helper::forFirstValue<copyType> copyFirstElement;
   typedef helper::getDataFromMXArray<copyFirstElement> getFirstElement;
   getFirstElement getter;
   copyType duplicator(&storage);
   copyFirstElement functor(duplicator);
   getter(functor, input);
}

template <class VECTORTYPE>
void IOMatlab::copyVector(VECTORTYPE& storage, const size_t numVariables, const mxArray*& input) {
   typedef helper::copyValue<typename VECTORTYPE::value_type, typename VECTORTYPE::iterator> copyType;
   typedef helper::forAllValues<copyType> copyAllElements;
   typedef helper::getDataFromMXArray<copyAllElements> getAllElements;
   // rescale vector
   storage.resize(numVariables);
   getAllElements getter;
   copyType duplicator(storage.begin());
   copyAllElements functor(duplicator);
   getter(functor, input);
}

template <>
inline bool IOMatlab::read(const BoolArgument& command) {
   bool isSet = false;
   if(getCommandOption(command) != NULL) {
      isSet = true;
   }
   command(isSet, isSet);
   return isSet;
}

template <>
inline bool IOMatlab::read(const ArgumentBase<std::string>& command) {
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL && mxIsChar(commandOption)) {
      std::string currentArgument = mxArrayToString(commandOption);
      if(currentArgument != " ") {
         command(currentArgument, true);
         return true;
      } else {
         return sanityCheck(command);
      }
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOMatlab::read(const StringArgument<>& command) {
   return read(static_cast<const ArgumentBase<std::string>& >(command));
}

template <>
inline bool IOMatlab::read(const ArgumentBase<int>& command) {
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL && mxIsNumeric(commandOption)) {
      int value;
      getFirstValue(value, commandOption);
      command(value, true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOMatlab::read(const IntArgument<>& command) {
   return read(static_cast<const ArgumentBase<int>& >(command));
}

template <>
inline bool IOMatlab::read(const ArgumentBase<unsigned int>& command) {
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL && mxIsNumeric(commandOption)) {
      unsigned int value;
      getFirstValue(value, commandOption);
      command(value, true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOMatlab::read(const ArgumentBase<size_t>& command) {
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL && mxIsNumeric(commandOption)) {
      size_t value;
      getFirstValue(value, commandOption);
      command(value, true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOMatlab::read(const Size_TArgument<>& command) {
   return read(static_cast<const ArgumentBase<size_t>& >(command));
}


template <>
inline bool IOMatlab::read(const ArgumentBase<double>& command) {
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL && mxIsNumeric(commandOption)) {
      double value;
      getFirstValue(value, commandOption);
      command(value, true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOMatlab::read(const DoubleArgument<>& command) {
   return read(static_cast<const ArgumentBase<double>& >(command));
}

template <>
inline bool IOMatlab::read(const ArgumentBase<float>& command) {
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL && mxIsNumeric(commandOption)) {
      float value;
      getFirstValue(value, commandOption);
      command(value, true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOMatlab::read(const FloatArgument<>& command) {
   return read(static_cast<const ArgumentBase<float>& >(command));
}

template <class VECTORTYPE, class CONTAINER>
inline bool IOMatlab::read(const VectorArgument<VECTORTYPE, CONTAINER>& command) {
   typedef typename VECTORTYPE::value_type value_type;
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL) {
      if(mxIsChar(commandOption)) {
         std::string currentArgument = mxArrayToString(commandOption);
         if(currentArgument != " ") {
            loadVector(currentArgument, command.getReference());
            command.markAsSet();
            return true;
         } else {
            return false;
         }
      } else {
         // deep copy data
         VECTORTYPE& vector = command.getReference();
         const size_t numVariables = mxGetNumberOfElements(commandOption);

         copyVector(vector, numVariables, commandOption);

         command.markAsSet();
         return true;
      }
   } else {
      //return sanityCheck(command);
      return false;
   }
}

template <class MARRAY, class CONTAINER>
inline bool IOMatlab::read(const MArrayArgument<MARRAY, CONTAINER>& command) {
   typedef typename MARRAY::value_type value_type;
   typedef typename MARRAY::index_type index_type;
   typedef typename MARRAY::iterator iterator;
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL) {
      if(mxIsChar(commandOption)) {
         std::string currentArgument = mxArrayToString(commandOption);
         if(currentArgument != " ") {
            loadMArray(currentArgument, command.getReference());
            command.markAsSet();
            return true;
         } else {
            return false;
         }
      } else {
         // deep copy data
         MARRAY& marray = command.getReference();

         // get dimension and resize marray
         const mwSize* shapeBegin = mxGetDimensions(commandOption);
         const mwSize* shapeEnd = mxGetDimensions(commandOption) + mxGetNumberOfDimensions(commandOption);

         marray.resize(shapeBegin, shapeEnd);

         // copy values
         typedef opengm::interface::helper::copyValue<value_type, iterator> copyFunctor;
         typedef opengm::interface::helper::forAllValues<copyFunctor> copyValues;
         typedef opengm::interface::helper::getDataFromMXArray<copyValues> coppyMARRAYValues;
         copyFunctor variableAdder(marray.begin());
         copyValues functor(variableAdder);
         coppyMARRAYValues()(functor, commandOption);

         command.markAsSet();
         return true;
      }
   } else {
      //return sanityCheck(command);
      return false;
   }
}

template <>
inline bool IOMatlab::read(const ArgumentBase<mxArray*>& command) {
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL) {
      command(const_cast<mxArray*>(commandOption), true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOMatlab::read(const mxArrayArgument<>& command) {
   return read(static_cast<const ArgumentBase<mxArray*>& >(command));
}

template <>
inline bool IOMatlab::read(const ArgumentBase<const mxArray*>& command) {
   const mxArray* commandOption(getCommandOption(command));
   if(commandOption != NULL) {
      command(commandOption, true);
      return true;
   } else {
      return sanityCheck(command);
   }
}

template <>
inline bool IOMatlab::read(const mxArrayConstArgument<>& command) {
   return read(static_cast<const ArgumentBase<const mxArray*>& >(command));
}

} // namespace interface

} // namespace opengm
#endif /* IO_MATLAB_HXX_ */
