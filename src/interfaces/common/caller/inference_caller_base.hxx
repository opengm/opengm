#ifndef INFERENCE_CALLER_BASE_HXX_
#define INFERENCE_CALLER_BASE_HXX_

#include <opengm/utilities/metaprogramming.hxx>
#include "../argument/argument.hxx"
#include "../argument/argument_executer.hxx"
#include "../io/io_base.hxx"

#ifdef WITH_MATLAB
   #include <mex.h>
   #include "../../matlab/opengm/mex-src/helper/mexHelper.hxx"
#endif

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC, class CHILD>
class InferenceCallerBase {
public:
   typedef GM GraphicalModelType;
   typedef ACC AccumulationType;
   InferenceCallerBase(const std::string& InferenceParserNameIn, const std::string& inferenceParserDescriptionIn, IO& ioIn, const size_t maxNumArguments = 100);
   virtual ~InferenceCallerBase();
   const std::string& getInferenceParserName();
   void printHelp(bool verboseRequested);
   template <class OUTPUTTYPE>
   void run(GM& model, OUTPUTTYPE& outputfile, const bool verbose);
protected:
   class OutputBase {
   public:

      typedef std::map<std::string, std::vector<double> >  ProtocolMapType;
      typedef std::vector<typename GM::LabelType> StatesType;

      virtual void storeProtocolMap(const ProtocolMapType& map) = 0;
      virtual void storeStates(const StatesType& states) = 0;

      OutputBase(IO& ioIn);
      template <class ARGUMENT>
      OutputBase(IO& ioIn, ARGUMENT& outputIn);
      virtual ~OutputBase();
   protected:
      IO& io_;

      template <class VECTOR>
      void store(const VECTOR& vec, const std::string& filename, const std::string& dataset) const;
      void checkFile(const std::string& fileLocation);
      template <class ARGUMENT, class REFERENCETYPE>
      REFERENCETYPE& getReference(ARGUMENT& arg);
   };

   class HDF5Output : public OutputBase {
   public:
      typedef typename OutputBase::ProtocolMapType ProtocolMapType;
      typedef typename OutputBase::StatesType StatesType;

      virtual void storeProtocolMap(const ProtocolMapType& map);
      virtual void storeStates(const StatesType& states);

      HDF5Output(IO& ioIn, StringArgument<>& outputIn);
      template <class ARGUMENT>
      HDF5Output(IO& ioIn, ARGUMENT& outputIn);
      virtual ~HDF5Output();
   protected:
      using OutputBase::io_;
      const std::string* output_;
   };

#ifdef WITH_MATLAB
   class MatlabOutput : public OutputBase {
   public:
      typedef typename OutputBase::ProtocolMapType ProtocolMapType;
      typedef typename OutputBase::StatesType StatesType;

      virtual void storeProtocolMap(const ProtocolMapType& map);
      virtual void storeStates(const StatesType& states);

      MatlabOutput(IO& ioIn, mxArrayArgument<>& outputIn);
      template <class ARGUMENT>
      MatlabOutput(IO& ioIn, ARGUMENT& outputIn);
      virtual ~MatlabOutput();
   protected:
      using OutputBase::io_;
      mxArray* output_;
      std::string fileLocation_;
      bool fileOutput_;
      template <class VECTYPE>
      mxArray* copyValues2mxArray(const VECTYPE& values);
   };
#endif

   std::string inferenceParserName_;
   std::string inferenceParserDescription_;
   IO& io_;
   ArgumentExecuter<IO> argumentContainer_;
   size_t protocolationInterval_;
   Size_TArgument<>* protocolate_;
   double timeLimit_;
   double gapLimit_;
   template <typename ARG>
   void addArgument(const ARG& argument);
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose) = 0;
   template <class VISITOR>
   void protocolate(const VISITOR& visitor, OutputBase& output) const;
   template <class INF, class VISITOR, class PARAMETER>
   void infer(GM& model, OutputBase& output, const bool verbose, const PARAMETER& param) const;
};

template <class IO, class GM, class ACC, class CHILD>
inline InferenceCallerBase<IO, GM, ACC, CHILD>::InferenceCallerBase(const std::string& InferenceParserNameIn, const std::string& inferenceParserDescriptionIn, IO& ioIn, const size_t maxNumArguments)
  : inferenceParserName_(InferenceParserNameIn), inferenceParserDescription_(inferenceParserDescriptionIn), io_(ioIn), argumentContainer_(io_, maxNumArguments) {
   protocolate_ = &argumentContainer_.addArgument(Size_TArgument<>(protocolationInterval_, "p", "protocolate", "used to enable protocolation mode. Usage: \"-p N\" where every Nth iteration step will be protocoled. If N = 0 only the final results will be protocoled.", size_t(0)));
   argumentContainer_.addArgument(DoubleArgument<>(timeLimit_, "", "timeout", "maximal runtime in seconds", std::numeric_limits<double>::infinity()));
   argumentContainer_.addArgument(DoubleArgument<>(gapLimit_, "", "gaplimit", "Inference will terminate if gap between bound and value is smaller or equal to gaplimit", 0.0));
}

template <class IO, class GM, class ACC, class CHILD>
inline InferenceCallerBase<IO, GM, ACC, CHILD>::~InferenceCallerBase() {

}

template <class IO, class GM, class ACC, class CHILD>
inline const std::string& InferenceCallerBase<IO, GM, ACC, CHILD>::getInferenceParserName() {
  return inferenceParserName_;
}

template <class IO, class GM, class ACC, class CHILD>
inline void InferenceCallerBase<IO, GM, ACC, CHILD>::printHelp(bool verboseRequested) {
   io_.standardStream() << "Printing Help for inference caller " << inferenceParserName_ << std::endl;
   io_.standardStream() << "Description:\n" << inferenceParserDescription_ << std::endl;
   if(argumentContainer_.size() != 0) {
      std::cout << "arguments:" << std::endl;
      std::cout << std::setw(12) << std::left << "  short name" << std::setw(29) << std::left << "  long name" << std::setw(8) << std::left << "needed" << "description" << std::endl;
      argumentContainer_.printHelp(io_.standardStream(), verboseRequested);
   } else {
      io_.standardStream() << inferenceParserName_ << " has no arguments." << std::endl;
   }
}

template <class IO, class GM, class ACC, class CHILD>
template <class OUTPUTTYPE>
inline void InferenceCallerBase<IO, GM, ACC, CHILD>::run(GM& model, OUTPUTTYPE& outputfile, const bool verbose) {
   argumentContainer_.read();
   if(meta::Compare<OUTPUTTYPE, StringArgument<> >::value) {
      HDF5Output output(io_, outputfile);
      runImpl(model, output, verbose);
   } else
#ifdef WITH_MATLAB
      if(meta::Compare<OUTPUTTYPE, mxArrayArgument<> >::value) {
      MatlabOutput output(io_, outputfile);
      runImpl(model, output, verbose);
   } else
#endif
   {
      throw RuntimeError("Unsupported Outputtype");
   }
}

template <class IO, class GM, class ACC, class CHILD>
template <typename ARG>
inline void InferenceCallerBase<IO, GM, ACC, CHILD>::addArgument(const ARG& argument) {
   argumentContainer_.addArgument(argument);
}

template <class IO, class GM, class ACC, class CHILD>
template <class VISITOR>
inline void InferenceCallerBase<IO, GM, ACC, CHILD>::protocolate(const VISITOR& visitor, OutputBase& output) const {
   if(protocolate_->isSet()) {
      output.storeProtocolMap(visitor.protocolMap());
   }
}

template <class IO, class GM, class ACC, class CHILD>
template <class INF, class VISITOR, class PARAMETER>
inline void InferenceCallerBase<IO, GM, ACC, CHILD>::infer(GM& model, OutputBase& output, const bool verbose, const PARAMETER& param) const {
   INF* inference = NULL;

   if(protocolate_->isSet()) {
      if(protocolate_->getValue() != 0) {
         VISITOR visitor(protocolate_->getValue(), 0, verbose, true, timeLimit_, gapLimit_);
         inference = new INF(model, param);
         if((inference->infer(visitor) == UNKNOWN)) {
            std::string error(inference->name() + " did not solve the problem.");
            io_.errorStream() << error << std::endl;
            delete inference;
            throw RuntimeError(error);
         }
         protocolate(visitor, output);
      } else {
         inference = new INF(model, param);
         if((inference->infer() == UNKNOWN)) {
            std::string error(inference->name() + " did not solve the problem.");
            io_.errorStream() << error << std::endl;
            delete inference;
            throw RuntimeError(error);
         }
      }
   } else {
      inference = new INF(model, param);
      if((inference->infer() == UNKNOWN)) {
         std::string error(inference->name() + " did not solve the problem.");
         io_.errorStream() << error << std::endl;
         delete inference;
         throw RuntimeError(error);
      }
   }

   std::vector<typename GM::LabelType> states;
   if(!(inference->arg(states) == NORMAL)) {
      std::string error(inference->name() + " could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      delete inference;
      throw RuntimeError(error);
   }

   output.storeStates(states);
   delete inference;
}

template <class IO, class GM, class ACC, class CHILD>
InferenceCallerBase<IO, GM, ACC, CHILD>::OutputBase::OutputBase(IO& ioIn) : io_(ioIn) {

}

template <class IO, class GM, class ACC, class CHILD>
template <class ARGUMENT>
InferenceCallerBase<IO, GM, ACC, CHILD>::OutputBase::OutputBase(IO& ioIn, ARGUMENT& outputIn) : io_(ioIn) {
   throw(RuntimeError("Unsupported output type."));
}

template <class IO, class GM, class ACC, class CHILD>
InferenceCallerBase<IO, GM, ACC, CHILD>::OutputBase::~OutputBase() {

}

template <class IO, class GM, class ACC, class CHILD>
template <class VECTOR>
inline void InferenceCallerBase<IO, GM, ACC, CHILD>::OutputBase::store(const VECTOR& vec, const std::string& filename, const std::string& dataset) const {
   std::string storage = filename;
   storage += ":";
   storage += dataset;
   io_.storeVector(storage, vec);
}

template <class IO, class GM, class ACC, class CHILD>
void InferenceCallerBase<IO, GM, ACC, CHILD>::OutputBase::checkFile(const std::string& fileLocation) {
   if(io_.fileExists(fileLocation)) {
      std::cout << "output file already exists, moving old file to " << fileLocation << "~!" << std::endl;
      if(rename(fileLocation.c_str(), std::string(fileLocation + "~").c_str()) != 0) {
         throw(RuntimeError("Failed to rename file."));
      }
   }
}

template <class IO, class GM, class ACC, class CHILD>
template <class ARGUMENT, class REFERENCETYPE>
REFERENCETYPE& InferenceCallerBase<IO, GM, ACC, CHILD>::OutputBase::getReference(ARGUMENT& arg) {
   io_.read(arg);
   return arg.getValue();
}

template <class IO, class GM, class ACC, class CHILD>
InferenceCallerBase<IO, GM, ACC, CHILD>::HDF5Output::HDF5Output(IO& ioIn, StringArgument<>& outputIn) : OutputBase(ioIn), output_(&(  (*this) . template getReference<StringArgument<>, const std::string>(outputIn))) {
   // check if selected outputfile already exists and move file
   this->checkFile(*output_);
}

template <class IO, class GM, class ACC, class CHILD>
template <class ARGUMENT>
InferenceCallerBase<IO, GM, ACC, CHILD>::HDF5Output::HDF5Output(IO& ioIn, ARGUMENT& outputIn) : OutputBase(ioIn, outputIn) {

}

template <class IO, class GM, class ACC, class CHILD>
InferenceCallerBase<IO, GM, ACC, CHILD>::HDF5Output::~HDF5Output() {

}

template <class IO, class GM, class ACC, class CHILD>
void InferenceCallerBase<IO, GM, ACC, CHILD>::HDF5Output::storeProtocolMap(const ProtocolMapType& map) {
   for(typename ProtocolMapType::const_iterator iter = map.begin(); iter != map.end(); iter++) {
      std::cout << "storing " << iter->first << " in file: " << *output_ << std::endl;
      this->store(iter->second, *output_, iter->first);
   }
}

template <class IO, class GM, class ACC, class CHILD>
void InferenceCallerBase<IO, GM, ACC, CHILD>::HDF5Output::storeStates(const StatesType& states) {
   std::cout << "storing optimal states in file: " << *output_ << std::endl;
   this->store(states, *output_, "states");
}

#ifdef WITH_MATLAB
template <class IO, class GM, class ACC, class CHILD>
InferenceCallerBase<IO, GM, ACC, CHILD>::MatlabOutput::MatlabOutput(IO& ioIn, mxArrayArgument<>& outputIn) : OutputBase(ioIn), output_(NULL) {
   io_.read(outputIn);
   output_ = outputIn.getValue();

   // check if mxArray is a string
   if(output_ && mxIsChar(output_)) {
      fileLocation_ = mxArrayToString(output_);
      fileOutput_ = true;
      // check if selected outputfile already exists and move file
      this->checkFile(fileLocation_);
   } else {
      fileLocation_ = "";
      fileOutput_ = false;
      // create empty struct
      const mwSize structDimensions[] = {1, 1};
      outputIn.getReference() = mxCreateStructArray(2, structDimensions, 0, NULL);
      output_ = outputIn.getReference();
   }
}

template <class IO, class GM, class ACC, class CHILD>
template <class ARGUMENT>
InferenceCallerBase<IO, GM, ACC, CHILD>::MatlabOutput::MatlabOutput(IO& ioIn, ARGUMENT& outputIn) : OutputBase(ioIn, outputIn) {

}

template <class IO, class GM, class ACC, class CHILD>
InferenceCallerBase<IO, GM, ACC, CHILD>::MatlabOutput::~MatlabOutput() {

}

template <class IO, class GM, class ACC, class CHILD>
void InferenceCallerBase<IO, GM, ACC, CHILD>::MatlabOutput::storeProtocolMap(const ProtocolMapType& map) {
   if(fileOutput_) {
      for(typename ProtocolMapType::const_iterator iter = map.begin(); iter != map.end(); iter++) {
         std::cout << "storing " << iter->first << " in file: " << fileLocation_ << std::endl;
         this->store(iter->second, fileLocation_, iter->first);
      }
   } else {
	  for(typename ProtocolMapType::const_iterator iter = map.begin(); iter != map.end(); iter++) {
		  int fieldIndex = mxAddField(output_, iter->first.c_str());
		  if(fieldIndex < 0) {
			  throw RuntimeError("mxArrayOutput could not add field");
		  }
		  mxArray* field = copyValues2mxArray(iter->second);
		  mxSetFieldByNumber(output_,0, fieldIndex, field);
	  }
   }
}

template <class IO, class GM, class ACC, class CHILD>
void InferenceCallerBase<IO, GM, ACC, CHILD>::MatlabOutput::storeStates(const StatesType& states) {
   if(fileOutput_) {
      std::cout << "storing optimal states in file: " << fileLocation_ << std::endl;
      this->store(states, fileLocation_, "states");
   } else {
	   int fieldIndex = mxAddField(output_, "states");
	   if(fieldIndex < 0) {
		   throw RuntimeError("mxArrayOutput could not add field");
	   }
	   mxArray* field = copyValues2mxArray(states);
	   mxSetFieldByNumber(output_,0, fieldIndex, field);
   }
}

template <class IO, class GM, class ACC, class CHILD>
template <class VECTYPE>
mxArray* InferenceCallerBase<IO, GM, ACC, CHILD>::MatlabOutput::copyValues2mxArray(const VECTYPE& values) {
   typedef typename VECTYPE::value_type value_type;
   const size_t numElements = values.size();
   // create new mxArray dependent on value_type
   mxArray* storage;
   if(meta::Compare<value_type, float>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxSINGLE_CLASS, mxREAL);
   } else if(meta::Compare<value_type, double>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxDOUBLE_CLASS, mxREAL);
   } else if(meta::Compare<value_type, int8_T>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxINT8_CLASS, mxREAL);
   } else if(meta::Compare<value_type, int16_T>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxINT16_CLASS, mxREAL);
   } else if(meta::Compare<value_type, int32_T>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxINT32_CLASS, mxREAL);
   } else if(meta::Compare<value_type, int64_T>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxINT64_CLASS, mxREAL);
   } else if(meta::Compare<value_type, uint8_T>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxUINT8_CLASS, mxREAL);
   } else if(meta::Compare<value_type, uint16_T>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxUINT16_CLASS, mxREAL);
   } else if(meta::Compare<value_type, uint32_T>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxUINT32_CLASS, mxREAL);
   } else if(meta::Compare<value_type, uint64_T>::value) {
      storage = mxCreateNumericMatrix(1, numElements, mxUINT64_CLASS, mxREAL);
   } else {
      throw RuntimeError("mxArrayOutput can not copy values to mxArray, unsupported value type");
   }

   // copy values
   typedef helper::storeValue<typename VECTYPE::value_type, typename VECTYPE::const_iterator> storeType;
   typedef helper::forAllValues<storeType> storeAllElements;
   typedef helper::getDataFromMXArray<storeAllElements> getAllElements;
   getAllElements getter;
   storeType duplicator(values.begin());
   storeAllElements functor(duplicator);
   getter(functor, storage);

   return storage;
}

#endif

} // namespace interface

} // namespace opengm

#endif /* INFERENCE_CALLER_BASE_HXX_ */
