#ifndef INFERENCE_CALLER_BASE_HXX_
#define INFERENCE_CALLER_BASE_HXX_

#include "../argument/argument.hxx"
#include "../argument/argument_executer.hxx"
#include "../io/io_base.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class InferenceCallerBase {
protected:
   std::string inferenceParserName_;
   std::string inferenceParserDescription_;
   IO& io_;
   ArgumentExecuter<IO> argumentContainer_;
   size_t protocolationInterval_;
   bool stateOutputRequested_;
   bool valueOutputRequested_;
   bool timingOutputRequested_;
   Size_TArgument<>* protocolate_;
   BoolArgument* stateOutput_;
   BoolArgument* valueOutput_;
   BoolArgument* timingOutput_;
   template <typename ARG>
   void addArgument(const ARG& argument);
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) = 0;
   template <class VISITOR>
   void protocolate(const VISITOR& visitor, const std::string& out) const;
   template <class VECTOR>
   void store(const VECTOR& vec, const std::string& filename, const std::string& dataset) const;
   template <class INF, class VISITOR, class PARAMETER>
   void infer(GM& model, StringArgument<>& outputfile, const bool verbose, const PARAMETER& param) const;
public:
   typedef GM GraphicalModelType;
   typedef ACC AccumulationType;
   InferenceCallerBase(const std::string& InferenceParserNameIn, const std::string& inferenceParserDescriptionIn, IO& ioIn, const size_t maxNumArguments = 20);
   const std::string& getInferenceParserName();
   void printHelp(bool verboseRequested);
   void run(GM& model, StringArgument<>& outputfile, const bool verbose);
};

template <class IO, class GM, class ACC>
inline InferenceCallerBase<IO, GM, ACC>::InferenceCallerBase(const std::string& InferenceParserNameIn, const std::string& inferenceParserDescriptionIn, IO& ioIn, const size_t maxNumArguments)
  : inferenceParserName_(InferenceParserNameIn), inferenceParserDescription_(inferenceParserDescriptionIn), io_(ioIn), argumentContainer_(io_, maxNumArguments) {
   protocolate_ = &argumentContainer_.addArgument(Size_TArgument<>(protocolationInterval_, "p", "protocolate", "used to enable protocolation mode. Usage: \"-p N\" where every Nth iteration step will be protocoled. If N = 0 only the final results will be protocoled.", size_t(0)));
}

template <class IO, class GM, class ACC>
inline const std::string& InferenceCallerBase<IO, GM, ACC>::getInferenceParserName() {
  return inferenceParserName_;
}

template <class IO, class GM, class ACC>
inline void InferenceCallerBase<IO, GM, ACC>::printHelp(bool verboseRequested) {
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

template <class IO, class GM, class ACC>
inline void InferenceCallerBase<IO, GM, ACC>::run(GM& model,StringArgument<>& outputfile, const bool verbose) {
   argumentContainer_.read();
   // check if selected outputfile already exists and move file
   io_.read(outputfile);
   if(outputfile.isSet()) {
      if(io_.fileExists(outputfile.getValue())) {
         std::cout << "output file already exists, moving old file to " << outputfile.getValue() << "~!" << std::endl;
         if(rename(outputfile.getValue().c_str(), std::string(outputfile.getValue() + "~").c_str()) != 0) {
            throw(RuntimeError("Failed to rename file."));
         }
      }
   }
   runImpl(model, outputfile, verbose);
}

template <class IO, class GM, class ACC>
template <typename ARG>
inline void InferenceCallerBase<IO, GM, ACC>::addArgument(const ARG& argument) {
   argumentContainer_.addArgument(argument);
}

template <class IO, class GM, class ACC>
template <class VISITOR>
inline void InferenceCallerBase<IO, GM, ACC>::protocolate(const VISITOR& visitor, const std::string& out) const {
   if(protocolate_->isSet()) {
      /*
      std::cout << "storing states in file: " << out << std::endl;
      throw("visitor does not support state protocolation");
      //store(visitor.getStates(), out, "states");
      */

      std::cout << "storing values in file: " << out << std::endl;
      store(visitor.getValues(), out, "values");
      std::cout << "storing bounds in file: " << out << std::endl;
      store(visitor.getBounds(), out, "bounds");

      std::cout << "storing times in file: " << out << std::endl;
      std::vector<double> times = visitor.getTimes();
      for(size_t i=1;i<times.size();++i)
         times[i] += times[i-1];
      store(times, out, "times");

      const typename VISITOR::LogMapType& logs = visitor.getLogsMap();

      for(typename VISITOR::LogMapType::const_iterator iter = logs.begin(); iter != logs.end(); iter++) {
         std::cout << "storing " << iter->first << " in file: " << out << std::endl;
         store(iter->second, out, iter->first);
      }

      std::cout << "storing corresponding iterations in file: " << out << std::endl;
      store(visitor.getIterations(), out, "iterations");
   }
}

template <class IO, class GM, class ACC>
template <class VECTOR>
inline void InferenceCallerBase<IO, GM, ACC>::store(const VECTOR& vec, const std::string& filename, const std::string& dataset) const {
   std::string storage = filename;
   if(filename != "PRINT ON SCREEN") {
      storage += ":";
      storage += dataset;
   }
   io_.storeVector(storage, vec);
}

template <class IO, class GM, class ACC>
template <class INF, class VISITOR, class PARAMETER>
inline void InferenceCallerBase<IO, GM, ACC>::infer(GM& model, StringArgument<>& outputfile, const bool verbose, const PARAMETER& param) const {
   INF inference(model, param);

   if(protocolate_->isSet()) {
      if(protocolate_->getValue() != 0) {
         VISITOR visitor(protocolate_->getValue(), 0, verbose);
         if(!(inference.infer(visitor) == NORMAL)) {
            std::string error(inference.name() + " did not solve the problem.");
            io_.errorStream() << error << std::endl;
            throw RuntimeError(error);
         }
         if(outputfile.isSet()) {
            protocolate(visitor, outputfile.getValue());
         }
      } else {

      }
   } else {
      if(!(inference.infer() == NORMAL)) {
         std::string error(inference.name() + " did not solve the problem.");
         io_.errorStream() << error << std::endl;
         throw RuntimeError(error);
      }
   }
   if(outputfile.isSet()) {

      std::vector<size_t> states;
      if(!(inference.arg(states) == NORMAL)) {
         std::string error(inference.name() + " could not return optimal argument.");
         io_.errorStream() << error << std::endl;
         throw RuntimeError(error);
      }

      std::cout << "storing optimal states in file: " << outputfile.getValue() << std::endl;
      store(states, outputfile.getValue(), "states");
   }
}

} // namespace interface

} // namespace opengm

#endif /* INFERENCE_CALLER_BASE_HXX_ */
