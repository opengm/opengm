#ifndef GRAPHCUT_CALLER_HXX_
#define GRAPHCUT_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif

#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

namespace opengm {

namespace interface {

struct GraphCutCallerStandAllone {
   template <class MINSTCUT, class GM>
   void runImplHelper(GM& model, StringArgument<>& outputfile, const bool verbose) {
      throw RuntimeError("Method runImplHelper() from class GraphCutCallerStandAllone should never be called");
   }
   const static std::string name_;
};
const std::string GraphCutCallerStandAllone::name_ = "This name should never be shown";

// using crtp to allow graph cut caller to be used with other caller (e.g. AlphaExpansionCaller)
template <class IO, class GM, class ACC, class CHILD = GraphCutCallerStandAllone>
class GraphCutCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:

   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   template <class MINSTCUT>
   void runImplHelper(GM& model, StringArgument<>& outputfile, const bool verbose);
   void callChild(GM& model, StringArgument<>& outputfile, const bool verbose);
   double scale_;
   std::string selectedMinSTCut_;
public:
   const static std::string name_;
   GraphCutCaller(IO& ioIn, const std::string& nameIn = name_, const std::string& descriptionIn = "detailed description of GraphCut caller...");
};

template <class IO, class GM, class ACC, class CHILD>
inline GraphCutCaller<IO, GM, ACC, CHILD>::GraphCutCaller(IO& ioIn, const std::string& nameIn, const std::string& descriptionIn)
   : InferenceCallerBase<IO, GM, ACC>(nameIn, descriptionIn, ioIn) {
#ifndef WITH_MAXFLOW
#ifndef WITH_BOOST
   throw RuntimeError("Unable to use Graph Cut. Recompile with \"WITH_BOOST\" or \"WITH_MAXFLOW\" enabled.");
#endif
#endif

   addArgument(DoubleArgument<>(scale_, "", "scale", "Add description for scale here!!!!.", 1.0));
   std::vector<std::string> possibleMinSTCuts;
#ifdef WITH_MAXFLOW
   possibleMinSTCuts.push_back("KOLMOGOROV");
#endif
#ifdef WITH_BOOST
   possibleMinSTCuts.push_back("BOOST_KOLMOGOROV");
   possibleMinSTCuts.push_back("BOOST_PUSH_RELABEL");
   possibleMinSTCuts.push_back("BOOST_EDMONDS_KARP");
#endif
   addArgument(StringArgument<>(selectedMinSTCut_, "mf", "maxflow", "Add description for MinSTCut here!!!!.", possibleMinSTCuts.front(), possibleMinSTCuts));
}

template <class IO, class GM, class ACC, class CHILD>
inline void GraphCutCaller<IO, GM, ACC, CHILD>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   if(meta::Compare<CHILD, GraphCutCallerStandAllone>::value) {
      std::cout << "running GraphCut caller" << std::endl;
   } else {
      std::cout << "running " << CHILD::name_ << " caller" << std::endl;
   }


#ifdef WITH_MAXFLOW
   if(selectedMinSTCut_ == "KOLMOGOROV") {
      typedef opengm::external::MinSTCutKolmogorov<size_t, typename GM::ValueType> MinStCutType;
      if(meta::Compare<CHILD, GraphCutCallerStandAllone>::value) {
         runImplHelper<MinStCutType>(model, outputfile, verbose);
      } else {
         reinterpret_cast<CHILD*>(this)->runImplHelper<MinStCutType>(model, outputfile, verbose);
      }
   } else
#endif
#ifdef WITH_BOOST
   if(selectedMinSTCut_ == "BOOST_PUSH_RELABEL") {
      typedef opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::PUSH_RELABEL> MinStCutType;
      if(meta::Compare<CHILD, GraphCutCallerStandAllone>::value) {
         runImplHelper<MinStCutType>(model, outputfile, verbose);
      } else {
         reinterpret_cast<CHILD*>(this)->runImplHelper<MinStCutType>(model, outputfile, verbose);
      }
   } else if(selectedMinSTCut_ == "BOOST_EDMONDS_KARP") {
      typedef opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::EDMONDS_KARP> MinStCutType;
      if(meta::Compare<CHILD, GraphCutCallerStandAllone>::value) {
         runImplHelper<MinStCutType>(model, outputfile, verbose);
      } else {
         reinterpret_cast<CHILD*>(this)->runImplHelper<MinStCutType>(model, outputfile, verbose);
      }
   }else if(selectedMinSTCut_ == "BOOST_KOLMOGOROV") {
      typedef opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::KOLMOGOROV> MinStCutType;
      if(meta::Compare<CHILD, GraphCutCallerStandAllone>::value) {
         runImplHelper<MinStCutType>(model, outputfile, verbose);
      } else {
         reinterpret_cast<CHILD*>(this)->runImplHelper<MinStCutType>(model, outputfile, verbose);
      }
   } else
#endif
   {
      throw RuntimeError("Unknown Min ST Cut!");
   }
}

template <class IO, class GM, class ACC, class CHILD>
template <class MINSTCUT>
void GraphCutCaller<IO, GM, ACC, CHILD>::runImplHelper(GM& model, StringArgument<>& outputfile, const bool verbose) {
   typedef GraphCut<GM, ACC, MINSTCUT> GraphCut;
   typename GraphCut::Parameter parameter_;
   parameter_.scale_ = scale_;

   typedef typename GraphCut::VerboseVisitorType VerboseVisitorType;
   typedef typename GraphCut::EmptyVisitorType EmptyVisitorType;
   typedef typename GraphCut::TimingVisitorType TimingVisitorType;

   this-> template infer<GraphCut, TimingVisitorType, typename GraphCut::Parameter>(model, outputfile, verbose, parameter_);
/*   GraphCut graphcut(model, parameter_);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(graphcut.infer() == NORMAL)) {
      std::string error("GraphCut did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(graphcut.arg(states) == NORMAL)) {
      std::string error("GraphCut could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);
   std::cout <<" E(x) = " << model.evaluate(states) <<std::endl;
*/
}

template <class IO, class GM, class ACC, class CHILD>
const std::string GraphCutCaller<IO, GM, ACC, CHILD>::name_ = "GRAPHCUT";

} // namespace interface

} // namespace opengm

#endif /* GRAPHCUT_CALLER_HXX_ */
