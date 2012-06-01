#ifndef ALPHABETASWAP_CALLER_HXX_
#define ALPHABETASWAP_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/graphcut.hxx>

#include "graphcut_caller.hxx"
#include "../argument/argument.hxx"

#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif

#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class AlphaBetaSwapCaller : public GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> > {
protected:

   using GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> >::addArgument;
   using GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> >::io_;
   using GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> >::scale_;
   using GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> >::infer;
   template <class MINSTCUT>
   void runImplHelper(GM& model, StringArgument<>& outputfile, const bool verbose);
   size_t maximalNumberOfIterations_;
public:
   const static std::string name_;
   AlphaBetaSwapCaller(IO& ioIn);

   friend class GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> >;
};

template <class IO, class GM, class ACC>
inline AlphaBetaSwapCaller<IO, GM, ACC>::AlphaBetaSwapCaller(IO& ioIn)
   : GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> >(ioIn, name_, "detailed description of AlphaBetaSwap caller...") {
   addArgument(Size_TArgument<>(maximalNumberOfIterations_, "", "maxIt", "Maximum number of iterations.", (size_t)1000));
}

template <class IO, class GM, class ACC>
template <class MINSTCUT>
void AlphaBetaSwapCaller<IO, GM, ACC>::runImplHelper(GM& model, StringArgument<>& outputfile, const bool verbose) {
   typedef GraphCut<GM, ACC, MINSTCUT> GraphCut;
   typename GraphCut::Parameter graphcutparameter;
   graphcutparameter.scale_ = scale_;

   typedef AlphaBetaSwap<GM, GraphCut> AlphaBetaSwap;
   typename AlphaBetaSwap::Parameter alphabetaswapparameter;
   alphabetaswapparameter.parameter_ = graphcutparameter;
   alphabetaswapparameter.maxNumberOfIterations_ = maximalNumberOfIterations_;

   typedef typename AlphaBetaSwap::VerboseVisitorType VerboseVisitorType;
   typedef typename AlphaBetaSwap::EmptyVisitorType EmptyVisitorType;
   typedef typename AlphaBetaSwap::TimingVisitorType TimingVisitorType;

   this-> template infer<AlphaBetaSwap, TimingVisitorType, typename AlphaBetaSwap::Parameter>(model, outputfile, verbose, alphabetaswapparameter);
/*   AlphaBetaSwap alphabetaswap(model, alphabetaswapparameter);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(alphabetaswap.infer() == NORMAL)) {
      std::string error("AlphaBetaSwap did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(alphabetaswap.arg(states) == NORMAL)) {
      std::string error("AlphaBetaSwap could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);*/
}

template <class IO, class GM, class ACC>
const std::string AlphaBetaSwapCaller<IO, GM, ACC>::name_ = "ALPHABETASWAP";

} // namespace interface

} // namespace opengm

#endif /* ALPHABETASWAP_CALLER_HXX_ */
