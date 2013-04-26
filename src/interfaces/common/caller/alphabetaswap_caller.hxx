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
public:
   typedef GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> > BaseClass;
   const static std::string name_;
   AlphaBetaSwapCaller(IO& ioIn);
   virtual ~AlphaBetaSwapCaller();

   friend class GraphCutCaller<IO, GM, ACC, AlphaBetaSwapCaller<IO, GM, ACC> >;
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::scale_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   template <class MINSTCUT>
   void runImplHelper(GM& model, OutputBase& output, const bool verbose);
   size_t maximalNumberOfIterations_;
};

template <class IO, class GM, class ACC>
inline AlphaBetaSwapCaller<IO, GM, ACC>::AlphaBetaSwapCaller(IO& ioIn)
   : BaseClass(ioIn, name_, "detailed description of AlphaBetaSwap caller...") {
   addArgument(Size_TArgument<>(maximalNumberOfIterations_, "", "maxIt", "Maximum number of iterations.", (size_t)1000));
}

template <class IO, class GM, class ACC>
inline AlphaBetaSwapCaller<IO, GM, ACC>::~AlphaBetaSwapCaller() {

}

template <class IO, class GM, class ACC>
template <class MINSTCUT>
void AlphaBetaSwapCaller<IO, GM, ACC>::runImplHelper(GM& model, OutputBase& output, const bool verbose) {
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

   this-> template infer<AlphaBetaSwap, TimingVisitorType, typename AlphaBetaSwap::Parameter>(model, output, verbose, alphabetaswapparameter);
}

template <class IO, class GM, class ACC>
const std::string AlphaBetaSwapCaller<IO, GM, ACC>::name_ = "ALPHABETASWAP";

} // namespace interface

} // namespace opengm

#endif /* ALPHABETASWAP_CALLER_HXX_ */
