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

template <class IO, class GM, class ACC>
struct GraphCutCallerStandAllone;

// using crtp to allow graph cut caller to be used with other caller (e.g. AlphaExpansionCaller)
template <class IO, class GM, class ACC, class CHILD = GraphCutCallerStandAllone<IO, GM, ACC> >
class GraphCutCaller : public InferenceCallerBase<IO, GM, ACC, GraphCutCaller<IO, GM, ACC, CHILD> > {
public:
   typedef InferenceCallerBase<IO, GM, ACC, GraphCutCaller<IO, GM, ACC, CHILD> > BaseClass;
#ifdef WITH_MAXFLOW
//   typedef visitors::VerboseVisitor<GraphCut<GM,ACC,opengm::external::MinSTCutKolmogorov<size_t, typename GM::ValueType> > >        VerboseVisitorType;
//   typedef visitors::TimingVisitor<GraphCut<GM,ACC,opengm::external::MinSTCutKolmogorov<size_t, typename GM::ValueType> > >         TimingVisitorType;
//   typedef visitors::EmptyVisitor<GraphCut<GM,ACC,opengm::external::MinSTCutKolmogorov<size_t, typename GM::ValueType> > >          EmptyVisitorType;
#else
#ifdef WITH_BOOST
//   typedef visitors::VerboseVisitor<GraphCut<GM,ACC,opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::PUSH_RELABEL> > >        VerboseVisitorType;
//   typedef visitors::TimingVisitor<GraphCut<GM,ACC,opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::PUSH_RELABEL> > >         TimingVisitorType;
//   typedef visitors::EmptyVisitor<GraphCut<GM,ACC,opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::PUSH_RELABEL> > >          EmptyVisitorType;
#else
#error "Unable to compile GraphCutCaller: Definition \"WITH_BOOST\" or \"WITH_MAXFLOW\" required. "
#endif
#endif
   const static std::string name_;
   GraphCutCaller(IO& ioIn, const std::string& nameIn = name_, const std::string& descriptionIn = "detailed description of GraphCut caller...");
   virtual ~GraphCutCaller();
protected:
   friend class GraphCutCallerStandAllone<IO, GM, ACC>;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
   template <class MINSTCUT>
   void runImplHelper(GM& model, OutputBase& output, const bool verbose);
   void callChild(GM& model, OutputBase& output, const bool verbose);
   double scale_;
   std::string selectedMinSTCut_;
};

template <class IO, class MODEL, class ACC>
struct GraphCutCallerStandAllone {
   typedef GraphCutCaller<IO, MODEL, ACC, GraphCutCallerStandAllone<IO, MODEL, ACC> > BaseClass;
   typedef typename BaseClass::OutputBase OutputBase;
   template <class MINSTCUT, class GM>
   void runImplHelper(GM& model, OutputBase& output, const bool verbose) {
      throw RuntimeError("Method runImplHelper() from class GraphCutCallerStandAllone should never be called");
   }
   const static std::string name_;
};

template <class IO, class MODEL, class ACC>
const std::string GraphCutCallerStandAllone<IO, MODEL, ACC>::name_ = "This name should never be shown";

template <class IO, class GM, class ACC, class CHILD>
inline GraphCutCaller<IO, GM, ACC, CHILD>::GraphCutCaller(IO& ioIn, const std::string& nameIn, const std::string& descriptionIn)
   : BaseClass(nameIn, descriptionIn, ioIn) {
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
inline GraphCutCaller<IO, GM, ACC, CHILD>::~GraphCutCaller() {

}

template <class IO, class GM, class ACC, class CHILD>
inline void GraphCutCaller<IO, GM, ACC, CHILD>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   if(meta::Compare<CHILD, GraphCutCallerStandAllone<IO, GM, ACC> >::value) {
      std::cout << "running GraphCut caller" << std::endl;
   } else {
      std::cout << "running " << CHILD::name_ << " caller" << std::endl;
   }


#ifdef WITH_MAXFLOW
   if(selectedMinSTCut_ == "KOLMOGOROV") {
      typedef opengm::external::MinSTCutKolmogorov<size_t, typename GM::ValueType> MinStCutType;
      if(meta::Compare<CHILD, GraphCutCallerStandAllone<IO, GM, ACC> >::value) {
         runImplHelper<MinStCutType>(model, output, verbose);
      } else {
         reinterpret_cast<CHILD*>(this)-> template runImplHelper<MinStCutType>(model, output, verbose);
      }
   } else
#endif
#ifdef WITH_BOOST
   if(selectedMinSTCut_ == "BOOST_PUSH_RELABEL") {
      typedef opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::PUSH_RELABEL> MinStCutType;
      if(meta::Compare<CHILD, GraphCutCallerStandAllone<IO, GM, ACC> >::value) {
         runImplHelper<MinStCutType>(model, output, verbose);
      } else {
         reinterpret_cast<CHILD*>(this)-> template runImplHelper<MinStCutType>(model, output, verbose);
      }
   } else if(selectedMinSTCut_ == "BOOST_EDMONDS_KARP") {
      typedef opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::EDMONDS_KARP> MinStCutType;
      if(meta::Compare<CHILD, GraphCutCallerStandAllone<IO, GM, ACC> >::value) {
         runImplHelper<MinStCutType>(model, output, verbose);
      } else {
         reinterpret_cast<CHILD*>(this)-> template runImplHelper<MinStCutType>(model, output, verbose);
      }
   }else if(selectedMinSTCut_ == "BOOST_KOLMOGOROV") {
      typedef opengm::MinSTCutBoost<size_t, typename GM::ValueType, opengm::KOLMOGOROV> MinStCutType;
      if(meta::Compare<CHILD, GraphCutCallerStandAllone<IO, GM, ACC> >::value) {
         runImplHelper<MinStCutType>(model, output, verbose);
      } else {
         reinterpret_cast<CHILD*>(this)-> template runImplHelper<MinStCutType>(model, output, verbose);
      }
   } else
#endif
   {
      throw RuntimeError("Unknown Min ST Cut!");
   }
}

template <class IO, class GM, class ACC, class CHILD>
template <class MINSTCUT>
void GraphCutCaller<IO, GM, ACC, CHILD>::runImplHelper(GM& model, OutputBase& output, const bool verbose) {
   typedef GraphCut<GM, ACC, MINSTCUT> GraphCut;
   typename GraphCut::Parameter parameter_;
   parameter_.scale_ = scale_;

   typedef typename GraphCut::VerboseVisitorType VerboseVisitorType;
   typedef typename GraphCut::EmptyVisitorType EmptyVisitorType;
   typedef typename GraphCut::TimingVisitorType TimingVisitorType;

   this-> template infer<GraphCut, TimingVisitorType, typename GraphCut::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC, class CHILD>
const std::string GraphCutCaller<IO, GM, ACC, CHILD>::name_ = "GRAPHCUT";

} // namespace interface

} // namespace opengm

#endif /* GRAPHCUT_CALLER_HXX_ */
