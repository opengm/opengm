#ifndef QPBO_CALLER_HXX_
#define QPBO_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/qpbo.hxx>
#include <opengm/inference/graphcut.hxx>

//#include "inference_caller_base.hxx"
#include "graphcut_caller.hxx"
#include "../argument/argument.hxx"

#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif

#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

#ifdef WITH_QPBO
#  include <opengm/inference/external/qpbo.hxx>
#endif

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class QPBOCaller : public GraphCutCaller<IO, GM, ACC, QPBOCaller<IO, GM, ACC> > {
public:
   typedef GraphCutCaller<IO, GM, ACC, QPBOCaller<IO, GM, ACC> > BaseClass;
   const static std::string name_;
   QPBOCaller(IO& ioIn);
   virtual ~QPBOCaller();

   friend class GraphCutCaller<IO, GM, ACC, QPBOCaller<IO, GM, ACC> >;
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::scale_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   template <class MINSTCUT>
   void runImplHelper(GM& model, OutputBase& output, const bool verbose);
   template <class QPBO>
   void runFinal(GM& model, OutputBase& output, const typename QPBO::Parameter& param, const bool verbose);
#ifdef WITH_QPBO
   bool externQPBO_;
   typedef opengm::external::QPBO<GM>  QPBO_EXTERN;
   typename QPBO_EXTERN::Parameter qpbo_externparameter;
   bool noStrongPersistency;

#endif
};

template <class IO, class GM, class ACC>
inline QPBOCaller<IO, GM, ACC>::QPBOCaller(IO& ioIn)
   : BaseClass(ioIn, name_, "detailed description of QPBO caller...") {
#ifdef WITH_QPBO
   addArgument(BoolArgument(externQPBO_, "", "extern", "Use extern QPBO. If extern QPBO is used only the arguments labeled for use with extern QPBO are considered."));
   addArgument(BoolArgument(qpbo_externparameter.strongPersistency_, "", "nostrong", "For use with extern QPBO: don't use strong persistency"));
   //addArgument(BoolArgument(qpbo_externparameter.useImproveing_, "", "improve", "For use with extern QPBO: use improving"));
   //addArgument(BoolArgument(qpbo_externparameter.useProbeing_, "", "probe", "For use with extern QPBO: use probing"));
   addArgument(VectorArgument<std::vector<size_t> >(qpbo_externparameter.label_, "", "label", "For use with extern QPBO: location of the file containing the initial configuration for improving"));
#endif
   //TODO remove scale argument which comes from GraphCutCaller but is unnecessary here
}

template <class IO, class GM, class ACC>
inline QPBOCaller<IO, GM, ACC>::~QPBOCaller() {

}

template <class IO, class GM, class ACC>
template <class MINSTCUT>
void QPBOCaller<IO, GM, ACC>::runImplHelper(GM& model, OutputBase& output, const bool verbose) {
#ifdef WITH_QPBO
   if(externQPBO_) {
      qpbo_externparameter.strongPersistency_ = !noStrongPersistency;
      runFinal<QPBO_EXTERN>(model, output, qpbo_externparameter, verbose);
   } else {
#endif
   typedef QPBO<GM, MINSTCUT> QPBO;
   typename QPBO::Parameter qpboparameter;
   runFinal<QPBO>(model, output, qpboparameter, verbose);
#ifdef WITH_QPBO
   }
#endif
}

template <class IO, class GM, class ACC>
template <class QPBO>
void QPBOCaller<IO, GM, ACC>::runFinal(GM& model, OutputBase& output, const typename QPBO::Parameter& param, const bool verbose) {
   typedef typename QPBO::VerboseVisitorType VerboseVisitorType;
   typedef typename QPBO::EmptyVisitorType EmptyVisitorType;
   typedef typename QPBO::TimingVisitorType TimingVisitorType;

   this-> template infer<QPBO, TimingVisitorType, typename QPBO::Parameter>(model, output, verbose, param);
}

template <class IO, class GM, class ACC>
const std::string QPBOCaller<IO, GM, ACC>::name_ = "QPBO";

} // namespace interface

} // namespace opengm

#endif /* QPBO_CALLER_HXX_ */
