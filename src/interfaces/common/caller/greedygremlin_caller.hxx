#ifndef GREEDYGREMLIN_CALLER_HXX_
#define GREEDYGREMLIN_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/greedygremlin.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class GreedyGremlinCaller : public InferenceCallerBase<IO, GM, ACC, GreedyGremlinCaller<IO, GM, ACC> > {
public:
   typedef InferenceCallerBase<IO, GM, ACC, GreedyGremlinCaller<IO, GM, ACC> > BaseClass;
   typedef GreedyGremlin<GM, ACC> GG;
   typedef typename GG::VerboseVisitorType VerboseVisitorType;
   typedef typename GG::EmptyVisitorType EmptyVisitorType;
   typedef typename GG::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   GreedyGremlinCaller(IO& ioIn);
   virtual ~GreedyGremlinCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename GG::Parameter ggParameter_;
};

template <class IO, class GM, class ACC>
inline GreedyGremlinCaller<IO, GM, ACC>::GreedyGremlinCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of GreedyGremlin caller...", ioIn) {
}

template <class IO, class GM, class ACC>
inline GreedyGremlinCaller<IO, GM, ACC>::~GreedyGremlinCaller() {

}

template <class IO, class GM, class ACC>
inline void GreedyGremlinCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running GreedyGremlin caller" << std::endl;
   this-> template infer<GG, TimingVisitorType, typename GG::Parameter>(model, output, verbose, ggParameter_);
}

template <class IO, class GM, class ACC>
const std::string GreedyGremlinCaller<IO, GM, ACC>::name_ = "GreedyGremlin";

} // namespace interface

} // namespace opengm

#endif /* GREEDYGREMLIN_CALLER_HXX_ */
