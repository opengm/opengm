#ifndef PARTITION_MOVE_CALLER_HXX_
#define PARTITION_MOVE_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/partition-move.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class PartitionMoveCaller : public InferenceCallerBase<IO, GM, ACC, PartitionMoveCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::PartitionMove<GM, ACC> InfType;
   typedef InferenceCallerBase<IO, GM, ACC, PartitionMoveCaller<IO, GM, ACC> > BaseClass;
   typedef typename InfType::VerboseVisitorType VerboseVisitorType;
   typedef typename InfType::EmptyVisitorType EmptyVisitorType;
   typedef typename InfType::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   PartitionMoveCaller(IO& ioIn);
   virtual ~PartitionMoveCaller();

protected: 
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

 
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
   typename InfType::Parameter parameter_;
};

template <class IO, class GM, class ACC>
inline PartitionMoveCaller<IO, GM, ACC>::PartitionMoveCaller(IO& ioIn)
   : BaseClass("PartitionMove", "detailed description of PartitionMove caller...", ioIn) {
   ;
}

template <class IO, class GM, class ACC>
inline  PartitionMoveCaller<IO, GM, ACC>::~PartitionMoveCaller() {

}

template <class IO, class GM, class ACC>
inline void PartitionMoveCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Partition Move caller" << std::endl;

   this-> template infer<InfType, TimingVisitorType, typename InfType::Parameter>(model, output, verbose, parameter_);

}

template <class IO, class GM, class ACC>
const std::string PartitionMoveCaller<IO, GM, ACC>::name_ = "PARTITIONMOVE";

} // namespace interface

} // namespace opengm

#endif /* PARTITION_MOVE_CALLER_HXX_ */
