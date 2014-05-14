#ifndef TRWSI_CALLER_HXX_
#define TRWSI_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/trws/trws_trws.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class TRWSiCaller : public InferenceCallerBase<IO, GM, ACC, TRWSiCaller<IO, GM, ACC> > {
protected:
   typedef InferenceCallerBase<IO, GM, ACC, TRWSiCaller<IO, GM, ACC> > BaseClass;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
 
   typedef TRWSi<GM, ACC> TRWSiType;
   typedef typename TRWSiType::VerboseVisitorType VerboseVisitorType;
   typedef typename TRWSiType::EmptyVisitorType EmptyVisitorType;
   typedef typename TRWSiType::TimingVisitorType TimingVisitorType;
   typename TRWSiType::Parameter trwsParameter_;

   size_t relativePrecision;
   std::string stringDecompositionType;
   size_t slowComputations;
   size_t noNormalization;
   //size_t maxNumberOfIterations;
   size_t treeAgreeMaxStableIter;
public:
   const static std::string name_;
   TRWSiCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline TRWSiCaller<IO, GM, ACC>::TRWSiCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of the internal TRWSi caller...", ioIn) {
	std::vector<size_t> boolVec(2); boolVec[0]=0; boolVec[1]=1;
	std::vector<std::string> stringVec(3); stringVec[0]="GENERAL"; stringVec[1]="GRID";  stringVec[2]="EDGE";
	addArgument(Size_TArgument<>(trwsParameter_.maxNumberOfIterations_, "", "maxIt", "Maximum number of iterations.",true));
	//addArgument(Size_TArgument<>(maxNumberOfIterations, "", "maxIt", "Maximum number of iterations.",true));
	addArgument(DoubleArgument<>(trwsParameter_.precision(), "", "precision", "Duality gap based absolute precision to be obtained. Default is 0.0. Use parameter --relative to select the relative one",(double)0.0));
	addArgument(Size_TArgument<>(relativePrecision, "", "relative", "If set to 1 , then the parameter --precision determines a relative precision value. Default is an absolute one",(size_t)0,boolVec));
	addArgument(DoubleArgument<>(trwsParameter_.minRelativeDualImprovement(), "", "minRelativeDualImprovement", "The minimal improvement of the dual function. If the actual improvement is less, it stops the solver",false));
	addArgument(StringArgument<>(stringDecompositionType, "d", "decomposition", "Select decomposition: GENERAL, GRID or EDGE. Default is GENERAL", false,stringVec));
	addArgument(Size_TArgument<>(slowComputations, "", "slowComputations", "If set to 1 the type of the pairwise factors (Potts for now, will be extended to truncated linear and quadratic) will NOT be used to speed up computations of the solver.",(size_t)0,boolVec));
	addArgument(Size_TArgument<>(noNormalization, "", "noNormalization", "If set to 1 the canonical normalization is NOT used in the solver.",(size_t)0,boolVec));
	addArgument(BoolArgument(trwsParameter_.verbose(), "", "debugverbose", "If set the solver will output debug information to the stdout"));
	//addArgument(Size_TArgument<>(trwsParameter_.treeAgreeMaxStableIter_, "", "treeAgreeMaxStableIter", "Maximum number of iterations after the last improvements of the tree agreement.",false));
	addArgument(Size_TArgument<>(treeAgreeMaxStableIter, "", "treeAgreeMaxStableIter", "Maximum number of iterations after the last improvements of the tree agreement.",(size_t)0));
}

template <class IO, class GM, class ACC>
inline void TRWSiCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running internal TRWSi caller" << std::endl;

   //trwsParameter_.setMaxNumberOfIterations(maxNumberOfIterations);
   trwsParameter_.isAbsolutePrecision()=(relativePrecision==0);
   trwsParameter_.decompositionType()=trws_base::DecompositionStorage<GM>::getStructureType(stringDecompositionType);
   trwsParameter_.fastComputations()=(slowComputations==0);
   trwsParameter_.canonicalNormalization()=(noNormalization==0);
   trwsParameter_.setTreeAgreeMaxStableIter(treeAgreeMaxStableIter);
   this-> template infer<TRWSiType, TimingVisitorType, typename TRWSiType::Parameter>(model, output, verbose, trwsParameter_);
}

template <class IO, class GM, class ACC>
const std::string TRWSiCaller<IO, GM, ACC>::name_ = "TRWSi";

} // namespace interface

} // namespace opengm

#endif /* TRWSI_CALLER_HXX_ */
