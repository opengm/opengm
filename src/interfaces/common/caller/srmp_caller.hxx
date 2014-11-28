#ifndef OPENGM_EXTERNAL_SRMP_CALLER_HXX_
#define OPENGM_EXTERNAL_SRMP_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/srmp.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class SRMPCaller : public InferenceCallerBase<IO, GM, ACC, SRMPCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::external::SRMP<GM> SRMP;
   typedef InferenceCallerBase<IO, GM, ACC, SRMPCaller<IO, GM, ACC> > BaseClass;
   typedef typename SRMP::VerboseVisitorType VerboseVisitorType;
   typedef typename SRMP::EmptyVisitorType EmptyVisitorType;
   typedef typename SRMP::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   SRMPCaller(IO& ioIn);
   virtual ~SRMPCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename SRMP::Parameter srmpParameter_;

   std::string selectedMethod_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

};

template <class IO, class GM, class ACC>
inline SRMPCaller<IO, GM, ACC>::SRMPCaller(IO& ioIn)
   : BaseClass("SRMP", "detailed description of SRMP Parser...", ioIn) {
   std::vector<std::string> possibleMethods;
   possibleMethods.push_back("SRMP");
   possibleMethods.push_back("MPLP");
   possibleMethods.push_back("MPLP_BW");
   possibleMethods.push_back("CMP");
   addArgument(StringArgument<>(selectedMethod_, "", "method", "Select desired inference method.", possibleMethods.front(), possibleMethods));
   addArgument(IntArgument<>(srmpParameter_.iter_max, "", "maxIter", "Maximum number of iterations.", srmpParameter_.iter_max));
   addArgument(DoubleArgument<>(srmpParameter_.time_max, "", "maxTime", "Maximum time in seconds.", srmpParameter_.time_max));
   addArgument(DoubleArgument<>(srmpParameter_.eps, "", "eps", "Stop if the increase of the lower during one iteration is less than eps.", srmpParameter_.eps));
   addArgument(IntArgument<>(srmpParameter_.compute_solution_period, "", "compute_solution_period", "Extract solution after every compute_solution_period iterations.", srmpParameter_.compute_solution_period));
   addArgument(BoolArgument(srmpParameter_.print_times, "", "printtimes", "Print times."));
   addArgument(IntArgument<>(srmpParameter_.sort_flag, "", "sort", "sort = -1: process factors in the order they were given (except that nodes in SRMP and CMP are always traversed first). sort = 0: sort factors according to the given node ordering. sort = 1: use an automatic greedy technique for sorting nodes, then sort factors accordingly. sort > 1: user random permutation of nodes, with sort as the seed.", srmpParameter_.sort_flag));
   addArgument(BoolArgument(srmpParameter_.verbose, "", "srmpverbose", "Enable srmp verbose mode."));
   addArgument(DoubleArgument<>(srmpParameter_.TRWS_weighting, "", "TRWS_weighting", "TRWS weighting in [0.0;1.0], 1.0 corresponds to TRW-S (for pairwise energies).", srmpParameter_.TRWS_weighting));
   addArgument(BoolArgument(srmpParameter_.BLPRelaxation_, "", "blpRelaxation", "Use blp relaxation."));
   addArgument(BoolArgument(srmpParameter_.FullRelaxation_, "", "fullRelaxation", "Use full relaxation."));

   std::vector<int> possibleFullRelaxationMethods;
   possibleFullRelaxationMethods.push_back(0);
   possibleFullRelaxationMethods.push_back(1);
   possibleFullRelaxationMethods.push_back(2);
   possibleFullRelaxationMethods.push_back(3);

   addArgument(IntArgument<>(srmpParameter_.FullRelaxationMethod_, "", "fullRelaxationMethod", "Select full relaxation method.", srmpParameter_.FullRelaxationMethod_, possibleFullRelaxationMethods));
   addArgument(BoolArgument(srmpParameter_.FullDualRelaxation_, "", "fulldualRelaxation", "Use full relaxation, by constructing dual graph."));
   addArgument(IntArgument<>(srmpParameter_.FullDualRelaxationMethod_, "", "fulldualRelaxationMethod", "Select full dual relaxation method  (has the same meaning as in sort).", srmpParameter_.FullDualRelaxationMethod_));
}

template <class IO, class GM, class ACC>
inline SRMPCaller<IO, GM, ACC>::~SRMPCaller() {

}

template <class IO, class GM, class ACC>
inline void SRMPCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running SRMP caller" << std::endl;

   if(selectedMethod_ == "SRMP") {
      srmpParameter_.method= SRMP::Parameter::SRMP;
   } else if(selectedMethod_ == "MPLP") {
      srmpParameter_.method = SRMP::Parameter::MPLP;
   } else if(selectedMethod_ == "MPLP_BW") {
      srmpParameter_.method = SRMP::Parameter::MPLP_BW;
   } else if(selectedMethod_ == "CMP") {
      srmpParameter_.method = SRMP::Parameter::CMP;
   } else {
      throw RuntimeError("Unknown inference method for SRMP");
   }

   this-> template infer<SRMP, TimingVisitorType, typename SRMP::Parameter>(model, output, verbose, srmpParameter_);
}

template <class IO, class GM, class ACC>
const std::string SRMPCaller<IO, GM, ACC>::name_ = "SRMP";

} // namespace interface

} // namespace opengm

#endif /* OPENGM_EXTERNAL_SRMP_CALLER_HXX_ */
