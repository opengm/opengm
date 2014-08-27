#ifndef FUSION_CALLER_HXX_
#define FUSION_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/fusion_based_inf.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class FusionCaller : public InferenceCallerBase<IO, GM, ACC, FusionCaller<IO, GM, ACC> > {
protected:

   typedef InferenceCallerBase<IO, GM, ACC, FusionCaller<IO, GM, ACC> > BaseClass;
   typedef opengm::FusionBasedInf<GM, ACC> INF;
   typedef typename INF::VerboseVisitorType VerboseVisitorType;
   typedef typename INF::EmptyVisitorType EmptyVisitorType;
   typedef typename INF::TimingVisitorType TimingVisitorType;

   typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer; 

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   size_t numIt_;
   size_t numStopIt_;
   int numberOfThreads_;
   std::string selectedGenType_;
   std::string selectedFusionType_;
   size_t maxSubgraphSize_; 
   bool useDirectInterface_;
   bool reducedInf_,connectedComponents_,tentacles_;
   float temperature_;
   double sigma_;
   
public:
   const static std::string name_;
   FusionCaller(IO& ioIn);
   ~FusionCaller();
};

template <class IO, class GM, class ACC>
inline  FusionCaller<IO, GM, ACC>::FusionCaller(IO& ioIn)
: BaseClass(name_, "detailed description of  SelfFusion caller...", ioIn) { 
   std::vector<std::string> fusion;
   fusion.push_back("QPBO");
   fusion.push_back("LF");
   fusion.push_back("ILP");
   std::vector<std::string> gen;  
   gen.push_back("A-EXP");
   gen.push_back("AB-SWAP");
   gen.push_back("RANDOM");
   gen.push_back("RANDOMLF");
   gen.push_back("NU_RANDOM");
   gen.push_back("BLUR"); 
   gen.push_back("ENERGYBLUR");
   gen.push_back("NU_ENERGYBLUR");      
   addArgument(StringArgument<>(selectedGenType_, "g", "gen", "Selected proposal generator", gen.front(), gen));
   addArgument(StringArgument<>(selectedFusionType_, "f", "fusion", "Select fusion method", fusion.front(), fusion));
   addArgument(IntArgument<>(numberOfThreads_, "", "threads", "number of threads", static_cast<int>(1)));
   addArgument(Size_TArgument<>(numIt_, "", "numIt", "number of iterations", static_cast<size_t>(100))); 
   addArgument(Size_TArgument<>(numStopIt_, "", "numStopIt", "number of iterations with no improvment that cause stopping (0=auto)", static_cast<size_t>(0))); 
   addArgument(Size_TArgument<>(maxSubgraphSize_, "", "maxSS", "maximal subgraph size for LF", static_cast<size_t>(2)));
   addArgument(BoolArgument(useDirectInterface_,"","useDirectInterface", "avoid a copy into a submodel for fusion"));

   addArgument(BoolArgument(reducedInf_,"","reducedInf", "use reduced inference"));
   addArgument(BoolArgument(connectedComponents_,"","connectedComponents", "use reduced inference connectedComponents"));
   addArgument(BoolArgument(tentacles_,"","tentacles", "use reduced inference tentacles"));

   addArgument(FloatArgument<>(temperature_, "temp", "temperature", "temperature for non uniform random proposal generator", static_cast<float>(1.0))); 
   addArgument(DoubleArgument<>(sigma_, "", "sigma", "standard devariation used for bluring", static_cast<double>(20.0)));
}

template <class IO, class GM, class ACC>
FusionCaller<IO, GM, ACC>::~FusionCaller()
{;}

template <class IO, class GM, class ACC>
inline void FusionCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Fusion caller" << std::endl;
   typename INF::Parameter para_;
 
   if(selectedFusionType_=="QPBO") para_.fusionSolver_ = INF::QpboFusion;
   if(selectedFusionType_=="LF")   para_.fusionSolver_ = INF::LazyFlipperFusion;
   if(selectedFusionType_=="ILP")  para_.fusionSolver_ = INF::CplexFusion;
  
   if(selectedGenType_=="A-EXP")      para_.proposalGen_ = INF::AlphaExpansion;
   if(selectedGenType_=="AB-SWAP")    para_.proposalGen_ = INF::AlphaBetaSwap;
   if(selectedGenType_=="RANDOM")     para_.proposalGen_ = INF::Random;
   if(selectedGenType_=="RANDOMLF")   para_.proposalGen_ = INF::RandomLF;
   if(selectedGenType_=="NU_RANDOM")  para_.proposalGen_ = INF::NonUniformRandom;
   if(selectedGenType_=="BLUR")       para_.proposalGen_ = INF::Blur;
   if(selectedGenType_=="ENERGYBLUR") para_.proposalGen_ = INF::EnergyBlur;
   if(selectedGenType_=="NU_ENERGYBLUR"){
      para_.proposalGen_ = INF::EnergyBlur;
      para_.useEstimatedMarginals_ =true;
   }

   para_.numIt_ = numIt_;
   para_.numStopIt_ = numStopIt_;  
   para_.maxSubgraphSize_= maxSubgraphSize_;
   para_.useDirectInterface_ = useDirectInterface_;

   para_.reducedInf_          = reducedInf_;
   para_.connectedComponents_ = connectedComponents_;
   para_.tentacles_           = tentacles_;
   
   para_.temperature_         = temperature_;
   para_.sigma_               = sigma_;

   this-> template infer<INF, TimingVisitorType, typename INF::Parameter>(model, output, verbose, para_);
}

template <class IO, class GM, class ACC>
const std::string  FusionCaller<IO, GM, ACC>::name_ = "Fusion";

} // namespace interface

} // namespace opengm

#endif /* SelfFusion_CALLER_HXX_ */
