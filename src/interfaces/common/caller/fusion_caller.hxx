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


   typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer; 

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   template<class INF>
   void setParam(typename INF::Parameter & param);

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
   gen.push_back("UP-DOWN");
   gen.push_back("RANDOM");
   gen.push_back("RANDOMLF");
   gen.push_back("NU_RANDOM");
   gen.push_back("BLUR"); 
   gen.push_back("ENERGYBLUR");
   gen.push_back("NU_ENERGYBLUR");      
   addArgument(StringArgument<>(selectedGenType_, "g", "gen", "Selected proposal generator", gen.front(), gen));
   addArgument(StringArgument<>(selectedFusionType_, "f", "fusion", "Select fusion method", fusion.front(), fusion));
   //addArgument(IntArgument<>(numberOfThreads_, "", "threads", "number of threads", static_cast<int>(1)));
   addArgument(Size_TArgument<>(numIt_, "", "numIt", "number of iterations", static_cast<size_t>(100))); 
   addArgument(Size_TArgument<>(numStopIt_, "", "numStopIt", "number of iterations with no improvment that cause stopping (0=auto)", static_cast<size_t>(0))); 
   addArgument(Size_TArgument<>(maxSubgraphSize_, "", "maxSS", "maximal subgraph size for LF", static_cast<size_t>(2)));
   //addArgument(BoolArgument(useDirectInterface_,"","useDirectInterface", "avoid a copy into a submodel for fusion"));
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
template <class INF>
inline void FusionCaller<IO, GM, ACC>::setParam(
   typename INF::Parameter & param
){


   if(selectedFusionType_=="QPBO") 
      param.fusionParam_.fusionSolver_ = INF::FusionMover::QpboFusion;
   if(selectedFusionType_=="LF")   
      param.fusionParam_.fusionSolver_ = INF::FusionMover::QpboFusion;
   if(selectedFusionType_=="ILP")  
      param.fusionParam_.fusionSolver_ = INF::FusionMover::CplexFuison;


   param.numIt_ = numIt_;
   param.numStopIt_ = numStopIt_;  
   param.fusionParam_.maxSubgraphSize_= maxSubgraphSize_;

   //param.useDirectInterface_ = useDirectInterface_;

   param.fusionParam_.reducedInf_          = reducedInf_;
   param.fusionParam_.connectedComponents_ = connectedComponents_;
   param.fusionParam_.tentacles_           = tentacles_;
}

template <class IO, class GM, class ACC>
inline void FusionCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Fusion caller" << std::endl;


   
   typedef opengm::proposal_gen::AlphaExpansionGen<GM, opengm::Minimizer> AEGen;
   typedef opengm::proposal_gen::AlphaBetaSwapGen<GM, opengm::Minimizer> ABGen;
   typedef opengm::proposal_gen::UpDownGen<GM, opengm::Minimizer> UDGen;
   typedef opengm::proposal_gen::RandomGen<GM, opengm::Minimizer> RGen;
   typedef opengm::proposal_gen::RandomLFGen<GM, opengm::Minimizer> RLFGen;
   typedef opengm::proposal_gen::NonUniformRandomGen<GM, opengm::Minimizer> NURGen;
   typedef opengm::proposal_gen::BlurGen<GM, opengm::Minimizer> BlurGen;
   typedef opengm::proposal_gen::EnergyBlurGen<GM, opengm::Minimizer> EBlurGen;


   if(selectedGenType_=="A-EXP"){
      typedef AEGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="AB-SWAP"){
      typedef ABGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="UP-DOWN"){
      typedef UDGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="RADOM"){
      typedef RGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="RANDOMLF"){
      typedef RLFGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="NU_RANDOM"){
      typedef NURGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      para.proposalParam_.temp_ = temperature_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="BLUR"){
      typedef BlurGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      para.proposalParam_.sigma_ = sigma_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="ENERGYBLUR"){
      typedef EBlurGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      para.proposalParam_.sigma_ = sigma_;
      para.proposalParam_.useLocalMargs_ = false;
      para.proposalParam_.temp_ = temperature_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="NU_ENERGYBLUR"){
      typedef EBlurGen Gen;
      typedef opengm::FusionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      para.proposalParam_.sigma_ = sigma_;
      para.proposalParam_.useLocalMargs_ = true;
      para.proposalParam_.temp_ = temperature_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }

}

template <class IO, class GM, class ACC>
const std::string  FusionCaller<IO, GM, ACC>::name_ = "Fusion";

} // namespace interface

} // namespace opengm

#endif /* SelfFusion_CALLER_HXX_ */
