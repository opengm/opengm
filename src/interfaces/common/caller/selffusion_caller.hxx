#ifndef SELFFUSION_CALLER_HXX_
#define SELFFUSION_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/self_fusion.hxx>


#include <opengm/inference/messagepassing/messagepassing.hxx>
#ifdef WITH_TRWS
#include <opengm/inference/external/trws.hxx>
#endif

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class SelfFusionCaller : public InferenceCallerBase<IO, GM, ACC, SelfFusionCaller<IO, GM, ACC> > {
protected:

   typedef InferenceCallerBase<IO, GM, ACC, SelfFusionCaller<IO, GM, ACC> > BaseClass;
   typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer; 

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   size_t numIt_; 
   size_t numStopIt_;
   int numberOfThreads_;
   std::string selectedInfType_;
   std::string selectedFusionType_; 
   std::string selectedEnergyType_;
   size_t maxSubgraphSize_;
   double lbpDamping_;
   bool reducedInf_,connectedComponents_,tentacles_;
   
public:
   const static std::string name_;
   SelfFusionCaller(IO& ioIn);
   ~SelfFusionCaller();
};

template <class IO, class GM, class ACC>
inline  SelfFusionCaller<IO, GM, ACC>::SelfFusionCaller(IO& ioIn)
: BaseClass(name_, "detailed description of  SelfFusion caller...", ioIn) { 
   std::vector<std::string> fusion;
   fusion.push_back("QPBO");
   fusion.push_back("LF");
   fusion.push_back("ILP");
   std::vector<std::string> inf;  
   inf.push_back("TRWS");
   inf.push_back("LBP"); 
   std::vector<std::string> possibleEnergyTypes;
   possibleEnergyTypes.push_back("VIEW");
   possibleEnergyTypes.push_back("TABLES");
   possibleEnergyTypes.push_back("TL1");
   possibleEnergyTypes.push_back("TL2");
   //possibleEnergyTypes.push_back("WEIGHTEDTABLE");
   addArgument(StringArgument<>(selectedEnergyType_, "", "energy", "Select desired energy type.", possibleEnergyTypes.front(), possibleEnergyTypes));  
   addArgument(StringArgument<>(selectedInfType_, "i", "inf", "Select inference method for proposals", inf.front(), inf));
   addArgument(StringArgument<>(selectedFusionType_, "f", "fusion", "Select fusion method", fusion.front(), fusion));
   addArgument(IntArgument<>(numberOfThreads_, "", "threads", "number of threads", static_cast<int>(1)));
   addArgument(Size_TArgument<>(numIt_, "", "numIt", "number of iterations", static_cast<size_t>(10))); 
   addArgument(Size_TArgument<>(numStopIt_, "", "numStopIt", "number of iterations without energy improvment which cause termination", static_cast<size_t>(10))); 
   addArgument(Size_TArgument<>(maxSubgraphSize_, "", "maxSS", "maximal subgraph size for LF", static_cast<size_t>(2))); 
   addArgument(DoubleArgument<>(lbpDamping_, "", "dampingLBP", "damping for LBP", static_cast<double>(0.5)));
   addArgument(BoolArgument(reducedInf_,"","reducedInf", "use reduced inference"));
   addArgument(BoolArgument(connectedComponents_,"","connectedComponents", "use reduced inference connectedComponents"));
   addArgument(BoolArgument(tentacles_,"","tentacles", "use reduced inference tentacles"));
}

template <class IO, class GM, class ACC>
SelfFusionCaller<IO, GM, ACC>::~SelfFusionCaller()
{;}

template <class IO, class GM, class ACC>
inline void SelfFusionCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running SelfFusion caller" << std::endl;


   if(selectedInfType_=="LBP"){
      typedef opengm::BeliefPropagationUpdateRules<GM,ACC> UpdateRulesType;
      typedef opengm::MessagePassing<GM, ACC,UpdateRulesType, opengm::MaxDistance> BP;
      typedef SelfFusion<BP> INF;
      typedef typename INF::VerboseVisitorType VerboseVisitorType;
      typedef typename INF::EmptyVisitorType EmptyVisitorType;
      typedef typename INF::TimingVisitorType TimingVisitorType; 

      typename INF::Parameter para_; 
      if(selectedFusionType_=="QPBO") para_.fusionSolver_ = INF::QpboFusion;
      if(selectedFusionType_=="LF")   para_.fusionSolver_ = INF::LazyFlipperFusion;
      if(selectedFusionType_=="ILP")  para_.fusionSolver_ = INF::CplexFusion;
      para_.infParam_.maximumNumberOfSteps_ = numIt_;  
      para_.numStopIt_ = numStopIt_;
      //para_.infParam_.saveMemory_ = true; 
      para_.infParam_.damping_ = lbpDamping_; 
      para_.maxSubgraphSize_= maxSubgraphSize_; 

      para_.reducedInf_ = reducedInf_;
      para_.connectedComponents_ = connectedComponents_;
      para_.tentacles_ = tentacles_;

      this-> template infer<INF, TimingVisitorType, typename INF::Parameter>(model, output, verbose, para_);
   } 
   else if(selectedInfType_=="TRWS"){
#ifdef WITH_TRWS
      typedef typename opengm::external::TRWS<GM> TRWSType;
      typedef SelfFusion<TRWSType> INF;
      typedef typename INF::VerboseVisitorType VerboseVisitorType;
      typedef typename INF::EmptyVisitorType EmptyVisitorType;
      typedef typename INF::TimingVisitorType TimingVisitorType;

      typename INF::Parameter para_; 
      if(selectedFusionType_=="QPBO") para_.fusionSolver_ = INF::QpboFusion;
      if(selectedFusionType_=="LF")   para_.fusionSolver_ = INF::LazyFlipperFusion;
      if(selectedFusionType_=="ILP")  para_.fusionSolver_ = INF::CplexFusion;
      para_.infParam_.numberOfIterations_ = numIt_; 
      para_.numStopIt_ = numStopIt_;
      para_.maxSubgraphSize_= maxSubgraphSize_; 

      para_.reducedInf_ = reducedInf_;
      para_.connectedComponents_ = connectedComponents_;
      para_.tentacles_ = tentacles_;


      if(selectedEnergyType_ == "VIEW") {
         para_.infParam_.energyType_= TRWSType::Parameter::VIEW;
      } else if(selectedEnergyType_ == "TABLES") {
         para_.infParam_.energyType_= TRWSType::Parameter::TABLES;
      } else if(selectedEnergyType_ == "TL1") {
         para_.infParam_.energyType_= TRWSType::Parameter::TL1;
      } else if(selectedEnergyType_ == "TL2") {
         para_.infParam_.energyType_= TRWSType::Parameter::TL2;
      } else {
         throw RuntimeError("Unknown energy type");
      }
      
      this-> template infer<INF, TimingVisitorType, typename INF::Parameter>(model, output, verbose, para_);
#else
      throw RuntimeError("TRWS is disabled!");
#endif
   }
   else{
      throw RuntimeError("Unknown Inference method for subproblems!");
   }
}

template <class IO, class GM, class ACC>
const std::string  SelfFusionCaller<IO, GM, ACC>::name_ = "SelfFusion";

} // namespace interface

} // namespace opengm

#endif /* SelfFusion_CALLER_HXX_ */
