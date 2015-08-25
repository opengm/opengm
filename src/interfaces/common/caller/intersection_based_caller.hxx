#ifndef INTERSECTION_BASED_CALLER
#define INTERSECTION_BASED_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/intersection_based_inf.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class IntersectionBasedCaller : public InferenceCallerBase<IO, GM, ACC, IntersectionBasedCaller<IO, GM, ACC> > {
protected:

   typedef InferenceCallerBase<IO, GM, ACC, IntersectionBasedCaller<IO, GM, ACC> > BaseClass;


   typedef opengm::proposal_gen::WeightRandomization<typename GM::ValueType> WRand;
   typedef typename  WRand::Parameter WRandParam;


   typedef PermutableLabelFusionMove<GM, ACC>  FusionMoverType;
   typedef typename FusionMoverType::Parameter FusionParameter;

   typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer; 

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   template<class INF>
   void setParam(typename INF::Parameter & param);

   size_t numIt_;
   size_t numStopIt_;
   size_t parallelProposals_;
   

   std::string selectedNoise_;
   WRandParam wRand_;



   int numberOfThreads_;
   std::string selectedGenType_;


   // fusion param 
   std::string selectedFusionType_;
   FusionParameter fusionParam_;
   bool planar_;

   // RHC SPECIFIC param
   float stopWeight_;
   float nodeStopNum_;

   // for ws
   float seedFraction_;
   bool ingoreNegativeWeights_;
   bool seedFromNegativeEdges_;

   bool cgcFinalization_;
   bool doCutMove_;

   bool acceptFirst_;
   bool warmStart_;
   std::string storagePrefix_;
   
public:
   const static std::string name_;
   IntersectionBasedCaller(IO& ioIn);
   ~IntersectionBasedCaller();
};

template <class IO, class GM, class ACC>
inline  IntersectionBasedCaller<IO, GM, ACC>::IntersectionBasedCaller(IO& ioIn)
: BaseClass(name_, "detailed description of  SelfFusion caller...", ioIn) { 

   std::vector<std::string> fusion;
   fusion.push_back("MC");
   fusion.push_back("MCN");
   fusion.push_back("CGC");
   fusion.push_back("HC");
   fusion.push_back("BASE");
   fusion.push_back("CF");

   std::vector<std::string> gen;  
   gen.push_back("RHC");
   gen.push_back("R2C");
   gen.push_back("RWS");

   std::vector<std::string> noise;  
   noise.push_back("NA");
   noise.push_back("UA");
   noise.push_back("NM");
   noise.push_back("NONE");

   addArgument(StringArgument<>(selectedGenType_, "g", "gen", "Selected proposal generator", gen.front(), gen));
   addArgument(StringArgument<>(selectedFusionType_, "f", "fusion", "Select fusion method", fusion.front(), fusion));
   addArgument(BoolArgument(fusionParam_.planar_,"pl","planar", "is problem planar"));
   addArgument(BoolArgument(fusionParam_.decompose_,"dc","decompose", "try to decompose subproblems"));
   addArgument(StringArgument<>(fusionParam_.workflow_, "", "workflow", "workflow of cutting-plane procedure", false));

   //addArgument(IntArgument<>(numberOfThreads_, "", "threads", "number of threads", static_cast<int>(1)));
   addArgument(Size_TArgument<>(numIt_, "", "numIt", "number of iterations", static_cast<size_t>(100))); 
   addArgument(Size_TArgument<>(numStopIt_, "", "numStopIt", "number of iterations with no improvment that cause stopping (0=auto)", static_cast<size_t>(20))); 
   addArgument(Size_TArgument<>(parallelProposals_, "pp", "parallelProposals", "number of parallel proposals (1)", static_cast<size_t>(1))); 

   addArgument(BoolArgument(warmStart_,"ws","warmStart", "use warm start"));


   addArgument(BoolArgument(acceptFirst_,"af","acceptFirst", ""));
   addArgument(BoolArgument(cgcFinalization_,"cgcf","cgcFinalization", "use cgc in the end"));
   addArgument(BoolArgument(doCutMove_,"dcm","doCutMove", "do the cut phase within cgc (better not, should be faster)"));


   // parameter for weight randomizer_
   addArgument(StringArgument<>(selectedNoise_, "nt", "noiseType", "selected noise type", noise.front(), noise));
   addArgument(FloatArgument<>(wRand_.noiseParam_, "np", "noiseParam", "parameter of noise type", 1.0f));
   addArgument(Size_TArgument<>(wRand_.seed_, "", "seed", "seed", size_t(42)));
   addArgument(BoolArgument(wRand_.ignoreSeed_,"is","ignoreSeed", "ignore seed and use auto generated seed (based on time )"));
   addArgument(BoolArgument(wRand_.autoScale_,"as","autoScale", "use the noise parameter in relation to weights min max range"));
   addArgument(FloatArgument<>(wRand_.permuteN_,"pn","permuteN", "permute relative or absolute number of weights",-1.0f));


   // parameter for h
   addArgument(FloatArgument<>(stopWeight_, "sw", "stopWeight", "stop hc merging when this weight is reached", 0.0f));
   addArgument(FloatArgument<>(nodeStopNum_, "snn", "stopNodeNum", "stop hc merging when this (maybe relativ) number of nodes is reached", 0.1f));

   // param for ws
   addArgument(FloatArgument<>(seedFraction_, "", "nSeeds", "(maybe relative) number of seeds ", 20.0f));
   addArgument(BoolArgument(seedFromNegativeEdges_, "sfn", "seedFromNegative", "use only nodes of negative edges as random seed"));

   addArgument(StringArgument<>(storagePrefix_, "sp", "storagePrefix", "storage prefix", std::string("")));
}

template <class IO, class GM, class ACC>
IntersectionBasedCaller<IO, GM, ACC>::~IntersectionBasedCaller()
{;}


template <class IO, class GM, class ACC>
template <class INF>
inline void IntersectionBasedCaller<IO, GM, ACC>::setParam(
   typename INF::Parameter & param
){

   param.planar_ = fusionParam_.planar_;
   param.numIt_ = numIt_;
   param.numStopIt_ = numStopIt_;  
   param.cgcFinalization_ = cgcFinalization_;
   param.doCutMove_ = doCutMove_;
   param.parallelProposals_ = parallelProposals_;
   param.fusionParam_ = fusionParam_;
   param.proposalParam_.randomizer_ = wRand_;
   param.storagePrefix_ = storagePrefix_;
   param.acceptFirst_ = acceptFirst_;
   param.warmStart_ = warmStart_;
}

template <class IO, class GM, class ACC>
inline void IntersectionBasedCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Intersection Based caller" << std::endl;


   
   typedef opengm::proposal_gen::RandomizedHierarchicalClustering<GM, opengm::Minimizer> RHC;
   typedef opengm::proposal_gen::RandomizedWatershed<GM, opengm::Minimizer> RWS;
   typedef opengm::proposal_gen::QpboBased<GM, opengm::Minimizer> R2C;

   // noise 
   if(selectedNoise_ == "NA"){
      wRand_.noiseType_ = WRandParam::NormalAdd;
   }
   else if(selectedNoise_ == "UA"){
      wRand_.noiseType_ = WRandParam::UniformAdd;
   }
   else if(selectedNoise_ == "NM"){
      wRand_.noiseType_ = WRandParam::NormalMult;
   }
   else if(selectedNoise_ == "NONE"){
      wRand_.noiseType_ = WRandParam::None;
   }

   // fusion solver
   if(selectedFusionType_ == "MC"){
      fusionParam_.fusionSolver_  = FusionMoverType::MulticutSolver;
   }
   else if(selectedFusionType_ == "MCN"){
      fusionParam_.fusionSolver_  = FusionMoverType::MulticutNativeSolver;
   }
   else if(selectedFusionType_ == "CGC"){
      fusionParam_.fusionSolver_  = FusionMoverType::CgcSolver;
   }
   else if(selectedFusionType_ == "HC"){
      fusionParam_.fusionSolver_  = FusionMoverType::HierachicalClusteringSolver;
   }
   else if(selectedFusionType_ == "BASE"){
      fusionParam_.fusionSolver_  = FusionMoverType::BaseSolver;
   }
   else if(selectedFusionType_ == "CF"){
      fusionParam_.fusionSolver_  = FusionMoverType::ClassicFusion;
   }


   // proposal
   if(selectedGenType_=="RHC"){
      typedef RHC Gen;
      typedef opengm::IntersectionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      para.proposalParam_.stopWeight_ = stopWeight_;
      para.proposalParam_.nodeStopNum_ = nodeStopNum_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="RWS"){
      typedef RWS Gen;
      typedef opengm::IntersectionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      para.proposalParam_.seedFraction_ = seedFraction_;
      para.proposalParam_.seedFromNegativeEdges_ = seedFromNegativeEdges_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="R2C"){
      typedef R2C Gen;
      typedef opengm::IntersectionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      //para.proposalParam_.sigma_ = sigma_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
}

template <class IO, class GM, class ACC>
const std::string  IntersectionBasedCaller<IO, GM, ACC>::name_ = "IBMC";

} // namespace interface

} // namespace opengm

#endif /* INTERSECTION_BASED_CALLER */
