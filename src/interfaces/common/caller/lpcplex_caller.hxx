#ifndef LPCPLEX_CALLER_HXX_
#define LPCPLEX_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/lpcplex.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LPCplexCaller : public InferenceCallerBase<IO, GM, ACC, LPCplexCaller<IO, GM, ACC> > {
public:
   typedef LPCplex<GM, ACC> LPCPLEX;
   typedef InferenceCallerBase<IO, GM, ACC, LPCplexCaller<IO, GM, ACC> > BaseClass;
   typedef typename LPCPLEX::VerboseVisitorType VerboseVisitorType;
   typedef typename LPCPLEX::EmptyVisitorType EmptyVisitorType;
   typedef typename LPCPLEX::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   LPCplexCaller(IO& ioIn);
   virtual ~LPCplexCaller();
protected:

   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename LPCPLEX::Parameter lpcplexParameter_;

   std::string rootAlg_;
   std::string nodeAlg_;
   std::string mipFocus_;
   std::string presolve_;
   std::string cutLevel_; 


   LPDef::MIP_CUT getCutLevel(std::string cl){
      if(cl == "DEFAULT"){
         return  LPDef::MIP_CUT_DEFAULT;
      } else if(cl == "AUTO"){
         return  LPDef::MIP_CUT_AUTO;
      } else if(cl == "OFF"){
         return  LPDef::MIP_CUT_OFF;
      } else if(cl == "ON"){ 
         return LPDef::MIP_CUT_ON;
      } else if(cl == "AGGRESSIVE"){
         return LPDef::MIP_CUT_AGGRESSIVE; 
      } else if(cl == "VERYAGGRESSIVE"){
         return  LPDef::MIP_CUT_VERYAGGRESSIVE;
      }else{
         throw RuntimeError("Unknown cutlevel");
      }
      return  LPDef::MIP_CUT_AUTO;
   };

};

template <class IO, class GM, class ACC>
inline LPCplexCaller<IO, GM, ACC>::LPCplexCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of LPCplex caller...", ioIn) {
   addArgument(BoolArgument(lpcplexParameter_.integerConstraint_, "ic", "integerconstraint", "use integer constraints"));
   addArgument(IntArgument<>(lpcplexParameter_.numberOfThreads_, "", "threads", "number of threads", lpcplexParameter_.numberOfThreads_));
   addArgument(BoolArgument(lpcplexParameter_.verbose_, "v", "verbose", "used to activate verbose output"));
   addArgument(DoubleArgument<>(lpcplexParameter_.cutUp_, "", "cutup", "cut up", lpcplexParameter_.cutUp_));
   addArgument(DoubleArgument<>(lpcplexParameter_.epOpt_, "", "epOpt", "Optimality tolerance", lpcplexParameter_.epOpt_));
   addArgument(DoubleArgument<>(lpcplexParameter_.epMrk_, "", "epMrk", "Markowitz tolerance", lpcplexParameter_.epMrk_));
   addArgument(DoubleArgument<>(lpcplexParameter_.epRHS_, "", "epRHS", "Feasibility tolerance", lpcplexParameter_.epRHS_));
   addArgument(DoubleArgument<>(lpcplexParameter_.epInt_, "", "epInt", "Integer feasibility tolerance", lpcplexParameter_.epInt_));
   addArgument(DoubleArgument<>(lpcplexParameter_.epGap_, "", "epGap", "Relative MIP gap tolerance", lpcplexParameter_.epGap_));
   addArgument(DoubleArgument<>(lpcplexParameter_.epAGap_, "", "epAGap", "Absolute MIP gap tolerance ", lpcplexParameter_.epAGap_));

   double timeout =604800.0;
   addArgument(DoubleArgument<>(lpcplexParameter_.timeLimit_,"","maxTime","maximal runtime in seconds",timeout)); //default 1 week
   addArgument(IntArgument<>(lpcplexParameter_.probeingLevel_,"","probing", "probing level (-1=no, 0=auto, 1=moderate, 2=agressive, 3=very agressive)",lpcplexParameter_.probeingLevel_));

   std::vector<std::string> possibleAlgs;
   possibleAlgs.push_back("AUTO");
   possibleAlgs.push_back("PRIMAL_SIMPLEX");
   possibleAlgs.push_back("DUAL_SIMPLEX");
   possibleAlgs.push_back("NETWORK_SIMPLEX");
   possibleAlgs.push_back("BARRIER");
   possibleAlgs.push_back("SIFTING");
   possibleAlgs.push_back("CONCURRENT");
   addArgument(StringArgument<>(rootAlg_, "", "rootAlg", "Select algorithm for LP and root of ILP", possibleAlgs.front(), possibleAlgs));
   addArgument(StringArgument<>(nodeAlg_, "", "nodeAlg", "Select algorithm for nodes of ILP", possibleAlgs.front(), possibleAlgs));

   std::vector<std::string> possibleMIPEmphasis;
   possibleMIPEmphasis.push_back("AUTO"); 
   possibleMIPEmphasis.push_back("FEASIBILITY");
   possibleMIPEmphasis.push_back("OPTIMALITY");
   possibleMIPEmphasis.push_back("BESTBOUND");
   possibleMIPEmphasis.push_back("HIDDEN"); 
   addArgument(StringArgument<>(mipFocus_, "", "mipFocus", "Select the focus/emphasis", possibleMIPEmphasis.front(), possibleMIPEmphasis));

   std::vector<std::string> possiblePresolve;
   possiblePresolve.push_back("AUTO");   
   possiblePresolve.push_back("OFF");
   possiblePresolve.push_back("CONSERVATIVE"); 
   possiblePresolve.push_back("AGRESSIVE"); 
   addArgument(StringArgument<>(presolve_, "", "presolve", "Select presolve level", possiblePresolve.front(), possiblePresolve));

  std::vector<std::string> possibleCutLevels;
   possibleCutLevels.push_back("DEFAULT"); 
   possibleCutLevels.push_back("AUTO"); 
   possibleCutLevels.push_back("OFF"); 
   possibleCutLevels.push_back("ON"); 
   possibleCutLevels.push_back("AGGRESSIVE");
   possibleCutLevels.push_back("VERYAGGRESSIVE");
   addArgument(StringArgument<>(cutLevel_, "", "cutLevel", "Select general cut level", possibleCutLevels.front(), possibleCutLevels)); 
   // addArgument(StringArgument<>(cliqueCutLevel_, "", "cliqueCutLevel", "Select clique cut level", possibleCutLevels.front(), possibleCutLevels)); 
   //addArgument(StringArgument<>(coverCutLevel_, "", "coverCutLevel", "Select cover cut level", possibleCutLevels.front(), possibleCutLevels)); 
   //addArgument(StringArgument<>(gubCutLevel_, "", "gubCutLevel", "Select generalized upper bound cut level", possibleCutLevels.front(), possibleCutLevels)); 
   //addArgument(StringArgument<>(mirCutLevel_, "", "mirCutLevel", "Select mixed integer rounding cut level", possibleCutLevels.front(), possibleCutLevels)); 
   //addArgument(StringArgument<>(iboundCutLevel_, "", "iboundCutLevel", "Select implied bound cut level", possibleCutLevels.front(), possibleCutLevels)); 
   //addArgument(StringArgument<>(flowcoverCutLevel_, "", "flowcoverCutLevel", "Select flow-cover cut level", possibleCutLevels.front(), possibleCutLevels)); 
   //addArgument(StringArgument<>(flowpathCutLevel_, "", "flowpathCutLevel", "Select flow-path cut level", possibleCutLevels.front(), possibleCutLevels)); 

}

template <class IO, class GM, class ACC>
inline LPCplexCaller<IO, GM, ACC>::~LPCplexCaller() {

}

template <class IO, class GM, class ACC>
inline void LPCplexCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LPCplex caller" << std::endl;

    // Root Algorithm
   if(rootAlg_=="AUTO"){
      lpcplexParameter_.rootAlg_ = LPCPLEX::LP_SOLVER_AUTO;
   } else if(rootAlg_== "PRIMAL_SIMPLEX"){
      lpcplexParameter_.rootAlg_ = LPCPLEX::LP_SOLVER_PRIMAL_SIMPLEX;
   } else if(rootAlg_== "DUAL_SIMPLEX"){
      lpcplexParameter_.rootAlg_ = LPCPLEX::LP_SOLVER_DUAL_SIMPLEX;
   } else if(rootAlg_== "NETWORK_SIMPLEX"){
      lpcplexParameter_.rootAlg_ = LPCPLEX::LP_SOLVER_NETWORK_SIMPLEX;
   } else if(rootAlg_== "BARRIER"){
      lpcplexParameter_.rootAlg_ = LPCPLEX::LP_SOLVER_BARRIER;
   } else if(rootAlg_== "SIFTING"){
      lpcplexParameter_.rootAlg_ = LPCPLEX::LP_SOLVER_SIFTING;
   } else if(rootAlg_=="CONCURRENT"){
      lpcplexParameter_.rootAlg_ = LPCPLEX::LP_SOLVER_CONCURRENT;
   }else{
      throw RuntimeError("Unknown root alg");
   }

   // Node Algorithm
   if(nodeAlg_=="AUTO"){
      lpcplexParameter_.nodeAlg_ = LPCPLEX::LP_SOLVER_AUTO;
   } else if(nodeAlg_== "PRIMAL_SIMPLEX"){
      lpcplexParameter_.nodeAlg_ = LPCPLEX::LP_SOLVER_PRIMAL_SIMPLEX;
   } else if(nodeAlg_== "DUAL_SIMPLEX"){
      lpcplexParameter_.nodeAlg_ = LPCPLEX::LP_SOLVER_DUAL_SIMPLEX;
   } else if(nodeAlg_== "NETWORK_SIMPLEX"){
      lpcplexParameter_.nodeAlg_ = LPCPLEX::LP_SOLVER_NETWORK_SIMPLEX;
   } else if(nodeAlg_== "BARRIER"){
      lpcplexParameter_.nodeAlg_ = LPCPLEX::LP_SOLVER_BARRIER;
   } else if(nodeAlg_== "SIFTING"){
      lpcplexParameter_.nodeAlg_ = LPCPLEX::LP_SOLVER_SIFTING;
   } else if(nodeAlg_=="CONCURRENT"){
      lpcplexParameter_.nodeAlg_ = LPCPLEX::LP_SOLVER_CONCURRENT;
   }else{
      throw RuntimeError("Unknown root alg");
   }


   // MIPFOCUS
   if(mipFocus_ == "AUTO"){ 
      lpcplexParameter_.mipEmphasis_ =  LPCPLEX::MIP_EMPHASIS_BALANCED;
   } else if(mipFocus_ == "FEASIBILITY"){
      lpcplexParameter_.mipEmphasis_ =  LPCPLEX::MIP_EMPHASIS_FEASIBILITY;
   } else if(mipFocus_ == "OPTIMALITY"){
      lpcplexParameter_.mipEmphasis_ =  LPCPLEX::MIP_EMPHASIS_OPTIMALITY;
   } else if(mipFocus_ == "BESTBOUND"){
      lpcplexParameter_.mipEmphasis_ =  LPCPLEX::MIP_EMPHASIS_BESTBOUND;
   } else if(mipFocus_ == "HIDDEN"){
      lpcplexParameter_.mipEmphasis_ =  LPCPLEX::MIP_EMPHASIS_HIDDENFEAS;
   } else{
      throw RuntimeError("Unknown MIP-Focus");
   }

   // presolve
   if(presolve_ == "AUTO"){
      lpcplexParameter_.presolve_ = LPCPLEX::LP_PRESOLVE_AUTO;
   } else if(presolve_ == "OFF"){
      lpcplexParameter_.presolve_ = LPCPLEX::LP_PRESOLVE_OFF;
   } else if(presolve_ == "CONSERVATIVE"){ 
      lpcplexParameter_.presolve_ = LPCPLEX::LP_PRESOLVE_CONSEVATIVE;
   } else if(presolve_ == "AGRESSIVE"){
      lpcplexParameter_.presolve_ = LPCPLEX::LP_PRESOLVE_AGRESSIVE;
   }else{
      throw RuntimeError("Unknown presolve");
   }

   lpcplexParameter_.cutLevel_ = getCutLevel(cutLevel_);

   this-> template infer<LPCPLEX, TimingVisitorType, typename LPCPLEX::Parameter>(model, output, verbose, lpcplexParameter_);
}

template <class IO, class GM, class ACC>
const std::string LPCplexCaller<IO, GM, ACC>::name_ = "LPCPLEX";

} // namespace interface

} // namespace opengm

#endif /* LPCPLEX_CALLER_HXX_ */
