#ifndef LPGUROBI_CALLER_HXX_
#define LPGUROBI_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/lpgurobi.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LPGurobiCaller : public InferenceCallerBase<IO, GM, ACC, LPGurobiCaller<IO, GM, ACC> > {
public:
   typedef LPGurobi<GM, ACC> LPGUROBI;
   typedef InferenceCallerBase<IO, GM, ACC, LPGurobiCaller<IO, GM, ACC> > BaseClass;
   typedef typename LPGUROBI::VerboseVisitorType VerboseVisitorType;
   typedef typename LPGUROBI::EmptyVisitorType EmptyVisitorType;
   typedef typename LPGUROBI::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   LPGurobiCaller(IO& ioIn);
   virtual ~LPGurobiCaller();
protected:

   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename LPGUROBI::Parameter parameter_; 

   std::string rootAlg_;
   std::string nodeAlg_;
   std::string mipFocus_;
   std::string presolve_;
   std::string cutLevel_;
   std::string cliqueCutLevel_;
   std::string coverCutLevel_;
   std::string gubCutLevel_;
   std::string mirCutLevel_;
   std::string iboundCutLevel_;
   std::string flowcoverCutLevel_;
   std::string flowpathCutLevel_;

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
inline LPGurobiCaller<IO, GM, ACC>::LPGurobiCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of LPGurobi caller...", ioIn) {
   addArgument(BoolArgument(parameter_.integerConstraint_, "ic", "integerconstraint", "use integer constraints"));
   addArgument(IntArgument<>(parameter_.numberOfThreads_, "", "threads", "number of threads", parameter_.numberOfThreads_)); 
   //addArgument(IntArgument<>(parameter_.presolveLevel_, "", "presolve", "Presolve level: auto (-1), off (0), conservative (1), or aggressive (2)", parameter_.presolveLevel_));
   addArgument(BoolArgument(parameter_.verbose_, "v", "verbose", "used to activate verbose output"));
   addArgument(DoubleArgument<>(parameter_.cutUp_, "", "cutup", "cut up", parameter_.cutUp_));
   addArgument(DoubleArgument<>(parameter_.epOpt_, "", "epOpt", "Optimality tolerance", parameter_.epOpt_));
   addArgument(DoubleArgument<>(parameter_.epMrk_, "", "epMrk", "Markowitz tolerance", parameter_.epMrk_));
   addArgument(DoubleArgument<>(parameter_.epRHS_, "", "epRHS", "Feasibility tolerance", parameter_.epRHS_));
   addArgument(DoubleArgument<>(parameter_.epInt_, "", "epInt", "Integer feasibility tolerance", parameter_.epInt_));
   addArgument(DoubleArgument<>(parameter_.epGap_, "", "epGap", "Relative MIP gap tolerance", parameter_.epGap_));
   addArgument(DoubleArgument<>(parameter_.epAGap_, "", "epAGap", "Absolute MIP gap tolerance ", parameter_.epAGap_));
   double timeout =604800.0;
   addArgument(DoubleArgument<>(parameter_.timeLimit_,"","timeout","maximal runtime in seconds",timeout)); //default 1 week 

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
   addArgument(StringArgument<>(cliqueCutLevel_, "", "cliqueCutLevel", "Select clique cut level", possibleCutLevels.front(), possibleCutLevels)); 
   addArgument(StringArgument<>(coverCutLevel_, "", "coverCutLevel", "Select cover cut level", possibleCutLevels.front(), possibleCutLevels)); 
   addArgument(StringArgument<>(gubCutLevel_, "", "gubCutLevel", "Select generalized upper bound cut level", possibleCutLevels.front(), possibleCutLevels)); 
   addArgument(StringArgument<>(mirCutLevel_, "", "mirCutLevel", "Select mixed integer rounding cut level", possibleCutLevels.front(), possibleCutLevels)); 
   addArgument(StringArgument<>(iboundCutLevel_, "", "iboundCutLevel", "Select implied bound cut level", possibleCutLevels.front(), possibleCutLevels)); 
   addArgument(StringArgument<>(flowcoverCutLevel_, "", "flowcoverCutLevel", "Select flow-cover cut level", possibleCutLevels.front(), possibleCutLevels)); 
   addArgument(StringArgument<>(flowpathCutLevel_, "", "flowpathCutLevel", "Select flow-path cut level", possibleCutLevels.front(), possibleCutLevels)); 

}

template <class IO, class GM, class ACC>
inline LPGurobiCaller<IO, GM, ACC>::~LPGurobiCaller() {

}

template <class IO, class GM, class ACC>
inline void LPGurobiCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LPGurobi caller" << std::endl;

  // Root Algorithm
   if(rootAlg_=="AUTO"){
      parameter_.rootAlg_ = LPGUROBI::LP_SOLVER_AUTO;
   } else if(rootAlg_== "PRIMAL_SIMPLEX"){
      parameter_.rootAlg_ = LPGUROBI::LP_SOLVER_PRIMAL_SIMPLEX;
   } else if(rootAlg_== "DUAL_SIMPLEX"){
      parameter_.rootAlg_ = LPGUROBI::LP_SOLVER_DUAL_SIMPLEX;
   } else if(rootAlg_== "NETWORK_SIMPLEX"){
      parameter_.rootAlg_ = LPGUROBI::LP_SOLVER_NETWORK_SIMPLEX;
   } else if(rootAlg_== "BARRIER"){
      parameter_.rootAlg_ = LPGUROBI::LP_SOLVER_BARRIER;
   } else if(rootAlg_== "SIFTING"){
      parameter_.rootAlg_ = LPGUROBI::LP_SOLVER_SIFTING;
   } else if(rootAlg_=="CONCURRENT"){
      parameter_.rootAlg_ = LPGUROBI::LP_SOLVER_CONCURRENT;
   }else{
      throw RuntimeError("Unknown root alg");
   }

   // Node Algorithm
   if(nodeAlg_=="AUTO"){
      parameter_.nodeAlg_ = LPGUROBI::LP_SOLVER_AUTO;
   } else if(nodeAlg_== "PRIMAL_SIMPLEX"){
      parameter_.nodeAlg_ = LPGUROBI::LP_SOLVER_PRIMAL_SIMPLEX;
   } else if(nodeAlg_== "DUAL_SIMPLEX"){
      parameter_.nodeAlg_ = LPGUROBI::LP_SOLVER_DUAL_SIMPLEX;
   } else if(nodeAlg_== "NETWORK_SIMPLEX"){
      parameter_.nodeAlg_ = LPGUROBI::LP_SOLVER_NETWORK_SIMPLEX;
   } else if(nodeAlg_== "BARRIER"){
      parameter_.nodeAlg_ = LPGUROBI::LP_SOLVER_BARRIER;
   } else if(nodeAlg_== "SIFTING"){
      parameter_.nodeAlg_ = LPGUROBI::LP_SOLVER_SIFTING;
   } else if(nodeAlg_=="CONCURRENT"){
      parameter_.nodeAlg_ = LPGUROBI::LP_SOLVER_CONCURRENT;
   }else{
      throw RuntimeError("Unknown root alg");
   }


   // MIPFOCUS
   if(mipFocus_ == "AUTO"){ 
      parameter_.mipFocus_ =  LPGUROBI::MIP_EMPHASIS_BALANCED;
   } else if(mipFocus_ == "FEASIBILITY"){
      parameter_.mipFocus_ =  LPGUROBI::MIP_EMPHASIS_FEASIBILITY;
   } else if(mipFocus_ == "OPTIMALITY"){
      parameter_.mipFocus_ =  LPGUROBI::MIP_EMPHASIS_OPTIMALITY;
   } else if(mipFocus_ == "BESTBOUND"){
      parameter_.mipFocus_ =  LPGUROBI::MIP_EMPHASIS_BESTBOUND;
   } else if(mipFocus_ == "HIDDEN"){
      parameter_.mipFocus_ =  LPGUROBI::MIP_EMPHASIS_HIDDENFEAS;
   } else{
      throw RuntimeError("Unknown MIP-Focus");
   }

   // presolve
   if(presolve_ == "AUTO"){
      parameter_.presolve_ = LPGUROBI::LP_PRESOLVE_AUTO;
   } else if(presolve_ == "OFF"){
      parameter_.presolve_ = LPGUROBI::LP_PRESOLVE_OFF;
   } else if(presolve_ == "CONSERVATIVE"){ 
      parameter_.presolve_ = LPGUROBI::LP_PRESOLVE_CONSEVATIVE;
   } else if(presolve_ == "AGRESSIVE"){
      parameter_.presolve_ = LPGUROBI::LP_PRESOLVE_AGRESSIVE;
   }else{
      throw RuntimeError("Unknown presolve");
   }

   // cutLevel
   parameter_.cutLevel_       =  getCutLevel(cutLevel_);
   parameter_.cliqueCutLevel_ =  getCutLevel(cliqueCutLevel_);
   parameter_.coverCutLevel_  =  getCutLevel(coverCutLevel_);
   parameter_.gubCutLevel_    =  getCutLevel(gubCutLevel_);
   parameter_.mirCutLevel_    =  getCutLevel(mirCutLevel_);
   parameter_.iboundCutLevel_ =  getCutLevel(iboundCutLevel_);
   parameter_.flowcoverCutLevel_  =  getCutLevel(flowcoverCutLevel_);
   parameter_.flowpathCutLevel_ =  getCutLevel(flowpathCutLevel_);
   
   this-> template infer<LPGUROBI, TimingVisitorType, typename LPGUROBI::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC>
const std::string LPGurobiCaller<IO, GM, ACC>::name_ = "LPGUROBI";

} // namespace interface

} // namespace opengm

#endif /* LPGUROBI_CALLER_HXX_ */
