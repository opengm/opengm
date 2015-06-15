#ifndef OPENGM_LP_INFERENCE_CALLER_BASE_HXX_
#define OPENGM_LP_INFERENCE_CALLER_BASE_HXX_

#include <opengm/opengm.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
class LPInferenceCallerBase : public InferenceCallerBase<IO, GM, ACC, LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC> > {
public:
   typedef LP_INFERENCE_TYPE                                                                        LPInferenceType;
   typedef InferenceCallerBase<IO, GM, ACC, LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC> > BaseClass;
   typedef typename LPInferenceType::VerboseVisitorType                                             VerboseVisitorType;
   typedef typename LPInferenceType::EmptyVisitorType                                               EmptyVisitorType;
   typedef typename LPInferenceType::TimingVisitorType                                              TimingVisitorType;

   LPInferenceCallerBase(IO& ioIn, const std::string& name, const std::string& description);
   virtual ~LPInferenceCallerBase();

protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename LPInferenceType::Parameter lpinferenceParameter_;

   std::string rootAlgorithm_;
   std::string nodeAlgorithm_;
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
   std::string disjunctCutLevel_;
   std::string gomoryCutLevel_;

   std::string relaxation_;
   std::string challengeHeuristic_;

   static LPDef::LP_SOLVER getAlgorithm(const std::string& algorithm);
   static LPDef::MIP_EMPHASIS getMIPEmphasis(const std::string& MIPEmphasis);
   static LPDef::LP_PRESOLVE getPresolve(const std::string& presolve);
   static LPDef::MIP_CUT getMIPCut(const std::string& cutLevel);

   static typename LPInferenceType::Parameter::Relaxation getRelaxation(const std::string& relaxation);
   static typename LPInferenceType::Parameter::ChallengeHeuristic getHeuristic(const std::string& heuristic);
};

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::LPInferenceCallerBase(IO& ioIn, const std::string& name, const std::string& description)
   : BaseClass(name, description, ioIn) {
   addArgument(IntArgument<>(lpinferenceParameter_.numberOfThreads_, "", "threads", "The number of threads used for Optimization (0 = autoselect).", lpinferenceParameter_.numberOfThreads_));
   addArgument(BoolArgument(lpinferenceParameter_.verbose_, "v", "verbose", "Used to activate verbose output."));
   addArgument(DoubleArgument<>(lpinferenceParameter_.cutUp_, "", "cutup", "Upper cutoff tolerance.", lpinferenceParameter_.cutUp_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.epOpt_, "", "epOpt", "Optimality tolerance.", lpinferenceParameter_.epOpt_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.epMrk_, "", "epMrk", "Markowitz tolerance.", lpinferenceParameter_.epMrk_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.epRHS_, "", "epRHS", "Feasibility tolerance.", lpinferenceParameter_.epRHS_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.epInt_, "", "epInt", "Amount by which an integer variable can differ from an integer.", lpinferenceParameter_.epInt_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.epAGap_, "", "epAGap", "Absolute MIP gap tolerance.", lpinferenceParameter_.epAGap_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.epGap_, "", "epGap", "Relative MIP gap tolerance.", lpinferenceParameter_.epGap_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.workMem_, "", "workMem", "Maximal amount of memory in MB used for workspace.", lpinferenceParameter_.workMem_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.treeMemoryLimit_, "", "treeMem", "Maximal amount of memory in MB used for tree.", lpinferenceParameter_.treeMemoryLimit_));
   addArgument(DoubleArgument<>(lpinferenceParameter_.timeLimit_, "", "maxTime", "Maximal time in seconds the solver has.", lpinferenceParameter_.timeLimit_));
   addArgument(IntArgument<>(lpinferenceParameter_.probingLevel_, "", "probingLevel", "Amount of probing on variables to be performed before MIP branching.", lpinferenceParameter_.probingLevel_));

   std::vector<std::string> possibleAlgorithms;
   possibleAlgorithms.push_back("AUTO");
   possibleAlgorithms.push_back("PRIMAL_SIMPLEX");
   possibleAlgorithms.push_back("DUAL_SIMPLEX");
   possibleAlgorithms.push_back("NETWORK_SIMPLEX");
   possibleAlgorithms.push_back("BARRIER");
   possibleAlgorithms.push_back("SIFTING");
   possibleAlgorithms.push_back("CONCURRENT");
   addArgument(StringArgument<>(rootAlgorithm_, "", "rootAlg", "Select which algorithm is used to solve continuous models or to solve the root relaxation of a MIP.", possibleAlgorithms.front(), possibleAlgorithms));
   addArgument(StringArgument<>(nodeAlgorithm_, "", "nodeAlg", "Select which algorithm is used to solve the subproblems in a MIP after the initial relaxation has been solved.", possibleAlgorithms.front(), possibleAlgorithms));

   std::vector<std::string> possibleMIPEmphasis;
   possibleMIPEmphasis.push_back("AUTO");
   possibleMIPEmphasis.push_back("FEASIBILITY");
   possibleMIPEmphasis.push_back("OPTIMALITY");
   possibleMIPEmphasis.push_back("BESTBOUND");
   possibleMIPEmphasis.push_back("HIDDEN");
   addArgument(StringArgument<>(mipFocus_, "", "mipFocus", "Controls trade-offs between speed, feasibility, optimality, and moving bounds in a MIP.", possibleMIPEmphasis.front(), possibleMIPEmphasis));

   std::vector<std::string> possiblePresolve;
   possiblePresolve.push_back("AUTO");
   possiblePresolve.push_back("OFF");
   possiblePresolve.push_back("CONSERVATIVE");
   possiblePresolve.push_back("AGGRESSIVE");
   addArgument(StringArgument<>(presolve_, "", "presolve", "Controls how aggressive presolve is performed during preprocessing.", possiblePresolve.front(), possiblePresolve));

   std::vector<std::string> possibleMIPCutLevels;
   possibleMIPCutLevels.push_back("DEFAULT");
   possibleMIPCutLevels.push_back("AUTO");
   possibleMIPCutLevels.push_back("OFF");
   possibleMIPCutLevels.push_back("ON");
   possibleMIPCutLevels.push_back("AGGRESSIVE");
   possibleMIPCutLevels.push_back("VERYAGGRESSIVE");
   addArgument(StringArgument<>(cutLevel_, "", "cutLevel", "Determines whether or not to generate cuts for the problem and how aggressively (will be overruled by specific ones).", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(cliqueCutLevel_, "", "cliqueCutLevel", "Determines whether or not to generate clique cuts for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(coverCutLevel_, "", "coverCutLevel", "Determines whether or not to generate cover cuts for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(gubCutLevel_, "", "gubCutLevel", "Determines whether or not to generate generalized upper bound (GUB) cuts for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(mirCutLevel_, "", "mirCutLevel", "Determines whether or not mixed integer rounding (MIR) cuts should be generated for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(iboundCutLevel_, "", "iboundCutLevel", "Determines whether or not to generate implied bound cuts for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(flowcoverCutLevel_, "", "flowcoverCutLevel", "Determines whether or not to generate flow cover cuts for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(flowpathCutLevel_, "", "flowpathCutLevel", "Determines whether or not to generate flow path cuts for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(disjunctCutLevel_, "", "disjunctCutLevel", "Determines whether or not to generate disjunctive cuts for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));
   addArgument(StringArgument<>(gomoryCutLevel_, "", "gomoryCutLevel", "Determines whether or not to generate gomory fractional cuts for the problem and how aggressively.", possibleMIPCutLevels.front(), possibleMIPCutLevels));

   addArgument(BoolArgument(lpinferenceParameter_.integerConstraintNodeVar_, "icnv", "integerConstraintNodeVar", "Use integer constraints for node variables."));
   addArgument(BoolArgument(lpinferenceParameter_.integerConstraintFactorVar_, "icfv", "integerConstraintFactorVar", "Use integer constraints for factor variables."));
   addArgument(BoolArgument(lpinferenceParameter_.useSoftConstraints_, "sc", "softConstraints", "If constraint factors are present in the model add them as soft constraints e.g. treat them as normal factors."));
   addArgument(BoolArgument(lpinferenceParameter_.useFunctionTransfer_, "ft", "functionTransfer", "Use function transfer if available to generate more efficient lp models."));
   addArgument(BoolArgument(lpinferenceParameter_.mergeParallelFactors_, "mpf", "mergeParallelFactors", "Merge factors which are connected to the same set of variables."));
   addArgument(BoolArgument(lpinferenceParameter_.nameConstraints_, "", "nameConstraints", "Create unique names for the linear constraints added to the model (might be helpful for debugging models)."));

   std::vector<std::string> possibleRelaxations;
   possibleRelaxations.push_back("LOCAL");
   possibleRelaxations.push_back("LOOSE");
   possibleRelaxations.push_back("TIGHT");
   addArgument(StringArgument<>(relaxation_, "", "relaxation", "Relaxation method.", possibleRelaxations.front(), possibleRelaxations));

   addArgument(Size_TArgument<>(lpinferenceParameter_.maxNumIterations_, "", "maxNumIterations", "Maximum number of tightening iterations (infinite if set to 0).", lpinferenceParameter_.maxNumIterations_));
   addArgument(Size_TArgument<>(lpinferenceParameter_.maxNumConstraintsPerIter_, "", "maxNumConstraintsPerIter", "Maximum number of added constraints per tightening iteration (all if set to 0).", lpinferenceParameter_.maxNumConstraintsPerIter_));

   std::vector<std::string> possibleChallengeHeuristics;
   possibleChallengeHeuristics.push_back("RANDOM");
   possibleChallengeHeuristics.push_back("WEIGHTED");
   addArgument(StringArgument<>(challengeHeuristic_, "", "challengeHeuristic", "Heuristic on how to select violated constraints.", possibleChallengeHeuristics.front(), possibleChallengeHeuristics));

   addArgument(DoubleArgument<>(lpinferenceParameter_.tolerance_, "", "tolerance", "Tolerance for violation of linear constraints.", lpinferenceParameter_.tolerance_));
}

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::~LPInferenceCallerBase() {

}

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline void LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LP inference caller" << std::endl;

   lpinferenceParameter_.rootAlg_ = getAlgorithm(rootAlgorithm_);
   lpinferenceParameter_.nodeAlg_ = getAlgorithm(nodeAlgorithm_);

   lpinferenceParameter_.mipEmphasis_ = getMIPEmphasis(mipFocus_);

   lpinferenceParameter_.presolve_ = getPresolve(presolve_);

   lpinferenceParameter_.cutLevel_ = getMIPCut(cutLevel_);
   lpinferenceParameter_.cliqueCutLevel_ = getMIPCut(cliqueCutLevel_);
   lpinferenceParameter_.coverCutLevel_ = getMIPCut(coverCutLevel_);
   lpinferenceParameter_.gubCutLevel_ = getMIPCut(gubCutLevel_);
   lpinferenceParameter_.mirCutLevel_ = getMIPCut(mirCutLevel_);
   lpinferenceParameter_.iboundCutLevel_ = getMIPCut(iboundCutLevel_);
   lpinferenceParameter_.flowcoverCutLevel_ = getMIPCut(flowcoverCutLevel_);
   lpinferenceParameter_.flowpathCutLevel_ = getMIPCut(flowpathCutLevel_);
   lpinferenceParameter_.disjunctCutLevel_ = getMIPCut(disjunctCutLevel_);
   lpinferenceParameter_.gomoryCutLevel_ = getMIPCut(gomoryCutLevel_);

   lpinferenceParameter_.relaxation_ = getRelaxation(relaxation_);
   lpinferenceParameter_.challengeHeuristic_ = getHeuristic(challengeHeuristic_);


   this-> template infer<LPInferenceType, TimingVisitorType, typename LPInferenceType::Parameter>(model, output, verbose, lpinferenceParameter_);
}

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline LPDef::LP_SOLVER LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::getAlgorithm(const std::string& algorithm) {
   if(algorithm == "AUTO") {
      return LPDef::LP_SOLVER_AUTO;
   } else if(algorithm == "PRIMAL_SIMPLEX") {
      return LPDef::LP_SOLVER_PRIMAL_SIMPLEX;
   } else if(algorithm == "DUAL_SIMPLEX") {
      return LPDef::LP_SOLVER_DUAL_SIMPLEX;
   } else if(algorithm == "NETWORK_SIMPLEX") {
      return LPDef::LP_SOLVER_NETWORK_SIMPLEX;
   } else if(algorithm == "BARRIER") {
      return LPDef::LP_SOLVER_BARRIER;
   } else if(algorithm == "SIFTING") {
      return LPDef::LP_SOLVER_SIFTING;
   } else if(algorithm == "CONCURRENT") {
      return LPDef::LP_SOLVER_CONCURRENT;
   } else {
      throw std::runtime_error("Unknown Algorithm.");
   }
}

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline LPDef::MIP_EMPHASIS LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::getMIPEmphasis(const std::string& MIPEmphasis) {
   if(MIPEmphasis == "AUTO") {
      return LPDef::MIP_EMPHASIS_BALANCED;
   } else if(MIPEmphasis == "FEASIBILITY") {
      return LPDef::MIP_EMPHASIS_FEASIBILITY;
   } else if(MIPEmphasis == "OPTIMALITY") {
      return LPDef::MIP_EMPHASIS_OPTIMALITY;
   } else if(MIPEmphasis == "BESTBOUND") {
      return LPDef::MIP_EMPHASIS_BESTBOUND;
   } else if(MIPEmphasis == "HIDDEN") {
      return LPDef::MIP_EMPHASIS_HIDDENFEAS;
   } else {
      throw std::runtime_error("Unknown MIP emphasis.");
   }
}

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline LPDef::LP_PRESOLVE LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::getPresolve(const std::string& presolve) {
   if(presolve == "AUTO") {
      return LPDef::LP_PRESOLVE_AUTO;
   } else if(presolve == "OFF") {
      return LPDef::LP_PRESOLVE_OFF;
   } else if(presolve == "CONSERVATIVE") {
      return LPDef::LP_PRESOLVE_CONSERVATIVE;
   } else if(presolve == "AGGRESSIVE") {
      return LPDef::LP_PRESOLVE_AGGRESSIVE;
   } else {
      throw std::runtime_error("Unknown presolve.");
   }
}

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline LPDef::MIP_CUT LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::getMIPCut(const std::string& cutLevel) {
   if(cutLevel == "DEFAULT") {
      return LPDef::MIP_CUT_DEFAULT;
   } else if(cutLevel == "AUTO") {
      return LPDef::MIP_CUT_AUTO;
   } else if(cutLevel == "OFF") {
      return LPDef::MIP_CUT_OFF;
   } else if(cutLevel == "ON") {
      return LPDef::MIP_CUT_ON;
   } else if(cutLevel == "AGGRESSIVE") {
      return LPDef::MIP_CUT_AGGRESSIVE;
   } else if(cutLevel == "VERYAGGRESSIVE") {
      return LPDef::MIP_CUT_VERYAGGRESSIVE;
   } else {
      throw std::runtime_error("Unknown MIP cut.");
   }
}

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline typename LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::LPInferenceType::Parameter::Relaxation LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::getRelaxation(const std::string& relaxation) {
   if(relaxation == "LOCAL") {
      return LPInferenceType::Parameter::LocalPolytope;
   } else if(relaxation == "LOOSE") {
      return LPInferenceType::Parameter::LoosePolytope;
   } else if(relaxation == "TIGHT") {
      return LPInferenceType::Parameter::TightPolytope;
   } else {
      throw std::runtime_error("Unknown relaxation.");
   }
}

template <class LP_INFERENCE_TYPE, class IO, class GM, class ACC>
inline typename LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::LPInferenceType::Parameter::ChallengeHeuristic LPInferenceCallerBase<LP_INFERENCE_TYPE, IO, GM, ACC>::getHeuristic(const std::string& heuristic) {
   if(heuristic == "RANDOM") {
      return LPInferenceType::Parameter::Random;
   } else if(heuristic == "WEIGHTED") {
      return LPInferenceType::Parameter::Weighted;
   } else {
      throw std::runtime_error("Unknown heuristic.");
   }
}

} // namespace interface

} // namespace opengm

#endif /* OPENGM_LP_INFERENCE_CALLER_BASE_HXX_ */
