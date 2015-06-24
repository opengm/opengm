#ifndef OPENGM_LPDEF_HXX_
#define OPENGM_LPDEF_HXX_

namespace opengm {
   class LPDef{
   public:
      enum LP_SOLVER {LP_SOLVER_AUTO, LP_SOLVER_PRIMAL_SIMPLEX, LP_SOLVER_DUAL_SIMPLEX, LP_SOLVER_NETWORK_SIMPLEX, LP_SOLVER_BARRIER, LP_SOLVER_SIFTING, LP_SOLVER_CONCURRENT};
      enum LP_PRESOLVE{LP_PRESOLVE_AUTO, LP_PRESOLVE_OFF, LP_PRESOLVE_CONSERVATIVE, LP_PRESOLVE_AGGRESSIVE};
      enum MIP_EMPHASIS{MIP_EMPHASIS_BALANCED, MIP_EMPHASIS_FEASIBILITY, MIP_EMPHASIS_OPTIMALITY, MIP_EMPHASIS_BESTBOUND, MIP_EMPHASIS_HIDDENFEAS};
      enum MIP_CUT{MIP_CUT_DEFAULT, MIP_CUT_AUTO, MIP_CUT_OFF, MIP_CUT_ON, MIP_CUT_AGGRESSIVE, MIP_CUT_VERYAGGRESSIVE};

      static const int          default_numberOfThreads_;   // number of threads (0=autoselect)
      static const bool         default_verbose_;           // switch on/off verbose mode
      static const double       default_cutUp_;             // upper cutoff
      static const double       default_epOpt_;             // Optimality tolerance
      static const double       default_epMrk_;             // Markowitz tolerance
      static const double       default_epRHS_;             // Feasibility Tolerance
      static const double       default_epInt_;             // amount by which an integer variable can differ from an integer
      static const double       default_epAGap_;            // Absolute MIP gap tolerance
      static const double       default_epGap_;             // Relative MIP gap tolerance
      static const double       default_workMem_;           // maximal amount of memory in MB used for workspace
      static const double       default_treeMemoryLimit_;   // maximal amount of memory in MB used for tree
      static const double       default_timeLimit_;         // maximal time in seconds the solver has
      static const int          default_probingLevel_;
      static const LP_SOLVER    default_rootAlg_;
      static const LP_SOLVER    default_nodeAlg_;
      static const MIP_EMPHASIS default_mipEmphasis_;
      static const LP_PRESOLVE  default_presolve_;
      static const MIP_CUT      default_cutLevel_;          // Determines whether or not to cuts for the problem and how aggressively (will be overruled by specific ones).
      static const MIP_CUT      default_cliqueCutLevel_;    // Determines whether or not to generate clique cuts for the problem and how aggressively.
      static const MIP_CUT      default_coverCutLevel_;     // Determines whether or not to generate cover cuts for the problem and how aggressively.
      static const MIP_CUT      default_gubCutLevel_;       // Determines whether or not to generate generalized upper bound (GUB) cuts for the problem and how aggressively.
      static const MIP_CUT      default_mirCutLevel_;       // Determines whether or not mixed integer rounding (MIR) cuts should be generated for the problem and how aggressively.
      static const MIP_CUT      default_iboundCutLevel_;    // Determines whether or not to generate implied bound cuts for the problem and how aggressively.
      static const MIP_CUT      default_flowcoverCutLevel_; // Determines whether or not to generate flow cover cuts for the problem and how aggressively.
      static const MIP_CUT      default_flowpathCutLevel_;  // Determines whether or not to generate flow path cuts for the problem and how aggressively.
      static const MIP_CUT      default_disjunctCutLevel_;  // Determines whether or not to generate disjunctive cuts for the problem and how aggressively.
      static const MIP_CUT      default_gomoryCutLevel_;    // Determines whether or not to generate gomory fractional cuts for the problem and how aggressively.
   };

#ifndef OPENGM_LPDEF_NO_SYMBOLS
   const int                 LPDef::default_numberOfThreads_(0);
   const bool                LPDef::default_verbose_(false);
   const double              LPDef::default_cutUp_(1.0e+75);
   const double              LPDef::default_epOpt_(1e-5);
   const double              LPDef::default_epMrk_(0.01);
   const double              LPDef::default_epRHS_(1e-5);
   const double              LPDef::default_epInt_(1e-5);
   const double              LPDef::default_epAGap_(0.0);
   const double              LPDef::default_epGap_(0.0);

   const double              LPDef::default_workMem_(128.0);
   const double              LPDef::default_treeMemoryLimit_(1e+75);
   const double              LPDef::default_timeLimit_(1e+75);
   const int                 LPDef::default_probingLevel_(0);
   const LPDef::LP_SOLVER    LPDef::default_rootAlg_(LP_SOLVER_AUTO);
   const LPDef::LP_SOLVER    LPDef::default_nodeAlg_(LP_SOLVER_AUTO);
   const LPDef::MIP_EMPHASIS LPDef::default_mipEmphasis_(MIP_EMPHASIS_BALANCED);
   const LPDef::LP_PRESOLVE  LPDef::default_presolve_(LP_PRESOLVE_AUTO);
   const LPDef::MIP_CUT      LPDef::default_cutLevel_(MIP_CUT_AUTO);
   const LPDef::MIP_CUT      LPDef::default_cliqueCutLevel_(MIP_CUT_DEFAULT);
   const LPDef::MIP_CUT      LPDef::default_coverCutLevel_(MIP_CUT_DEFAULT);
   const LPDef::MIP_CUT      LPDef::default_gubCutLevel_(MIP_CUT_DEFAULT);
   const LPDef::MIP_CUT      LPDef::default_mirCutLevel_(MIP_CUT_DEFAULT);
   const LPDef::MIP_CUT      LPDef::default_iboundCutLevel_(MIP_CUT_DEFAULT);
   const LPDef::MIP_CUT      LPDef::default_flowcoverCutLevel_(MIP_CUT_DEFAULT);
   const LPDef::MIP_CUT      LPDef::default_flowpathCutLevel_(MIP_CUT_DEFAULT);
   const LPDef::MIP_CUT      LPDef::default_disjunctCutLevel_(MIP_CUT_DEFAULT);
   const LPDef::MIP_CUT      LPDef::default_gomoryCutLevel_(MIP_CUT_DEFAULT);
#endif
   
}

#endif
