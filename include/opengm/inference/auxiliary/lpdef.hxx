#ifndef OPENGM_LPDEF_HXX_
#define OPENGM_LPDEF_HXX_

namespace opengm {
   class LPDef{
   public:
      enum LP_SOLVER {LP_SOLVER_AUTO,  LP_SOLVER_PRIMAL_SIMPLEX,  LP_SOLVER_DUAL_SIMPLEX,  LP_SOLVER_NETWORK_SIMPLEX,  LP_SOLVER_BARRIER,  LP_SOLVER_SIFTING,  LP_SOLVER_CONCURRENT};
      enum LP_PRESOLVE{LP_PRESOLVE_AUTO, LP_PRESOLVE_OFF,  LP_PRESOLVE_CONSEVATIVE,  LP_PRESOLVE_AGRESSIVE}; 
      enum MIP_EMPHASIS{MIP_EMPHASIS_BALANCED, MIP_EMPHASIS_FEASIBILITY, MIP_EMPHASIS_OPTIMALITY, MIP_EMPHASIS_BESTBOUND, MIP_EMPHASIS_HIDDENFEAS};
      enum MIP_CUT{MIP_CUT_DEFAULT,MIP_CUT_AUTO,MIP_CUT_OFF,MIP_CUT_ON,MIP_CUT_AGGRESSIVE,MIP_CUT_VERYAGGRESSIVE};


      double default_cutUp_; // upper cutoff
       
      double default_epOpt_; // Optimality tolerance  
      double default_epMrk_; // Markowitz tolerance 
      double default_epRHS_; // Feasibility Tolerance 
      double default_epInt_; // amount by which an integer variable can differ from an integer 
      double default_epAGap_;// Absolute MIP gap tolerance 
      double default_epGap_; // Relative MIP gap tolerance

      LPDef():
         default_cutUp_( 1.0e+75 ),
         default_epOpt_( 1e-5    ),
         default_epMrk_( 0.01    ),
         default_epRHS_( 1e-5    ),
         default_epInt_( 1e-5    ),
         default_epAGap_(0.0     ),
         default_epGap_( 0.0     )
         {}
  

   };
}
#endif 
