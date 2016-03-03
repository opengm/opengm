#pragma once
#ifndef OPENGM_LP_CPLEX_HXX
#define OPENGM_LP_CPLEX_HXX

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <typeinfo>

#include <ilcplex/ilocplex.h>

#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/opengm.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/auxiliary/lpdef.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// \brief Optimization by Linear Programming (LP) or Integer LP using IBM ILOG CPLEX\n\n
/// http://www.ilog.com/products/cplex/
///
/// The optimization problem is reformulated as an LP or ILP.
/// For the LP, a first order local polytope approximation of the
/// marginal polytope is used, i.e. the affine instead of the convex 
/// hull.
/// 
/// IBM ILOG CPLEX is a commercial product that is 
/// free for accadamical use.
///
/// \ingroup inference 
template<class GM, class ACC>
class LPCplex : public Inference<GM, ACC>, public LPDef {
public:
   typedef ACC AccumulationType; 
   typedef ACC AccumulatorType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS; 
   typedef visitors::VerboseVisitor<LPCplex<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<LPCplex<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<LPCplex<GM,ACC> >  TimingVisitorType;

    template<class _GM>
    struct RebindGm{
        typedef LPCplex<_GM, ACC> type;
    };

    template<class _GM,class _ACC>
    struct RebindGmAndAcc{
        typedef LPCplex<_GM, _ACC > type;
    };


//   enum LP_SOLVER {LP_SOLVER_AUTO,  LP_SOLVER_PRIMAL_SIMPLEX,  LP_SOLVER_DUAL_SIMPLEX,  LP_SOLVER_NETWORK_SIMPLEX,  LP_SOLVER_BARRIER,  LP_SOLVER_SIFTING,  LP_SOLVER_CONCURRENT};
//   enum LP_PRESOLVE{LP_PRESOLVE_AUTO, LP_PRESOLVE_OFF,  LP_PRESOLVE_CONSEVATIVE,  LP_PRESOLVE_AGRESSIVE}; 
//   enum MIP_EMPHASIS{MIP_EMPHASIS_BALANCED, MIP_EMPHASIS_FEASIBILITY, MIP_EMPHASIS_OPTIMALITY, MIP_EMPHASIS_BESTBOUND, MIP_EMPHASIS_HIDDENFEAS};
 
   class Parameter {
   public:
      bool   integerConstraint_; // ILP=true, 1order-LP=false
      int    numberOfThreads_;    // number of threads (0=autosect)
      bool   verbose_;           // switch on/off verbode mode 
      double cutUp_;           // upper cutoff
      double epOpt_;           // Optimality tolerance  
      double epMrk_;           // Markowitz tolerance 
      double epRHS_;           // Feasibility Tolerance 
      double epInt_;           // amount by which an integer variable can differ from an integer 
      double epAGap_;          // Absolute MIP gap tolerance 
      double epGap_;           // Relative MIP gap tolerance
      double workMem_;         // maximal ammount of memory in MB used for workspace
      double treeMemoryLimit_; // maximal ammount of memory in MB used for treee
      double timeLimit_;       // maximal time in seconds the solver has
      int    probeingLevel_;
      //int coverCutLevel_;
      //int disjunctiverCutLevel_;
      //int cliqueCutLevel_;
      //int MIRCutLevel_;
      LP_SOLVER    rootAlg_;
      LP_SOLVER    nodeAlg_;
      LP_PRESOLVE  presolve_;
      MIP_EMPHASIS mipEmphasis_;
      MIP_CUT      cutLevel_;       // Determines whether or not to cuts for the problem and how aggressively (will be overruled by specific ones). 
      MIP_CUT      cliqueCutLevel_; // Determines whether or not to generate clique cuts for the problem and how aggressively. 
      MIP_CUT      coverCutLevel_;  // Determines whether or not to generate cover cuts for the problem and how aggressively. 
      MIP_CUT      gubCutLevel_;    // Determines whether or not to generate generalized upper bound (GUB) cuts for the problem and how aggressively. 
      MIP_CUT      mirCutLevel_;    // Determines whether or not mixed integer rounding (MIR) cuts should be generated for the problem and how aggressively.  
      MIP_CUT      iboundCutLevel_; // Determines whether or not to generate implied bound cuts for the problem and how aggressively.
      MIP_CUT      flowcoverCutLevel_; //Determines whether or not to generate flow cover cuts for the problem and how aggressively. 
      MIP_CUT      flowpathCutLevel_; //Determines whether or not to generate flow path cuts for the problem and how aggressively.
      MIP_CUT      disjunctCutLevel_; // Determines whether or not to generate disjunctive cuts for the problem and how aggressively.
      MIP_CUT      gomoryCutLevel_; // Determines whether or not to generate gomory fractional cuts for the problem and how aggressively. 

      template<class P>
      Parameter(
        const P & p
      ):

        integerConstraint_(p.integerConstraint_),
        numberOfThreads_(p.numberOfThreads_),
        verbose_(p.verbose_),
        cutUp_(p.cutUp_),
        epOpt_(p.epOpt_),
        epMrk_(p.epMrk_),
        epRHS_(p.epRHS_),
        epInt_(p.epInt_),
        epAGap_(p.epAGap_),
        epGap_(p.epGap_),
        workMem_(p.workMem_),
        treeMemoryLimit_(p.treeMemoryLimit_),
        timeLimit_(p.timeLimit_),
        probeingLevel_(p.probeingLevel_),
        rootAlg_(p.rootAlg_),
        nodeAlg_(p.nodeAlg_),
        presolve_(p.presolve_),
        mipEmphasis_(p.mipEmphasis_),
        cutLevel_(p.cutLevel_),
        cliqueCutLevel_(p.cliqueCutLevel_),
        coverCutLevel_(p.coverCutLevel_),
        gubCutLevel_(p.gubCutLevel_),
        mirCutLevel_(p.mirCutLevel_),
        iboundCutLevel_(p.iboundCutLevel_),
        flowcoverCutLevel_(p.flowcoverCutLevel_),
        flowpathCutLevel_(p.flowpathCutLevel_),
        disjunctCutLevel_(p.disjunctCutLevel_),
        gomoryCutLevel_(p.gomoryCutLevel_)
      {

      }

      /// constructor
      /// \param cutUp upper cutoff - assume that: min_x f(x) <= cutUp 
      /// \param epGap relative stopping criterion: |bestnode-bestinteger| / (1e-10 + |bestinteger|) <= epGap
      Parameter
      (
         int numberOfThreads = 0
      )
      :  numberOfThreads_(numberOfThreads), 
         //integerConstraint_(false), 
         verbose_(false),
         workMem_(128.0),
         treeMemoryLimit_(1e+75),
         timeLimit_(1e+75),
         probeingLevel_(0),
         //  coverCutLevel_(0),
         //disjunctiverCutLevel_(0),
         //cliqueCutLevel_(0),
         //MIRCutLevel_(0),
         rootAlg_(LP_SOLVER_AUTO),
         nodeAlg_(LP_SOLVER_AUTO),
         presolve_(LP_PRESOLVE_AUTO),
         mipEmphasis_(MIP_EMPHASIS_BALANCED),
         cutLevel_(MIP_CUT_AUTO), 
         cliqueCutLevel_(MIP_CUT_AUTO),
         coverCutLevel_(MIP_CUT_AUTO), 
         gubCutLevel_(MIP_CUT_AUTO),
         mirCutLevel_(MIP_CUT_AUTO), 
         iboundCutLevel_(MIP_CUT_AUTO),
         flowcoverCutLevel_(MIP_CUT_AUTO), 
         flowpathCutLevel_(MIP_CUT_AUTO),
         disjunctCutLevel_(MIP_CUT_AUTO), 
         gomoryCutLevel_(MIP_CUT_AUTO)
         {


            
            numberOfThreads_   = numberOfThreads; 
            integerConstraint_ = false; 
            LPDef lpdef;
            cutUp_ = lpdef.default_cutUp_;
            epOpt_ = lpdef.default_epOpt_;
            epMrk_ = lpdef.default_epMrk_;
            epRHS_ = lpdef.default_epRHS_;
            epInt_ = lpdef.default_epInt_;
            epAGap_= lpdef.default_epAGap_;
            epGap_ = lpdef.default_epGap_;
         };

      int getCutLevel(MIP_CUT cl){
         switch(cl){
         case MIP_CUT_AUTO:
            return 0;
         case MIP_CUT_OFF:
            return -1;
         case  MIP_CUT_ON:
            return 1;
         case MIP_CUT_AGGRESSIVE:
            return 2;
         case MIP_CUT_VERYAGGRESSIVE:
            return 3;
         }
         return 0;
      }
   };

   LPCplex(const GraphicalModelType&, const Parameter& = Parameter());
   ~LPCplex();
   virtual std::string name() const 
      { return "LPCplex"; }
   const GraphicalModelType& graphicalModel() const;
   virtual InferenceTermination infer();
   template<class VisitorType>
   InferenceTermination infer(VisitorType&);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   virtual InferenceTermination args(std::vector<std::vector<LabelType> >&) const 
      { return UNKNOWN; };
   void variable(const size_t, IndependentFactorType& out) const;     
   void factorVariable(const size_t, IndependentFactorType& out) const;
   typename GM::ValueType bound() const; 
   typename GM::ValueType value() const;
   void setStartingPoint( typename std::vector<LabelType>::const_iterator );


   size_t lpNodeVi(const IndexType variableIndex,const LabelType label)const;
   size_t lpFactorVi(const IndexType factorIndex,const size_t labelingIndex)const;
   template<class LABELING_ITERATOR>
   size_t lpFactorVi(const IndexType factorIndex,LABELING_ITERATOR labelingBegin,LABELING_ITERATOR labelingEnd)const;
   template<class LPVariableIndexIterator, class CoefficientIterator>
   void addConstraint(LPVariableIndexIterator, LPVariableIndexIterator, CoefficientIterator,const ValueType&, const ValueType&, const char * name=0);

private:
   const GraphicalModelType& gm_;
   Parameter parameter_;
   std::vector<size_t> idNodesBegin_; 
   std::vector<size_t> idFactorsBegin_; 
   std::vector<std::vector<size_t> > unaryFactors_;
   bool inferenceStarted_;
    
   IloEnv env_;
   IloModel model_;
   IloNumVarArray x_;
   IloRangeArray c_;
   IloObjective obj_;
   IloNumArray sol_;
   IloCplex cplex_;
   ValueType constValue_;
};

template<class GM, class ACC>
LPCplex<GM, ACC>::LPCplex
(
   const GraphicalModelType& gm, 
   const Parameter& para
)
:  gm_(gm), inferenceStarted_(false)
{
   if(typeid(OperatorType) != typeid(opengm::Adder)) {
      throw RuntimeError("This implementation does only supports Min-Plus-Semiring and Max-Plus-Semiring.");
   }     
   parameter_ = para;
   idNodesBegin_.resize(gm_.numberOfVariables());
   unaryFactors_.resize(gm_.numberOfVariables());
   idFactorsBegin_.resize(gm_.numberOfFactors());
  
   // temporal variables
   IloInt numberOfElements = 0;
   IloInt numberOfVariableElements = 0;
   IloInt numberOfFactorElements   = 0;
   // enumerate variables
   size_t idCounter = 0;
   for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
      numberOfVariableElements += gm_.numberOfLabels(node);
      idNodesBegin_[node] = idCounter;
      idCounter += gm_.numberOfLabels(node);
   }
   // enumerate factors
   constValue_ = 0;
   for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables() == 0) {
         LabelType l = 0;
         constValue_ += gm_[f](&l);
      }
      else if(gm_[f].numberOfVariables() == 1) {
         size_t node = gm_[f].variableIndex(0);
         unaryFactors_[node].push_back(f);
         idFactorsBegin_[f] = idNodesBegin_[node];
      }
      else {
         idFactorsBegin_[f] = idCounter;
         idCounter += gm_[f].size();
         numberOfFactorElements += gm_[f].size();
      }
   }
   numberOfElements = numberOfVariableElements + numberOfFactorElements;
   // build LP
   model_ = IloModel(env_);
   x_ = IloNumVarArray(env_);
   c_ = IloRangeArray(env_);
   sol_ = IloNumArray(env_);

   if(typeid(ACC) == typeid(opengm::Minimizer)) {
     obj_ = IloMinimize(env_);
   } else if(typeid(ACC) == typeid(opengm::Maximizer)){
     obj_ = IloMaximize(env_);
   } else {
     throw RuntimeError("This implementation does only support Minimizer or Maximizer accumulators");
   }     
   // set variables and objective
   if(parameter_.integerConstraint_) {
      x_.add(IloNumVarArray(env_, numberOfVariableElements, 0, 1, ILOBOOL));
   }
   else {
      x_.add(IloNumVarArray(env_, numberOfVariableElements, 0, 1));
   }
   x_.add(IloNumVarArray(env_, numberOfFactorElements, 0, 1));
   IloNumArray obj(env_, numberOfElements);

   for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
      for(size_t i = 0; i < gm_.numberOfLabels(node); ++i) {
         ValueType t = 0;
         for(size_t n=0; n<unaryFactors_[node].size();++n) {
            t += gm_[unaryFactors_[node][n]](&i); 
         }
         OPENGM_ASSERT_OP(idNodesBegin_[node]+i,<,numberOfElements);
         obj[idNodesBegin_[node]+i] = t;
      } 
   }
   for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables() == 2) {
         size_t index[2];
         size_t counter = idFactorsBegin_[f];
         for(index[1]=0; index[1]<gm_[f].numberOfLabels(1);++index[1])
            for(index[0]=0; index[0]<gm_[f].numberOfLabels(0);++index[0])
               obj[counter++] = gm_[f](index);
      }
      else if(gm_[f].numberOfVariables() == 3) {
         size_t index[3];
         size_t counter = idFactorsBegin_[f] ;
         for(index[2]=0; index[2]<gm_[f].numberOfLabels(2);++index[2])
            for(index[1]=0; index[1]<gm_[f].numberOfLabels(1);++index[1])
               for(index[0]=0; index[0]<gm_[f].numberOfLabels(0);++index[0])
                  obj[counter++] = gm_[f](index);
      } 
      else if(gm_[f].numberOfVariables() == 4) {
         size_t index[4];
         size_t counter = idFactorsBegin_[f];
         for(index[3]=0; index[3]<gm_[f].numberOfLabels(3);++index[3])
            for(index[2]=0; index[2]<gm_[f].numberOfLabels(2);++index[2])
               for(index[1]=0; index[1]<gm_[f].numberOfLabels(1);++index[1])
                  for(index[0]=0; index[0]<gm_[f].numberOfLabels(0);++index[0])
                     obj[counter++] = gm_[f](index);
      }
      else if(gm_[f].numberOfVariables() > 4) {
         size_t counter = idFactorsBegin_[f];
         std::vector<size_t> coordinate(gm_[f].numberOfVariables());   
         marray::Marray<bool> temp(gm_[f].shapeBegin(), gm_[f].shapeEnd());
         for(marray::Marray<bool>::iterator mit = temp.begin(); mit != temp.end(); ++mit) {
            mit.coordinate(coordinate.begin());
            obj[counter++] = gm_[f](coordinate.begin());
         }
      }
   } 
   obj_.setLinearCoefs(x_, obj);
   // set constraints
   size_t constraintCounter = 0;
   // \sum_i \mu_i = 1
   for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
      c_.add(IloRange(env_, 1, 1));
      for(size_t i = 0; i < gm_.numberOfLabels(node); ++i) {
         c_[constraintCounter].setLinearCoef(x_[idNodesBegin_[node]+i], 1);
      }
      ++constraintCounter;
   } 
   // \sum_i \mu_{f;i_1,...,i_n} - \mu{b;j}= 0
   for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables() > 1) {
         marray::Marray<size_t> temp(gm_[f].shapeBegin(), gm_[f].shapeEnd());
         size_t counter = idFactorsBegin_[f];
         for(marray::Marray<size_t>::iterator mit = temp.begin(); mit != temp.end(); ++mit) {
            *mit = counter++;
         }
         for(size_t n = 0; n < gm_[f].numberOfVariables(); ++n) {
            size_t node = gm_[f].variableIndex(n);
            for(size_t i = 0; i < gm_.numberOfLabels(node); ++i) {
               c_.add(IloRange(env_, 0, 0));
               c_[constraintCounter].setLinearCoef(x_[idNodesBegin_[node]+i], -1);
               marray::View<size_t> view = temp.boundView(n, i);
               for(marray::View<size_t>::iterator vit = view.begin(); vit != view.end(); ++vit) {
                  c_[constraintCounter].setLinearCoef(x_[*vit], 1);
               }
               ++constraintCounter;
            }
         }
      }
   }  
   model_.add(obj_);
   model_.add(c_);
   // initialize solver
   try {
      cplex_ = IloCplex(model_);
   }
   catch(IloCplex::Exception& e) {
	throw std::runtime_error("CPLEX exception");
   } 
}

template <class GM, class ACC>
InferenceTermination
LPCplex<GM, ACC>::infer() {
   EmptyVisitorType v; 
   return infer(v); 
}

template<class GM, class ACC>
template<class VisitorType>
InferenceTermination 
LPCplex<GM, ACC>::infer
(
   VisitorType& visitor
) { 
   visitor.begin(*this);
   inferenceStarted_ = true;
   try {
      // Root Algorithm
      switch(parameter_.rootAlg_) {
      case LP_SOLVER_AUTO:
         cplex_.setParam(IloCplex::RootAlg, 0);
         break;
      case LP_SOLVER_PRIMAL_SIMPLEX:
         cplex_.setParam(IloCplex::RootAlg, 1);
         break;
      case LP_SOLVER_DUAL_SIMPLEX:
         cplex_.setParam(IloCplex::RootAlg, 2);
         break;
      case LP_SOLVER_NETWORK_SIMPLEX:
         cplex_.setParam(IloCplex::RootAlg, 3);
         break;
      case LP_SOLVER_BARRIER:
         cplex_.setParam(IloCplex::RootAlg, 4);
         break;
      case LP_SOLVER_SIFTING:
         cplex_.setParam(IloCplex::RootAlg, 5);
         break;
      case LP_SOLVER_CONCURRENT:
         cplex_.setParam(IloCplex::RootAlg, 6);
         break;
      }

      // Node Algorithm
      switch(parameter_.nodeAlg_) {
      case LP_SOLVER_AUTO:
         cplex_.setParam(IloCplex::NodeAlg, 0);
         break;
      case LP_SOLVER_PRIMAL_SIMPLEX:
         cplex_.setParam(IloCplex::NodeAlg, 1);
         break;
      case LP_SOLVER_DUAL_SIMPLEX:
         cplex_.setParam(IloCplex::NodeAlg, 2);
         break;
      case LP_SOLVER_NETWORK_SIMPLEX:
         cplex_.setParam(IloCplex::NodeAlg, 3);
         break;
      case LP_SOLVER_BARRIER:
         cplex_.setParam(IloCplex::NodeAlg, 4);
         break;
      case LP_SOLVER_SIFTING:
         cplex_.setParam(IloCplex::NodeAlg, 5);
         break;
      case LP_SOLVER_CONCURRENT:
         cplex_.setParam(IloCplex::NodeAlg, 6);
         break;
      }

      // presolve
      switch(parameter_.presolve_) {
      case LP_PRESOLVE_AUTO: 
         cplex_.setParam(IloCplex::PreInd, CPX_ON);
         cplex_.setParam(IloCplex::RelaxPreInd, -1);
         break;
      case LP_PRESOLVE_OFF:  
         cplex_.setParam(IloCplex::PreInd, CPX_OFF);
         cplex_.setParam(IloCplex::RelaxPreInd, 0);
         break;
      case LP_PRESOLVE_CONSERVATIVE:
         cplex_.setParam(IloCplex::PreInd, CPX_ON);
         cplex_.setParam(IloCplex::RelaxPreInd, -1);
         break;
      case LP_PRESOLVE_AGGRESSIVE:
         cplex_.setParam(IloCplex::PreInd, CPX_ON);
         cplex_.setParam(IloCplex::RelaxPreInd, 1);
         break; 
      }

      // MIP EMPHASIS
      switch(parameter_.mipEmphasis_) {
      case MIP_EMPHASIS_BALANCED:
         cplex_.setParam(IloCplex::MIPEmphasis, 0);
         break;
      case  MIP_EMPHASIS_FEASIBILITY:
            cplex_.setParam(IloCplex::MIPEmphasis, 1);
         break;
      case MIP_EMPHASIS_OPTIMALITY:
         cplex_.setParam(IloCplex::MIPEmphasis, 2);
         break;
      case MIP_EMPHASIS_BESTBOUND:
         cplex_.setParam(IloCplex::MIPEmphasis, 3);
         break;
      case MIP_EMPHASIS_HIDDENFEAS:
         cplex_.setParam(IloCplex::MIPEmphasis, 4);
         break;
      }

      // verbose options
      if(parameter_.verbose_ == false) {
       cplex_.setParam(IloCplex::MIPDisplay, 0);
       cplex_.setParam(IloCplex::BarDisplay, 0);
       cplex_.setParam(IloCplex::SimDisplay, 0);
       cplex_.setParam(IloCplex::NetDisplay, 0);
       cplex_.setParam(IloCplex::SiftDisplay, 0);
      } 
         
      // tolarance settings
      cplex_.setParam(IloCplex::EpOpt,  parameter_.epOpt_); // Optimality Tolerance
      cplex_.setParam(IloCplex::EpMrk,  parameter_.epMrk_); // Markowitz tolerance
      cplex_.setParam(IloCplex::EpRHS,  parameter_.epRHS_);  // Feasibility Tolerance
      cplex_.setParam(IloCplex::EpInt,  parameter_.epInt_);    // amount by which an integer variable can differ from an integer
      cplex_.setParam(IloCplex::EpAGap, parameter_.epAGap_);   // Absolute MIP gap tolerance
      cplex_.setParam(IloCplex::EpGap,  parameter_.epGap_); // Relative MIP gap tolerance

      // set hints
      cplex_.setParam(IloCplex::CutUp, parameter_.cutUp_);

      // memory setting
      cplex_.setParam(IloCplex::WorkMem, parameter_.workMem_);
      cplex_.setParam(IloCplex::ClockType,2);//wall-clock-time=2 cpu-time=1
      cplex_.setParam(IloCplex::TreLim,parameter_.treeMemoryLimit_);
      cplex_.setParam(IloCplex::MemoryEmphasis, 1);

      // time limit
      cplex_.setParam(IloCplex::TiLim, parameter_.timeLimit_);

      // multo-threading options
      cplex_.setParam(IloCplex::Threads, parameter_.numberOfThreads_);

      // Tuning
      cplex_.setParam(IloCplex::Probe, parameter_.probeingLevel_);
      if(parameter_.cutLevel_ != MIP_CUT_DEFAULT){
         int cl = parameter_.getCutLevel(parameter_.cutLevel_);
         cplex_.setParam(IloCplex::Covers, cl); 
         cplex_.setParam(IloCplex::Cliques, cl);
         cplex_.setParam(IloCplex::DisjCuts, cl);
         cplex_.setParam(IloCplex::Cliques, cl);
         cplex_.setParam(IloCplex::MIRCuts, cl);
         cplex_.setParam(IloCplex::GUBCovers, cl);
         cplex_.setParam(IloCplex::FlowCovers, cl);
         cplex_.setParam(IloCplex::FlowPaths, cl);
         cplex_.setParam(IloCplex::ImplBd, cl);
         cplex_.setParam(IloCplex::FracCuts, cl);
      }

      // cplex_.setParam(IloCplex::Covers, parameter_.coverCutLevel_);
      //cplex_.setParam(IloCplex::DisjCuts, parameter_.disjunctiverCutLevel_);
      //cplex_.setParam(IloCplex::Cliques, parameter_.cliqueCutLevel_);
      //cplex_.setParam(IloCplex::MIRCuts, parameter_.MIRCutLevel_);
  
      // solve problem

      if(!cplex_.solve()) {
         std::cout << "failed to optimize. " <<cplex_.getStatus() << std::endl;
         inferenceStarted_ = 0;
         return UNKNOWN;
      } 
      cplex_.getValues(sol_, x_);
   }
   catch(IloCplex::Exception e) {
      std::cout << "caught CPLEX exception: " << e << std::endl;
      return UNKNOWN;
   } 
   visitor.end(*this);
   return NORMAL;
}
 
template <class GM, class ACC>
LPCplex<GM, ACC>::~LPCplex() {
   env_.end();
}

template <class GM, class ACC>
inline InferenceTermination
LPCplex<GM, ACC>::arg
(
   std::vector<typename LPCplex<GM, ACC>::LabelType>& x, 
   const size_t N
) const {
   x.resize(gm_.numberOfVariables());
   if(inferenceStarted_) {
      for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
         ValueType value = sol_[idNodesBegin_[node]];
         size_t state = 0;
         for(size_t i = 1; i < gm_.numberOfLabels(node); ++i) {
            if(sol_[idNodesBegin_[node]+i] > value) {
               value = sol_[idNodesBegin_[node]+i];
               state = i;
            }
         }
         x[node] = state;
      }
      return NORMAL;
   } else {
      for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
         x[node] = 0;
      }
      return UNKNOWN;
   }

}

template <class GM, class ACC>
void LPCplex<GM, ACC>::variable
(
   const size_t nodeId, 
   IndependentFactorType& out
) const {
   size_t var[] = {nodeId};
   size_t shape[] = {gm_.numberOfLabels(nodeId)};
   out.assign(var, var + 1, shape, shape + 1);
   for(size_t i = 0; i < gm_.numberOfLabels(nodeId); ++i) {
      out(i) = sol_[idNodesBegin_[nodeId]+i];
   }
   //return UNKNOWN;
}

template <class GM, class ACC>
void LPCplex<GM, ACC>::factorVariable
(
   const size_t factorId, 
   IndependentFactorType& out
) const {
   std::vector<size_t> var(gm_[factorId].numberOfVariables());
   std::vector<size_t> shape(gm_[factorId].numberOfVariables());
   for(size_t i = 0; i < gm_[factorId].numberOfVariables(); ++i) {
      var[i] = gm_[factorId].variableIndex(i);
      shape[i] = gm_[factorId].shape(i);
   }
   out.assign(var.begin(), var.end(), shape.begin(), shape.end());
   if(gm_[factorId].numberOfVariables() == 1) {
      size_t nodeId = gm_[factorId].variableIndex(0);
      marginal(nodeId, out);
   }
   else {
      size_t c = 0;
      for(size_t n = idFactorsBegin_[factorId]; n<idFactorsBegin_[factorId]+gm_[factorId].size(); ++n) {
         out(c++) = sol_[n];
      }
   }
   //return UNKNOWN;
}

template<class GM, class ACC>
inline void
LPCplex<GM, ACC>::setStartingPoint(
   typename std::vector<typename LPCplex<GM, ACC>::LabelType>::const_iterator begin
) {
   IloNumVarArray vars(env_);
   IloNumArray values(env_);
   for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
      const IloNumVar lpvar = x_[lpNodeVi(var,*(begin+var))];
      vars.add(lpvar);
      values.add(1);
   }
   cplex_.addMIPStart(vars, values);
}



template<class GM, class ACC>
inline const typename LPCplex<GM, ACC>::GraphicalModelType&
LPCplex<GM, ACC>::graphicalModel() const 
{
   return gm_;
}

template<class GM, class ACC>
typename GM::ValueType LPCplex<GM, ACC>::value() const { 
   std::vector<LabelType> states;
   arg(states);
   return gm_.evaluate(states);
}

template<class GM, class ACC>
typename GM::ValueType LPCplex<GM, ACC>::bound() const { 
   if(inferenceStarted_) {
      if(parameter_.integerConstraint_) {
         return cplex_.getBestObjValue()+constValue_;
      }
      else{
         return  cplex_.getObjValue()+constValue_;
      }
   }
   else{
      return ACC::template ineutral<ValueType>();
   }
}


template <class GM, class ACC>
inline size_t 
LPCplex<GM, ACC>::lpNodeVi
(
   const typename LPCplex<GM, ACC>::IndexType variableIndex,
   const typename LPCplex<GM, ACC>::LabelType label
)const{
   OPENGM_ASSERT(variableIndex<gm_.numberOfVariables());
   OPENGM_ASSERT(label<gm_.numberOfLabels(variableIndex));
   return idNodesBegin_[variableIndex]+label;
}


template <class GM, class ACC>
inline size_t 
LPCplex<GM, ACC>::lpFactorVi
(
   const typename LPCplex<GM, ACC>::IndexType factorIndex,
   const size_t labelingIndex
)const{
   OPENGM_ASSERT(factorIndex<gm_.numberOfFactors());
   OPENGM_ASSERT(labelingIndex<gm_[factorIndex].size());
   return idFactorsBegin_[factorIndex]+labelingIndex;
}


template <class GM, class ACC>
template<class LABELING_ITERATOR>
inline size_t 
LPCplex<GM, ACC>::lpFactorVi
(
   const typename LPCplex<GM, ACC>::IndexType factorIndex,
   LABELING_ITERATOR labelingBegin,
   LABELING_ITERATOR labelingEnd
)const{
   OPENGM_ASSERT(factorIndex<gm_.numberOfFactors());
   OPENGM_ASSERT(std::distance(labelingBegin,labelingEnd)==gm_[factorIndex].numberOfVariables());
   const size_t numVar=gm_[factorIndex].numberOfVariables();
   size_t labelingIndex=labelingBegin[0];
   size_t strides=gm_[factorIndex].numberOfLabels(0);
   for(size_t vi=1;vi<numVar;++vi){
      OPENGM_ASSERT(labelingBegin[vi]<gm_[factorIndex].numberOfLabels(vi));
      labelingIndex+=strides*labelingBegin[vi];
      strides*=gm_[factorIndex].numberOfLabels(vi);
   }
   return idFactorsBegin_[factorIndex]+labelingIndex;
}




/// \brief add constraint
/// \param viBegin iterator to the beginning of a sequence of variable indices
/// \param viEnd iterator to the end of a sequence of variable indices
/// \param coefficient iterator to the beginning of a sequence of coefficients
/// \param lowerBound lower bound
/// \param upperBound upper bound
///
/// variable indices refer to variables of the LP that is set up
/// in the constructor of LPCplex (NOT to the variables of the 
/// graphical model).
///
template<class GM, class ACC>
template<class LPVariableIndexIterator, class CoefficientIterator>
inline void LPCplex<GM, ACC>::addConstraint(
   LPVariableIndexIterator viBegin, 
   LPVariableIndexIterator viEnd, 
   CoefficientIterator coefficient, 
   const ValueType& lowerBound, 
   const ValueType& upperBound,
   const char * name
) {
   IloRange constraint(env_, lowerBound, upperBound, name);
   while(viBegin != viEnd) {
      constraint.setLinearCoef(x_[*viBegin], *coefficient);
      ++viBegin;
      ++coefficient;
   }
   model_.add(constraint);
   // adding constraints does not require a re-initialization of the
   // object cplex_. cplex_ is initialized in the constructor.
}

} // end namespace opengm

#endif
