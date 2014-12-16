#pragma once
#ifndef OPENGM_LP_GURPBI_HXX
#define OPENGM_LP_GURPBI_HXX

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <typeinfo>

#include "gurobi_c++.h"

#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/opengm.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/auxiliary/lpdef.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// \brief Optimization by Linear Programming (LP) or Integer LP using Guroi\n\n
///http://www.gurobi.com
///
/// The optimization problem is reformulated as an LP or ILP.
/// For the LP, a first order local polytope approximation of the
/// marginal polytope is used, i.e. the affine instead of the convex 
/// hull.
/// 
/// Gurobi is a commercial product that is 
/// free for accadamical use.
///
/// \ingroup inference 
template<class GM, class ACC>
class LPGurobi : public Inference<GM, ACC>, public LPDef {
public:
   typedef ACC AccumulationType; 
   typedef ACC AccumulatorType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS; 
   typedef visitors::VerboseVisitor<LPGurobi<GM, ACC> > VerboseVisitorType;
   typedef visitors::TimingVisitor<LPGurobi<GM, ACC> > TimingVisitorType;
   typedef visitors::EmptyVisitor< LPGurobi<GM, ACC> > EmptyVisitorType;
 
 
   class Parameter {
   public:
      bool integerConstraint_; // ILP=true, 1order-LP=false
      int numberOfThreads_;    // number of threads (0=autosect)
      bool verbose_;           // switch on/off verbode mode 
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
      int probeingLevel_;
      //int coverCutLevel_;
      //int disjunctiverCutLevel_;
      //int cliqueCutLevel_;
      //int MIRCutLevel_;
      //int presolveLevel_;
      LP_SOLVER rootAlg_;  
      LP_SOLVER nodeAlg_;
      MIP_EMPHASIS mipFocus_;
      LP_PRESOLVE presolve_;
      MIP_CUT cutLevel_;       // Determines whether or not to cuts for the problem and how aggressively (will be overruled by specific ones). 
      MIP_CUT cliqueCutLevel_; // Determines whether or not to generate clique cuts for the problem and how aggressively. 
      MIP_CUT coverCutLevel_;  // Determines whether or not to generate cover cuts for the problem and how aggressively. 
      MIP_CUT gubCutLevel_;    // Determines whether or not to generate generalized upper bound (GUB) cuts for the problem and how aggressively. 
      MIP_CUT mirCutLevel_;    // Determines whether or not mixed integer rounding (MIR) cuts should be generated for the problem and how aggressively.  
      MIP_CUT iboundCutLevel_; // Determines whether or not to generate implied bound cuts for the problem and how aggressively.
      MIP_CUT flowcoverCutLevel_; //Determines whether or not to generate flow cover cuts for the problem and how aggressively. 
      MIP_CUT flowpathCutLevel_; //Determines whether or not to generate flow path cuts for the problem and how aggressively.
      MIP_CUT disjunctCutLevel_; // Determines whether or not to generate disjunctive cuts for the problem and how aggressively.
      MIP_CUT gomoryCutLevel_; // Determines whether or not to generate gomory fractional cuts for the problem and how aggressively. 

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
         //coverCutLevel_(0),
         //disjunctiverCutLevel_(0),
         //cliqueCutLevel_(0),
         //MIRCutLevel_(0),
         //presolveLevel_(-1),
         rootAlg_(LP_SOLVER_AUTO),
         nodeAlg_(LP_SOLVER_AUTO),
         mipFocus_(MIP_EMPHASIS_BALANCED),
         presolve_(LP_PRESOLVE_AUTO),
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
            return -1;
         case MIP_CUT_OFF:
            return 0;
         case  MIP_CUT_ON:
            return 1;
         case MIP_CUT_AGGRESSIVE:
            return 2;
         case MIP_CUT_VERYAGGRESSIVE:
            return 3;
         }
         return  -1;
      };
   };

   LPGurobi(const GraphicalModelType&, const Parameter& = Parameter());
   ~LPGurobi();
   virtual std::string name() const 
      { return "LPGurobi"; }
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

   size_t lpNodeVi(const IndexType variableIndex,const LabelType label)const;
   size_t lpFactorVi(const IndexType factorIndex,const size_t labelingIndex)const;
   template<class LABELING_ITERATOR>
   size_t lpFactorVi(const IndexType factorIndex,LABELING_ITERATOR labelingBegin,LABELING_ITERATOR labelingEnd)const;
   template<class LPVariableIndexIterator, class CoefficientIterator>
   void addConstraint(LPVariableIndexIterator, LPVariableIndexIterator, CoefficientIterator,const ValueType&, const ValueType&, const char * name=0);


   void writeModelToDisk(const std::string & filename)const{
      try {
         if( filename.size()!=0)
            model_->write(filename);
      }
      catch(GRBException e) {
            std::cout << "**Error code = " << e.getErrorCode() << "\n";
            std::cout << e.getMessage() <<"\n";
            throw  opengm::RuntimeError( e.getMessage() );
         } 
         catch(...) {
            std::cout << "Exception during write" <<"\n";
            throw  opengm::RuntimeError( "Exception during write" );
      }

   }

private:
   void updateIfDirty();

   const GraphicalModelType& gm_;
   Parameter param_;
   std::vector<size_t> idNodesBegin_; 
   std::vector<size_t> idFactorsBegin_; 
   std::vector<std::vector<size_t> > unaryFactors_; 
   bool inferenceStarted_;

   bool dirty_;
   
   std::vector<double>    lpArg_;
   std::vector<LabelType> arg_;
   size_t                 nLpVar_;
   // gurobi members
   GRBEnv   * env_ ;
   GRBModel * model_;
   GRBVar   * vars_;

   // 
   ValueType bound_;
   ValueType value_;
};



template<class GM, class ACC>
LPGurobi<GM, ACC>::LPGurobi
(
   const GraphicalModelType& gm, 
   const Parameter& para
)
:  gm_(gm),
   param_(para),
   idNodesBegin_(gm_.numberOfVariables()),
   idFactorsBegin_(gm_.numberOfFactors()),
   unaryFactors_(gm_.numberOfVariables()),
   inferenceStarted_(false),
   dirty_(false),
   lpArg_(),
   arg_(gm_.numberOfVariables(),0),
   nLpVar_(0),
   env_(),
   model_(),
   vars_(),
   bound_(),
   value_()
{

   ACC::neutral(value_);
   ACC::ineutral(bound_); 
   //std::cout<<"setup basic env\n";
   try {
      env_   = new GRBEnv();
      env_->set(GRB_IntParam_LogToConsole,int(param_.verbose_));  

      // Root Algorithm
      switch(param_.nodeAlg_) {
      case LP_SOLVER_AUTO:
         env_->set(GRB_IntParam_NodeMethod,1);
         break;
      case LP_SOLVER_PRIMAL_SIMPLEX:
         env_->set(GRB_IntParam_NodeMethod,0);
         break;
      case LP_SOLVER_DUAL_SIMPLEX:
         env_->set(GRB_IntParam_NodeMethod,1);
         break;
      case LP_SOLVER_NETWORK_SIMPLEX:
         throw RuntimeError("Gurobi does not support Network Simplex");
         break;
      case LP_SOLVER_BARRIER:
         env_->set(GRB_IntParam_NodeMethod,2);
         break;
      case LP_SOLVER_SIFTING:
          throw RuntimeError("Gurobi does not support Sifting");
         break;
      case LP_SOLVER_CONCURRENT:
         throw RuntimeError("Gurobi does not concurrent solvers");
         break;
      }

      // Node Algorithm
      switch(param_.rootAlg_) {
      case LP_SOLVER_AUTO:
         env_->set(GRB_IntParam_Method,-1);
         break;
      case LP_SOLVER_PRIMAL_SIMPLEX:
         env_->set(GRB_IntParam_Method,0);
         break;
      case LP_SOLVER_DUAL_SIMPLEX:
         env_->set(GRB_IntParam_Method,1);
         break;
      case LP_SOLVER_NETWORK_SIMPLEX:
         throw RuntimeError("Gurobi does not support Network Simplex");
         break;
      case LP_SOLVER_BARRIER:
         env_->set(GRB_IntParam_Method,2);
         break;
      case LP_SOLVER_SIFTING:
         env_->set(GRB_IntParam_Method,1);
         env_->set(GRB_IntParam_SiftMethod,1);
         break;
      case LP_SOLVER_CONCURRENT:
         env_->set(GRB_IntParam_Method,4);
         break;
      } 

      // presolve
      switch(param_.presolve_) {
      case LP_PRESOLVE_AUTO:
         env_->set(GRB_IntParam_Presolve,-1); 
         break;
      case LP_PRESOLVE_OFF:
         env_->set(GRB_IntParam_Presolve,0);   
         break;
      case LP_PRESOLVE_CONSEVATIVE:
         env_->set(GRB_IntParam_Presolve,1); 
         break;
      case LP_PRESOLVE_AGRESSIVE: 
         env_->set(GRB_IntParam_Presolve,2); 
         break; 
      }

      // MIP FOCUS 
      switch(param_.mipFocus_) {
      case MIP_EMPHASIS_BALANCED:
         env_->set(GRB_IntParam_MIPFocus,0);
         break;
      case  MIP_EMPHASIS_FEASIBILITY:
         env_->set(GRB_IntParam_MIPFocus,1);
         break;
      case MIP_EMPHASIS_OPTIMALITY:
         env_->set(GRB_IntParam_MIPFocus,2);
         break;
      case MIP_EMPHASIS_BESTBOUND:
         env_->set(GRB_IntParam_MIPFocus,3);
         break;
      case MIP_EMPHASIS_HIDDENFEAS:
         throw RuntimeError("Gurobi does not support hidden feasibility as MIP-focus");
         break;
      }

      // tolarance settings
      env_->set(GRB_DoubleParam_Cutoff        ,param_.cutUp_); // Optimality Tolerance
      env_->set(GRB_DoubleParam_OptimalityTol ,param_.epOpt_); // Optimality Tolerance
      env_->set(GRB_DoubleParam_IntFeasTol    ,param_.epInt_); // amount by which an integer variable can differ from an integer
      env_->set(GRB_DoubleParam_MIPGapAbs     ,param_.epAGap_); // Absolute MIP gap tolerance
      env_->set(GRB_DoubleParam_MIPGap        ,param_.epGap_); // Relative MIP gap tolerance
      env_->set(GRB_DoubleParam_FeasibilityTol,param_.epRHS_);
      env_->set(GRB_DoubleParam_MarkowitzTol  ,param_.epMrk_);

      // set hints 
      // CutUp is missing http://www.gurobi.com/resources/switching-to-gurobi/switching-from-cplex#setting

      // memory settings 
      // -missing
     
      // time limit
      env_->set(GRB_DoubleParam_TimeLimit       ,param_.timeLimit_); // time limit

      // threadding
      if(param_.numberOfThreads_!=0)
         env_->set(GRB_IntParam_Threads       ,param_.numberOfThreads_); // threads


      // tuning
      // *Probe missing
      // *DisjCuts missing
      if(param_.cutLevel_ != MIP_CUT_DEFAULT)
         env_->set(GRB_IntParam_Cuts            ,param_.getCutLevel(param_.cutLevel_));
      if(param_.cliqueCutLevel_ != MIP_CUT_DEFAULT) 
         env_->set(GRB_IntParam_CliqueCuts      ,param_.getCutLevel(param_.cliqueCutLevel_)); 
      if(param_.coverCutLevel_ != MIP_CUT_DEFAULT)
         env_->set(GRB_IntParam_CoverCuts       ,param_.getCutLevel(param_.coverCutLevel_)); 
      if(param_.gubCutLevel_ != MIP_CUT_DEFAULT)
         env_->set(GRB_IntParam_GUBCoverCuts    ,param_.getCutLevel(param_.gubCutLevel_)); 
      if(param_.mirCutLevel_ != MIP_CUT_DEFAULT)
         env_->set(GRB_IntParam_MIRCuts         ,param_.getCutLevel(param_.mirCutLevel_));
      if(param_.iboundCutLevel_ != MIP_CUT_DEFAULT)
         env_->set(GRB_IntParam_ImpliedCuts     ,param_.getCutLevel(param_.iboundCutLevel_));
      if(param_.flowcoverCutLevel_ != MIP_CUT_DEFAULT)
         env_->set(GRB_IntParam_FlowCoverCuts   ,param_.getCutLevel(param_.flowcoverCutLevel_));
      if(param_.flowpathCutLevel_ != MIP_CUT_DEFAULT)
         env_->set(GRB_IntParam_FlowPathCuts    ,param_.getCutLevel(param_.flowpathCutLevel_));
      // *DisjCuts missing
      // *Gomory missing    
      model_ = new GRBModel(*env_);
   }
   catch(GRBException e) {
      std::cout << "Error code = " << e.getErrorCode() << "\n";
      std::cout << e.getMessage() <<"\n";
      throw  opengm::RuntimeError( e.getMessage() );
   } catch(...) {
      std::cout << "Exception during construction of gurobi solver" <<"\n";
      throw  opengm::RuntimeError( "Exception during construction of gurobi solver" );
   }


   if(typeid(OperatorType) != typeid(opengm::Adder)) {
      throw RuntimeError("This implementation does only supports Min-Plus-Semiring");
   }
   //std::cout<<"enumerate stuff\n";    
   param_ = para;
   idNodesBegin_.resize(gm_.numberOfVariables());
   unaryFactors_.resize(gm_.numberOfVariables());
   idFactorsBegin_.resize(gm_.numberOfFactors());

   // temporal variables
   size_t numberOfElements = 0;
   size_t numberOfVariableElements = 0;
   size_t numberOfFactorElements   = 0;
   size_t maxLabel                 = 0 ;
   size_t maxFacSize               = 0;
   // enumerate variables
   size_t idCounter = 0;
   for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
      numberOfVariableElements += gm_.numberOfLabels(node);
      maxLabel=std::max(size_t(gm_.numberOfLabels(node)),maxLabel);

      idNodesBegin_[node] = idCounter;
      idCounter += gm_.numberOfLabels(node);
   }
   // enumerate factors
   for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables() == 1) {
         size_t node = gm_[f].variableIndex(0);
         unaryFactors_[node].push_back(f);
         idFactorsBegin_[f] = idNodesBegin_[node];
      }
      else {
         idFactorsBegin_[f] = idCounter;
         idCounter += gm_[f].size();
         maxFacSize=std::max(size_t(gm_[f].size()),maxFacSize);
         numberOfFactorElements += gm_[f].size();
      }
   }
   numberOfElements = numberOfVariableElements + numberOfFactorElements;
   nLpVar_=numberOfElements; // refactor me

   if(typeid(ACC) == typeid(opengm::Minimizer)) {
   }
   else {
     throw RuntimeError("This implementation does only support Minimizer or Maximizer accumulators");
   }
   
   //std::cout<<"fill obj ptrs \n";    
   lpArg_.resize(nLpVar_);
   std::vector<double> lb(numberOfElements,0.0);
   std::vector<double> ub(numberOfElements,1.0);
   std::vector<double> obj(numberOfElements);
   std::vector<char>   vtype(numberOfElements,GRB_CONTINUOUS);
   // set variables and objective
   if(param_.integerConstraint_) {
      std::fill(vtype.begin(),vtype.begin()+numberOfVariableElements,GRB_BINARY);
   }


   for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
      for(size_t i = 0; i < gm_.numberOfLabels(node); ++i) {
         ValueType t = 0;
         for(size_t n=0; n<unaryFactors_[node].size();++n) {
            t += gm_[unaryFactors_[node][n]](&i);
         }
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

   //std::cout<<"add obj ptrs \n"; 
   try {
      // add all variables at once with an allready setup objective
      vars_ = model_->addVars(&lb[0],&ub[0],&obj[0],&vtype[0],NULL,numberOfElements);
      //integrate new variales
      model_->update();
   }
   catch(GRBException e) {
      std::cout << "**Error code = " << e.getErrorCode() << "\n";
      std::cout << e.getMessage() <<"\n";
      throw  opengm::RuntimeError( e.getMessage() );
   } catch(...) {
      std::cout << "Exception during construction of gurobi model" <<"\n";
      throw  opengm::RuntimeError( "Exception during construction of gurobi model" );
   }

   //std::cout<<"count constr \n"; 
   // count the needed constraints
   size_t constraintCounter = 0;
   // \sum_i \mu_i = 1
   for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
      ++constraintCounter;
   }
   
   // \sum_i \mu_{f;i_1,...,i_n} - \mu{b;j}= 0
   for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables() > 1) {
         for(size_t n = 0; n < gm_[f].numberOfVariables(); ++n) {
            size_t node = gm_[f].variableIndex(n);
            for(size_t i = 0; i < gm_.numberOfLabels(node); ++i) {
               ++constraintCounter;
            }
         }
      }
   } 
   




   std::vector<GRBLinExpr>    lhsExprs(constraintCounter);
   std::vector<char>          sense(constraintCounter,GRB_EQUAL);
   std::vector<double>        rhsVals(constraintCounter,0.0);
   std::vector<std::string>   names(constraintCounter,std::string());

   std::fill(rhsVals.begin(),rhsVals.begin()+gm_.numberOfVariables(),1.0);



   //std::cout<<"setup constr \n"; 

   // set constraints
   constraintCounter = 0;
   // \sum_i \mu_i = 1

   const size_t buffferSize =  std::max(maxLabel,size_t(maxFacSize+1));
   std::vector<GRBVar> localVars(buffferSize);
   std::vector<double> localVal(buffferSize,1.0);

   for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
      for(size_t i = 0; i < gm_.numberOfLabels(node); ++i) {
         localVars[i]=vars_[idNodesBegin_[node]+i];
      }
      lhsExprs[constraintCounter].addTerms(&localVal[0],&localVars[0],gm_.numberOfLabels(node));
      ++constraintCounter;
   }
   
   localVal[0]=-1.0;

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
               //c_.add(IloRange(env_, 0, 0));
               //c_[constraintCounter].setLinearCoef(x_[idNodesBegin_[node]+i], -1);
               //double mone =-1.0;
               //lhsExprs[constraintCounter].addTerms(&mone,&vars_[idNodesBegin_[node]+i],1);
               size_t localCounter=1;
               localVars[0]=vars_[idNodesBegin_[node]+i];
               marray::View<size_t> view = temp.boundView(n, i);
               for(marray::View<size_t>::iterator vit = view.begin(); vit != view.end(); ++vit) {
                  //c_[constraintCounter].setLinearCoef(x_[*vit], 1);
                  //double one =1.0;
                  //lhsExprs[constraintCounter].addTerms(&one,&vars_[*vit],1);
                  localVars[localCounter]=vars_[*vit];
                  ++localCounter;
               }
               lhsExprs[constraintCounter].addTerms(&localVal[0],&localVars[0],localCounter);
               ++constraintCounter;
            }
         }
      }
   } 
   

   try {

      //std::cout<<"add constr \n"; 
      // add all constraints at once to the model
      GRBConstr* constr = model_->addConstrs(&lhsExprs[0],&sense[0],&rhsVals[0],&names[0],constraintCounter);
      //std::cout<<"done\n"; 
   }
   catch(GRBException e) {
      std::cout << "**Error code = " << e.getErrorCode() << "\n";
      std::cout << e.getMessage() <<"\n";
      throw  opengm::RuntimeError( e.getMessage() );
   } catch(...) {
      std::cout << "Exception during adding constring to gurobi model" <<"\n";
      throw  opengm::RuntimeError( "Exception during adding constring to gurobi model" );
   }

   // test if it help for write model to file
   model_->update();
}

template <class GM, class ACC>
InferenceTermination
LPGurobi<GM, ACC>::infer() {
   EmptyVisitorType v; 
   return infer(v); 
}

template<class GM, class ACC>
template<class VisitorType>
InferenceTermination 
LPGurobi<GM, ACC>::infer
(
   VisitorType& visitor
) { 
   updateIfDirty();
   visitor.begin(*this);
   inferenceStarted_ = true;
   try {
      model_->optimize();
      if(param_.integerConstraint_){
           bound_ = model_->get(GRB_DoubleAttr_ObjBound);
      }
      else{
         bound_ = model_->get(GRB_DoubleAttr_ObjVal);
      }
      //std::cout << "Bound: " <<bound_ << "\n";
      for(size_t lpvi=0;lpvi<nLpVar_;++lpvi){
         lpArg_[lpvi]=vars_[lpvi].get(GRB_DoubleAttr_X);
         //td::cout<<"lpvi "<<lpvi<<" "<<lpArg_[lpvi]<<"\n";
      }

   }
   catch(GRBException e) {
      std::cout << "Error code = " << e.getErrorCode() << "\n";
      std::cout << e.getMessage() <<"\n";
   } catch(...) {
      std::cout << "Exception during optimization" <<"\n";
   }
   visitor.end(*this);
   return NORMAL;
}
 
template <class GM, class ACC>
LPGurobi<GM, ACC>::~LPGurobi() {
   delete model_;
   delete env_;
}

template <class GM, class ACC>
inline InferenceTermination
LPGurobi<GM, ACC>::arg
(
   std::vector<typename LPGurobi<GM, ACC>::LabelType>& x, 
   const size_t N
) const {
   
   x.resize(gm_.numberOfVariables()); 
   if(inferenceStarted_) {
      for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
         ValueType value = lpArg_[idNodesBegin_[node]];
         size_t state = 0;
         for(size_t i = 1; i < gm_.numberOfLabels(node); ++i) {
            if(lpArg_[idNodesBegin_[node]+i] > value) {
               value = lpArg_[idNodesBegin_[node]+i];
               state = i;
            }
         }
         x[node] = state;
      }
      return NORMAL;
   }
   else{
      for(size_t node = 0; node < gm_.numberOfVariables(); ++node) {
         x[node] = 0;
      }
      return UNKNOWN;
   }  
}

template <class GM, class ACC>
void LPGurobi<GM, ACC>::variable
(
   const size_t nodeId, 
   IndependentFactorType& out
) const {
   
   size_t var[] = {nodeId};
   size_t shape[] = {gm_.numberOfLabels(nodeId)};
   out.assign(var, var + 1, shape, shape + 1);
   for(size_t i = 0; i < gm_.numberOfLabels(nodeId); ++i) {
      out(i) = lpArg_[idNodesBegin_[nodeId]+i];
   }
   //return UNKNOWN;
   
}

template <class GM, class ACC>
void LPGurobi<GM, ACC>::factorVariable
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
         out(c++) = lpArg_[n];
      }
   }
   //return UNKNOWN;
}

template<class GM, class ACC>
inline const typename LPGurobi<GM, ACC>::GraphicalModelType&
LPGurobi<GM, ACC>::graphicalModel() const 
{
   return gm_;
}

template<class GM, class ACC>
typename GM::ValueType LPGurobi<GM, ACC>::value() const { 
   std::vector<LabelType> states;
   arg(states);
   return gm_.evaluate(states);
}

template<class GM, class ACC>
typename GM::ValueType LPGurobi<GM, ACC>::bound() const {
   
   if(param_.integerConstraint_) {
      return bound_;
   }
   else{
      return  bound_;
   }
   
}


template <class GM, class ACC>
inline size_t 
LPGurobi<GM, ACC>::lpNodeVi
(
   const typename LPGurobi<GM, ACC>::IndexType variableIndex,
   const typename LPGurobi<GM, ACC>::LabelType label
)const{
   OPENGM_ASSERT(variableIndex<gm_.numberOfVariables());
   OPENGM_ASSERT(label<gm_.numberOfLabels(variableIndex));
   return idNodesBegin_[variableIndex]+label;
}


template <class GM, class ACC>
inline size_t 
LPGurobi<GM, ACC>::lpFactorVi
(
   const typename LPGurobi<GM, ACC>::IndexType factorIndex,
   const size_t labelingIndex
)const{
   OPENGM_ASSERT(factorIndex<gm_.numberOfFactors());
   OPENGM_ASSERT(labelingIndex<gm_[factorIndex].size());
   return idFactorsBegin_[factorIndex]+labelingIndex;
}


template <class GM, class ACC>
template<class LABELING_ITERATOR>
inline size_t 
LPGurobi<GM, ACC>::lpFactorVi
(
   const typename LPGurobi<GM, ACC>::IndexType factorIndex,
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
/// in the constructor of LPGurobi (NOT to the variables of the 
/// graphical model).
///
template<class GM, class ACC>
template<class LPVariableIndexIterator, class CoefficientIterator>
inline void LPGurobi<GM, ACC>::addConstraint(
   LPVariableIndexIterator viBegin, 
   LPVariableIndexIterator viEnd, 
   CoefficientIterator coefficient, 
   const ValueType& lowerBound, 
   const ValueType& upperBound,
   const char * name
) {
   // construct linear constraint expression
   GRBLinExpr expr;
   while(viBegin != viEnd) {
      expr += vars_[*viBegin] * (*coefficient);
      ++viBegin;
      ++coefficient;
   }

   // add constraints for upper and lower bound
   model_->addConstr(expr, GRB_LESS_EQUAL, upperBound, name);
   model_->addConstr(expr, GRB_GREATER_EQUAL, lowerBound, name);

   // Gurobi needs a model update after adding a constraint
   dirty_ = true;
}

template<class GM, class ACC>
inline void LPGurobi<GM, ACC>::updateIfDirty()
{
   if(dirty_) {
      model_->update();
      dirty_ = false;
   }
}

} // end namespace opengm

#endif
