#pragma once
#ifndef OPENGM_LP_SOLVER_CPLEX_HXX
#define OPENGM_LP_SOLVER_CPLEX_HXX

#include <ilcplex/ilocplex.h>
#include "lp_solver_interface.hxx"



namespace opengm{


class LpSolverCplex : public LpSolverInterface
{
public:
	typedef double LpValueType;
	typedef int    LpIndexType;

	class Parameter {
	public:
		/// constructor
		/// \param cutUp upper cutoff - assume that: min_x f(x) <= cutUp 
		/// \param epGap relative stopping criterion: |bestnode-bestinteger| / (1e-10 + |bestinteger|) <= epGap
		Parameter
		(
			int numberOfThreads = 0, 
			double cutUp = 1.0e+75,
			double epGap=0
		)
		:  	numberOfThreads_(numberOfThreads), 
			//integerConstraint_(false), 
			verbose_(false), 
			cutUp_(cutUp),
			epGap_(epGap),
			workMem_(128.0),
			treeMemoryLimit_(1e+75),
			timeLimit_(1e+75),
			probeingLevel_(0),
			coverCutLevel_(0),
			disjunctiverCutLevel_(0),
			cliqueCutLevel_(0),
			MIRCutLevel_(0)
		{
			numberOfThreads_   = numberOfThreads; 
			integerConstraint_ = false;
		};

		bool integerConstraint_; // ILP=true, 1order-LP=false
		int numberOfThreads_;    // number of threads (0=autosect)
		bool verbose_;           // switch on/off verbode mode 
		double cutUp_;           // upper cutoff
		double epGap_;           // relative optimality gap tolerance
		double workMem_;         // maximal ammount of memory in MB used for workspace
		double treeMemoryLimit_; // maximal ammount of memory in MB used for treee
		double timeLimit_;       // maximal time in seconds the solver has
		int probeingLevel_;
		int coverCutLevel_;
		int disjunctiverCutLevel_;
		int cliqueCutLevel_;
		int MIRCutLevel_;
	};


	// costructor 
	LpSolverCplex(const Parameter & parameter = Parameter())
	:
		env_(),
		model_(env_),
		x_(env_),
		c_(env_),
		obj_(IloMinimize(env_)),
		sol_(env_),
		cplex_(),
		param_(parameter),
		numVar_(0)
    {    

    }


	void addVariables(
		const UInt64Type 	numVar,
		const LpVarType  	varType,
		const LpValueType   lowerBound = 1.0,
		const LpValueType	upperBound = 1.0
	){
		if(varType==Continous){
			x_.add(IloNumVarArray(env_, numVar, lowerBound, upperBound));
		}
		else if(varType==Binary ){
			x_.add(IloNumVarArray(env_, numVar, lowerBound, upperBound, ILOBOOL));
		}
		else{
			OPENGM_CHECK(false,"not yet implemented");
		}
		numVar_+=numVar;
	}


	void setObjective(const UInt64Type lpVi,const LpValueType obj){
        obj_.setLinearCoef(x_[lpVi],obj);
    }


    template<class LPVariableIndexIterator,class CoefficientIterator>
    void addConstraint(
        LPVariableIndexIterator lpVarBegin, 
        LPVariableIndexIterator lpVarEnd, 
        CoefficientIterator     coeffBegin,
        const LpValueType lowerBound, 
        const LpValueType upperBound, 
        const std::string & name  = std::string()
    ){
	   IloRange constraint(env_, lowerBound, upperBound, name.c_str());
	   while(lpVarBegin != lpVarEnd) {
	      constraint.setLinearCoef(x_[*lpVarBegin], *coeffBegin);
	      ++lpVarBegin;
	      ++coeffBegin;
	   }
	   model_.add(constraint);
   	   // adding constraints does not require a re-initialization of the
   	   // object cplex_. cplex_ is initialized in the constructor.
    }

    void setupFinished(){
        model_.add(obj_);
        model_.add(c_);
        // initialize solver
        try {
            cplex_ = IloCplex(model_);
        }
            catch(IloCplex::Exception& e) {
            std::cout << e << std::endl;
            throw RuntimeError("CPLEX exception");
        } 
    }

    void updateObjective(){

    }
    void updateConstraints(){

    }
    UInt64Type numberOfVariables() const {
        return numVar_;
    }

    void optimize() {
		try {

            
			// verbose options
			if(param_.verbose_ == false) {
				cplex_.setParam(IloCplex::MIPDisplay, 0);
				cplex_.setParam(IloCplex::SimDisplay, 0);
				cplex_.setParam(IloCplex::SiftDisplay, 0);
			} 
            /*
			// tolarance settings
			cplex_.setParam(IloCplex::EpOpt, 1e-9); // Optimality Tolerance
			cplex_.setParam(IloCplex::EpInt, 0);    // amount by which an integer variable can differ from an integer
			cplex_.setParam(IloCplex::EpAGap, 0);   // Absolute MIP gap tolerance
			cplex_.setParam(IloCplex::EpGap, param_.epGap_); // Relative MIP gap tolerance

			// set hints
			cplex_.setParam(IloCplex::CutUp, param_.cutUp_);

			// memory setting
			cplex_.setParam(IloCplex::WorkMem, param_.workMem_);
			cplex_.setParam(IloCplex::ClockType,2);//wall-clock-time=2 cpu-time=1
			cplex_.setParam(IloCplex::TiLim,param_.treeMemoryLimit_);
			cplex_.setParam(IloCplex::MemoryEmphasis, 1);

			// time limit
			cplex_.setParam(IloCplex::TiLim, param_.timeLimit_);

			// multo-threading options
			cplex_.setParam(IloCplex::Threads, param_.numberOfThreads_);

			// Tuning
			cplex_.setParam(IloCplex::Probe, param_.probeingLevel_);
			cplex_.setParam(IloCplex::Covers, param_.coverCutLevel_);
			cplex_.setParam(IloCplex::DisjCuts, param_.disjunctiverCutLevel_);
			cplex_.setParam(IloCplex::Cliques, param_.cliqueCutLevel_);
			cplex_.setParam(IloCplex::MIRCuts, param_.MIRCutLevel_);
            */
			// solve problem
			if(!cplex_.solve()) {
				throw RuntimeError( "failed to optimize.");
			}
			cplex_.getValues(sol_, x_);
		}
		catch(IloCplex::Exception e) {
			std::cout << "caught CPLEX exception: " << e << std::endl;
			throw RuntimeError( "caught CPLEX exception:");
		} 
    }

    LpValueType lpArg(const LpIndexType lpVi)const{
        return  sol_[lpVi];
    }

    LpValueType lpValue()const{
        return  cplex_.getObjValue();
        //return cplex_.getBestObjValue();
    }

    LpValueType bestLpValue()const{
    	return cplex_.getBestObjValue();
    }

private:

    // mebers of cplex itself
	IloEnv env_;
	IloModel model_;
	IloNumVarArray x_;
	IloRangeArray c_;
	IloObjective obj_;
	IloNumArray sol_;
	IloCplex cplex_;

	//IloNumArray objBuffer_;
    // param 
    Parameter param_;
    UInt64Type numVar_;


};



}
#endif