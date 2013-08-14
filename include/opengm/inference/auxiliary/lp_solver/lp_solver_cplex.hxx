#pragma once
#ifndef OPENGM_LP_SOLVER_CPLEX_HXX
#define OPENGM_LP_SOLVER_CPLEX_HXX


#include <lp_solver_interface.hxx>

#include <ilcplex/ilocplex.h>


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
		param_(parameter)
    {    

    }


	template<class OBJ_ITER>
	void addVariables(
		const UInt64Type 	numVar,
		const LpVarType  	varType,
		const LpValueType   lowerBound,
		const LpValueType	upperBound
	){
		if(varType==Continous){
			x_.add(IloNumVarArray(env_, numVar, lowerBound, upperBound));
		}
		else if(varType==Binary ){
			x_.add(IloNumVarArray(env_, numVar, lowerBound, upperBound, ILOBOOL));
		}
		else{
			OPENGM_CHECK(false);
		}
	}


	const LpValueType & operator[](const UInt64Type lpVi)const{
		return obj[]
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

    }


    void updateLinearCoefs(){
    	obj_.setLinearCoefs(x_, obj);
    }

    UInt64Type numberOfVariables() const {
        return numVar_;
    }

    void optimize() {

    }

    LpValueType lpArg(const LpIndexType lpVi)const{
        GRBVar * gvars = grbModel_.getVars();
        return gvars[lpVi].get(GRB_DoubleAttr_X);
    }

    LpValueType lpValue()const{
        const double objval = grbModel_.get(GRB_DoubleAttr_ObjVal);
        return static_cast<LpValueType>(objval);
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

	IloNumArray obBuffer_;
    // param 
    Parameter param_;


};



}
#endif