#pragma once
#ifndef OPENGM_LP_SOLVER_GUROBI_HXX
#define OPENGM_LP_SOLVER_GUROBI_HXX


#include "gurobi_c++.h"


namespace opengm{


class LpSolverGurobi{
public:
	typedef double LpValueType;
	typedef int    LpIndexType;

	struct  Parameter{



        struct Termination{
            Termination(
                const double cutoff         = std::numeric_limits<double>::infinity(),
                const double iterationLimit = std::numeric_limits<double>::infinity(),
                const double nodeLimit      = std::numeric_limits<double>::infinity(),
                const double solutionLimit  = std::numeric_limits<int>::max(),
                const double timeLimit      = std::numeric_limits<double>::infinity(),
                const double barIterLimit   = std::numeric_limits<int>::max()
            )
            :
            cutoff_(cutoff),
            iterationLimit_(iterationLimit),
            nodeLimit_(nodeLimit),
            solutionLimit_(solutionLimit),
            timeLimit_(timeLimit),
            barIterLimit_(barIterLimit)
            {

            }

            double cutoff_;
            double iterationLimit_;
            double nodeLimit_;
            double solutionLimit_;
            double timeLimit_;
            int    barIterLimit_;
        };

        struct Tolerances{
            Tolerances(

            ){

            }
        };

        struct Simplex{
            Simplex(

            ){

            }
        };

        struct Barrier{
            Barrier(

            ){

            }
        };

        struct MIP{
            MIP(

            ){

            }  
        };

        struct MIPCuts{
            MIPCuts(

            ){

            }
        };

        struct Others{
            Others(

            ){

            } 
        };

        Parameter(
            const bool integerConstraint    = false,
            const Termination & termination = Termination(),
            const Tolerances & tolerances   = Tolerances(),
            const Simplex & simplex         = Simplex(),
            const Barrier & barrier         = Barrier(),
            const MIP & mip                 = MIP(),
            const MIPCuts & mipCuts         = MIPCuts(),
            const Others & other            = Others()
        )
        :   integerConstraint_(integerConstraint),
            termination_(termination),
            tolerances_(tolerances), 
            simplex_(simplex),
            barrier_(barrier),
            mip_(mip),
            mipCuts_(mipCuts),
            other_(other)
        {
        }


		bool integerConstraint_;
        Termination termination_;
        Tolerances tolerances_;  
        Simplex simplex_;
        Barrier barrier_;
        MIP mip_;
        MIPCuts mipCuts_;
        Others other_;

	};


	// costructor 
	LpSolverGurobi(const Parameter & parameter = Parameter())
	:
    grbEnv_(),
    grbModel_(grbEnv_),
    param_(parameter),
    objBuffer_(),
    lbBuffer_(),
    ubBuffer_(),
    varTypeBuffer_(),
    numVar_(0)
    {    

    }

    void addVariable(const LpValueType lb,const LpValueType ub,const LpValueType obj){
        grbModel_.addVar(lb,ub,obj,param_.integerConstraint_ ? GRB_BINARY : GRB_CONTINUOUS);  
    }

	template<class OBJ_ITER,class LB_ITER,class UP_ITER>
	void addVariables(
		OBJ_ITER objectiveBegin,
		OBJ_ITER objectiveEnd,
		LB_ITER lowerBoundBegin,
		UP_ITER upperBoundBegin
	){
		// ensure buffers have the correct size
		const UInt64Type nInputVar = std::distance(objectiveBegin,objectiveEnd);
		ensureBufferSize(nInputVar);

		// copy the values from iterator to
		// the needed pointer data types
		std::copy(objectiveBegin,  objectiveEnd,	objBuffer_.begin());
		std::copy(lowerBoundBegin, lowerBoundBegin,	lbBuffer_.begin() );
		std::copy(objectiveBegin,  objectiveEnd,	ubBuffer_.begin() );
        std::fill(varTypeBuffer_.begin(),varTypeBuffer_.begin()+nInputVar,
            param_.integerConstraint_ ? GRB_BINARY : GRB_CONTINUOUS);
		// add the variables to the model
        grbModel_.addVars (  
            & lbBuffer_[0],
            & ubBuffer_[0],
            & objBuffer_[0], 
            & varTypeBuffer_[0],
            NULL,                         
            static_cast<int>(nInputVar)     // number of variables to add
        );
        numVar_+=nInputVar;
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
        GRBVar * gvars = grbModel_.getVars();

        if(upperBound-lowerBound < 0.000000001){
            GRBLinExpr linExp;// = new GRBLinExpr();
            while(lpVarBegin!=lpVarEnd){
                const LpIndexType lpVi  = static_cast<LpIndexType>(*lpVarBegin);
                const LpValueType coeff = static_cast<LpValueType>(*coeffBegin);
                linExp.addTerms(&coeff,&gvars[lpVi],1);
                ++lpVarBegin;
                ++coeffBegin;
            }
            if(name.size()>0){
                grbModel_.addConstr(linExp,GRB_EQUAL,static_cast<LpValueType>(lowerBound),name.c_str());
            }
            else{
                grbModel_.addConstr(linExp,GRB_EQUAL,static_cast<LpValueType>(lowerBound));
            }
        }
        else{
            GRBLinExpr linExpLower;// = new GRBLinExpr();
            GRBLinExpr linExpUpper;// = new GRBLinExpr();
            while(lpVarBegin!=lpVarEnd){
                const LpIndexType lpVi  = static_cast<LpIndexType>(*lpVarBegin);
                const LpValueType coeff = static_cast<LpValueType>(*coeffBegin);
                linExpLower.addTerms(&coeff,&gvars[lpVi],1);
                linExpUpper.addTerms(&coeff,&gvars[lpVi],1);
                ++lpVarBegin;
                ++coeffBegin;
            }
            if(name.size()>0){
                std::string nameLower = name + std::string("_lower");
                std::string nameUpper = name + std::string("_upper");
                grbModel_.addConstr(linExpLower,GRB_GREATER_EQUAL ,static_cast<LpValueType>(lowerBound),nameLower);
                grbModel_.addConstr(linExpUpper,GRB_LESS_EQUAL    ,static_cast<LpValueType>(upperBound),nameUpper);
            }
            else{
                grbModel_.addConstr(linExpLower,GRB_GREATER_EQUAL ,static_cast<LpValueType>(lowerBound));
                grbModel_.addConstr(linExpUpper,GRB_LESS_EQUAL    ,static_cast<LpValueType>(upperBound));
            }
        }
    }


    void updateModel(){
        grbModel_.update();
    }

    UInt64Type numberOfVariables() const {
        return numVar_;
    }

    void optimize() {
        try{
            grbModel_.optimize();
        }
        catch(GRBException e) {
            std::cout << "Error code = " << e.getErrorCode() << "\n";
            std::cout << e.getMessage() <<"\n";
            throw RuntimeError("Exception during gurobi optimization");
        } 
        catch(...) {
            throw RuntimeError("Exception during gurobi optimization");
        }
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

	void ensureBufferSize(const UInt64Type size){
		if(objBuffer_.size()<size)
			objBuffer_.resize(size);
		if(lbBuffer_.size()<size)
			lbBuffer_.resize(size);
		if(ubBuffer_.size()<size)
			ubBuffer_.resize(size);
        if(varTypeBuffer_.size()<size)
            varTypeBuffer_.resize(size);
	}

    // mebers of gurobi itself
    GRBEnv     grbEnv_;
    GRBModel   grbModel_;

    // param 
    Parameter param_;

	// members for interface 
	std::vector<LpValueType> objBuffer_;
	std::vector<LpValueType> lbBuffer_;
	std::vector<LpValueType> ubBuffer_;
    std::vector<char>        varTypeBuffer_;
	UInt64Type numVar_;

};



}
#endif