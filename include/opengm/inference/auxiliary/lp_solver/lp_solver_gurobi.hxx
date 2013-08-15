#pragma once
#ifndef OPENGM_LP_SOLVER_GUROBI_HXX
#define OPENGM_LP_SOLVER_GUROBI_HXX

#include "gurobi_c++.h"
#include "lp_solver_interface.hxx"


namespace opengm{


class LpSolverGurobi :public LpSolverInterface{
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
            const Termination & termination = Termination(),
            const Tolerances & tolerances   = Tolerances(),
            const Simplex & simplex         = Simplex(),
            const Barrier & barrier         = Barrier(),
            const MIP & mip                 = MIP(),
            const MIPCuts & mipCuts         = MIPCuts(),
            const Others & other            = Others()
        )
        :   termination_(termination),
            tolerances_(tolerances), 
            simplex_(simplex),
            barrier_(barrier),
            mip_(mip),
            mipCuts_(mipCuts),
            other_(other)
        {
        }

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
    numVar_(0)
    {    

    }


	void addVariables(
        const UInt64Type numVar,
        const LpVarType varType,
		const LpValueType lowerBound = 0.0,
        const LpValueType upperBound = 1.0
	){
        if(varType==Continous){
            for(UInt64Type i=0;i<numVar;++i)
                grbModel_.addVar(lowerBound,upperBound,0.0,GRB_CONTINUOUS);
        }
        else if (varType == Binary){
            for(UInt64Type i=0;i<numVar;++i)
                grbModel_.addVar(lowerBound,upperBound,0.0,GRB_BINARY);
        }
        else{
            throw RuntimeError("General Integer VarType is not yet implemented");
        }
        numVar_+=numVar;
	}

    void setObjective(const UInt64Type lpVi,const LpValueType obj){
        grbModel_.getVars()[lpVi].set(GRB_DoubleAttr_Obj, obj); 
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


    void updateObjective(){
        grbModel_.update();
    }

    void updateConstraints(){

    }

    void setupFinished(){

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

    LpValueType bestLpValue()const{
        return this->lpValue();
    }

private:


    // mebers of gurobi itself
    GRBEnv     grbEnv_;
    GRBModel   grbModel_;

    // param 
    Parameter param_;
	UInt64Type numVar_;

};



}
#endif