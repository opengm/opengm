#ifndef OPENGM_LEARNING_SOLVER_CPLEX_SOLVER_H__
#define OPENGM_LEARNING_SOLVER_CPLEX_SOLVER_H__

#ifdef WITH_CPLEX

#include <string>
#include <vector>

#include <ilcplex/ilocplex.h>

#include "LinearConstraints.h"
#include "QuadraticObjective.h"
#include "QuadraticSolverBackend.h"
#include "Sense.h"
#include "Solution.h"

namespace opengm {
namespace learning {
namespace solver {

/**
 * Cplex interface to solve the following (integer) quadratic program:
 *
 * min  <a,x> + xQx
 * s.t. Ax  == b
 *      Cx  <= d
 *      optionally: x_i \in {0,1} for all i
 *
 * Where (A,b) describes all linear equality constraints, (C,d) all linear
 * inequality constraints and x is the solution vector. a is a real-valued
 * vector denoting the coefficients of the objective and Q a PSD matrix giving
 * the quadratic coefficients of the objective.
 */
class CplexBackend : public QuadraticSolverBackend {

public:

    struct Parameter {

        Parameter() :
            mipGap(0.0001),
            mipFocus(0),
            numThreads(0),
            verbose(false) {}

        // The Gurobi relative optimality gap.
        double mipGap;

        // The Gurobi MIP focus: 0 = balanced, 1 = feasible solutions, 2 =
        // optimal solution, 3 = bound.
        unsigned int mipFocus;

        // The number of threads to be used by Gurobi. The default (0) uses all
        // available CPUs.
        unsigned int numThreads;

        // Show the gurobi output.
        bool verbose;
    };

    CplexBackend(const Parameter& parameter = Parameter());

    virtual ~CplexBackend();

    ///////////////////////////////////
    // solver backend implementation //
    ///////////////////////////////////

    void initialize(
            unsigned int numVariables,
            VariableType variableType);

    void initialize(
            unsigned int                                numVariables,
            VariableType                                defaultVariableType,
            const std::map<unsigned int, VariableType>& specialVariableTypes);

    void setObjective(const LinearObjective& objective);

    void setObjective(const QuadraticObjective& objective);

    void setConstraints(const LinearConstraints& constraints);

    void addConstraint(const LinearConstraint& constraint);

    bool solve(Solution& solution, double& value, std::string& message);

private:

    //////////////
    // internal //
    //////////////

    // set the optimality gap
    void setMIPGap(double gap);

    // set the mpi focus
    void setMIPFocus(unsigned int focus);

    // set the number of threads to use
    void setNumThreads(unsigned int numThreads);

    // create a CPLEX constraint from a linear constraint
    IloRange createConstraint(const LinearConstraint &constraint);

    /**
     * Enable solver output.
     */
    void setVerbose(bool verbose);

    // size of a and x
    unsigned int _numVariables;

    // rows in A
    unsigned int _numEqConstraints;

    // rows in C
    unsigned int _numIneqConstraints;

    Parameter _parameter;

    // the verbosity of the output
    int _verbosity;

    // a value by which to scale the objective
    double _scale;

    // Objective, constraints and cplex environment:
    IloEnv env_;
    IloModel model_;
    IloNumVarArray x_;
    IloRangeArray c_;
    IloObjective obj_;
    IloNumArray sol_;
    IloCplex cplex_;
    double constValue_;

    typedef std::vector<IloExtractable> ConstraintVector;
    ConstraintVector _constraints;
};

inline CplexBackend::CplexBackend(const Parameter& parameter) :
    _parameter(parameter),
    model_(env_),
    x_(env_),
    c_(env_),
    obj_(env_),
    sol_(env_)
{
    std::cout << "constructing cplex solver" << std::endl;
}

inline CplexBackend::~CplexBackend() {
    std::cout << "destructing cplex solver..." << std::endl;
}

inline void
CplexBackend::initialize(
        unsigned int numVariables,
        VariableType variableType) {

    initialize(numVariables, variableType, std::map<unsigned int, VariableType>());
}

inline void
CplexBackend::initialize(
        unsigned int                                numVariables,
        VariableType                                defaultVariableType,
        const std::map<unsigned int, VariableType>& specialVariableTypes) {

    _numVariables = numVariables;

    // delete previous variables
    x_.clear();

    // add new variables to the model
    if (defaultVariableType == Binary) {
        std::cout << "creating " << _numVariables << " binary variables" << std::endl;
        x_.add(IloNumVarArray(env_, _numVariables, 0, 1, ILOBOOL));
    } else if (defaultVariableType == Continuous) {
        std::cout << "creating " << _numVariables << " continuous variables" << std::endl;
        x_.add(IloNumVarArray(env_, _numVariables, -IloInfinity, IloInfinity));
    } else if (defaultVariableType == Integer) {
        x_.add(IloNumVarArray(env_, _numVariables, -IloInfinity, IloInfinity, ILOINT));
    }

    // TODO: port me!
//    // handle special variable types
//    typedef std::map<unsigned int, VariableType>::const_iterator VarTypeIt;
//    for (VarTypeIt i = specialVariableTypes.begin(); i != specialVariableTypes.end(); i++) {

//        unsigned int v = i->first;
//        VariableType type = i->second;

//        char t = (type == Binary ? 'B' : (type == Integer ? 'I' : 'C'));
//        _variables[v].set(GRB_CharAttr_VType, t);
//    }

    std::cout << "creating " << _numVariables << " ceofficients" << std::endl;
}

inline void
CplexBackend::setObjective(const LinearObjective& objective) {

    setObjective((QuadraticObjective)objective);
}

inline void
CplexBackend::setObjective(const QuadraticObjective& objective) {

    try {

        // set sense of objective
        if (objective.getSense() == Minimize)
            obj_ = IloMinimize(env_);
        else
            obj_ = IloMaximize(env_);

        // set the constant value of the objective
        obj_.setConstant(objective.getConstant());

        std::cout << "setting linear coefficients" << std::endl;

        for(size_t i = 0; i < _numVariables; i++)
        {
            obj_.setLinearCoef(x_[i], objective.getCoefficients()[i]);
        }

        // set the quadratic coefficients for all pairs of variables
        std::cout << "setting quadratic coefficients" << std::endl;

        typedef std::map<std::pair<unsigned int, unsigned int>, double>::const_iterator QuadCoefIt;
        for (QuadCoefIt i = objective.getQuadraticCoefficients().begin(); i != objective.getQuadraticCoefficients().end(); i++) {

            const std::pair<unsigned int, unsigned int>& variables = i->first;
            float value = i->second;

            if (value != 0)
                obj_.setQuadCoef(x_[variables.first], x_[variables.second], value);
        }

        model_.add(obj_);

    } catch (IloCplex::Exception e) {

        std::cerr << "CPLEX error: " << e.getMessage() << std::endl;
    }
}

inline void
CplexBackend::setConstraints(const LinearConstraints& constraints) {

    // remove previous constraints
    for (ConstraintVector::iterator constraint = _constraints.begin(); constraint != _constraints.end(); constraint++)
        model_.remove(*constraint);
    _constraints.clear();

    // allocate memory for new constraints
    _constraints.reserve(constraints.size());

    try {
        std::cout << "setting " << constraints.size() << " constraints" << std::endl;

        IloExtractableArray cplex_constraints(env_);
        for (LinearConstraints::const_iterator constraint = constraints.begin(); constraint != constraints.end(); constraint++) {
            IloRange linearConstraint = createConstraint(*constraint);
            _constraints.push_back(linearConstraint);
            cplex_constraints.add(linearConstraint);
        }

        // add all constraints as batch to the model
        model_.add(cplex_constraints);

    } catch (IloCplex::Exception e) {

        std::cerr << "error: " << e.getMessage() << std::endl;
    }
}

inline void
CplexBackend::addConstraint(const LinearConstraint& constraint) {

    try {
        std::cout << "adding a constraint" << std::endl;

        // add to the model
        _constraints.push_back(model_.add(createConstraint(constraint)));

    } catch (IloCplex::Exception e) {

        std::cerr << "error: " << e.getMessage() << std::endl;
    }
}

inline IloRange
CplexBackend::createConstraint(const LinearConstraint& constraint) {
    // create the lhs expression
    IloExpr linearExpr(env_);

    // set the coefficients
    typedef std::map<unsigned int, double>::const_iterator CoefIt;
    for (CoefIt pair = constraint.getCoefficients().begin(); pair != constraint.getCoefficients().end(); pair++)
    {
        linearExpr.setLinearCoef(x_[pair->first], pair->second);
    }

    switch(constraint.getRelation())
    {
        case LessEqual:
            return IloRange(env_, linearExpr, constraint.getValue());
            break;
        case GreaterEqual:
            return IloRange(env_, constraint.getValue(), linearExpr);
            break;
    }
}

inline bool
CplexBackend::solve(Solution& x, double& value, std::string& msg) {

    try {
        cplex_ = IloCplex(model_);
        setVerbose(_parameter.verbose);

        setMIPGap(_parameter.mipGap);

        if (_parameter.mipFocus <= 3)
            setMIPFocus(_parameter.mipFocus);
        else
            std::cerr << "Invalid value for MIP focus!" << std::endl;

        setNumThreads(_parameter.numThreads);

        if(!cplex_.solve()) {
           std::cout << "failed to optimize. " << cplex_.getStatus() << std::endl;
           msg = "Optimal solution *NOT* found";
           return false;
        }
        else
            msg = "Optimal solution found";

        // extract solution
        cplex_.getValues(sol_, x_);
        x.resize(_numVariables);
        for (unsigned int i = 0; i < _numVariables; i++)
            x[i] = sol_[i];

        // get current value of the objective
        value = cplex_.getObjValue();

        x.setValue(value);

    } catch (IloCplex::Exception& e) {

        std::cerr << "error: " << e.getMessage() << std::endl;

        msg = e.getMessage();

        return false;
    }

    return true;
}

inline void
CplexBackend::setMIPGap(double gap) {
     cplex_.setParam(IloCplex::EpGap, gap);
}

inline void
CplexBackend::setMIPFocus(unsigned int focus) {
    /*
     * GUROBI and CPLEX have the same meaning for the values of the MIPFocus and MIPEmphasis parameter:
     *
     * GUROBI docs:
     * If you are more interested in finding feasible solutions quickly, you can select MIPFocus=1.
     * If you believe the solver is having no trouble finding good quality solutions,
     * and wish to focus more attention on proving optimality, select MIPFocus=2.
     * If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3
     * to focus on the bound.
     *
     * CPLEX params:
     * switch(focus) {
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
        }
     */

    cplex_.setParam(IloCplex::MIPEmphasis, focus);
}

inline void
CplexBackend::setNumThreads(unsigned int numThreads) {
    cplex_.setParam(IloCplex::Threads, numThreads);
}

inline void
CplexBackend::setVerbose(bool verbose) {

    // setup GRB environment
    if (verbose)
    {
        cplex_.setParam(IloCplex::MIPDisplay, 1);
        cplex_.setParam(IloCplex::SimDisplay, 1);
        cplex_.setParam(IloCplex::SiftDisplay, 1);
    }
    else
    {
        cplex_.setParam(IloCplex::MIPDisplay, 0);
        cplex_.setParam(IloCplex::SimDisplay, 0);
        cplex_.setParam(IloCplex::SiftDisplay, 0);
    }
}

}}} // namespace opengm::learning::solver

#endif // WITH_CPLEX

#endif // CPLEX_OPENGM_LEARNING_SOLVER_SOLVER_H__
