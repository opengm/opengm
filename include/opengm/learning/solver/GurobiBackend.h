#ifndef OPENGM_LEARNING_SOLVER_GUROBI_SOLVER_H__
#define OPENGM_LEARNING_SOLVER_GUROBI_SOLVER_H__

#ifdef WITH_GUROBI

#include <string>
#include <vector>

#include <gurobi_c++.h>

#include "LinearConstraints.h"
#include "QuadraticObjective.h"
#include "QuadraticSolverBackend.h"
#include "Sense.h"
#include "Solution.h"

namespace opengm {
namespace learning {
namespace solver {

/**
 * Gurobi interface to solve the following (integer) quadratic program:
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
class GurobiBackend : public QuadraticSolverBackend {

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

	GurobiBackend(const Parameter& parameter = Parameter());

	virtual ~GurobiBackend();

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

	// dump the current problem to a file
	void dumpProblem(std::string filename);

	// set the optimality gap
	void setMIPGap(double gap);

	// set the mpi focus
	void setMIPFocus(unsigned int focus);

	// set the number of threads to use
	void setNumThreads(unsigned int numThreads);

    // create a gurobi constraint from a linear constraint
    GRBConstr createConstraint(const LinearConstraint &constraint);

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

	// the GRB environment
	GRBEnv _env;

	// the (binary) variables x
	GRBVar* _variables;

	// the objective
	GRBQuadExpr _objective;

	std::vector<GRBConstr> _constraints;

	// the GRB model containing the objective and constraints
	GRBModel _model;

	// the verbosity of the output
	int _verbosity;

	// a value by which to scale the objective
	double _scale;
};

inline GurobiBackend::GurobiBackend(const Parameter& parameter) :
	_parameter(parameter),
	_variables(0),
	_model(_env) {
}

inline GurobiBackend::~GurobiBackend() {

	std::cout << "destructing gurobi solver..." << std::endl;

	if (_variables)
		delete[] _variables;
}

inline void
GurobiBackend::initialize(
		unsigned int numVariables,
		VariableType variableType) {

	initialize(numVariables, variableType, std::map<unsigned int, VariableType>());
}

inline void
GurobiBackend::initialize(
		unsigned int                                numVariables,
		VariableType                                defaultVariableType,
		const std::map<unsigned int, VariableType>& specialVariableTypes) {

	if (_parameter.verbose)
		setVerbose(true);
	else
		setVerbose(false);

	setMIPGap(_parameter.mipGap);

	if (_parameter.mipFocus <= 3)
		setMIPFocus(_parameter.mipFocus);
	else
		std::cerr << "Invalid value for MPI focus!" << std::endl;

	setNumThreads(_parameter.numThreads);

	_numVariables = numVariables;

	// delete previous variables
	if (_variables)
		delete[] _variables;

	// add new variables to the model
	if (defaultVariableType == Binary) {

		std::cout << "creating " << _numVariables << " binary variables" << std::endl;

		_variables = _model.addVars(_numVariables, GRB_BINARY);

		_model.update();

	} else if (defaultVariableType == Continuous) {

		std::cout << "creating " << _numVariables << " continuous variables" << std::endl;

		_variables = _model.addVars(_numVariables, GRB_CONTINUOUS);

		_model.update();

		// remove default lower bound on variables
		for (unsigned int i = 0; i < _numVariables; i++)
			_variables[i].set(GRB_DoubleAttr_LB, -GRB_INFINITY);

	} else if (defaultVariableType == Integer) {

		std::cout << "creating " << _numVariables << " integer variables" << std::endl;

		_variables = _model.addVars(_numVariables, GRB_INTEGER);

		_model.update();

		// remove default lower bound on variables
		for (unsigned int i = 0; i < _numVariables; i++)
			_variables[i].set(GRB_DoubleAttr_LB, -GRB_INFINITY);
	}

	// handle special variable types
	typedef std::map<unsigned int, VariableType>::const_iterator VarTypeIt;
	for (VarTypeIt i = specialVariableTypes.begin(); i != specialVariableTypes.end(); i++) {

		unsigned int v = i->first;
		VariableType type = i->second;

		char t = (type == Binary ? 'B' : (type == Integer ? 'I' : 'C'));
		_variables[v].set(GRB_CharAttr_VType, t);
	}

	std::cout << "creating " << _numVariables << " ceofficients" << std::endl;
}

inline void
GurobiBackend::setObjective(const LinearObjective& objective) {

	setObjective((QuadraticObjective)objective);
}

inline void
GurobiBackend::setObjective(const QuadraticObjective& objective) {

	try {

		// set sense of objective
		if (objective.getSense() == Minimize)
			_model.set(GRB_IntAttr_ModelSense, 1);
		else
			_model.set(GRB_IntAttr_ModelSense, -1);

		// set the constant value of the objective
		_objective = objective.getConstant();

		std::cout << "setting linear coefficients" << std::endl;

		_objective.addTerms(&objective.getCoefficients()[0], _variables, _numVariables);

		// set the quadratic coefficients for all pairs of variables
		std::cout << "setting quadratic coefficients" << std::endl;

		typedef std::map<std::pair<unsigned int, unsigned int>, double>::const_iterator QuadCoefIt;
		for (QuadCoefIt i = objective.getQuadraticCoefficients().begin(); i != objective.getQuadraticCoefficients().end(); i++) {

			const std::pair<unsigned int, unsigned int>& variables = i->first;
			float value = i->second;

			if (value != 0)
				_objective += _variables[variables.first]*_variables[variables.second]*value;
		}

		_model.setObjective(_objective);

		_model.update();

	} catch (GRBException e) {

		std::cerr << "error: " << e.getMessage() << std::endl;
	}
}

inline void
GurobiBackend::setConstraints(const LinearConstraints& constraints) {

	// remove previous constraints
	for (std::vector<GRBConstr>::iterator constraint = _constraints.begin(); constraint != _constraints.end(); constraint++)
		_model.remove(*constraint);
	_constraints.clear();

	_model.update();

	// allocate memory for new constraints
	_constraints.reserve(constraints.size());

	try {

		std::cout << "setting " << constraints.size() << " constraints" << std::endl;

		for (LinearConstraints::const_iterator constraint = constraints.begin(); constraint != constraints.end(); constraint++) {
            _constraints.push_back(createConstraint(*constraint));
		}

		_model.update();

	} catch (GRBException e) {

		std::cerr << "error: " << e.getMessage() << std::endl;
	}
}

inline void
GurobiBackend::addConstraint(const LinearConstraint& constraint) {

    try {

        std::cout << "adding a constraint" << std::endl;

        _constraints.push_back(createConstraint(constraint));
        _model.update();

    } catch (GRBException e) {
        std::cerr << "error: " << e.getMessage() << std::endl;
    }
}

inline GRBConstr
GurobiBackend::createConstraint(const LinearConstraint& constraint)
{
    // create the lhs expression
    GRBLinExpr lhsExpr;

    // set the coefficients
    typedef std::map<unsigned int, double>::const_iterator CoefIt;
    for (CoefIt pair = constraint.getCoefficients().begin(); pair != constraint.getCoefficients().end(); pair++)
        lhsExpr += pair->second * _variables[pair->first];

    // construct constraint
    return _model.addConstr(
                lhsExpr,
                (constraint.getRelation() == LessEqual ? GRB_LESS_EQUAL :
                                                          (constraint.getRelation() == GreaterEqual ? GRB_GREATER_EQUAL :
                                                                                                       GRB_EQUAL)),
                constraint.getValue());
}

inline bool
GurobiBackend::solve(Solution& x, double& value, std::string& msg) {

	try {

		_model.optimize();

		int status = _model.get(GRB_IntAttr_Status);

		if (status != GRB_OPTIMAL) {
			msg = "Optimal solution *NOT* found";
			return false;
		} else
			msg = "Optimal solution found";

		// extract solution

		x.resize(_numVariables);
		for (unsigned int i = 0; i < _numVariables; i++)
			x[i] = _variables[i].get(GRB_DoubleAttr_X);

		// get current value of the objective
		value = _model.get(GRB_DoubleAttr_ObjVal);

		x.setValue(value);

	} catch (GRBException e) {

		std::cerr << "error: " << e.getMessage() << std::endl;

		msg = e.getMessage();

		return false;
	}

	return true;
}

inline void
GurobiBackend::setMIPGap(double gap) {

	_model.getEnv().set(GRB_DoubleParam_MIPGap, gap);
}

inline void
GurobiBackend::setMIPFocus(unsigned int focus) {

	_model.getEnv().set(GRB_IntParam_MIPFocus, focus);
}

inline void
GurobiBackend::setNumThreads(unsigned int numThreads) {

	_model.getEnv().set(GRB_IntParam_Threads, numThreads);
}

inline void
GurobiBackend::setVerbose(bool verbose) {

	// setup GRB environment
	if (verbose)
		_model.getEnv().set(GRB_IntParam_OutputFlag, 1);
	else
		_model.getEnv().set(GRB_IntParam_OutputFlag, 0);
}

inline void
GurobiBackend::dumpProblem(std::string filename) {

	try {

		_model.write(filename);

	} catch (GRBException e) {

		std::cerr << "error: " << e.getMessage() << std::endl;
	}
}

}}} // namespace opengm::learning::solver

#endif // WITH_GUROBI

#endif // GUROBI_OPENGM_LEARNING_SOLVER_SOLVER_H__


