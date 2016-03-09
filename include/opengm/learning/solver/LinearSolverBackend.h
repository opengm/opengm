#ifndef INFERENCE_LINEAR_SOLVER_BACKEND_H__
#define INFERENCE_LINEAR_SOLVER_BACKEND_H__

#include "LinearObjective.h"
#include "LinearConstraints.h"
#include "Solution.h"
#include "VariableType.h"

namespace opengm {
namespace learning {
namespace solver {

class LinearSolverBackend {

public:

	virtual ~LinearSolverBackend() {};

	/**
	 * Initialise the linear solver for the given type of variables.
	 *
	 * @param numVariables The number of variables in the problem.
	 * @param variableType The type of the variables (Continuous, Integer,
	 *                     Binary).
	 */
	virtual void initialize(
			unsigned int numVariables,
			VariableType variableType) = 0;

	/**
	 * Initialise the linear solver for the given type of variables.
	 *
	 * @param numVariables
	 *             The number of variables in the problem.
	 * 
	 * @param defaultVariableType
	 *             The default type of the variables (Continuous, Integer, 
	 *             Binary).
	 *
	 * @param specialVariableTypes
	 *             A map of variable numbers to variable types to override the 
	 *             default.
	 */
	virtual void initialize(
			unsigned int                                numVariables,
			VariableType                                defaultVariableType,
			const std::map<unsigned int, VariableType>& specialVariableTypes) = 0;

	/**
	 * Set the objective.
	 *
	 * @param objective A linear objective.
	 */
	virtual void setObjective(const LinearObjective& objective) = 0;

	/**
	 * Set the linear (in)equality constraints.
	 *
	 * @param constraints A set of linear constraints.
	 */
	virtual void setConstraints(const LinearConstraints& constraints) = 0;

	/**
	 * Add a single linear constraint.
	 *
	 * @param constraint The constraint to add.
	 */
	virtual void addConstraint(const LinearConstraint& constraint) = 0;

	/**
	 * Solve the problem.
	 *
	 * @param solution A solution object to write the solution to.
	 * @param value The optimal value of the objective.
	 * @param message A status message from the solver.
	 * @return true, if the optimal value was found.
	 */
	virtual bool solve(Solution& solution, double& value, std::string& message) = 0;
};

}}} // namespace opengm::learning::solver

#endif // INFERENCE_LINEAR_SOLVER_BACKEND_H__

