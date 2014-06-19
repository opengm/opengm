#ifndef OPENGM_LEARNING_SOLVER_QUADRATIC_SOLVER_FACTORY_H__
#define OPENGM_LEARNING_SOLVER_QUADRATIC_SOLVER_FACTORY_H__

#ifdef WITH_GUROBI
#include "GurobiBackend.h"
#endif

namespace opengm {
namespace learning {
namespace solver {

class QuadraticSolverFactory {

public:

	static QuadraticSolverBackend* Create() {

#ifdef WITH_GUROBI
		return new GurobiBackend();
#endif

      throw opengm::RuntimeError("No quadratic solver available.");
	}
};

}}} // namespace opengm::learning::solver

#endif // OPENGM_LEARNING_SOLVER_QUADRATIC_SOLVER_FACTORY_H__

