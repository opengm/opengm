#pragma once
#ifndef OPENGM_LEARNING_BUNDLE_OPTIMIZER_HXX
#define OPENGM_LEARNING_BUNDLE_OPTIMIZER_HXX

#include "solver/QuadraticSolverFactory.h"

namespace opengm {

namespace learning {

enum OptimizerResult {

	// the minimal optimization gap was reached
	ReachedMinGap,

	// the requested number of steps was exceeded
	ReachedSteps,

	// something went wrong
	Error
};

template <typename ValueType>
class BundleOptimizer {

public:

	struct Parameter {

		Parameter() :
			lambda(1.0),
			min_gap(1e-5),
			steps(0) {}

		// regularizer weight
		double lambda;

		// stopping criteria of the bundle method optimization
		ValueType min_gap;

		// the maximal number of steps to perform, 0 = no limit
		unsigned int steps;
	};

	BundleOptimizer(const Parameter& parameter = Parameter());

	~BundleOptimizer();

	/**
	 * Start the bundle method optimization on the given dataset. It is assumed 
	 * that the models in the dataset were already augmented by the loss.
	 */
	template <typename DatasetType>
	OptimizerResult optimize(const DatasetType& dataset, typename DatasetType::ModelParameters& w);

private:

	template <typename ModelParameters>
	void setupQp(const ModelParameters& w);

	void findMinLowerBound(std::vector<ValueType>& w, ValueType& value);

	ValueType dot(const std::vector<ValueType>& a, const std::vector<ValueType>& b);

	Parameter _parameter;

	solver::QuadraticSolverBackend* _solver;
};

template <typename T>
BundleOptimizer<T>::BundleOptimizer(const Parameter& parameter) :
	_parameter(parameter),
	_solver(0) {}

template <typename T>
BundleOptimizer<T>::~BundleOptimizer() {

	if (_solver)
		delete _solver;
}

template <typename T>
template <typename DatasetType>
OptimizerResult
BundleOptimizer<T>::optimize(const DatasetType& dataset, typename DatasetType::ModelParameters& w) {

	setupQp(w);

	/*
	  1. w_0 = 0, t = 0
	  2. t++
	  3. compute a_t = ∂L(w_t-1)/∂w
	  4. compute b_t =  L(w_t-1) - <w_t-1,a_t>
	  5. ℒ_t(w) = max_i <w,a_i> + b_i
	  6. w_t = argmin λ½|w|² + ℒ_t(w)
	  7. ε_t = min_i [ λ½|w_i|² + L(w_i) ] - [ λ½|w_t|² + ℒ_t(w_t) ]
			   ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
				 smallest L(w) ever seen    current min of lower bound
	  8. if ε_t > ε, goto 2
	  9. return w_t
	*/

	//std::vector<T> w(_dims, 0.0);
	//T minValue = std::numeric_limits<T>::infinity();

	//unsigned int t = 0;

	//while (true) {

		//t++;

		//LOG_USER(bundlelog) << std::endl << "----------------- iteration " << t << std::endl;

		//std::vector<T> w_tm1 = w;

		//LOG_DEBUG(bundlelog) << "current w is " << w_tm1 << std::endl;

		//// value of L at current w
		//T L_w_tm1 = 0.0;

		//// gradient of L at current w
		//std::vector<T> a_t(_dims, 0.0);

		//// get current value and gradient
		//_valueGradientCallback(w_tm1, L_w_tm1, a_t);

		//LOG_DEBUG(bundlelog) << "       L(w)              is: " << L_w_tm1 << std::endl;
		//LOG_ALL(bundlelog)   << "      ∂L(w)/∂            is: " << a_t << std::endl;

		//// update smallest observed value of regularized L
		//minValue = std::min(minValue, L_w_tm1 + _lambda*0.5*dot(w_tm1, w_tm1));

		//LOG_DEBUG(bundlelog) << " min_i L(w_i) + ½λ|w_i|² is: " << minValue << std::endl;

		//// compute hyperplane offset
		//T b_t = L_w_tm1 - dot(w_tm1, a_t);

		//LOG_ALL(bundlelog) << "adding hyperplane " << a_t << "*w + " << b_t << std::endl;

		//// update lower bound
		//_bundleCollector->addHyperplane(a_t, b_t);

		//// minimal value of lower bound
		//T minLower;

		//// update w and get minimal value
		//findMinLowerBound(w, minLower);

		//LOG_DEBUG(bundlelog) << " min_w ℒ(w)   + ½λ|w|²   is: " << minLower << std::endl;
		//LOG_DEBUG(bundlelog) << " w* of ℒ(w)   + ½λ|w|²   is: "  << w << std::endl;

		//// compute gap
		//T eps_t = minValue - minLower;

		//LOG_USER(bundlelog)  << "          ε   is: " << eps_t << std::endl;

		//// converged?
		//if (eps_t <= _eps) {

			//if (eps_t >= 0) {

				//LOG_USER(bundlelog) << "converged!" << std::endl;

			//} else {

				//LOG_ERROR(bundlelog) << "ε < 0 -- something went wrong" << std::endl;
			//}

			//break;
		//}
	//}

	return ReachedMinGap;
}

template <typename T>
template <typename ModelParameters>
void
BundleOptimizer<T>::setupQp(const ModelParameters& w) {

	/*
	  w* = argmin λ½|w|² + ξ, s.t. <w,a_i> + b_i ≤ ξ ∀i
	*/

	if (!_solver)
		_solver = solver::QuadraticSolverFactory::Create();

	// one variable for each component of w and for ξ
	solver::QuadraticObjective obj(w.numberOfParameters() + 1);

	// regularizer
	for (unsigned int i = 0; i < w.numberOfParameters(); i++)
		obj.setQuadraticCoefficient(i, i, 0.5*_parameter.lambda);

	// ξ
	obj.setCoefficient(w.numberOfParameters(), 1.0);

	// we minimize
	obj.setSense(solver::Minimize);

	// we are done with the objective -- this does not change anymore
	_solver->setObjective(obj);
}

template <typename T>
void
BundleOptimizer<T>::findMinLowerBound(std::vector<T>& w, T& value) {

	// read the solution (pipeline magic!)
	//for (unsigned int i = 0; i < _dims; i++)
		//w[i] = (*_qpSolution)[i];

	//value = _qpSolution->getValue();
}

template <typename T>
T
BundleOptimizer<T>::dot(const std::vector<T>& a, const std::vector<T>& b) {

	OPENGM_ASSERT(a.size() == b.size());

	T d = 0.0;
	typename std::vector<T>::const_iterator i, j;
	for (i = a.begin(), j = b.begin(); i != a.end(); i++, j++)
		d += (*i)*(*j);

	return d;
}

} // learning

} // opengm

#endif // OPENGM_LEARNING_BUNDLE_OPTIMIZER_HXX

