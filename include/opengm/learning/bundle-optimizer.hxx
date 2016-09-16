#pragma once
#ifndef OPENGM_LEARNING_BUNDLE_OPTIMIZER_HXX
#define OPENGM_LEARNING_BUNDLE_OPTIMIZER_HXX

#include "solver/BundleCollector.h"
#include "solver/QuadraticSolverFactory.h"

namespace opengm {

namespace learning {

//template <typename T>
//std::ostream& operator<<(std::ostream& out, Weights<T>& w) {

//    out << "[";
//    for (int i = 0; i < w.numberOfWeights(); i++) {

//        if (i > 0)
//            out << ", ";
//        out << w[i];
//    }
//    out << "]";
//}

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

	enum EpsStrategy {

		/**
		 * Compute the eps from the gap estimate between the lower bound and the 
		 * target objective. The gap estimate will only be correct for oracle 
		 * calls that perform exact inference.
		 */
		EpsFromGap,

		/**
		 * Compute the eps from the change of the minimum of the lower bound.  
		 * This version does also work for approximate (but deterministic) 
		 * inference methods.
		 */
		EpsFromChange
	};

	struct Parameter {

		Parameter() :
			lambda(1.0),
			min_eps(1e-5),
			steps(0),
            epsStrategy(EpsFromChange),
            nonNegativeWeights(false){}

		// regularizer weight
		double lambda;

		// the maximal number of steps to perform, 0 = no limit
		unsigned int steps;

		// bundle method stops if eps is smaller than this value
		ValueType min_eps;

		// how to compute the eps for the stopping criterion
		EpsStrategy epsStrategy;
        bool verbose_;
        bool nonNegativeWeights;
	};

	BundleOptimizer(const Parameter& parameter = Parameter());

	~BundleOptimizer();

	/**
	 * Start the bundle method optimization on the given oracle. The oracle has 
	 * to model:
	 *
     *   Weights current;
     *   Weights gradient;
	 *   double          value;
	 *
	 *   valueAndGradient = oracle(current, value, gradient);
	 *
	 * and should return the value and gradient of the objective function 
	 * (passed by reference) at point 'current'.
	 */
    template <typename Oracle, typename Weights>
    OptimizerResult optimize(Oracle& oracle, Weights& w);

private:

    template <typename Weights>
    void setupQp(const Weights& w);

	template <typename ModelWeights>
	void findMinLowerBound(ModelWeights& w, ValueType& value);

	template <typename ModelWeights>
	ValueType dot(const ModelWeights& a, const ModelWeights& b);

	Parameter _parameter;

	solver::BundleCollector _bundleCollector;

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
template <typename Oracle, typename Weights>
OptimizerResult
BundleOptimizer<T>::optimize(Oracle& oracle, Weights& w) {

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

	T minValue     =  std::numeric_limits<T>::infinity();
	T lastMinLower = -std::numeric_limits<T>::infinity();

    if (_parameter.nonNegativeWeights){
        if (_parameter.verbose_)
            std::cout << "creating " << w.size() << " non-negative constraints" << std::endl;
        for(size_t i=0; i<w.size(); ++i){
            Weights vecLHS(w.numberOfWeights());
            vecLHS[i]=1.0;
            _bundleCollector.addNonNegative(vecLHS);
            if (_parameter.verbose_){
                std::cout << "adding non-negativity constraint" << i << ": (";
                for(size_t j=0; j<vecLHS.size(); ++j){
                    std::cout << vecLHS[j] << " ";
                }
                std::cout << ")*w >= " << 0.0 << std::endl;
            }
        }
    }

    unsigned int t = 0;

    while (true) {

		t++;

        if(_parameter.verbose_)
            std::cout << std::endl << "----------------- iteration      " << t << std::endl;

        Weights w_tm1 = w;

        if(_parameter.verbose_){
            std::cout << "w: ";
            for(size_t i=0; i<w_tm1.size(); ++i)
                std::cout << w_tm1[i] << " ";
            std::cout << std::endl;
        }

		// value of L at current w
		T L_w_tm1 = 0.0;

		// gradient of L at current w
        Weights a_t(w.numberOfWeights());

		// get current value and gradient
		oracle(w_tm1, L_w_tm1, a_t);

        if(_parameter.verbose_){
            std::cout << "       L(w)              is: " << L_w_tm1 << std::endl;
            std::cout << "∂L(w)/∂:  (";
            for(size_t i=0; i<a_t.size(); ++i)
                std::cout << a_t[i] << " ";
            std::cout << ")" << std::endl;
        }

		// update smallest observed value of regularized L
		minValue = std::min(minValue, L_w_tm1 + _parameter.lambda*0.5*dot(w_tm1, w_tm1));

        if(_parameter.verbose_)
            std::cout << " min_i L(w_i) + ½λ|w_i|² is: " << minValue << std::endl;

		// compute hyperplane offset
		T b_t = L_w_tm1 - dot(w_tm1, a_t);

        if(_parameter.verbose_){
            std::cout << "adding hyperplane: ( ";
            for(size_t i=0; i<a_t.size(); ++i)
                std::cout << a_t[i] << " ";
            std::cout << ")*w + " << b_t << std::endl;
        }

		// update lower bound
		_bundleCollector.addHyperplane(a_t, b_t);

		// minimal value of lower bound
		T minLower;

        // update w and get minimal value
		findMinLowerBound(w, minLower);

        // norm of w
        double norm = 0.0;
        for(size_t i=0; i<w.size(); ++i)
            norm += w[i]*w[i];
        norm = std::sqrt(norm);

        if(_parameter.verbose_){
            std::cout << " min_w ℒ(w)   + ½λ|w|²   is: " << minLower << std::endl;
            std::cout << " w* of ℒ(w)   + ½λ|w|²   is: (";
            for(size_t i=0; i<w.size(); ++i)
                std::cout << w[i] << " ";
            std::cout << ")              normalized: (";
            for(size_t i=0; i<w.size(); ++i)
                std::cout << w[i]/norm << " ";
            std::cout << ")" << std::endl;
        }

		// compute gap
		T eps_t;
		if (_parameter.epsStrategy == EpsFromGap)
			eps_t = minValue - minLower;
		else
			eps_t = minLower - lastMinLower;

		lastMinLower = minLower;

        if(_parameter.verbose_)
            std::cout  << "          ε   is: " << eps_t << std::endl;

		// converged?
		if (eps_t <= _parameter.min_eps)
			break;
	}

	return ReachedMinGap;
}

template <typename T>
template <typename Weights>
void
BundleOptimizer<T>::setupQp(const Weights& w) {

	/*
	  w* = argmin λ½|w|² + ξ, s.t. <w,a_i> + b_i ≤ ξ ∀i
	*/

	if (!_solver)
		_solver = solver::QuadraticSolverFactory::Create();

	_solver->initialize(w.numberOfWeights() + 1, solver::Continuous);

	// one variable for each component of w and for ξ
    solver::QuadraticObjective obj(w.numberOfWeights() + 1);

	// regularizer
    for (unsigned int i = 0; i < w.numberOfWeights(); i++)
		obj.setQuadraticCoefficient(i, i, 0.5*_parameter.lambda);

	// ξ
    obj.setCoefficient(w.numberOfWeights(), 1.0);

	// we minimize
	obj.setSense(solver::Minimize);

	// we are done with the objective -- this does not change anymore
	_solver->setObjective(obj);
}

template <typename T>
template <typename ModelWeights>
void
BundleOptimizer<T>::findMinLowerBound(ModelWeights& w, T& value) {

	_solver->setConstraints(_bundleCollector.getConstraints());

	solver::Solution x;
	std::string msg;
	bool optimal = _solver->solve(x, value, msg);

	if (!optimal) {

		std::cerr
				<< "[BundleOptimizer] QP could not be solved to optimality: "
				<< msg << std::endl;

		return;
	}

	for (size_t i = 0; i < w.numberOfWeights(); i++)
		w[i] = x[i];
}

template <typename T>
template <typename ModelWeights>
T
BundleOptimizer<T>::dot(const ModelWeights& a, const ModelWeights& b) {

	OPENGM_ASSERT(a.numberOfWeights() == b.numberOfWeights());

	T d = 0.0;
	for (size_t i = 0; i < a.numberOfWeights(); i++)
		d += a[i]*b[i];

	return d;
}

} // learning

} // opengm

#endif // OPENGM_LEARNING_BUNDLE_OPTIMIZER_HXX

