#ifndef INFERENCE_BUNDLE_COLLECTOR_H__
#define INFERENCE_BUNDLE_COLLECTOR_H__

#include "LinearConstraints.h"

namespace opengm {
namespace learning {
namespace solver {

class BundleCollector {

public:

	template <typename ModelParameters>
	void addHyperplane(const ModelParameters& a, double b);

	const LinearConstraints& getConstraints() const { return _constraints; }

private:

	LinearConstraints _constraints;
};

template <typename ModelParameters>
void
BundleCollector::addHyperplane(const ModelParameters& a, double b) {
	/*
	  <w,a> + b ≤  ξ
	        <=>
	  <w,a> - ξ ≤ -b
	*/

	unsigned int dims = a.numberOfParameters();

	LinearConstraint constraint;

	for (unsigned int i = 0; i < dims; i++)
		constraint.setCoefficient(i, a[i]);
	constraint.setCoefficient(dims, -1.0);
	constraint.setRelation(LessEqual);
	constraint.setValue(-b);

	_constraints.add(constraint);
}

}}} // namespace opengm::learning::solver

#endif // INFERENCE_BUNDLE_COLLECTOR_H__

