#ifndef INFERENCE_LINEAR_CONSTRAINTS_H__
#define INFERENCE_LINEAR_CONSTRAINTS_H__

#include "LinearConstraint.h"

namespace opengm {
namespace learning {
namespace solver {

class LinearConstraints {

	typedef std::vector<LinearConstraint> linear_constraints_type;

public:

	typedef linear_constraints_type::iterator       iterator;

	typedef linear_constraints_type::const_iterator const_iterator;

	/**
	 * Create a new set of linear constraints and allocate enough memory to hold
	 * 'size' linear constraints. More or less constraints can be added, but
	 * memory might be wasted (if more allocated then necessary) or unnecessary
	 * reallocations might occur (if more added than allocated).
	 *
	 * @param size The number of linear constraints to reserve memory for.
	 */
	LinearConstraints(size_t size = 0);

	/**
	 * Remove all constraints from this set of linear constraints.
	 */
	void clear() { _linearConstraints.clear(); }

	/**
	 * Add a linear constraint.
	 *
	 * @param linearConstraint The linear constraint to add.
	 */
	void add(const LinearConstraint& linearConstraint);

	/**
	 * Add a set of linear constraints.
	 *
	 * @param linearConstraints The set of linear constraints to add.
	 */
	void addAll(const LinearConstraints& linearConstraints);

	/**
	 * @return The number of linear constraints in this set.
	 */
	unsigned int size() const { return _linearConstraints.size(); }

	const const_iterator begin() const { return _linearConstraints.begin(); }

	iterator begin() { return _linearConstraints.begin(); }

	const const_iterator end() const { return _linearConstraints.end(); }

	iterator end() { return _linearConstraints.end(); }

	const LinearConstraint& operator[](size_t i) const { return _linearConstraints[i]; }

	LinearConstraint& operator[](size_t i) { return _linearConstraints[i]; }

	/**
	 * Get a list of indices of linear constraints that use the given variables.
	 */
	std::vector<unsigned int> getConstraints(const std::vector<unsigned int>& variableIds);

private:

	linear_constraints_type _linearConstraints;
};

inline
LinearConstraints::LinearConstraints(size_t size) {

	_linearConstraints.resize(size);
}

inline void
LinearConstraints::add(const LinearConstraint& linearConstraint) {

	_linearConstraints.push_back(linearConstraint);
}

inline void
LinearConstraints::addAll(const LinearConstraints& linearConstraints) {

	_linearConstraints.insert(_linearConstraints.end(), linearConstraints.begin(), linearConstraints.end());
}

inline std::vector<unsigned int>
LinearConstraints::getConstraints(const std::vector<unsigned int>& variableIds) {

	std::vector<unsigned int> indices;

	for (unsigned int i = 0; i < size(); i++) {

		LinearConstraint& constraint = _linearConstraints[i];

		for (std::vector<unsigned int>::const_iterator v = variableIds.begin(); v != variableIds.end(); v++) {

			if (constraint.getCoefficients().count(*v) != 0) {

				indices.push_back(i);
				break;
			}
		}
	}

	return indices;
}

}}} // namespace opengm::learning::solver

#endif // INFERENCE_LINEAR_CONSTRAINTS_H__

