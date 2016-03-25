#ifndef INFERENCE_QUADRATIC_OBJECTIVE_H__
#define INFERENCE_QUADRATIC_OBJECTIVE_H__

#include <map>

#include "Sense.h"

namespace opengm {
namespace learning {
namespace solver {

class QuadraticObjective {

public:

	/**
	 * Create a new quadratic objective for 'size' varibales.
	 *
	 * @param size The number of coefficients in the objective.
	 */
	QuadraticObjective(unsigned int size = 0);

	/**
	 * Set the constant value of the expression.
	 *
	 * @param constant The value of the constant part of the objective.
	 */
	void setConstant(double constant);

	/**
	 * @return The value of the constant part of the objective.
	 */
	double getConstant() const;

	/**
	 * Add a coefficient.
	 *
	 * @param varNum The number of the variable to add the coefficient for.
	 * @param coef The value of the coefficient.
	 */
	void setCoefficient(unsigned int varNum, double coef);

	/**
	 * Get the linear coefficients of this objective as a map of variable
	 * numbers to coefficient values.
	 *
	 * @return A map from variable numbers to coefficient values.
	 */
	const std::vector<double>& getCoefficients() const;

	/**
	 * Add a quadratic coefficient. Use this to fill the Q matrix in the
	 * objective <a,x> + xQx.
	 *
	 * @param varNum1 The row of Q.
	 * @param varNum2 The columnt of Q.
	 * @param coef The value of the coefficient.
	 */
	void setQuadraticCoefficient(unsigned int varNum1, unsigned int varNum2, double coef);

	/**
	 * Get the quadratic coefficients of this objective as a map of pairs of variable
	 * numbers to coefficient values.
	 *
	 * @return A map from pairs of variable numbers to coefficient values.
	 */
	const std::map<std::pair<unsigned int, unsigned int>, double>& getQuadraticCoefficients() const;

	/**
	 * Set the sense of the objective.
	 *
	 * @param sense Minimize or Maximize.
	 */
	void setSense(Sense sense);

	/**
	 * Get the sense of this objective.
	 *
	 * @return Minimize or Maximize.
	 */
	Sense getSense() const;

	/**
	 * Resize the objective. New coefficients will be set to zero.
	 *
	 * @param The new size of the objective.
	 */
	void resize(unsigned int size);

	/**
	 * Get the number of variables in this objective.
	 *
	 * @return The number of variables in this objective.
	 */
	unsigned int size() const { return _coefs.size(); }

private:

	Sense _sense;

	double _constant;

	// linear coefficients are assumed to be dense, therefore we use a vector
	std::vector<double> _coefs;

	std::map<std::pair<unsigned int, unsigned int>, double> _quadraticCoefs;
};

inline
QuadraticObjective::QuadraticObjective(unsigned int size) :
	_sense(Minimize),
	_constant(0) {

	resize(size);
}

inline void
QuadraticObjective::setConstant(double constant) {

	_constant = constant;
}

inline double
QuadraticObjective::getConstant() const {

	return _constant;
}

inline void
QuadraticObjective::setCoefficient(unsigned int varNum, double coef) {

	_coefs[varNum] = coef;
}

inline const std::vector<double>&
QuadraticObjective::getCoefficients() const {

	return _coefs;
}

inline void
QuadraticObjective::setQuadraticCoefficient(unsigned int varNum1, unsigned int varNum2, double coef) {

	if (coef == 0) {

		_quadraticCoefs.erase(_quadraticCoefs.find(std::make_pair(varNum1, varNum2)));

	} else {

		_quadraticCoefs[std::make_pair(varNum1, varNum2)] = coef;
	}
}

inline const std::map<std::pair<unsigned int, unsigned int>, double>&
QuadraticObjective::getQuadraticCoefficients() const {

	return _quadraticCoefs;
}

inline void
QuadraticObjective::setSense(Sense sense) {

	_sense = sense;
}

inline Sense
QuadraticObjective::getSense() const {

	return _sense;
}

inline void
QuadraticObjective::resize(unsigned int size) {

	_coefs.resize(size, 0.0);
}

}}} // namespace opengm::learning::solver

#endif // INFERENCE_QUADRATIC_OBJECTIVE_H__

