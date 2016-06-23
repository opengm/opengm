#ifndef INFERENCE_LINEAR_OBJECTIVE_H__
#define INFERENCE_LINEAR_OBJECTIVE_H__

#include "QuadraticObjective.h"

namespace opengm {
namespace learning {
namespace solver {

class LinearObjective : public QuadraticObjective {

public:

	LinearObjective(unsigned int size = 0) : QuadraticObjective(size) {}

private:

	using QuadraticObjective::setQuadraticCoefficient;
};

}}} // namspace opengm::learning::solver

#endif // INFERENCE_OBJECTIVE_H__

