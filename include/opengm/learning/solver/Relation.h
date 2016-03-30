#ifndef INFERENCE_RELATION_H__
#define INFERENCE_RELATION_H__

namespace opengm {
namespace learning {
namespace solver {

/** Used to indicate the relation of a linear constraint.
 */
enum Relation {

	LessEqual,
	Equal,
	GreaterEqual
};

}}} // namspace opengm::learning::solver

#endif // INFERENCE_RELATION_H__

