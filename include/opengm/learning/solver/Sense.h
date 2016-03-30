#ifndef SENSE_H__
#define SENSE_H__

namespace opengm {
namespace learning {
namespace solver {

/** Used to indicate whether an objective is supposed to be minimized or
 * maximized.
 */
enum Sense {

	Minimize,
	Maximize
};

}}} // namspace opengm::learning::solver

#endif // SENSE_H__

