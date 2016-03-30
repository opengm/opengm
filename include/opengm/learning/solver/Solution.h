#ifndef INFERENCE_SOLUTION_H__
#define INFERENCE_SOLUTION_H__

namespace opengm {
namespace learning {
namespace solver {

class Solution {

public:

	Solution(unsigned int size = 0);

	void resize(unsigned int size);

	unsigned int size() const { return _solution.size(); }

	const double& operator[](unsigned int i) const { return _solution[i]; }

	double& operator[](unsigned int i) { return _solution[i]; }

	std::vector<double>& getVector() { return _solution; }

	void setValue(double value) { _value = value; }

	double getValue() { return _value; }

private:

	std::vector<double> _solution;

	double _value;
};

inline Solution::Solution(unsigned int size) {

	resize(size);
}

inline void
Solution::resize(unsigned int size) {

	_solution.resize(size);
}

}}} // namspace opengm::learning::solver

#endif // INFERENCE_SOLUTION_H__

