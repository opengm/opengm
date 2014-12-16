#ifndef OPENGM_LEARNING_GRADIENT_ACCUMULATOR_H__
#define OPENGM_LEARNING_GRADIENT_ACCUMULATOR_H__

namespace opengm {
namespace learning {

/**
 * Model function visitor to accumulate the gradient for each model weight, 
 * given a configuration.
 */
template <typename ModelWeights, typename ConfigurationType>
class GradientAccumulator {

public:

	/**
	 * How to accumulate the gradient on the provided ModelWeights.
	 */
	enum Mode {

		Add,

		Subtract
	};

	/**
	 * @param gradient
	 *              ModelWeights reference to store the gradients.
	 * @param configuration
	 *              Current configuration of the variables in the model.
	 */
	GradientAccumulator(ModelWeights& gradient, const ConfigurationType& configuration, Mode mode = Add) :
		_gradient(gradient),
		_configuration(configuration),
		_mode(mode) {

		for (size_t i = 0; i < gradient.numberOfWeights(); i++)
			gradient[i] = 0;
	}

	template <typename FunctionType>
	void operator()(const FunctionType& function) {

		for (int i = 0; i < function.numberOfWeights(); i++) {

			int index = function.weightIndex(i);

			double g = function.weightGradient(i, _configuration.begin());
			if (_mode == Add)
				_gradient[index] += g;
			else
				_gradient[index] -= g;
		}
	}

private:

	ModelWeights& _gradient;
	const ConfigurationType& _configuration;
	Mode _mode;
};

}} // namespace opengm::learning

#endif // OPENGM_LEARNING_GRADIENT_ACCUMULATOR_H__

