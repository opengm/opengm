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
	 * @param gradient
	 *              ModelWeights reference to store the gradients.
	 * @param configuration
	 *              Current configuration of the variables in the model.
	 */
	GradientAccumulator(ModelWeights& gradient, ConfigurationType& configuration) :
		_gradient(gradient),
		_configuration(configuration) {

		for (size_t i = 0; i < gradient.numberOfWeights(); i++)
			gradient[i] = 0;
	}

	template <typename FunctionType>
	void operator()(const FunctionType& function) {

		for (int i = 0; i < function.numberOfWeights(); i++) {

			int index = function.weightIndex(i);

			_gradient[index] += function.weightGradient(i, _configuration.begin());
		}
	}

private:

	ModelWeights& _gradient;
	ConfigurationType& _configuration;
};

}} // namespace opengm::learning

#endif // OPENGM_LEARNING_GRADIENT_ACCUMULATOR_H__

