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
	 *              ModelWeights reference to store the gradients. Gradient 
	 *              values will only be added (or subtracted, if mode == 
	 *              Subtract), so you have to make sure gradient is properly 
	 *              initialized to zero.
	 *
	 * @param configuration
	 *              Configuration of the variables in the model, to evaluate the 
	 *              gradient for.
	 *
	 * @param mode
	 *              Add or Subtract the weight gradients from gradient.
	 */
	GradientAccumulator(ModelWeights& gradient, const ConfigurationType& configuration, Mode mode = Add) :
		_gradient(gradient),
		_configuration(configuration),
		_mode(mode) {}

	template <typename Iterator, typename FunctionType>
	void operator()(Iterator begin, Iterator end, const FunctionType& function) {

		ConfigurationType localConfiguration;
		for (Iterator j = begin; j != end; j++)
			localConfiguration.push_back(_configuration[*j]);

		for (int i = 0; i < function.numberOfWeights(); i++) {

			int index = function.weightIndex(i);

			double g = function.weightGradient(i, localConfiguration.begin());

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

