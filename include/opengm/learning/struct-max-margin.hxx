#pragma once
#ifndef OPENGM_LEARNING_STRUCT_MAX_MARGIN_HXX
#define OPENGM_LEARNING_STRUCT_MAX_MARGIN_HXX

#include "bundle-optimizer.hxx"
#include "gradient-accumulator.hxx"

namespace opengm {

namespace learning {

template <
		typename DS,
		typename LG,
		typename O = BundleOptimizer<typename DS::ValueType> >
class StructMaxMargin {

public:

	typedef DS DatasetType;
	typedef LG LossGeneratorType;
	typedef O  OptimizerType;

	typedef typename DatasetType::ValueType       ValueType;
    typedef typename DatasetType::Weights         Weights;

	struct Parameter {

		Parameter() :
			regularizerWeight(1.0) {}

		typedef typename OptimizerType::Parameter OptimizerParameter;

		ValueType regularizerWeight;

		OptimizerParameter optimizerParameter;
	};

	StructMaxMargin(DatasetType& dataset, const Parameter& parameter = Parameter()) :
		_dataset(dataset),
		_parameter(parameter) {}

	Parameter& parameter() { return _parameter; }

	template <typename InferenceType>
	void learn(typename InferenceType::Parameter& parameter);

    const Weights& getWeights() { return _weights; }

private:

	template <typename InferenceType>
	class Oracle {

		public:

			Oracle(DatasetType& dataset) :
				_dataset(dataset) {}

			/**
			 * Evaluate the loss-augmented energy value of the dataset and its 
			 * gradient at w.
			 */
            void operator()(const Weights& w, double& value, Weights& gradient) {

				typedef std::vector<typename InferenceType::LabelType> ConfigurationType;

				// initialize gradient with zero

				for (int i = 0; i < _dataset.getNumberOfModels(); i++) {

					// NOT IMPLEMENTED, YET
					//_dataset.lockModel(i);
					//const typename DatasetType::GMWITHLOSS& gm = _dataset.getModelWithLoss(i);
					const typename DatasetType::GMType& gm = _dataset.getModel(i);

					_dataset.getWeights() = w;

					InferenceType inference(gm);

					ConfigurationType configuration;
					inference.infer();
					inference.arg(configuration);

					GradientAccumulator<Weights, ConfigurationType> ga(gradient, configuration);
					for (size_t i = 0; i < gm.numberOfFactors(); i++)
						gm[i].callFunctor(ga);

					// NOT IMPLEMENTED, YET
					//_dataset.unlockModel(i);
				}
			}

		private:

			DatasetType& _dataset;
	};

	DatasetType& _dataset;

	Parameter _parameter;

	OptimizerType _optimizer;

    Weights _weights;
};

template <typename DS, typename LG, typename O>
template <typename InfereneType>
void
StructMaxMargin<DS, LG, O>::learn(typename InfereneType::Parameter& infParams) {

	Oracle<InfereneType> oracle(_dataset);

	_weights = _dataset.getWeights();

	// minimize structured loss
    OptimizerResult result = _optimizer.optimize(oracle, _weights);

	if (result == Error)
		throw opengm::RuntimeError("optimizer did not succeed");

	if (result == ReachedMinGap)
		std::cout << "optimization converged to requested precision" << std::endl;

	if (result == ReachedSteps)
		std::cout << "optimization stopped after " << parameter().optimizerParameter.steps << " iterations" << std::endl;
}

} // namespace learning

} // namespace opengm

#endif // OPENGM_LEARNING_STRUCT_MAX_MARGIN_HXX

