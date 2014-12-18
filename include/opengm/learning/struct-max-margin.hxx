#pragma once
#ifndef OPENGM_LEARNING_STRUCT_MAX_MARGIN_HXX
#define OPENGM_LEARNING_STRUCT_MAX_MARGIN_HXX

#include "bundle-optimizer.hxx"
#include "gradient-accumulator.hxx"

namespace opengm {

namespace learning {

template <
		typename DS,
		typename O = BundleOptimizer<typename DS::ValueType> >
class StructMaxMargin {

public:

	typedef DS DatasetType;
	typedef O  OptimizerType;

	typedef typename DatasetType::ValueType       ValueType;
    typedef typename DatasetType::Weights         Weights;

	struct Parameter {
        typedef typename OptimizerType::Parameter OptimizerParameter;
        OptimizerParameter optimizerParameter_;
	};

	StructMaxMargin(DatasetType& dataset, const Parameter& parameter = Parameter()) :
		_dataset(dataset),
        _parameter(parameter),
        _optimizer(parameter.optimizerParameter_)
    {}

	Parameter& parameter() { return _parameter; }

	template <typename InferenceType>
	void learn(typename InferenceType::Parameter& parameter);

    const Weights& getWeights() { return _weights; }

private:

	template <typename InferenceType>
	class Oracle {

		public:

            Oracle(DatasetType& dataset, typename InferenceType::Parameter& infParam) :
                _dataset(dataset),
                _infParam(infParam)
            {}

			/**
			 * Evaluate the loss-augmented energy value of the dataset and its 
			 * gradient at w.
			 */
            void operator()(const Weights& w, double& value, Weights& gradient) {

				typedef std::vector<typename InferenceType::LabelType> ConfigurationType;

				// initialize gradient and value with zero
				for (int i = 0; i < gradient.numberOfWeights(); i++)
					gradient[i] = 0;
				value = 0;

				// For each model E(y,w), we have to compute the value and 
				// gradient of
				//
				//   max_y E(y',w) - E(y,w) + Δ(y',y)            (1)
				//   =
				//   max_y L(y,w)
				//
				// where y' is the best-effort solution (also known as 
				// groundtruth) and w are the current weights. The loss 
				// augmented model given by the dataset is
				//
				//   F(y,w) = E(y,w) - Δ(y',y).
				//
				// Let c = E(y',w) be the constant contribution of the 
				// best-effort solution. (1) is equal to
				//
				//  -min_y -c + F(y,w).
				//
				// The gradient of the maximand in (1) at y* is
				//
				//   ∂L(y,w)/∂w = ∂E(y',w)/∂w -
				//                ∂E(y,w)/∂w
				//
				//              = Σ_θ ∂θ(y'_θ,w)/∂w -
				//                Σ_θ ∂θ(y_θ,w)/∂w,
				//
				// which is a positive gradient contribution for the 
				// best-effort, and a negative contribution for the maximizer 
				// y*.

				for (int i = 0; i < _dataset.getNumberOfModels(); i++) {

					// get E(x,y) and F(x,y)
					_dataset.lockModel(i);
					const typename DatasetType::GMType&     gm  = _dataset.getModel(i);
					const typename DatasetType::GMWITHLOSS& gml = _dataset.getModelWithLoss(i);

					// set the weights w in E(x,y) and F(x,y)
					_dataset.getWeights() = w;

					// get the best-effort solution y'
					const ConfigurationType& bestEffort = _dataset.getGT(i);

					// compute constant c for current w
					ValueType c = gm.evaluate(bestEffort);

					// find the minimizer y* of F(y,w)
					ConfigurationType mostViolated;
                    InferenceType inference(gml, _infParam);
					inference.infer();
					inference.arg(mostViolated);

					// the optimal value of (1) is now c - F(y*,w)
					value += c - gml.evaluate(mostViolated);

					// the gradients are
					typedef GradientAccumulator<Weights, ConfigurationType> GA;
					GA gaBestEffort(gradient, bestEffort, GA::Add);
					GA gaMostViolated(gradient, mostViolated, GA::Subtract);
					for (size_t j = 0; j < gm.numberOfFactors(); j++) {

						gm[j].callViFunctor(gaBestEffort);
						gm[j].callViFunctor(gaMostViolated);
					}

					_dataset.unlockModel(i);
				}
			}

		private:

			DatasetType& _dataset;
            typename InferenceType::Parameter& _infParam;
	};

	DatasetType& _dataset;

	Parameter _parameter;

	OptimizerType _optimizer;

    Weights _weights;
};

template <typename DS, typename O>
template <typename InfereneType>
void
StructMaxMargin<DS, O>::learn(typename InfereneType::Parameter& infParams) {

    Oracle<InfereneType> oracle(_dataset, infParams);

	_weights = _dataset.getWeights();

	// minimize structured loss
    OptimizerResult result = _optimizer.optimize(oracle, _weights);

	if (result == Error)
		throw opengm::RuntimeError("optimizer did not succeed");

	if (result == ReachedMinGap)
		std::cout << "optimization converged to requested precision" << std::endl;

	if (result == ReachedSteps)
        std::cout << "optimization stopped after " << parameter().optimizerParameter_.steps << " iterations" << std::endl;
}

} // namespace learning

} // namespace opengm

#endif // OPENGM_LEARNING_STRUCT_MAX_MARGIN_HXX

