#pragma once
#ifndef OPENGM_LEARNING_STRUCT_MAX_MARGIN_HXX
#define OPENGM_LEARNING_STRUCT_MAX_MARGIN_HXX

// uncomment when dataset is done
//#include "dataset.hxx"
#include "bundle-optimizer.hxx"

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
	typedef typename DatasetType::ModelParameters ModelParameters;

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

	const ModelParameters& getModelParameters() { return _learntParameters; }

private:

	DatasetType& _dataset;

	Parameter _parameter;

	OptimizerType _optimizer;

	ModelParameters _learntParameters;
};

template <typename DS, typename LG, typename O>
template <typename InfereneType>
void
StructMaxMargin<DS, LG, O>::learn(typename InfereneType::Parameter& infParams) {

	// create a loss-augmented copy of the dataset
	DS augmentedDataset = _dataset;
	LossGeneratorType loss;
	for (unsigned int i = 0; i < augmentedDataset.getNumberOfModels(); i++)
		loss.addLoss(augmentedDataset.getModel(i), augmentedDataset.getGT(i).begin());

	// minimize structured loss
	OptimizerResult result = _optimizer.optimize(augmentedDataset, _learntParameters);

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

