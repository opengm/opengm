#include <vector>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include <opengm/functions/learnable/lpotts.hxx>
#include <opengm/functions/learnable/sum_of_experts.hxx>
#include <opengm/learning/maximum-likelihood-learning.hxx>
#include <opengm/learning/loss/hammingloss.hxx>
#include <opengm/learning/dataset/testdatasets.hxx>


//*************************************

typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType; 
typedef opengm::meta::TypeListGenerator<
    opengm::ExplicitFunction<ValueType,IndexType,LabelType>,
    opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>,
    opengm::functions::learnable::SumOfExperts<ValueType,IndexType,LabelType>
>::type FunctionListType;

typedef opengm::GraphicalModel<
    ValueType,opengm::Adder,
    FunctionListType,
    opengm::DiscreteSpace<IndexType,LabelType>
> GM;

typedef opengm::learning::HammingLoss     LOSS;
typedef opengm::datasets::TestDataset0<GM,LOSS> DS0;
typedef opengm::datasets::TestDataset1<GM,LOSS> DS1;
typedef opengm::datasets::TestDataset2<GM,LOSS> DS2;
typedef opengm::datasets::TestDatasetSimple<GM,LOSS> DSSimple;
//typedef opengm::ICM<GM,opengm::Minimizer> INF;

//*************************************


int main() {
   std::cout << " Includes are fine :-) " << std::endl; 
   /*
   {
      DS0 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::learning::MaximumLikelihoodLearner<DS0,LOSS>::Weight weight;
      opengm::learning::MaximumLikelihoodLearner<DS0,LOSS> learner(dataset,weight);
      INF::Parameter infWeight;
      learner.learn<INF>(infWeight);

   }
*/

   {
      DS1 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::learning::MaximumLikelihoodLearner<DS1,LOSS>::Weight weight;
      opengm::learning::MaximumLikelihoodLearner<DS1,LOSS> learner(dataset,weight);
      INF::Parameter infWeight;
      learner.learn<INF>(infWeight);
      
   }
/*

   {
      DS2 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::learning::MaximumLikelihoodLearner<DS2,LOSS>::Weight weight;
      opengm::learning::MaximumLikelihoodLearner<DS2,LOSS> learner(dataset,weight);
      INF::Parameter infWeight;
      learner.learn<INF>(infWeight);
   }

/*
   {
      DSSimple dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::learning::MaximumLikelihoodLearner<DSSimple,LOSS>::Weight weight;
      opengm::learning::MaximumLikelihoodLearner<DSSimple,LOSS> learner(dataset,weight);
      INF::Parameter infWeight;
      learner.learn<INF>(infWeight);
   }
*/
}
