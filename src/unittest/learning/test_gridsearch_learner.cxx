#include <vector>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include <opengm/functions/learnable/lpotts.hxx>
#include <opengm/functions/learnable/lsum_of_experts.hxx>
#include <opengm/learning/gridsearch-learning.hxx>
#include <opengm/learning/loss/hammingloss.hxx>
//#include <opengm/learning/dataset/testdataset.hxx>
//#include <opengm/learning/dataset/testdataset2.hxx>
#include <opengm/learning/dataset/testdatasets.hxx>


//*************************************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType; 
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType>, opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>, opengm::functions::learnable::LSumOfExperts<ValueType,IndexType,LabelType> >::type FunctionListType;
typedef opengm::GraphicalModel<ValueType,opengm::Adder, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GM; 
//typedef opengm::datasets::TestDataset<GM>  DS;
//typedef opengm::datasets::TestDataset2<GM> DS2;
typedef opengm::learning::HammingLoss     LOSS;
typedef opengm::ICM<GM,opengm::Minimizer> INF;
typedef opengm::datasets::TestDataset1<GM,LOSS> DS1;
typedef opengm::datasets::TestDataset2<GM,LOSS> DS2;

//*************************************


int main() {
   std::cout << " Includes are fine :-) " << std::endl; 
   /* 
   {
      DS dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      
      
      opengm::learning::GridSearchLearner<DS,LOSS>::Parameter para;
      para.parameterUpperbound_.resize(1,1);
      para.parameterLowerbound_.resize(1,0);
      para.testingPoints_.resize(1,10);
      opengm::learning::GridSearchLearner<DS,LOSS> learner(dataset,para);
      
      
      INF::Parameter infPara;
      learner.learn<INF>(infPara);
      
   } 
   */
   {
      DS1 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      
      
      opengm::learning::GridSearchLearner<DS1>::Parameter para;
      para.parameterUpperbound_.resize(1,1);
      para.parameterLowerbound_.resize(1,0);
      para.testingPoints_.resize(1,10);
      opengm::learning::GridSearchLearner<DS1> learner(dataset,para);
      
      
      INF::Parameter infPara;
      learner.learn<INF>(infPara);
      
   }

   {
      DS2 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      
      
      opengm::learning::GridSearchLearner<DS2>::Parameter para;
      para.parameterUpperbound_.resize(3,1);
      para.parameterLowerbound_.resize(3,0);
      para.testingPoints_.resize(3,5);
      opengm::learning::GridSearchLearner<DS2> learner(dataset,para);
      
      
      INF::Parameter infPara;
      learner.learn<INF>(infPara);
   }


}
