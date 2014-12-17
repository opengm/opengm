#include <vector>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/external/trws.hxx>
#include <opengm/inference/lpgurobi.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include <opengm/functions/learnable/lpotts.hxx>
#include <opengm/functions/learnable/sum_of_experts.hxx>
#include <opengm/learning/struct-max-margin.hxx>
#include <opengm/learning/loss/hammingloss.hxx>
//#include <opengm/learning/dataset/testdataset.hxx>
//#include <opengm/learning/dataset/testdataset2.hxx>
#include <opengm/learning/dataset/testdatasets.hxx>


//*************************************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType; 
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType>, opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>, opengm::functions::learnable::SumOfExperts<ValueType,IndexType,LabelType> >::type FunctionListType;
typedef opengm::GraphicalModel<ValueType,opengm::Adder, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GM; 
typedef opengm::learning::HammingLoss     LOSS;
typedef opengm::LPGurobi<GM,opengm::Minimizer> INF;
typedef opengm::datasets::TestDataset1<GM,LOSS> DS1;
typedef opengm::datasets::TestDataset2<GM,LOSS> DS2;
typedef opengm::datasets::TestDatasetSimple<GM,LOSS> DSS;

//*************************************


int main() {
   std::cout << " Includes are fine :-) " << std::endl; 

   {
	  DSS dataset(1);
	  std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
	  
	  
	  opengm::learning::StructMaxMargin<DSS,LOSS>::Parameter para;
	  opengm::learning::StructMaxMargin<DSS,LOSS> learner(dataset,para);
	  
	  
	  INF::Parameter infPara;
	  infPara.integerConstraint_ = true;
	  learner.learn<INF>(infPara);
   }

   {
	  DS1 dataset(1);
	  std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
	  
	  
	  opengm::learning::StructMaxMargin<DS1,LOSS>::Parameter para;
	  opengm::learning::StructMaxMargin<DS1,LOSS> learner(dataset,para);
	  
	  
	  INF::Parameter infPara;
	  infPara.integerConstraint_ = true;
	  learner.learn<INF>(infPara);
	  
   }

   //{
	  //DS2 dataset(1);
	  //std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
	  
	  
	  //opengm::learning::StructMaxMargin<DS2,LOSS>::Parameter para;
	  //opengm::learning::StructMaxMargin<DS2,LOSS> learner(dataset,para);
	  
	  
	  //INF::Parameter infPara;
	  //infPara.integerConstraint_ = true;
	  //learner.learn<INF>(infPara);
   //}


}

