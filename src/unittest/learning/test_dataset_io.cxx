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
#include <opengm/learning/dataset/testdataset.hxx>
#include <opengm/learning/dataset/testdataset2.hxx>
#include <opengm/learning/dataset/dataset_io.hxx>
#include <opengm/learning/dataset/dataset.hxx>


//*************************************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType; 
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType>, opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>, opengm::functions::learnable::SumOfExperts<ValueType,IndexType,LabelType> >::type FunctionListType;
typedef opengm::GraphicalModel<ValueType,opengm::Adder, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GM; 
typedef opengm::datasets::TestDataset<GM>  DS1;
typedef opengm::datasets::TestDataset2<GM> DS2;
typedef opengm::datasets::Dataset<GM>      DS;

//*************************************


int main() {
   std::cout << " Includes are fine :-) " << std::endl; 
  
   {
      DS1 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfParameters() << " parameters."<<std::endl; 
      opengm::save(dataset,"./","dataset1_");
   }
  
   {
      DS2 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfParameters() << " parameters."<<std::endl; 
      opengm::save(dataset,"./","dataset2_");
   }

   {
      DS ds;
      ds.load("./","dataset2_");
   }

}
