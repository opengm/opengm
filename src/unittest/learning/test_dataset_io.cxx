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
//#include <opengm/learning/dataset/testdataset.hxx>
//#include <opengm/learning/dataset/testdataset2.hxx>
#include <opengm/learning/dataset/dataset_io.hxx>
#include <opengm/learning/dataset/dataset.hxx>
#include <opengm/learning/dataset/testdatasets.hxx>
#include <opengm/learning/loss/noloss.hxx>
#include <opengm/learning/loss/hammingloss.hxx>
#include <opengm/learning/loss/generalized-hammingloss.hxx>


//*************************************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType; 
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType>, opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>, opengm::functions::learnable::LSumOfExperts<ValueType,IndexType,LabelType> >::type FunctionListType;
typedef opengm::GraphicalModel<ValueType,opengm::Adder, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GM; 
typedef opengm::learning::NoLoss                 LOSS1;
typedef opengm::learning::HammingLoss            LOSS2;
typedef opengm::learning::GeneralizedHammingLoss LOSS3;
typedef opengm::datasets::TestDataset1<GM,LOSS1>  DS11;
typedef opengm::datasets::TestDataset2<GM,LOSS1>  DS21;
typedef opengm::datasets::TestDataset1<GM,LOSS2>  DS12;
typedef opengm::datasets::TestDataset2<GM,LOSS2>  DS22;
typedef opengm::datasets::TestDataset1<GM,LOSS3>  DS13;
typedef opengm::datasets::Dataset<GM,LOSS1>       DS1;
typedef opengm::datasets::Dataset<GM,LOSS2>       DS2;
typedef opengm::datasets::Dataset<GM,LOSS3>       DS3;

//*************************************


int main() {
   std::cout << " Includes are fine :-) " << std::endl; 
  
   {
      DS11 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::datasets::DatasetSerialization::save(dataset,"./","dataset11_");
      std::cout <<"done!" <<std::endl;
   }
   {
      DS12 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::datasets::DatasetSerialization::save(dataset,"./","dataset12_");
      std::cout <<"done!" <<std::endl;
   }
   {
      DS21 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::datasets::DatasetSerialization::save(dataset,"./","dataset21_");
      std::cout <<"done!" <<std::endl;
   }
   {
      DS22 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::datasets::DatasetSerialization::save(dataset,"./","dataset22_");
      std::cout <<"done!" <<std::endl;
   }
   {
      DS13 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      opengm::datasets::DatasetSerialization::save(dataset,"./","dataset13_");
      std::cout <<"done!" <<std::endl;
   }

   {
      DS1 ds;
      opengm::datasets::DatasetSerialization::loadAll("./","dataset11_",ds);
   }
   {
      DS1 ds;
      opengm::datasets::DatasetSerialization::loadAll("./","dataset21_",ds);
   }
   {
      DS2 ds;
      opengm::datasets::DatasetSerialization::loadAll("./","dataset12_",ds);
   }
   {
      DS2 ds;
      opengm::datasets::DatasetSerialization::loadAll("./","dataset22_",ds);
   }
   {
      DS3 ds;
      opengm::datasets::DatasetSerialization::loadAll("./","dataset13_",ds);
   }
   std::cout << "test successful." << std::endl;
}
