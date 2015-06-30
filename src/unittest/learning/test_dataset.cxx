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


//*************************************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType; 
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType>, opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>, opengm::functions::learnable::LSumOfExperts<ValueType,IndexType,LabelType> >::type FunctionListType;
typedef opengm::GraphicalModel<ValueType,opengm::Adder, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GM; 
//typedef opengm::datasets::TestDataset<GM>  DS1;
//typedef opengm::datasets::Dataset<GM>      DS;

typedef opengm::learning::HammingLoss     LOSS;
//typedef opengm::learning::NoLoss                 LOSS;
typedef opengm::datasets::TestDataset1<GM,LOSS>  DS0;
typedef opengm::datasets::TestDataset1<GM,LOSS>  DS1;
typedef opengm::datasets::TestDataset2<GM,LOSS>  DS2;
typedef opengm::datasets::Dataset<GM,LOSS>       DS;

//*************************************

template<class DatasetType>
struct DatasetTest {

   DatasetType& dataset_;

   DatasetTest(DatasetType& data): dataset_(data) {}

   void testInitialization() {
      std::cout << "Initialize Model:" << std::endl;
      // create a new dataset
      DatasetType dataset;
   }

   void callModelFunctions(){

         std::cout << "calling Model functions:" << std::endl;
         std::cout << "\tlocking all available Models" << std::endl;

         for(int i; i<dataset_.getNumberOfModels();i++)
         {
            dataset_.lockModel(i);
            dataset_.unlockModel(i);
         }

         std::cout << "\tgetModel with and without loss" << std::endl;
         for(int i; i<dataset_.getNumberOfModels();i++)
         {
            OPENGM_TEST(dataset_.getModel(i).numberOfVariables() == dataset_.getModelWithLoss(i).numberOfVariables());
            OPENGM_TEST(dataset_.getModel(i).numberOfFactors() <=  dataset_.getModelWithLoss(i).numberOfFactors());
         }

         std::cout << "\tgetGT" << std::endl;
         for(int i; i<dataset_.getNumberOfModels();i++)
         {
            std::cout << dataset_.getGT(i).size() << std::endl;
         }

      }

   void getInfo(){
         std::cout << "Info of data size:" << std::endl;
         std::cout << "\tnumberOfWeights\t" << dataset_.getNumberOfWeights() << std::endl;
         std::cout << "\tnumberOfModels\t" << dataset_.getNumberOfModels() << std::endl;

         opengm::learning::Weights<ValueType> weights = dataset_.getWeights();
         std::cout << "Beginning of weight vector: ";
         for(int i; i<std::min(dataset_.getNumberOfWeights(),size_t(10));i++)
         {
            std::cout << dataset_.getWeights()[i] << " ";
         }
         std::cout << std::endl;
   }

   void run() {
      this->testInitialization();
      this->getInfo();
      this->callModelFunctions();
   }
};


int main() {
   std::cout << " Includes are fine :-) " << std::endl; 
  
   {
      // initialize your data here
      // eventually you need to load it from file
      DS data;
  
      std::cout << "Start test DS" <<std::endl;
      //run tests on dataset
      DatasetTest<DS >t(data);
      t.run();
   } 

   {
      // initialize your data here
      // eventually you need to load it from file
      DS0 data;

      std::cout << "Start test DS0" <<std::endl;
      //run tests on dataset
      DatasetTest<DS0 >t(data);
      t.run();

   }
   {
      // initialize your data here
      // eventually you need to load it from file
      DS1 data;

      std::cout << "Start test DS1" <<std::endl;
      //run tests on dataset
      DatasetTest<DS1 >t(data);
      t.run();

   }
   {
      // initialize your data here
      // eventually you need to load it from file
      DS2 data;

      std::cout << "Start test DS2" <<std::endl;
      //run tests on dataset
      DatasetTest<DS2 >t(data);
      t.run();

   }


}
