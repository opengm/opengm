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
#include <opengm/learning/struct-max-margin.hxx>
#include <opengm/learning/loss/hammingloss.hxx>
#include <opengm/learning/dataset/testdataset.hxx>
#include <opengm/learning/dataset/testdataset2.hxx>


//*************************************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType; 
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType>, opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>, opengm::functions::learnable::SumOfExperts<ValueType,IndexType,LabelType> >::type FunctionListType;
typedef opengm::GraphicalModel<ValueType,opengm::Adder, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GM; 
typedef opengm::datasets::TestDataset<GM>  DS;
typedef opengm::datasets::TestDataset2<GM> DS2;
typedef opengm::learning::HammingLoss     LOSS;
typedef opengm::ICM<GM,opengm::Minimizer> INF;

//*************************************

template<class T>
struct LearningTest{

    typedef T                                                                  ValueType;
    typedef OPENGM_TYPELIST_2(
        opengm::ExplicitFunction<T>,
        opengm::functions::learnable::LPotts<T>)                               FunctionTypeList;
    typedef opengm::GraphicalModel<ValueType, opengm::Adder, FunctionTypeList> GraphicalModelType;
    typedef opengm::datasets::TestDataset<GraphicalModelType>                  DatasetType;
    typedef typename DatasetType::Weights                                      Weights;
    typedef opengm::learning::HammingLoss                                      LossGeneratorType;
    typedef opengm::Bruteforce<GraphicalModelType, opengm::Minimizer>          InferenceType;

    void testStructMaxMargin()
    {

        // create a dataset
        DatasetType dataset;

        // create a learning algorithm
        opengm::learning::StructMaxMargin<DatasetType, LossGeneratorType> structMaxMargin(dataset);

        // train
        typename InferenceType::Parameter infParams;
        structMaxMargin.template learn<InferenceType>(infParams);

        // get the result
        const Weights &learntParameters = structMaxMargin.getWeights();
        std::cout << learntParameters.numberOfWeights() << std::endl;
        for (size_t i = 0; i < learntParameters.numberOfWeights(); ++i)
            std::cout << learntParameters[i] << " ";
        std::cout << std::endl;
    }


    void testStructMaxMargin_prediction()
    {

        // create a dataset
        DatasetType dataset;

        std::vector< std::vector<size_t> >GTSolutionVector;

        std::cout << "inference with fixed, arbitrary weights to generate solution" << std::endl;

        Weights weightVector = dataset.getWeights();
        // std::srand(std::time(0));

        for (int i = 0; i < weightVector.numberOfWeights(); i++)
        {
            weightVector.setWeight(i, double(std::rand()) / RAND_MAX * 100);
            std::cout << weightVector[i] << std::endl;
        }

        for (size_t modelIndex = 0; modelIndex < dataset.getNumberOfModels(); modelIndex++)
        {

            std::cout << "starting inference on GM " << modelIndex << std::endl;
            InferenceType solver(dataset.getModel(modelIndex));
            solver.infer();
            std::vector<size_t> sol1;
            OPENGM_TEST(solver.arg(sol1) == opengm::NORMAL);
            GTSolutionVector.push_back(sol1);
            std::cout << "add solution to GM " << modelIndex << std::endl;
            for (size_t j = 0; j < sol1.size(); j++)
            {
                // TODO: find way to set GT weights
                // dataset.getGT(modelIndex)[j] = sol1[j]; does not work
            }
        }

        std::cout << "learn weights (without regularization)" << std::endl;
        // create a learning algorithm
        opengm::learning::StructMaxMargin<DatasetType, LossGeneratorType> structMaxMargin(dataset);
        // train
        typename InferenceType::Parameter infParams;
        structMaxMargin.template learn<InferenceType>(infParams);

        // get the result
        const Weights &learntParameters = structMaxMargin.getWeights();
        std::cout << learntParameters.numberOfWeights() << std::endl;
        std::cout << "learntParameters: ";
        for (size_t i = 0; i < learntParameters.numberOfWeights(); ++i)
        {
            std::cout << learntParameters[i] << " ";
            weightVector.setWeight(i, learntParameters[i]);
        }
        std::cout << std::endl;

        std::cout << "inference with new weights" << std::endl;
        for (size_t modelIndex = 0; modelIndex < dataset.getNumberOfModels(); modelIndex++)
        {
            std::cout << "starting inference on GM " << modelIndex << "with learned weights" << std::endl;
            InferenceType solver(dataset.getModel(modelIndex));
            solver.infer();
            std::vector<size_t> sol2;
            OPENGM_TEST(solver.arg(sol2) == opengm::NORMAL);
            for (size_t j = 0; j < sol2.size(); j++)
            {
                OPENGM_TEST(sol2[j] == GTSolutionVector[modelIndex][j]);
            }
        }
    }

    void run()
    {
        this->testStructMaxMargin();
        this->testStructMaxMargin_prediction();
    }
};

int main() {
   std::cout << " Includes are fine :-) " << std::endl; 

    //  {
    //  LearningTest<double >t;
    //  t.run();
    // }

    {
      DS dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      
      
      opengm::learning::StructMaxMargin<DS,LOSS> learner(dataset);
      
      
      INF::Parameter infPara;
      learner.learn<INF>(infPara);
   }

   {
      DS2 dataset;
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      
      
      opengm::learning::StructMaxMargin<DS2,LOSS> learner(dataset);
      
      
      INF::Parameter infPara;
      learner.learn<INF>(infPara);
   }
}