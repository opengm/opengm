#include <vector>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/multicut.hxx>
#include <opengm/inference/external/trws.hxx>



#include <opengm/functions/learnable/lpotts.hxx>
#include <opengm/functions/learnable/lsum_of_experts.hxx>
#include <opengm/learning/subgradient_ssvm.hxx>
#include <opengm/learning/loss/hammingloss.hxx>
//#include <opengm/learning/dataset/testdataset.hxx>
//#include <opengm/learning/dataset/testdataset2.hxx>
#include <opengm/learning/dataset/testdatasets.hxx>
#include <opengm/learning/dataset/editabledataset.hxx>


//*************************************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType; 
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType>, opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType>, opengm::functions::learnable::LSumOfExperts<ValueType,IndexType,LabelType> >::type FunctionListType;
typedef opengm::GraphicalModel<ValueType,opengm::Adder, FunctionListType, opengm::DiscreteSpace<IndexType,LabelType> > GM; 
typedef opengm::learning::HammingLoss     LOSS;

typedef opengm::Multicut<GM,opengm::Minimizer> Multicut;
typedef opengm::LPCplex<GM,opengm::Minimizer> INFCPLEX;
typedef opengm::external::TRWS<GM> INFTRWS;

typedef opengm::datasets::EditableTestDataset<GM,LOSS> EDS;
typedef opengm::datasets::TestDataset1<GM,LOSS> DS1;
typedef opengm::datasets::TestDataset2<GM,LOSS> DS2;
typedef opengm::datasets::TestDatasetSimple<GM,LOSS> DSS;

//*************************************


int main() {
   {
      DSS dataset(5);
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      
      
      opengm::learning::SubgradientSSVM<DSS>::Parameter para;
      para.maxIterations_ = 50;
      para.C_ = 100.0;
      para.learningRate_ = 0.1;
      opengm::learning::SubgradientSSVM<DSS> learner(dataset,para);
      
      
      INFCPLEX::Parameter infPara;
      infPara.integerConstraint_ = true;
      learner.learn<INFCPLEX>(infPara); 
          const DSS::Weights& weights = learner.getWeights();
          std::cout <<"Weights: ";
          for (size_t i=0; i<weights.numberOfWeights(); ++i)
             std::cout << weights[i] <<" ";
          std::cout <<std::endl;
   }

   {
      DS1 dataset(4);
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      
      
      opengm::learning::SubgradientSSVM<DS1>::Parameter para;
      para.maxIterations_ = 10;
      para.C_ = 10.0;
      para.learningRate_ = 0.01;

      opengm::learning::SubgradientSSVM<DS1> learner(dataset,para);
      
      
      INFTRWS::Parameter infPara;
      //infPara.integerConstraint_ = true;
      learner.learn<INFTRWS>(infPara);
      const DS1::Weights& weights = learner.getWeights();
      std::cout <<"Weights: ";
      for (size_t i=0; i<weights.numberOfWeights(); ++i)
         std::cout << weights[i] <<" ";
      std::cout <<std::endl;
      
   }

   {
      DS2 dataset(4);
      std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
      
      
      opengm::learning::SubgradientSSVM<DS2>::Parameter para;
      para.maxIterations_ = 10;
      para.C_ = 10.0;
      para.learningRate_ = 0.01;
      opengm::learning::SubgradientSSVM<DS2> learner(dataset,para);
      
      
      INFTRWS::Parameter infPara;
      //infPara.integerConstraint_ = true;
      learner.learn<INFTRWS>(infPara);
          const DS2::Weights& weights = learner.getWeights();
          std::cout <<"Weights: ";
          for (size_t i=0; i<weights.numberOfWeights(); ++i)
             std::cout << weights[i] <<" ";
          std::cout <<std::endl;
   }

/* ?!?!?
   {
        // create editable dataset
        EDS learningdataset;

        INFTRWS::Parameter infPara;


        std::vector< std::vector< LabelType > >GTSolutionVector;

        std::cout << "inference with fixed, arbitrary weights to generate solution" << std::endl;

        EDS::Weights learningWeightVector = learningdataset.getWeights();
        EDS::Weights randomWeights(learningdataset.getNumberOfWeights());


        // opengm::learning::SubgradientSSVM<EDS>::Parameter para0;
        // para0.optimizerParameter_.lambda = 1;
        // opengm::learning::SubgradientSSVM<EDS> learner0(learningdataset,para0);

        // // // learn
        // learner0.learn<INFTRWS>(infPara);

        // std::srand(std::time(0));
        for (int i = 0; i < learningWeightVector.numberOfWeights(); ++i)
        {
            randomWeights[i] = 1.0;

            std::cout << randomWeights[i] << " --->  "  << learningWeightVector[i] << std::endl;
            learningWeightVector.setWeight(i, randomWeights[i]);//double(std::rand()) / RAND_MAX * 100);
        }

        for (size_t modelIndex = 0; modelIndex < learningdataset.getNumberOfModels(); modelIndex++)
        {

            std::cout << "starting inference on GM " << modelIndex << std::endl;

            // INFTRWS inference(learningdataset.getModel(modelIndex), infPara);
            // inference.infer();
            // std::vector< LabelType > sol1;
            
            // OPENGM_TEST(inference.arg(sol1) == opengm::NORMAL);

            INFTRWS solver(learningdataset.getModel(modelIndex),infPara);
            solver.infer();
            std::vector< LabelType > sol1;
            OPENGM_TEST(solver.arg(sol1) == opengm::NORMAL);


            std::cout << "add solution "<< modelIndex <<" to new dataset" << std::endl;
            learningdataset.setGT(modelIndex,sol1);

            for (size_t j = 0; j < sol1.size(); j++)
            {
              std::cout << sol1[j];
            }
            std::cout << std::endl;
            GTSolutionVector.push_back(sol1);
        }


        std::cout << "learn weights (without regularization)" << std::endl;

        std::cout << "weight vector size " << learningdataset.getNumberOfWeights() << std::endl;
        // Parameter
        opengm::learning::SubgradientSSVM<EDS>::Parameter para;
          para.maxIterations_ = 500;
          para.C_ = 10000.0;
          para.learningRate_ = 0.1;
        opengm::learning::SubgradientSSVM<EDS> learner(learningdataset,para);

        // learn
        learner.learn<INFTRWS>(infPara);

        // get the result
        const EDS::Weights &learnedParameters = learner.getWeights();
        std::cout << learnedParameters.numberOfWeights() << std::endl;
        std::cout << "set learnedParameters as new Weights: ";
        for (size_t i = 0; i < learnedParameters.numberOfWeights(); ++i)
        {
            std::cout << learnedParameters[i] << " ";
            learningWeightVector.setWeight(i, learnedParameters[i]);
        }
        std::cout << std::endl;

        std::cout << "new weights: ";
        for (int i = 0; i < learningWeightVector.numberOfWeights(); i++)
        {
            std::cout << learningWeightVector[i] << ", ";
        }
        std::cout << std::endl;


        std::cout << "inference with new weights" << std::endl;
        for (size_t modelIndex = 0; modelIndex < learningdataset.getNumberOfModels(); modelIndex++)
        {
            std::cout << "starting inference on GM " << modelIndex << " with learned weights" << std::endl;
            INFTRWS solver(learningdataset.getModel(modelIndex),infPara);
            solver.infer();
            std::vector< LabelType > sol2;
            OPENGM_TEST(solver.arg(sol2) == opengm::NORMAL);
            for (size_t j = 0; j < sol2.size(); j++)
            {
               std::cout << "sol2["<<j<<"]:" << sol2[j] << "   GTSolutionVector["<<modelIndex<<"]["<<j<<"]:" << GTSolutionVector[modelIndex][j] << std::endl; 
               OPENGM_TEST(sol2[j] == GTSolutionVector[modelIndex][j]);
            }
            // for (size_t j = 0; j < sol2.size(); j++)
            // {
            //    std::cout << sol2[j]; 
            //    // OPENGM_TEST(sol2[j] == GTSolutionVector[modelIndex][j]);
            // }
            // std::cout << std::endl<< std::endl;
            // for (size_t j = 0; j < sol2.size(); j++)
            // {
            //    std::cout <<  GTSolutionVector[modelIndex][j]; 
            //    // OPENGM_TEST(sol2[j] == GTSolutionVector[modelIndex][j]);
            // }
            std::cout << "all " << sol2.size() << " solutions are correct" << std::endl;
        }

    }
*/
}

