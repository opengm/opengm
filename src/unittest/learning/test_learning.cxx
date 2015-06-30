#include <vector>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/external/trws.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#ifdef WITH_GUROBI
#include <opengm/inference/lpgurobi.hxx>
#else
#include <opengm/inference/lpcplex.hxx>
#endif


#include <opengm/functions/learnable/lpotts.hxx>
#include <opengm/functions/learnable/lsum_of_experts.hxx>
#include <opengm/learning/struct-max-margin.hxx>
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

#ifdef WITH_GUROBI
typedef opengm::LPGurobi<GM,opengm::Minimizer> INF;
#else
typedef opengm::LPCplex<GM,opengm::Minimizer> INF;
#endif
typedef opengm::datasets::EditableTestDataset<GM,LOSS> EDS;
typedef opengm::datasets::TestDataset1<GM,LOSS> DS1;
typedef opengm::datasets::TestDataset2<GM,LOSS> DS2;
typedef opengm::datasets::TestDatasetSimple<GM,LOSS> DSS;

//*************************************


int main() {
   std::cout << " Includes are fine :-) " << std::endl; 

   {
	  DSS dataset(5);
	  std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
	  
	  
	  opengm::learning::StructMaxMargin<DSS>::Parameter para;
	  opengm::learning::StructMaxMargin<DSS> learner(dataset,para);
	  
	  
	  INF::Parameter infPara;
	  infPara.integerConstraint_ = true;
	  learner.learn<INF>(infPara); 
		  const DSS::Weights& weights = learner.getWeights();
		  std::cout <<"Weights: ";
		  for (size_t i=0; i<weights.numberOfWeights(); ++i)
			 std::cout << weights[i] <<" ";
		  std::cout <<std::endl;
   }

   {
	  DS1 dataset(4);
	  std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
	  
	  
      opengm::learning::StructMaxMargin<DS1>::Parameter para;
      opengm::learning::StructMaxMargin<DS1> learner(dataset,para);
	  
	  
	  INF::Parameter infPara;
	  infPara.integerConstraint_ = true;
	  learner.learn<INF>(infPara);
          const DS1::Weights& weights = learner.getWeights();
          std::cout <<"Weights: ";
          for (size_t i=0; i<weights.numberOfWeights(); ++i)
             std::cout << weights[i] <<" ";
          std::cout <<std::endl;
	  
   }

   {
	  DS2 dataset(4);
	  std::cout << "Dataset includes " << dataset.getNumberOfModels() << " instances and has " << dataset.getNumberOfWeights() << " parameters."<<std::endl;
	  
	  
	  opengm::learning::StructMaxMargin<DS2>::Parameter para;
	  para.optimizerParameter_.lambda = 1000.0;
	  opengm::learning::StructMaxMargin<DS2> learner(dataset,para);
	  
	  
	  INF::Parameter infPara;
	  infPara.integerConstraint_ = true;
	  learner.learn<INF>(infPara);
		  const DS2::Weights& weights = learner.getWeights();
		  std::cout <<"Weights: ";
		  for (size_t i=0; i<weights.numberOfWeights(); ++i)
			 std::cout << weights[i] <<" ";
		  std::cout <<std::endl;
   }

/* Does this make sence?!?
   {
        // create editable dataset
        EDS learningdataset;

        INF::Parameter infPara;
        infPara.integerConstraint_ = true;

        std::vector< std::vector< LabelType > >GTSolutionVector;

        std::cout << "inference with fixed, arbitrary weights to generate solution" << std::endl;

        EDS::Weights learningWeightVector = learningdataset.getWeights();
        EDS::Weights randomWeights(learningdataset.getNumberOfWeights());


        // opengm::learning::StructMaxMargin<EDS>::Parameter para0;
        // para0.optimizerParameter_.lambda = 1;
        // opengm::learning::StructMaxMargin<EDS> learner0(learningdataset,para0);

        // // // learn
        // learner0.learn<INF>(infPara);

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

            // INF inference(learningdataset.getModel(modelIndex), infPara);
            // inference.infer();
            // std::vector< LabelType > sol1;
            
            // OPENGM_TEST(inference.arg(sol1) == opengm::NORMAL);

            INF solver(learningdataset.getModel(modelIndex),infPara);
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
        opengm::learning::StructMaxMargin<EDS>::Parameter para;
        para.optimizerParameter_.lambda = 0.000000001;
        opengm::learning::StructMaxMargin<EDS> learner(learningdataset,para);

        // learn
        learner.learn<INF>(infPara);

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
            INF solver(learningdataset.getModel(modelIndex),infPara);
            solver.infer();
            std::vector< LabelType > sol2;
            OPENGM_TEST(solver.arg(sol2) == opengm::NORMAL);
            //for (size_t j = 0; j < sol2.size(); j++)
            //{
            //std::cout << "sol2["<<j<<"]:" << sol2[j] << "   GTSolutionVector["<<modelIndex<<"]["<<j<<"]:" << GTSolutionVector[modelIndex][j] << std::endl; 
            //  //!may not be true! OPENGM_TEST(sol2[j] == GTSolutionVector[modelIndex][j]);
            //}
            OPENGM_TEST( learningdataset.getModel(modelIndex).evaluate(sol2) ==  learningdataset.getModel(modelIndex).evaluate(GTSolutionVector[modelIndex]) );
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

