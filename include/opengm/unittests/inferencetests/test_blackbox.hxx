#pragma once
#ifndef OPENGM_TEST_TEST_BLACKBOX_HXX
#define OPENGM_TEST_TEST_BLACKBOX_HXX


#include <opengm/graphicalmodel/modelgenerators/syntheticmodelgenerator.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/minimizer.hxx>
#include <cmath>
/// \cond HIDDEN_SYMBOLS
namespace opengm {
   namespace test{
      enum BlackBoxFunction
      {
         RANDOM, POS_POTTS, NEG_POTTS, MIX_POTTS
      };

      enum BlackBoxModel
      {
         GRID, FULL, STAR
      };

      /// \brief TestBlackBox<INF> 
      /// Generate models from a spesific class and check if the algorithm INF show  
      /// the correct behaviour.
      /// This test is an easy way to get code coverage. 
      template <class INF>
      class TestBlackBox : public TestBase<INF>
      {
      public:
         typedef typename INF::GraphicalModelType GraphicalModelType;
      
         //const size_t gridHeight_;
         //const size_t gridWidth_;
         const size_t numberOfVariables_;
         const int    numberOfStates_;
         const size_t potentialFlag_;
         
         const BlackBoxModel    bbModel_;
         const BlackBoxFunction bbFunction_;
         const TestBehaviour    behaviour_;
         std::string            testName_;
      
         const size_t modelIdOffset_;
         const size_t numberOfTests_; 

         TestBlackBox(); 
         //TestBlackBox(const BlackBoxModel, const size_t, const size_t, const int, const size_t, 
         //             const BlackBoxFunction, const TestBehaviour, const std::string,
         //             const size_t, const size_t modelIdOffset=0);
         TestBlackBox(const BlackBoxModel, const size_t, const int, const size_t, 
                      const BlackBoxFunction, const TestBehaviour, const std::string,
                      const size_t, const size_t modelIdOffset=0);
         virtual void test(typename INF::Parameter);
         
      private:
         GraphicalModelType getModel(size_t);
      };
      
      // ****************
      // Implementation
      // ****************  


      /// \brief TestBlackBox Constructor
      /// \param bbModel Black-Box-ModelType [GRID, FULL, STAR]
      /// \param numberOfVariables number of variables of the model
      /// \param numberOfLabels    number of states for all variables - a negative number implies arbitrary statespace size between 1 and |numberOfLabels| for each variable independently.
      /// \param potentialFlag     orders that should be included: 3=1st and 2nd, 4= only 3rd, 7=1st, 2nd and 3rd
      /// \param bbFunction        function type of higher order factors [RANDOM, POS_POTTS, NEG_POTTS, MIX_POTTS]
      /// \param behaviour         behaviour the algorithm should show
      /// \param infoText          text that should be printed for this test
      /// \param numberOfTests     number of models that should be generated and tested
      /// \param modelIdOffset     offset for the model-id
      template <class INF>
      TestBlackBox<INF>::TestBlackBox(
         const BlackBoxModel bbModel,
         const size_t numberOfVariables, 
         const int numberOfLabels, 
         const size_t potentialFlag, 
         const BlackBoxFunction bbFunction, 
         const TestBehaviour behaviour, 
         const std::string infoText,
         const size_t numberOfTests,
         const size_t modelIdOffset)
         : bbModel_(bbModel), numberOfVariables_(numberOfVariables),
           numberOfStates_(numberOfLabels), potentialFlag_(potentialFlag),
           bbFunction_(bbFunction), behaviour_(behaviour), testName_(infoText), 
           numberOfTests_(numberOfTests), modelIdOffset_(modelIdOffset)
      {
         std::string s = numberOfStates_<0 ? "~" : "";
         std::string funcString;
         std::string modelString; 
         std::string orderString;

         switch (bbFunction_) {
         case RANDOM:      funcString = "random"; break;
         case POS_POTTS:   funcString = "potts"; break;
         case NEG_POTTS:   funcString = "negative potts"; break;
         case MIX_POTTS:   funcString = "general potts"; break;
         default:          funcString = "unknown";
         };
         switch(bbModel_) {
         case GRID:        modelString = "grid"; break; 
         case FULL:        modelString = "full"; break; 
         case STAR:        modelString = "star"; break; 
         default:          modelString = "unknown";
         }    


         size_t temp  = 1024*1024;
         size_t o     = potentialFlag;
         bool   print = false;
         while(temp>0) {
            if(o>=temp) {
               o -= temp;
               orderString += "1";
               print = true;
            } 
            else{
               if(print)  orderString += "0";
            }
            temp /= 2;
         }
         

         std::stringstream oss;
         oss << "Blackbox-Test: " << modelString << " model (order=" << orderString << ")";
         oss << " with " << numberOfVariables_ << " variables and " << s << abs(numberOfStates_) << " states ";
         oss << "(functionType = " << funcString << ")";
         testName_ = oss.str() + testName_; 
      }
      
      template <class INF>
      typename INF::GraphicalModelType TestBlackBox<INF>::getModel(size_t id)
      {
         typedef opengm::SyntheticModelGenerator2<GraphicalModelType> ModelGeneratorType;
         typename ModelGeneratorType::Parameter modelGeneratorPara;
       
         ModelGeneratorType modelGenerator;
         size_t orderFlag = potentialFlag_;

   
         if(potentialFlag_ && 1) {
            modelGeneratorPara.functionTypes_[0] = ModelGeneratorType::URANDOM;
            modelGeneratorPara.functionParameters_[0].resize(2);
            modelGeneratorPara.functionParameters_[0][0] = 1;
            modelGeneratorPara.functionParameters_[0][1] = 2;
            orderFlag -= 1;
         }else{
            modelGeneratorPara.functionTypes_[0] = ModelGeneratorType::EMPTY;
         }
         
         size_t t = 2;
         size_t c = 1;
         while(orderFlag !=0) {
            modelGeneratorPara.functionTypes_[c] = ModelGeneratorType::EMPTY;
            if(orderFlag && t) {
               orderFlag -= t;
               switch (bbFunction_) {
               case RANDOM:
                  modelGeneratorPara.functionTypes_[c] = ModelGeneratorType::URANDOM;
                  modelGeneratorPara.functionParameters_[c].resize(2);
                  modelGeneratorPara.functionParameters_[c][0] = 1;
                  modelGeneratorPara.functionParameters_[c][1] = 2;
                  break;
               case POS_POTTS:
                  modelGeneratorPara.functionTypes_[c] = ModelGeneratorType::GPOTTS;
                  modelGeneratorPara.functionParameters_[c].resize(1);
                  modelGeneratorPara.functionParameters_[c][0] = 3;
                  modelGeneratorPara.functionParameters_[c][0] = 3;
                  break;
               case NEG_POTTS:
                  modelGeneratorPara.functionTypes_[c] = ModelGeneratorType::GPOTTS;
                  modelGeneratorPara.functionParameters_[c].resize(1);
                  modelGeneratorPara.functionParameters_[c][0] = -3;
                  modelGeneratorPara.functionParameters_[c][0] = -3;
                  break; 
               case MIX_POTTS:
                  modelGeneratorPara.functionTypes_[c] = ModelGeneratorType::GPOTTS;
                  modelGeneratorPara.functionParameters_[c].resize(1);
                  modelGeneratorPara.functionParameters_[c][0] = -3;
                  modelGeneratorPara.functionParameters_[c][0] =  3;
                  break;
               }
            }
            ++c;
            t *= 2;
         }

         if(numberOfStates_ < 0)
            modelGeneratorPara.randomNumberOfStates_ = true;
         else
            modelGeneratorPara.randomNumberOfStates_ = false;
         

         switch(bbModel_) {
         case GRID:
         {
            size_t gridSize = (size_t)(std::sqrt((double)(numberOfVariables_))+0.001);
            OPENGM_ASSERT(numberOfVariables_ == gridSize*gridSize);
            return modelGenerator.buildGrid(id+modelIdOffset_, gridSize, gridSize, abs(numberOfStates_), modelGeneratorPara);
         }
         case FULL:
            return modelGenerator.buildFull(id+modelIdOffset_, numberOfVariables_, abs(numberOfStates_), modelGeneratorPara);
         case STAR:
            return modelGenerator.buildStar(id+modelIdOffset_, numberOfVariables_, abs(numberOfStates_), modelGeneratorPara);
         }
      }


      /// \brief test<INF> start test with algorithm INF
      /// \param para parameters of algorithm
      template <class INF>
      void TestBlackBox<INF>::test(typename INF::Parameter para) {
         typedef typename INF::AccumulationType AccType;
         typedef typename INF::GraphicalModelType           GraphicalModelType;
         typedef typename GraphicalModelType::ValueType     ValueType;

         if(numberOfTests_==0) {
            std::cout << "  - " << testName_ << " ... is empty!" << std::endl;
            return;
         } 
         std::cout << "  - " << testName_ << " ...   " << std::flush;
         for(size_t m=0; m<numberOfTests_; ++m) {
            std::cout << "*" << std::flush;
            GraphicalModelType gm = getModel(m);

            // Run Algorithm
            bool exceptionFlag = false; 
            try{
               INF inf(gm, para); 
               // ACC = MAX | MIN
               OPENGM_TEST(inf.infer()==opengm::NORMAL);
               ValueType value = inf.value();
               ValueType bound = inf.bound();
               OPENGM_TEST(AccType::bop(bound,value));
               if(typeid(AccType) == typeid(opengm::Minimizer) || typeid(AccType) == typeid(opengm::Maximizer))
               { 
                  std::vector<size_t> state;
                  OPENGM_TEST(inf.arg(state)==opengm::NORMAL);
                  OPENGM_TEST(state.size()==gm.numberOfVariables());
                  for(size_t varId = 0; varId < gm.numberOfVariables(); ++varId) {
                     OPENGM_TEST(state[varId]<gm.numberOfLabels(varId));
                  } 
                  if(behaviour_ == opengm::test::OPTIMAL) {
                     std::vector<size_t> optimalState;
                     opengm::Bruteforce<GraphicalModelType, AccType> bf(gm);

                     OPENGM_TEST(bf.infer()==opengm::NORMAL);
                     OPENGM_TEST(bf.arg(optimalState)==opengm::NORMAL);                     
                     OPENGM_TEST(optimalState.size()==gm.numberOfVariables());
                     for(size_t i = 0; i < gm.numberOfVariables(); ++i) {
                        OPENGM_TEST(optimalState[i]<gm.numberOfLabels(i));
                     }
                     OPENGM_TEST_EQUAL_TOLERANCE(gm.evaluate(state), gm.evaluate(optimalState), 0.00001);
                  }
               }

            }catch (std::exception& e)
            {
               exceptionFlag = true;
               std::cout << e.what() << std::endl;
            }

            // Check if exception has been thrown 
            if(behaviour_ == opengm::test::FAIL) {
               OPENGM_TEST(exceptionFlag);
            }else{
               OPENGM_TEST(!exceptionFlag);
            }
         }
         std::cout << "  done. " <<std::endl;
      }
   }
}
/// \endcond
#endif

