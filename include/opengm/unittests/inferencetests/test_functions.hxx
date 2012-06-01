#pragma once
#ifndef OPENGM_TEST_TEST_FUNCTIONS_HXX
#define OPENGM_TEST_TEST_FUNCTIONS_HXX

#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include <opengm/functions/absolute_difference.hxx>
#include <opengm/functions/squared_difference.hxx>
#include <opengm/functions/truncated_absolute_difference.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>
/// \cond HIDDEN_SYMBOLS
namespace opengm {
   namespace test{
      /// \brief TestFunctions<INF> 
      /// Build a second order binary submodular 3x3-grid-model with several functions 
      /// and check if the algorithm has the correct behaviour. 
      /// This test helps to check the compatibility of the solver with opengm-functions empirically.
      template <class INF>
      class TestFunctions : public TestBase<INF>
      {
      public:
         TestFunctions(TestBehaviour);
         virtual void test(typename INF::Parameter);
      private:
         TestBehaviour behaviour_;
      };

      /// \brief TestMultiMode Constructor
      /// \param behaviour expected behaviour of the algorithm
      template <class INF>
      TestFunctions<INF>::TestFunctions(TestBehaviour behaviour) : behaviour_(behaviour)
      {;}
 
      /// \brief test<INF> start test with algorithm INF
      /// \param para parameters of algorithm
      template <class INF>
      void TestFunctions<INF>::test(typename INF::Parameter para) {
         typedef typename INF::GraphicalModelType GraphicalModelType;
         typedef typename GraphicalModelType::ValueType ValueType;

         std::cout << "  - FunctionTest ... " << std::flush;

         std::vector<size_t> numberOfLabels(9, 2);
         GraphicalModelType gm(numberOfLabels.begin(), numberOfLabels.end());

         size_t var[2];

         // construct 3x3 grid using various functions

         // all pairwise potentials are (0,1,1,0) except edge (5,8) which is (0,2,2,0)
         typename GraphicalModelType::ExplicitFunctionType fExplicit(numberOfLabels.begin(), numberOfLabels.begin() + 2, 0);
         fExplicit(0, 1) = 1;
         fExplicit(1, 0) = 1;
         typename GraphicalModelType::FunctionIdentifier fExplicitId = gm.addFunction(fExplicit);
         var[0] = 0;
         var[1] = 1;
         gm.addFactor(fExplicitId, var, var + 2);

         opengm::PottsFunction<ValueType> fPotts(2, 2, 0, 1);
         typename GraphicalModelType::FunctionIdentifier fPottsId = gm.addFunction(fPotts);
         var[0] = 1;
         var[1] = 2;
         gm.addFactor(fPottsId, var, var + 2);

         opengm::PottsNFunction<ValueType> fPottsN(numberOfLabels.begin(), numberOfLabels.begin() + 2, 0, 1);
         typename GraphicalModelType::FunctionIdentifier fPottsNId = gm.addFunction(fPottsN);
         var[0] = 3;
         var[1] = 4;
         gm.addFactor(fPottsNId, var, var + 2);

         size_t values[2];
         values[0] = 0;
         values[1] = 1;
         opengm::PottsGFunction<ValueType> fPottsG(numberOfLabels.begin(), numberOfLabels.begin() + 2, values);
         typename GraphicalModelType::FunctionIdentifier fPottsGId = gm.addFunction(fPottsG);
         var[0] = 4;
         var[1] = 5;
         gm.addFactor(fPottsGId, var, var + 2);

         opengm::AbsoluteDifferenceFunction<ValueType> fAbsoluteDifference(numberOfLabels[0], numberOfLabels[1]);
         typename GraphicalModelType::FunctionIdentifier fAbsoluteDifferenceId = gm.addFunction(fAbsoluteDifference);
         var[0] = 6;
         var[1] = 7;
         gm.addFactor(fAbsoluteDifferenceId, var, var + 2);

         opengm::SquaredDifferenceFunction<ValueType> fSquaredDifference(numberOfLabels[0], numberOfLabels[1]);
         typename GraphicalModelType::FunctionIdentifier fSquaredDifferenceId = gm.addFunction(fSquaredDifference);
         var[0] = 7;
         var[1] = 8;
         gm.addFactor(fSquaredDifferenceId, var, var + 2);

         opengm::TruncatedAbsoluteDifferenceFunction<ValueType> fTruncatedAbsoluteDifference(numberOfLabels[0], numberOfLabels[1], 1, 1);
         typename GraphicalModelType::FunctionIdentifier fTruncatedAbsoluteDifferenceId = gm.addFunction(fTruncatedAbsoluteDifference);
         var[0] = 0;
         var[1] = 3;
         gm.addFactor(fTruncatedAbsoluteDifferenceId, var, var + 2);

         opengm::TruncatedSquaredDifferenceFunction<ValueType> fTruncatedSquaredDifference(numberOfLabels[0], numberOfLabels[1], 1, 1);
         typename GraphicalModelType::FunctionIdentifier fTruncatedSquaredDifferenceId = gm.addFunction(fTruncatedSquaredDifference);
         var[0] = 3;
         var[1] = 6;
         gm.addFactor(fTruncatedSquaredDifferenceId, var, var + 2);

         var[0] = 1;
         var[1] = 4;
         gm.addFactor(fExplicitId, var, var + 2);

         var[0] = 4;
         var[1] = 7;
         gm.addFactor(fExplicitId, var, var + 2);

         var[0] = 2;
         var[1] = 5;
         gm.addFactor(fExplicitId, var, var + 2);

         fExplicit(0, 0) = 0;
         fExplicit(0, 1) = 2;
         fExplicit(1, 0) = 2;
         fExplicit(1, 1) = 0;
         typename GraphicalModelType::FunctionIdentifier fExplicitId2 = gm.addFunction(fExplicit);
         var[0] = 5;
         var[1] = 8;
         gm.addFactor(fExplicitId2, var, var + 2);

         // add some unary potentials to nodes 0, 4, 6 and 8
         typename GraphicalModelType::ExplicitFunctionType fUnary(numberOfLabels.begin(), numberOfLabels.begin() + 1);

         fUnary(0) = 0;
         fUnary(1) = 2;
         typename GraphicalModelType::FunctionIdentifier fUnaryId = gm.addFunction(fUnary);
         var[0] = 0;
         gm.addFactor(fUnaryId, var, var + 1);

         fUnary(0) = 6;
         fUnary(1) = 0;
         typename GraphicalModelType::FunctionIdentifier fUnaryId2 = gm.addFunction(fUnary);
         var[0] = 4;
         gm.addFactor(fUnaryId2, var, var + 1);

         fUnary(0) = 1;
         fUnary(1) = 0;
         typename GraphicalModelType::FunctionIdentifier fUnaryId3 = gm.addFunction(fUnary);
         var[0] = 6;
         gm.addFactor(fUnaryId3, var, var + 1);

         fUnary(0) = 0;
         fUnary(1) = 2;
         typename GraphicalModelType::FunctionIdentifier fUnaryId4 = gm.addFunction(fUnary);
         var[0] = 8;
         gm.addFactor(fUnaryId4, var, var + 1);

         bool fail = false; 
         std::vector<size_t> sol;
         try {
            INF inf(gm);
            inf.infer(para);
            inf.arg(sol);
            
            bool optTest = (sol[0] == 0) && (sol[1] == 0) && (sol[2] == 0)
                        && (sol[3] == 1) && (sol[4] == 1) && (sol[5] == 0)
                        && (sol[6] == 1) && (sol[7] == 1) && (sol[8] == 0);

            OPENGM_TEST(state.size() == gm_.numberOfVariables());
            if(behaviour_ == opengm::test::OPTIMAL) {
               OPENGM_TEST(optTest)
            }
         }
         catch (std::exception& error) {
            std::cout << error.what() << std::endl;
            fail = true;
         }

         // Check if exception has been thrown
         if(behaviour_ == opengm::test::FAIL) {
            OPENGM_TEST(fail);
         }else{
            OPENGM_TEST(!fail);
         }

         std::cout <<"done!"<<std::endl;
      }
   }
}
/// \endcond
#endif
