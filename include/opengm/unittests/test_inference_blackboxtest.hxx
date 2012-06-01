#pragma once
#ifndef OPENGM_TEST_INFERENCE_BLACKBOX_HXX
#define OPENGM_TEST_INFERENCE_BLACKBOX_HXX

#include <vector>

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/models/syntheticmodelgenerator.hxx>
#ifdef WITH_CPLEX
#   include <opengm/inference/lpcplex.hxx>
#endif

/// \cond HIDDEN_SYMBOLS
namespace opengm {

   template<class GM, class INF>
   class InferenceBlackBoxTest {
   public:
      typedef GM GraphicalModelType;
      typedef INF InfType;
      typedef typename GraphicalModelType::ExplicitFunctionType ExplicitFunctionType;
      //typedef typename GraphicalModelType::SparseFunctionType SparseFunctionType;
      //typedef typename GraphicalModelType::ImplicitFunctionType ImplicitFunctionType;
      typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
      typedef typename GraphicalModelType::ValueType ValueType;
      typedef typename InfType::AccumulationType AccType;
      typedef typename InfType::Parameter InfParaType;

      InferenceBlackBoxTest();
      InferenceBlackBoxTest(InfParaType);
      void testPottsGrid2(size_t width,size_t height,size_t numStates, unsigned int id, bool opt=false);
      void testPottsFull2(size_t numVar,size_t numStates, unsigned int id, bool opt=false);
      void testPottsGrid02(size_t width,size_t height,size_t numStates, unsigned int id, bool opt=false);
      void testPottsFull02(size_t numVar,size_t numStates, unsigned int id, bool opt=false);
      void testGPottsGrid02(size_t width,size_t height,size_t numStates, unsigned int id, bool opt=false);
      void testGPottsFull02(size_t numVar,size_t numStates, unsigned int id, bool opt=false);
      void testPottsGrid2Int(size_t width,size_t height,size_t numStates, unsigned int id, bool opt=false);
      void testPottsFull2Int(size_t numVar,size_t numStates, unsigned int id, bool opt=false);
      void testRandomGrid2(size_t width,size_t height,size_t numStates, unsigned int id, bool opt=false);
      void testRandomFull2(size_t numVar,size_t numStates, unsigned int id, bool opt=false);
      void testRandomStar2(size_t numVar, size_t numStates, unsigned int id, bool opt=false);
      void testVarPottsGrid2(size_t width,size_t height,size_t numStates, unsigned int id, bool opt=false);
      void testVarPottsFull2(size_t numVar,size_t numStates, unsigned int id, bool opt=false);
      void testVarRandomGrid2(size_t width,size_t height,size_t numStates, unsigned int id, bool opt=false);
      void testVarRandomFull2(size_t numVar,size_t numStates, unsigned int id, bool opt=false);
      void testVarRandomStar2(size_t numVar, size_t numStates, unsigned int id, bool opt=false);
      void test(const GraphicalModelType& gm, bool opt=false);

   private:
      SyntheticModelGenerator<GraphicalModelType> modelGenerator_;
      InfParaType infPara_;
   };

   template<class GM, class INF>
   InferenceBlackBoxTest<GM, INF>::InferenceBlackBoxTest()
   {}

   template<class GM, class INF>
   InferenceBlackBoxTest<GM, INF>::InferenceBlackBoxTest(InfParaType infPara)
   {
      infPara_ = infPara;
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::test(const GM& gm, bool opt)
   {
      std::vector<size_t> states1;
      std::vector<size_t> states2;
      InfType inf(gm, infPara_);
      OPENGM_TEST(inf.infer()==opengm::NORMAL);
      OPENGM_TEST(inf.arg(states1)==opengm::NORMAL);
      OPENGM_TEST(states1.size()==gm.numberOfVariables());
      for(size_t i=0; i<gm.numberOfVariables();++i) {
         OPENGM_TEST(states1[i]<gm.numberOfLabels(i));
      }
      if(opt) {
         opengm::Bruteforce<GraphicalModelType, AccType> bf(gm);
         OPENGM_TEST(bf.infer()==opengm::NORMAL);
         OPENGM_TEST(bf.arg(states2)==opengm::NORMAL);
         OPENGM_TEST(states2.size()==gm.numberOfVariables());
         for(size_t i=0; i<gm.numberOfVariables();++i)
            OPENGM_TEST(states2[i]<gm.numberOfLabels(i));
         OPENGM_TEST_EQUAL_TOLERANCE(gm.evaluate(states1), gm.evaluate(states2), 0.00001);
         //testEqualSequence(states1.begin(), states1.end(), states2.begin());
      }
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testRandomFull2(size_t numVar, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildRandomFull2(numVar, numStates, id, 1, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testPottsFull2(size_t numVar, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildPottsFull2(numVar, numStates, id, numVar, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testGPottsFull02(size_t numVar, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildGPottsFull02(numVar, numStates, id, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testPottsFull02(size_t numVar, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildPottsFull02(numVar, numStates, id, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testPottsFull2Int(size_t numVar, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildPottsFull2(numVar, numStates, id, numVar*1000000, 1000000);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testRandomGrid2(size_t height, size_t width, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildRandomGrid2(height, width, numStates, id, 1, 1);
      OPENGM_ASSERT(gm.numberOfVariables() == height*width);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testPottsGrid2(size_t height, size_t width, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildPottsGrid2(height, width, numStates, id, 4, 1);
      OPENGM_ASSERT(gm.numberOfVariables() == height*width);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testGPottsGrid02(size_t height, size_t width, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildGPottsGrid02(height, width, numStates, id,  1);
      OPENGM_ASSERT(gm.numberOfVariables() == height*width);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testPottsGrid02(size_t height, size_t width, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildPottsGrid02(height, width, numStates, id,  1);
      OPENGM_ASSERT(gm.numberOfVariables() == height*width);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testPottsGrid2Int(size_t height, size_t width, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_.buildPottsGrid2(height, width, numStates, id, 4000000, 1000000);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testRandomStar2(size_t numVar, size_t numStates, unsigned int id, bool opt)
   {
      modelGenerator_.randomNumberOfStates_ = false;
      GraphicalModelType gm = modelGenerator_. buildRandomStar2(numVar, numStates, id, 1, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testVarRandomStar2(size_t numVar, size_t numStates, unsigned int id, bool opt)
   {
      modelGenerator_.randomNumberOfStates_ = true;
      GraphicalModelType gm = modelGenerator_. buildRandomStar2(numVar, numStates, id, 1, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testVarRandomFull2(size_t numVar, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = true;
      GraphicalModelType gm = modelGenerator_.buildRandomFull2(numVar, numStates, id, 1, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testVarPottsFull2(size_t numVar, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = true;
      GraphicalModelType gm = modelGenerator_.buildPottsFull2(numVar, numStates, id, numVar, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testVarRandomGrid2(size_t height, size_t width, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = true;
      GraphicalModelType gm = modelGenerator_.buildRandomGrid2(height, width, numStates, id, 1, 1);
      test(gm,opt);
   }

   template<class GM, class INF>
   void InferenceBlackBoxTest<GM, INF>::testVarPottsGrid2(size_t height, size_t width, size_t numStates, unsigned int id, bool opt) {
      modelGenerator_.randomNumberOfStates_ = true;
      GraphicalModelType gm = modelGenerator_.buildPottsGrid2(height, width, numStates, id, 3, 1);
      test(gm,opt);
   }
}
/// \endcond
#endif // #ifndef OPENGM_TEST_INFERENCE_BLACKBOX_HXX

