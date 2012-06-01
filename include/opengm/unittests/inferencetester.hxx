#pragma once
#ifndef OPENGM_TEST_INFERENCE_TESTER_HXX
#define OPENGM_TEST_INFERENCE_TESTER_HXX

#include <vector>
#include <typeinfo>

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/unittests/inferencetests/test_base.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/integrator.hxx>

/// \cond HIDDEN_SYMBOLS
namespace opengm {
   namespace test {
      template <class INF>
      class InferenceTester
      {
      public:
         typedef typename INF::GraphicalModelType GraphicalModelType;
         void test(const typename INF::Parameter&);
         //void addTest(TestBase<INF>*); 
         template<class TEST> void addTest(const TEST&);
         InferenceTester();
         ~InferenceTester();

      private:
         std::vector<TestBase<INF>*> testList;
      };

      //***************
      //IMPLEMENTATION
      //***************
      template<class INF>
      InferenceTester<INF>::InferenceTester()
      {
      }

      template<class INF>
      InferenceTester<INF>::~InferenceTester()
      {
         for(size_t testId = 0; testId < testList.size(); ++testId) {
            delete testList[testId];
         }
      }

      template<class INF>
      void InferenceTester<INF>::test(const typename INF::Parameter& infPara)
      {
         typedef typename GraphicalModelType::ValueType ValueType;
         typedef typename GraphicalModelType::OperatorType OperatorType;
         typedef typename INF::AccumulationType AccType;

         for(size_t testId = 0; testId < testList.size(); ++testId) {
            (*testList[testId]).test(infPara);
         }
      }

      //template<class INF>
      //void InferenceTester<INF>::addTest(TestBase<INF>* test)
      //{
      //   testList.push_back(test);
      //}

      template<class INF>
      template<class TEST>
      void InferenceTester<INF>::addTest(const TEST& test)
      {
         testList.push_back(new TEST(test));
         //*(testList.back()) = test;
      }
   }
}

/// \endcond

#endif 
