#pragma once
#ifndef OPENGM_TEST_TEST_BASE_HXX
#define OPENGM_TEST_TEST_BASE_HXX

/// \cond HIDDEN_SYMBOLS
namespace opengm {
   namespace test{
      enum TestBehaviour
      {
         PASS, OPTIMAL, FAIL
      };

      template <class INF>
      class TestBase
      {
      public:
         virtual void test(typename INF::Parameter) = 0;
      };
   }
}
/// \endcond
#endif

