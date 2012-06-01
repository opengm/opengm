#pragma once
#ifndef OPENGM_TEST_TEST_EMPTY_HXX
#define OPENGM_TEST_TEST_EMPTY_HXX

#include <opengm/unittests/inferencetests/test_base.hxx>
/// \cond HIDDEN_SYMBOLS
namespace opengm {
   namespace test{
      /// \brief TestEmpty<INF> 
      /// An empty test that test nothing 
      template<class INF>
      class TestEmpty : public TestBase<INF>
      {
      public:
         /// \brief test<INF> start an empty test with algorithm INF
         /// \param para parameters of algorithm
         virtual void test(typename INF::Parameter para) {
            std::cout << "  - Empty Test ... done!" << std::endl;
         };
      };
   }
}
/// \endcond
#endif

