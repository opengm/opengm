#pragma once
#ifndef OPENGM_TEST_INFERENCE_BLACKBOXTEST_BASE_HXX
#define OPENGM_TEST_INFERENCE_BLACKBOXTEST_BASE_HXX

/// \cond HIDDEN_SYMBOLS

namespace opengm {

   enum BlackBoxBehaviour
   {
      PASS, OPTIMAL, FAIL
   };

   template<class GM>
   class BlackBoxTestBase
   {
   public:
      virtual ~BlackBoxTestBase(){}
      virtual std::string infoText() = 0;
      virtual GM getModel(size_t) = 0;
      virtual size_t numberOfTests() = 0;
      virtual BlackBoxBehaviour behaviour() = 0;
   };

} // namepsace opengm

/// \endcond

#endif // #ifndef OPENGM_TEST_INFERENCE_BLACKBOXTEST_BASE_HXX
