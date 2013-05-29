#pragma once
#ifndef OPENGM_TEST_INFERENCE_BLACKBOXTEST_STAR_HXX
#define OPENGM_TEST_INFERENCE_BLACKBOXTEST_STAR_HXX

#include <vector>
#include <sstream>
#include <string>

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/modelgenerators/syntheticmodelgenerator.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestbase.hxx>

/// \cond HIDDEN_SYMBOLS

namespace opengm {

   template<class GM>
   class BlackBoxTestStar : public BlackBoxTestBase<GM>
   {
   public:
      enum BlackBoxFunction
      {
         RANDOM, POTTS, IPOTTS, L1
      };
      const size_t numVar_;
      const size_t numStates_;
      const bool varStates_;
      const bool withUnary_;
      const BlackBoxFunction function_;
      const BlackBoxBehaviour behaviour_;
      const size_t numTests_;
      const size_t modelIdOffset_;

      BlackBoxTestStar(const size_t numVar, const size_t numStates, const bool varStates, const bool withUnary, const BlackBoxFunction f, const BlackBoxBehaviour b, const size_t numTests, const size_t modelIdOffset=0);
      virtual std::string infoText();
      virtual GM getModel(size_t);
      virtual size_t numberOfTests();
      virtual BlackBoxBehaviour behaviour();

   private:
      std::string functionString(BlackBoxFunction f);
   };

   template<class GM>
   size_t BlackBoxTestStar<GM>::numberOfTests()
   {
      return numTests_;
   }

   template<class GM>
   BlackBoxBehaviour BlackBoxTestStar<GM>::behaviour()
   {
      return behaviour_;
   }

   template<class GM>
   BlackBoxTestStar<GM>::BlackBoxTestStar(
      const size_t numVar, const size_t numStates,
      const bool varStates, const bool withUnary,
      const BlackBoxFunction f, const BlackBoxBehaviour b,
      const size_t numTests, const size_t modelIdOffset)
      :  numVar_(numVar), numStates_(numStates),
         varStates_(varStates), withUnary_(withUnary),
         function_(f), behaviour_(b),
         numTests_(numTests), modelIdOffset_(modelIdOffset) 
   {
   }

   template<class GM>
   std::string BlackBoxTestStar<GM>::infoText()
   {
      std::string s = varStates_ ? "<" : "";
      std::string u = withUnary_ ? " with unary " : " without unary ";
      std::stringstream oss;
      oss << "    - 2nd order star model with " << numVar_ << " variables and " << s << numStates_ << " states ";
      oss << "(functionType = " << functionString(function_) << u << ")";
      return oss.str();
   }

   template<class GM>
   GM BlackBoxTestStar<GM>::getModel(size_t id)
   {
      typedef opengm::SyntheticModelGenerator2<GM> ModelGeneratorType;

      ModelGeneratorType modelGenerator;
      typename ModelGeneratorType::Parameter modelGeneratorPara;

      if(withUnary_) {
         modelGeneratorPara.functionTypes_[0] = ModelGeneratorType::URANDOM;
         modelGeneratorPara.functionParameters_[0].resize(2);
         modelGeneratorPara.functionParameters_[0][0] = 1;
         modelGeneratorPara.functionParameters_[0][1] = 2;
      }else{
         modelGeneratorPara.functionTypes_[0] = ModelGeneratorType::EMPTY;
      }

      switch (function_) {
      case RANDOM:
         modelGeneratorPara.functionTypes_[1] = ModelGeneratorType::URANDOM;
         modelGeneratorPara.functionParameters_[1].resize(2);
         modelGeneratorPara.functionParameters_[1][0] = 1;
         modelGeneratorPara.functionParameters_[1][1] = 2;
         break;
      case POTTS:
         modelGeneratorPara.functionTypes_[1] = ModelGeneratorType::GPOTTS;
         modelGeneratorPara.functionParameters_[1].resize(1);
         modelGeneratorPara.functionParameters_[1][0] = 1;
         modelGeneratorPara.functionParameters_[1][0] = 1;
         break;
      case IPOTTS:
         modelGeneratorPara.functionTypes_[1] = ModelGeneratorType::GPOTTS;
         modelGeneratorPara.functionParameters_[1].resize(1);
         modelGeneratorPara.functionParameters_[1][0] = -1;
         modelGeneratorPara.functionParameters_[1][0] = -1;
         break;
      case L1:
         modelGeneratorPara.functionTypes_[1] = ModelGeneratorType::L1;
         modelGeneratorPara.functionParameters_[1].resize(1);
         modelGeneratorPara.functionParameters_[1][0] = 1;
         modelGeneratorPara.functionParameters_[1][0] = 1;
         break;
      }
      modelGeneratorPara.randomNumberOfStates_ = varStates_;
      return modelGenerator.buildStar(id + modelIdOffset_, numVar_, numStates_, modelGeneratorPara);
   }

   template<class GM>
   std::string BlackBoxTestStar<GM>::functionString(BlackBoxFunction f)
   {
      switch (f) {
      case RANDOM:
         return "random";
      case POTTS:
         return "potts";
      case IPOTTS:
         return "negative potts";
      case L1:
         return "L1";
      default:
         return "";
      };
      return "";
   }

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_TEST_INFERENCE_BLACKBOXTEST_STAR_HXX
