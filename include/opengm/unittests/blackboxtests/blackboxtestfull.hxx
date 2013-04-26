#pragma once
#ifndef OPENGM_TEST_INFERENCE_BLACKBOXTEST_FULL2_HXX
#define OPENGM_TEST_INFERENCE_BLACKBOXTEST_FULL2_HXX

#include <vector>
#include <sstream>
#include <string>
#include <cmath>

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/modelgenerators/syntheticmodelgenerator.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestbase.hxx>

/// \cond HIDDEN_SYMBOLS

namespace opengm {
	inline double mylog2(const double x) {return std::log(x)/std::log(2.0);}
   template<class GM> class BlackBoxTestBase;

   template<class GM>
   class BlackBoxTestFull : public BlackBoxTestBase<GM>
   {
   public:
      enum BlackBoxFunction
      {
         RANDOM, POTTS, IPOTTS
      };
      const size_t numVar_;
      const size_t numStates_;
      const bool varStates_;
      const size_t orders_;
      const BlackBoxFunction function_;
      const BlackBoxBehaviour behaviour_;
      const size_t numTests_;
      const size_t modelIdOffset_;
      

      BlackBoxTestFull(const size_t var, const size_t numStates, const bool varStates, const size_t orders, const BlackBoxFunction f, const BlackBoxBehaviour b, const size_t numTests, const size_t modelIdOffset=0);
      virtual std::string infoText();
      virtual GM getModel(size_t);
      virtual size_t numberOfTests();
      virtual BlackBoxBehaviour behaviour();

   private:
      std::string functionString(BlackBoxFunction f);
   };

   template<class GM>
   BlackBoxTestFull<GM>::BlackBoxTestFull(
      const size_t numVar, const size_t numStates, const bool varStates, const size_t orders,
      const BlackBoxFunction f, const BlackBoxBehaviour b,
      const size_t numTests, const size_t modelIdOffset)
      : numVar_(numVar), numStates_(numStates), varStates_(varStates), orders_(orders),
        function_(f), behaviour_(b),
        numTests_(numTests), modelIdOffset_(modelIdOffset)
   {
   }

   template<class GM>
   size_t BlackBoxTestFull<GM>::numberOfTests()
   {
      return numTests_;
   }

   template<class GM>
   BlackBoxBehaviour BlackBoxTestFull<GM>::behaviour()
   {
      return behaviour_;
   }

   template<class GM>
   std::string BlackBoxTestFull<GM>::infoText()
   {
      size_t order = (size_t) mylog2((double) (orders_)) + 1;
      std::string s = varStates_ ? "<" : "";
      std::string u = orders_ % 2 == 1 ? " with unary " : " without unary ";

      std::stringstream oss;
      oss << "    - "<< order << "nd order full model with " << s << numVar_ << " variables ";
      oss << "and "<< s << numStates_ << " states ";
      oss << "(functionType = " << functionString(function_) << u << ")";
      return oss.str();
   }

   template<class GM>
   GM BlackBoxTestFull<GM>::getModel(size_t id)
   {
      typedef opengm::SyntheticModelGenerator2<GM> ModelGeneratorType;

      ModelGeneratorType modelGenerator;
      typename ModelGeneratorType::Parameter modelGeneratorPara;

      size_t tester = 1;
      size_t order = (size_t) mylog2((double) (orders_)) + 1;
    
      for(size_t i = 0; i < order; ++i) {
         if((orders_ / tester) % 2 == 0) {
            modelGeneratorPara.functionTypes_[i] = ModelGeneratorType::EMPTY;
         }else{
            if(i == 0) { 
               modelGeneratorPara.functionTypes_[i] = ModelGeneratorType::URANDOM;
               modelGeneratorPara.functionParameters_[i].resize(2);
               modelGeneratorPara.functionParameters_[i][0] = 1;
               modelGeneratorPara.functionParameters_[i][1] = 2;
            }else{
               switch (function_) {
               case RANDOM:
                  modelGeneratorPara.functionTypes_[i] = ModelGeneratorType::URANDOM;
                  modelGeneratorPara.functionParameters_[i].resize(2);
                  modelGeneratorPara.functionParameters_[i][0] = 1;
                  modelGeneratorPara.functionParameters_[i][1] = 2;
                  break;
               case POTTS:
                  modelGeneratorPara.functionTypes_[i] = ModelGeneratorType::GPOTTS;
                  modelGeneratorPara.functionParameters_[i].resize(1);
                  modelGeneratorPara.functionParameters_[i][0] = numVar_;
                  modelGeneratorPara.functionParameters_[i][0] = numVar_;
                  break;
               case IPOTTS:
                  modelGeneratorPara.functionTypes_[i] = ModelGeneratorType::GPOTTS;
                  modelGeneratorPara.functionParameters_[i].resize(1);
                  modelGeneratorPara.functionParameters_[i][0] = -numVar_;
                  modelGeneratorPara.functionParameters_[i][0] = -numVar_;
                  break; 
               }
            }
         }
         tester *= 2;
      }
      modelGeneratorPara.randomNumberOfStates_ = varStates_; 
      return modelGenerator.buildFull(id + modelIdOffset_, numVar_, numStates_, modelGeneratorPara);
   }

   template<class GM>
   std::string BlackBoxTestFull<GM>::functionString(BlackBoxFunction f)
   {
      switch (f) {
      case RANDOM:
         return "random";
      case POTTS:
         return "potts";
      case IPOTTS:
         return "negative potts";
      default:
         return "";
      };
      return "";
   }

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_TEST_INFERENCE_BLACKBOXTEST_FULL2_HXX
