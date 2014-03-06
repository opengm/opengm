#pragma once
#ifndef OPENGM_TEST_INFERENCE_BLACKBOXTESTER_HXX
#define OPENGM_TEST_INFERENCE_BLACKBOXTESTER_HXX

#include <vector>
#include <typeinfo>

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestbase.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/integrator.hxx>

/// \cond HIDDEN_SYMBOLS

namespace opengm {
   template<class GM>
   class InferenceBlackBoxTester
   {
   public:
      typedef GM GraphicalModelType;
      template<class INF> void test(const typename INF::Parameter&, bool tValue=true, bool tArg=false, bool tMarg=false, bool tFacMarg=false);
      void addTest(BlackBoxTestBase<GraphicalModelType>*);
      ~InferenceBlackBoxTester();

   private:
      std::vector<BlackBoxTestBase<GraphicalModelType>*> testList;
   };

   //***************
   //IMPLEMENTATION
   //***************

   template<class GM>
   InferenceBlackBoxTester<GM>::~InferenceBlackBoxTester()
   {
      for(size_t testId = 0; testId < testList.size(); ++testId) {
         delete testList[testId];
      }
   }

   template<class GM>
   template<class INF>
   void InferenceBlackBoxTester<GM>::test(const typename INF::Parameter& infPara, bool tValue, bool tArg, bool tMarg, bool tFacMarg)
   {
      typedef typename GraphicalModelType::ValueType ValueType;
      typedef typename GraphicalModelType::OperatorType OperatorType;
      typedef typename INF::AccumulationType AccType;

      for(size_t testId = 0; testId < testList.size(); ++testId) {
         size_t numTests = testList[testId]->numberOfTests();
         BlackBoxBehaviour behaviour = testList[testId]->behaviour();
         std::cout << testList[testId]->infoText();
         std::cout << " " << std::flush;
         for(size_t n = 0; n < numTests; ++n) {
            std::cout << "*" << std::flush;
            GraphicalModelType gm = testList[testId]->getModel(n);
            //Run Algorithm
            bool exceptionFlag = false;
            std::vector<typename GM::LabelType> state;
            try{      
               INF inf(gm, infPara);
               InferenceTermination returnValue=inf.infer();
               OPENGM_TEST((returnValue==opengm::NORMAL) || (returnValue==opengm::CONVERGENCE)); 
               if(typeid(AccType) == typeid(opengm::Minimizer) || typeid(AccType) == typeid(opengm::Maximizer)) {
                  OPENGM_TEST(inf.arg(state)==opengm::NORMAL);
                  OPENGM_TEST(state.size()==gm.numberOfVariables());
                  for(size_t varId = 0; varId < gm.numberOfVariables(); ++varId) {
                     OPENGM_TEST(state[varId]<gm.numberOfLabels(varId));
                  }
                  { 
                     ValueType bound = inf.bound();
                     ValueType value = inf.value();
                     ValueType value2 = 0;
                     if(typeid(AccType) == typeid(opengm::Minimizer))
                        value2 = value + std::min<ValueType>(1e20,std::max<ValueType>(1e-4,fabs(value)))*1e-6;
                     if(typeid(AccType) == typeid(opengm::Maximizer))
                        value2 = value - std::min<ValueType>(1e20,std::max<ValueType>(1e-4,fabs(value)))*1e-6;
                   
                     std::cout << "value = " << value << "  ,  bound = " << bound << std::endl;
                     OPENGM_TEST(AccType::bop(bound,value2)|| bound==value2);
                  }
                  if(behaviour == opengm::OPTIMAL) {
                     std::vector<typename GM::LabelType> optimalState;
                     opengm::Bruteforce<GraphicalModelType, AccType> bf(gm);
                     OPENGM_TEST(bf.infer()==opengm::NORMAL);
                     OPENGM_TEST(bf.arg(optimalState)==opengm::NORMAL);
                     OPENGM_TEST(optimalState.size()==gm.numberOfVariables());
                     for(size_t i = 0; i < gm.numberOfVariables(); ++i) {
                        OPENGM_TEST(optimalState[i]<gm.numberOfLabels(i));
                     }
                     OPENGM_TEST_EQUAL_TOLERANCE(gm.evaluate(state), gm.evaluate(optimalState), 0.00001);
                     //testEqualSequence(states1.begin(), states1.end(), states2.begin());
                  }
               }
               //if(typeid(AccType) == typeid(opengm::Integrator)) {
                  //for(size_t varId = 0; varId < gm.numberOfVariables(); ++varId) {
                  //   OPENGM_TEST(inf.marginal(varId)==opengm::NORMAL);
                  //}
                  //for(size_t factorId = 0; factorId < gm.numberOfFactors(); ++factorId) {
                  //   OPENGM_TEST(inf.factorMarginal(factorId)==opengm::NORMAL);
                  //}
               //}
               } catch(std::exception& e) {
                 exceptionFlag = true;
                 std::cout << e.what() <<std::endl;
               }
            if(behaviour == opengm::FAIL) {
               OPENGM_TEST(exceptionFlag);
            }else{
               OPENGM_TEST(!exceptionFlag);
            }
         }
         if(behaviour == opengm::OPTIMAL) {
            std::cout << " OPTIMAL!" << std::endl;
         }else if(behaviour == opengm::PASS) {
            std::cout << " PASS!" << std::endl;
         }else{
            std::cout << " OK!" << std::endl;
         }
      }
   }

   template<class GM>
   void InferenceBlackBoxTester<GM>::addTest(BlackBoxTestBase<GraphicalModelType>* test)
   {
      testList.push_back(test);
   }

/*
 template<class GM, class INF>
 void InferenceBlackBoxTest::test
 (
 const INF::Parameter& infPara,
 const GM& gm,
 const BlackBoxBehaviour behaviour) const
 {
 typedef GM GraphicalModelType;
 typedef INF InfType;
 typedef typename GraphicalModelType::ExplicitFunctionType ExplicitFunctionType;
 typedef typename GraphicalModelType::SparseFunctionType SparseFunctionType;
 typedef typename GraphicalModelType::ImplicitFunctionType ImplicitFunctionType;
 typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
 typedef typename GraphicalModelType::ValueType ValueType;
 typedef typename GraphicalModelType::OperatorType OperatorType;
 typedef typename InfType::AccumulationType AccType;
 typedef typename InfType::Parameter InfParaType;
 //Run Algorithm
 bool exceptionFlag = false;
 std::vector<size_t> state;
 try{
 InfType inf(gm, infPara);
 OPENGM_TEST(inf.infer()==opengm::NORMAL);
 if(typeid(AccType) == typeid(opengm::Minimizer) || typeid(AccType) == typeid(opengm::Maximizer)) {
 OPENGM_TEST(inf.arg(state)==opengm::NORMAL);
 OPENGM_TEST(state.size()==gm.numberOfVariabl      {
 std::string str;
 str = "    - 2nd order grid model (" + height_ + "x" + width_;
 str += ", " + varStates_ ? "~" : "" + numStates_ + ") ";
 str += withUnary_ ? "with unary" : "without unary and " + functionString(function_);
 return str;
 }es());
 for(size_t i = 0; i < gm.numberOfVariables(); ++i) {
 OPENGM_TEST(state[i]<gm.numberOfLabels(i));
 }
 {
 const ValueType value = gm.evaluate(state);
 const ValueType bound = inf.bound();
 OPENGM_TEST(AccType::bop(value,bound));
 }
 }
 if(typeid(AccType) == typeid(opengm::Integrator)) {
 for(size_t varId = 0; varId < gm.numberOfVariables; ++varId) {
 OPENGM_TEST(inf.marginal(varId)==opengm::NORMAL);
 }
 for(size_t factorId = 0; factorId < gm.numberOfFactors; ++factorId) {
 OPENGM_TEST(inf.factorMarginal(factorId)==opengm::NORMAL);
 }
 }
 } catch(std::exception& e) {
 exceptionFlag = true;
 }
 if(behaviour == FAIL) {
 OPENGM_ASSERT(exceptionFlag);
 }else{
 OPENGM_ASSERT(!exceptionFlag);
 }

 if(behaviour == OPTIMAL) {
 std::vector<size_t> optimalState;
 opengm::Bruteforce<GraphicalModelType, AccType> bf(gm);
 OPENGM_TEST(bf.infer()==opengm::NORMAL);
 OPENGM_TEST(bf.arg(optimalState)==opengm::NORMAL);
 OPENGM_TEST(optimalState.size()==gm.numberOfVariables());
 for(size_t i = 0; i < gm.numberOfVariables(); ++i)
 OPENGM_TEST(optimalState[i]<gm.numberOfLabels(i));

 OPENGM_TEST_EQUAL_TOLERANCE(gm.evaluate(state), gm.evaluate(optimalState), 0.00001);
 //testEqualSequence(states1.begin(), states1.end(), states2.begin());
 }
 }

 template<class GM, class INF>
 void InferenceBlackBoxTest::
 test(
 const INF::Parameter& para,
 const BlackBoxTest test,
 const BlackBoxFunction function,
 const BlackBoxVar var,
 const BlackBoxBehaviour behaviour,
 const size_t orders) const
 {
 opengm::SyntheticModelGenerator2 modelGenerator;
 opengm::SyntheticModelGenerator2::Parameter modelGeneratorPara;
 size_t order = (size_t) log2((double) (orders)) + 1;
 modelGeneratorPara.functionParameters_.resize(order);
 modelGeneratorPara.functionTypes_.resize(order, opengm::SyntheticModelGenerator2::URANDOM);
 modelGeneratorPara.sharedFunctions_.resize(order, false);

 size_t tester = 1;
 for(size_t i = 0; i < order; ++i) {
 if(orders & tester == 0) {
 modelGeneratorPara.functionTypes_[i] = opengm::SyntheticModelGenerator2::EMPTY;
 }else{
 if(i > 0) {
 modelGeneratorPara.functionTypes_[i] = opengm::SyntheticModelGenerator2::URANDOM;
 }
 }
 tester *= 2;
 }

 if(test == GRID)
 numberOfVariables_ = height_ * width_;
 std::vector<size_t> numberOfLabels;
 switch (var) {
 case RANDOM:
 numberOfLabels.resize(numberOfVariables_);
 for(size_t i = 0; i < numberOfLabels.size(); ++i) {
 numberOfLabels[i] = rand() % numberOfStates_ + 1;
 }
 break;
 case FIX:
 numberOfLabels.resize(numberOfVariables_, numberOfStates_);
 break;
 case BINARY:
 numberOfLabels.resize(numberOfVariables_, 2);
 break;
 }

 switch (test) {
 case STAR:
 GraphicalModelType gm = modelGenerator.buildStar(numberOfVariables_, numberOfLabels, modelGeneratorPara);
 break;
 case TREE:
 break;
 case GRID:
 GraphicalModelType gm = modelGenerator.buildGrid(height_, width_, numberOfLabels, modelGeneratorPara);
 break;
 case FULL:
 GraphicalModelType gm = modelGenerator.buildFull(numberOfVariables_, numberOfLabels, modelGeneratorPara);
 break;
 }
 }
 */
}

/// \endcond

#endif // #ifndef OPENGM_TEST_INFERENCE_BLACKBOXTESTER_HXX

