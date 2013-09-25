#include <iostream>
#ifdef WITH_QPBO
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/reducedinference.hxx>
#include <opengm/inference/bruteforce.hxx>


#include <opengm/operations/adder.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

struct RINFTest
{
  typedef opengm::GraphicalModel<double,opengm::Adder  > GraphicalModelType;
  typedef GraphicalModelType::ValueType ValueType;
  typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType ;	//explicit Factorfunction(=dense marray)
  typedef GraphicalModelType::IndependentFactorType IndependentFactorType ;	//independet Factor (detached from the graphical model)
  typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;

  template<class RINF>
  void test(typename RINF::Parameter para) {
   typedef opengm::GraphicalModel<ValueType, opengm::Adder> SumGmType;
   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
   typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

   opengm::InferenceBlackBoxTester<SumGmType> sumTester;
   sumTester.addTest(new SumFullTest(4,    2, false,    2, SumFullTest::POTTS, opengm::OPTIMAL, 1));
   sumTester.addTest(new SumFullTest(5,    2, false,    2, SumFullTest::POTTS, opengm::OPTIMAL, 1)); 
   sumTester.addTest(new SumStarTest(5,    5, false,    2, SumStarTest::RANDOM, opengm::OPTIMAL, 1));
   sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::POTTS, opengm::OPTIMAL, 1));
   sumTester.addTest(new SumGridTest(3, 3, 2, false, true, SumGridTest::POTTS, opengm::OPTIMAL, 3));
   sumTester.template test<RINF>(para);

  }

 void run()
  {
    std::cout << std::endl;
    std::cout << "  * Start Black-Box Tests ..."<<std::endl;
    typedef opengm::GraphicalModel<ValueType,opengm::Adder>                   GmType;
    typedef opengm::ReducedInferenceHelper<GmType>::InfGmType                 InfGmType;
    typedef opengm::Bruteforce<InfGmType, opengm::Minimizer>               SubInfType;
    typedef opengm::ReducedInference<GmType, opengm::Minimizer,SubInfType> InfType;

    InfType::Parameter para; 
    para.Persistency_ = false;
    std::cout << "    - Minimization/Adder (with no reduction) ..."<<std::endl; 
    this->test<InfType>(para);
    para.Persistency_ = true;
    std::cout << "    - Minimization/Adder (with persistency) ..."<<std::endl;
    this->test<InfType>(para);
    para.ConnectedComponents_ = true;
    std::cout << "    - Minimization/Adder (with persistency and CC) ..."<<std::endl;
    this->test<InfType>(para);
  };
};
#endif
int main() {
   #ifdef WITH_QPBO
   std::cout << "Reduced Inference Tests ..." << std::endl;
   {
      RINFTest t; t.run();
   }
   #else
   std::cout << "Compiled withput QPBO , RINF-Tests  is disabled..." << std::endl;
   #endif
   return 0;
}
