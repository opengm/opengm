#include <iostream>
#ifdef WITH_QPBO
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/external/qpbo.hxx>
#include <opengm/inference/bruteforce.hxx>


#include <opengm/operations/adder.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

struct QPBOTest
{
  typedef opengm::GraphicalModel<float,opengm::Adder  > GraphicalModelType;
  typedef GraphicalModelType::ValueType ValueType;
  typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType ;	//explicit Factorfunction(=dense marray)
  typedef GraphicalModelType::IndependentFactorType IndependentFactorType ;	//independet Factor (detached from the graphical model)
  typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;

  template<class QPBO>
  void test(typename QPBO::Parameter para) {
   typedef opengm::GraphicalModel<float, opengm::Adder> SumGmType;
   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
   typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

   opengm::InferenceBlackBoxTester<SumGmType> sumTester;
   sumTester.addTest(new SumFullTest(4,    2, false,    2, SumFullTest::POTTS, opengm::OPTIMAL, 1));
   sumTester.addTest(new SumFullTest(5,    2, false,    2, SumFullTest::POTTS, opengm::OPTIMAL, 1));
   sumTester.template test<QPBO>(para);

  }

 void run()
  {
    std::cout << std::endl;
    std::cout << "  * Start Black-Box Tests ..."<<std::endl;
    typedef opengm::GraphicalModel<float,opengm::Adder  > GraphicalModelType;
    typedef opengm::external::QPBO<GraphicalModelType>  QPBO;
    QPBO::Parameter para;

    std::cout << "    - Minimization/Adder with strong percistency..."<<std::flush;
    para.strongPersistency_ = true;
    this->test<QPBO>(para);
    std::cout << "    - Minimization/Adder with weak percistency ..."<<std::flush;
    para.strongPersistency_ = false;
    test<QPBO>(para);

    std::cout << "    - Minimization/Adder with weak percistency + I ..."<<std::flush;
    para.strongPersistency_ = false;
    para.useImproveing_     = true;
    test<QPBO>(para);

    std::cout << "    - Minimization/Adder with weak percistency + P ..."<<std::flush;
    para.strongPersistency_ = false;
    para.useImproveing_     = false;
    para.useProbeing_       = true;
    test<QPBO>(para);

    std::cout << "    - Minimization/Adder with weak percistency + PI ..."<<std::flush;
    para.strongPersistency_ = false;
    para.useImproveing_     = true;
    para.useProbeing_       = true;
    test<QPBO>(para);


  };
};
#endif
int main() {
   #ifdef WITH_QPBO
   std::cout << "QPBO  Tests ..." << std::endl;
   {
      QPBOTest t; t.run();
   }
   #else
   std::cout << "Compiled withput QPBO ,QPBO  Tests  is disabled..." << std::endl;
   #endif
   return 0;
}
