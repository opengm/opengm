
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/qpbo.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif


/*
struct QPBOTest
{
  typedef opengm::GraphicalModel<float,opengm::Adder  > GraphicalModelType;
  typedef GraphicalModelType::ValueType ValueType;
  typedef GraphicalModelType::ExplicitFunctionType ExplicitFunctionType ;	//explicit Factorfunction(=dense marray)
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
};
*/
int main() {
//void run() {
   typedef opengm::GraphicalModel<float, opengm::Adder> GraphicalModelType;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
   typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
   typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;
   
   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester;
   minTester.addTest(new GridTest(4, 4, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 1));
   minTester.addTest(new GridTest(3, 3, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new GridTest(3, 3, 2, false, false,GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new StarTest(5,    2, false, true, StarTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new FullTest(5,    2, false, 3,    FullTest::POTTS, opengm::OPTIMAL, 3));
   
   
   std::cout << "Test QPBO ..." << std::endl;
   
#ifdef WITH_MAXFLOW
   std::cout << "  * Test Min-Sum with Kolmogorov" << std::endl;
   {
      typedef opengm::external::MinSTCutKolmogorov<size_t, float> MinStCutType;
      typedef opengm::QPBO<GraphicalModelType, MinStCutType> MinQPBO;
      MinQPBO::Parameter para;
      minTester.test<MinQPBO>(para);
   }
#endif
   
#ifdef WITH_BOOST
   std::cout << "  * Test Min-Sum with BOOST-Push-Relabel" << std::endl;
   {
      typedef opengm::MinSTCutBoost<size_t, float, opengm::PUSH_RELABEL> MinStCutType;
      typedef opengm::QPBO<GraphicalModelType, MinStCutType> MinQPBO;
      MinQPBO::Parameter para;
      minTester.test<MinQPBO>(para);
   }
   std::cout << "  * Test Min-Sum with BOOST-Edmonds-Karp" << std::endl;
   {
      typedef opengm::MinSTCutBoost<size_t, float, opengm::EDMONDS_KARP> MinStCutType;
      typedef opengm::QPBO<GraphicalModelType,  MinStCutType> MinQPBO;
      MinQPBO::Parameter para;
      minTester.test<MinQPBO>(para);
   }
   std::cout << "  * Test Min-Sum with BOOST-Kolmogorov" << std::endl;
   {
      typedef opengm::MinSTCutBoost<size_t, float, opengm::KOLMOGOROV> MinStCutType;
      typedef opengm::QPBO<GraphicalModelType, MinStCutType> MinQPBO;
      MinQPBO::Parameter para;
      minTester.test<MinQPBO>(para);
   }
#endif
//}
//   QPBOTest test;
//   test.run();
   return 0;
}
