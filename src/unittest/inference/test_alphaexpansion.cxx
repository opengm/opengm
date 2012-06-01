#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphaexpansion.hxx>

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

int main() {
   typedef opengm::GraphicalModel<float, opengm::Adder > GraphicalModelType;
   typedef opengm::GraphicalModel<float, opengm::Adder, opengm::ExplicitFunction<float,unsigned int, unsigned char>, opengm::DiscreteSpace<unsigned int, unsigned char> > GraphicalModelType2;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
   typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
   typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType2> GridTest2;

   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester;
   minTester.addTest(new GridTest(4, 4, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 1));
   minTester.addTest(new GridTest(3, 3, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new GridTest(3, 3, 2, false, false,GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new StarTest(5,    2, false, true, StarTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new FullTest(5,    2, false, 3,    FullTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new GridTest(4, 4, 9, false, true, GridTest::POTTS, opengm::PASS,   10));
   minTester.addTest(new GridTest(4, 4, 9, false, false,GridTest::POTTS, opengm::PASS,   10));
   minTester.addTest(new FullTest(6,    4, false, 3,    FullTest::POTTS, opengm::PASS,   10));

   opengm::InferenceBlackBoxTester<GraphicalModelType2> minTester2;
   minTester2.addTest(new GridTest2(4, 4, 2, false, true, GridTest2::POTTS, opengm::OPTIMAL, 1));
  
   opengm::InferenceBlackBoxTester<GraphicalModelType> maxTester;
   maxTester.addTest(new GridTest(4, 4, 2, false, true, GridTest::IPOTTS, opengm::OPTIMAL, 1));
   maxTester.addTest(new GridTest(3, 3, 2, false, true, GridTest::IPOTTS, opengm::OPTIMAL, 3));
   maxTester.addTest(new GridTest(3, 3, 2, false, false,GridTest::IPOTTS, opengm::OPTIMAL, 3));
   maxTester.addTest(new StarTest(5,    2, false, true, StarTest::IPOTTS, opengm::OPTIMAL, 3));
   maxTester.addTest(new FullTest(5,    2, false, 3,    FullTest::IPOTTS, opengm::OPTIMAL, 3));
   maxTester.addTest(new GridTest(4, 4, 9, false, true, GridTest::IPOTTS, opengm::PASS,   10));
   maxTester.addTest(new GridTest(4, 4, 9, false, false,GridTest::IPOTTS, opengm::PASS,   10));
   maxTester.addTest(new FullTest(6,    4, false, 3,    FullTest::IPOTTS, opengm::PASS,   10));

   std::cout << "Test Alpha-Expansion ..." << std::endl;

#ifdef WITH_MAXFLOW
   std::cout << "  * Test Min-Sum with Kolmogorov" << std::endl;
   {
      typedef opengm::external::MinSTCutKolmogorov<size_t, float> MinStCutType;
      typedef opengm::GraphCut<GraphicalModelType, opengm::Minimizer, MinStCutType> MinGraphCut;
      typedef opengm::AlphaExpansion<GraphicalModelType, MinGraphCut> MinAlphaExpansion;
      MinAlphaExpansion::Parameter para;
      minTester.test<MinAlphaExpansion>(para);
   } 

   std::cout << "  * Test Min-Sum with Kolmogorov (float, uint16,uint8)" << std::endl;
   {
      typedef opengm::external::MinSTCutKolmogorov<size_t, float> MinStCutType;
      typedef opengm::GraphCut<GraphicalModelType2, opengm::Minimizer, MinStCutType> MinGraphCut;
      typedef opengm::AlphaExpansion<GraphicalModelType2, MinGraphCut> MinAlphaExpansion;
      MinAlphaExpansion::Parameter para;
      minTester2.test<MinAlphaExpansion>(para);
   }
#endif

#ifdef WITH_BOOST
   std::cout << "  * Test Min-Sum with BOOST-Push-Relabel" << std::endl;
   {
      typedef opengm::MinSTCutBoost<size_t, float, opengm::PUSH_RELABEL> MinStCutType;
      typedef opengm::GraphCut<GraphicalModelType, opengm::Minimizer, MinStCutType> MinGraphCut;
      typedef opengm::AlphaExpansion<GraphicalModelType, MinGraphCut> MinAlphaExpansion;
      MinAlphaExpansion::Parameter para;
      minTester.test<MinAlphaExpansion>(para);
   }
   std::cout << "  * Test Min-Sum with BOOST-Edmonds-Karp" << std::endl;
   {
      typedef opengm::MinSTCutBoost<size_t, float, opengm::EDMONDS_KARP> MinStCutType;
      typedef opengm::GraphCut<GraphicalModelType, opengm::Minimizer, MinStCutType> MinGraphCut;
      typedef opengm::AlphaExpansion<GraphicalModelType, MinGraphCut> MinAlphaExpansion;
      MinAlphaExpansion::Parameter para;
      minTester.test<MinAlphaExpansion>(para);
   }
   std::cout << "  * Test Min-Sum with BOOST-Kolmogorov" << std::endl;
   {
      typedef opengm::MinSTCutBoost<size_t, float, opengm::KOLMOGOROV> MinStCutType;
      typedef opengm::GraphCut<GraphicalModelType, opengm::Minimizer, MinStCutType> MinGraphCut;
      typedef opengm::AlphaExpansion<GraphicalModelType, MinGraphCut> MinAlphaExpansion;
      MinAlphaExpansion::Parameter para;
      minTester.test<MinAlphaExpansion>(para);
   }
   {
      typedef opengm::MinSTCutBoost<size_t, float, opengm::KOLMOGOROV> MinStCutType;
      typedef opengm::GraphCut<GraphicalModelType, opengm::Minimizer, MinStCutType> MinGraphCut;
      typedef opengm::AlphaExpansion<GraphicalModelType, MinGraphCut> MinAlphaExpansion;
      {
         std::cout << "  * Test Min-Sum with BOOST-Kolmogorov with random label initialization" << std::endl;
         MinAlphaExpansion::Parameter para;
         para.labelInitialType_=  MinAlphaExpansion::Parameter::RANDOM_LABEL;
         minTester.test<MinAlphaExpansion>(para);
      }
      {
         std::cout << "  * Minimization/Adder with GraphCut-Push-Relabel with local optimal label initialization..."<<std::endl;
         MinAlphaExpansion::Parameter para;
         para.labelInitialType_=  MinAlphaExpansion::Parameter::LOCALOPT_LABEL;
         minTester.test<MinAlphaExpansion>(para);
      }
      {
         std::cout << "  * Minimization/Adder with GraphCut-Push-Relabel with random order..."<<std::endl;
         MinAlphaExpansion::Parameter para;
         para.orderType_=  MinAlphaExpansion::Parameter::RANDOM_ORDER;
         para.randSeedOrder_=0;
         minTester.test<MinAlphaExpansion>(para);
      }
      {
         std::cout << "  * Minimization/Adder with GraphCut-Push-Relabel with fixed random order..."<<std::endl;
         MinAlphaExpansion::Parameter para;
         para.orderType_=  MinAlphaExpansion::Parameter::RANDOM_ORDER;
         para.randSeedOrder_=8;
         minTester.test<MinAlphaExpansion>(para);
      }
   }
#endif

#ifdef WITH_MAXFLOW
   std::cout << "  * Test Max-Sum with Kolmogorov" << std::endl;
   {
      typedef opengm::external::MinSTCutKolmogorov<size_t, float> MinStCutType;
      typedef opengm::GraphCut<GraphicalModelType, opengm::Maximizer, MinStCutType> MinGraphCut;
      typedef opengm::AlphaExpansion<GraphicalModelType, MinGraphCut> MinAlphaExpansion;
      MinAlphaExpansion::Parameter para;
      maxTester.test<MinAlphaExpansion>(para);
   }
#endif

   return 0;
}

