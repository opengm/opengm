#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/alphaexpansionfusion.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>


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

   std::cout << "Test Alpha-Expansion-Fusion ..." << std::endl;
   std::cout << "  * Test Min-Sum" << std::endl;
   {

      typedef opengm::AlphaExpansionFusion<GraphicalModelType, opengm::Minimizer> MinAlphaExpansionFusion;
      MinAlphaExpansionFusion::Parameter para;
      minTester.test<MinAlphaExpansionFusion>(para);
   } 
/*
   std::cout << "  * Test Max-Sum " << std::endl;
   {
      typedef opengm::AlphaExpansionFusion<GraphicalModelType, opengm::Maximizer> MaxAlphaExpansionFusion;
      MaxAlphaExpansionFusion::Parameter para;
      maxTester.test<MaxAlphaExpansionFusion>(para);
   }
*/
   return 0;
}

