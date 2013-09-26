#ifdef WITH_MPLP
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/external/mplp.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>
#endif
#include <iostream>

int main() {
#ifdef WITH_MPLP
   typedef opengm::GraphicalModel<double, opengm::Adder > GraphicalModelType;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
   typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;

   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester;
   minTester.addTest(new GridTest(4, 4, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 1));
   minTester.addTest(new GridTest(3, 3, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new GridTest(3, 3, 2, false, false,GridTest::POTTS, opengm::OPTIMAL, 3));

   minTester.addTest(new StarTest(16, 2, false, true, StarTest::RANDOM, opengm::OPTIMAL, 1));
   minTester.addTest(new StarTest(8, 2, false, true, StarTest::RANDOM, opengm::OPTIMAL, 20));

   std::cout << "Test MPLP External ..." << std::endl;
   typedef opengm::external::MPLP<GraphicalModelType> MPLP;
   MPLP::Parameter para;
   minTester.test<MPLP>(para);

   std::cout << "done!"<<std::endl;
#else
   std::cout << "MPLP External test is disabled (compiled without MPLP) "<< std::endl;
#endif
   return 0;
}
