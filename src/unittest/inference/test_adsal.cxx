
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#include <opengm/inference/trws/trws_adsal.hxx>

int main() {
	   typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
	   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
	   typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
	   typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;

	   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester;
	   minTester.addTest(new GridTest(3, 2, 3, true, true, GridTest::RANDOM, opengm::PASS, 10));
	   minTester.addTest(new GridTest(3, 2, 2, true, true, GridTest::POTTS, opengm::OPTIMAL, 10));
	   minTester.addTest(new GridTest(3, 2, 3, true, false, GridTest::POTTS, opengm::FAIL, 1));
	   minTester.addTest(new FullTest(5,    5, true, 3,    FullTest::POTTS, opengm::PASS, 10));
	   minTester.addTest(new FullTest(4,    4, true, 2,    FullTest::POTTS, opengm::FAIL, 1));
	   minTester.addTest(new FullTest(5,    5, true, 3,    FullTest::RANDOM, opengm::PASS, 10));
	   minTester.addTest(new StarTest(6,    5, true, true,  StarTest::RANDOM, opengm::OPTIMAL, 3));

   std::cout << "Test ADSal ..." << std::endl;

   {
      typedef opengm::ADSal<GraphicalModelType,opengm::Minimizer> AdsalSolverType;
      AdsalSolverType::Parameter para(100);
      para.precision()=1e-12;
      minTester.test<AdsalSolverType>(para);
   }

   return 0;
}
