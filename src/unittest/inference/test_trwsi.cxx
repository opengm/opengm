
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

#include <opengm/inference/trws/trws_trws.hxx>

int main() {
   typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
   typedef opengm::GraphicalModel<float, opengm::Adder, opengm::ExplicitFunction<float,unsigned int, unsigned char>, opengm::DiscreteSpace<unsigned int, unsigned char> >  GraphicalModelType2;

   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
   typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
   typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;

   typedef opengm::BlackBoxTestGrid<GraphicalModelType2> GridTest2;
   typedef opengm::BlackBoxTestFull<GraphicalModelType2> FullTest2;
   typedef opengm::BlackBoxTestStar<GraphicalModelType2> StarTest2;

   bool randomLabelSize = true;
   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester;
   minTester.addTest(new GridTest(3, 2, 3, randomLabelSize, true, GridTest::RANDOM, opengm::PASS, 10));
   minTester.addTest(new GridTest(3, 2, 2, randomLabelSize, true, GridTest::POTTS, opengm::OPTIMAL, 10));
   minTester.addTest(new GridTest(3, 2, 3, randomLabelSize, false, GridTest::POTTS, opengm::FAIL, 1));
   minTester.addTest(new FullTest(5,    5, randomLabelSize, 3,    FullTest::POTTS, opengm::PASS, 10));
   minTester.addTest(new FullTest(4,    4, randomLabelSize, 2,    FullTest::POTTS, opengm::FAIL, 1));
   minTester.addTest(new FullTest(5,    5, randomLabelSize, 3,    FullTest::RANDOM, opengm::PASS, 10));
   minTester.addTest(new StarTest(6,    5, randomLabelSize, true,  StarTest::RANDOM, opengm::OPTIMAL, 3));

   opengm::InferenceBlackBoxTester<GraphicalModelType2> minTester2;
   minTester2.addTest(new GridTest2(3, 2, 3, randomLabelSize, true, GridTest2::RANDOM, opengm::PASS, 10));
   minTester2.addTest(new GridTest2(3, 2, 2, randomLabelSize, true, GridTest2::POTTS, opengm::OPTIMAL, 10));
   minTester2.addTest(new GridTest2(3, 2, 3, randomLabelSize, false, GridTest2::POTTS, opengm::FAIL, 1));
   minTester2.addTest(new FullTest2(5,    5, randomLabelSize, 3,    FullTest2::POTTS, opengm::PASS, 10));
   minTester2.addTest(new FullTest2(4,    4, randomLabelSize, 2,    FullTest2::POTTS, opengm::FAIL, 1));
   minTester2.addTest(new FullTest2(5,    5, randomLabelSize, 3,    FullTest2::RANDOM, opengm::PASS, 10));
   minTester2.addTest(new StarTest2(6,    5, randomLabelSize, true,  StarTest2::RANDOM, opengm::OPTIMAL, 3));


   std::cout << "Test TRWSi ..." << std::endl;

   {
      typedef opengm::TRWSi<GraphicalModelType,opengm::Minimizer> TRWSiSolverType;
      TRWSiSolverType::Parameter para(100);
      para.precision_=1e-12;
      minTester.test<TRWSiSolverType>(para);
   }

   {
      typedef opengm::TRWSi<GraphicalModelType2,opengm::Minimizer> TRWSiSolverType;
      TRWSiSolverType::Parameter para(100);
      para.precision_=1e-12;
      minTester2.test<TRWSiSolverType>(para);
   }


   return 0;
}





