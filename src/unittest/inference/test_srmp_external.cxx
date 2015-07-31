#ifdef WITH_SRMP

#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/external/srmp.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>
#include <opengm/unittests/blackboxtests/blackboxtesthigherordergrid.hxx>
#endif
#include <iostream>

int main() {
#ifdef WITH_SRMP
   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;

   typedef opengm::meta::TypeListGenerator<
      opengm::ConstantFunction<ValueType, IndexType, LabelType>,
      opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
      opengm::PottsFunction<ValueType, IndexType, LabelType>,
      opengm::AbsoluteDifferenceFunction<ValueType, IndexType, LabelType>
      >::type FunctionTypeList;

   typedef opengm::GraphicalModel<ValueType, opengm::Adder, FunctionTypeList> GraphicalModelType;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
   typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;
   typedef opengm::BlackBoxTestHigherOrderGrid<GraphicalModelType> HigherOrderGridTest;

   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester;
   minTester.addTest(new GridTest(4, 4, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 1));
   minTester.addTest(new GridTest(3, 3, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new GridTest(3, 3, 2, false, false,GridTest::POTTS, opengm::OPTIMAL, 3));

   minTester.addTest(new StarTest(16, 2, false, true, StarTest::RANDOM, opengm::OPTIMAL, 1));
   minTester.addTest(new StarTest(8, 2, false, true, StarTest::RANDOM, opengm::OPTIMAL, 20));
   minTester.addTest(new StarTest(8, 4, true, true, StarTest::RANDOM, opengm::OPTIMAL, 20));

   minTester.addTest(new HigherOrderGridTest(3, 1, 3, 3, false, false, false, HigherOrderGridTest::RANDOM, opengm::OPTIMAL, 5));
   minTester.addTest(new HigherOrderGridTest(8, 4, 3, 4, false, true, false, HigherOrderGridTest::RANDOM, opengm::PASS, 5));
   minTester.addTest(new HigherOrderGridTest(8, 4, 3, 4, true, false, false, HigherOrderGridTest::RANDOM, opengm::PASS, 5));
   minTester.addTest(new HigherOrderGridTest(8, 4, 3, 4, true, true, true, HigherOrderGridTest::RANDOM, opengm::PASS, 5));

   std::cout << "Test SRMP External ..." << std::endl;
   typedef opengm::external::SRMP<GraphicalModelType> SRMP;
   SRMP::Parameter para;
   minTester.test<SRMP>(para);

   std::cout << "done!"<<std::endl;
#else
   std::cout << "SRMP External test is disabled (compiled without SRMP) "<< std::endl;
#endif
   return 0;
}
