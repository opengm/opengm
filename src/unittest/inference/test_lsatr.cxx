#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/inference/lsatr.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

int main() {
   typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;
   typedef opengm::GraphicalModel<float, opengm::Adder, opengm::ExplicitFunction<float,unsigned int, unsigned char>, opengm::DiscreteSpace<unsigned int, unsigned char> > SumGmType2;

   typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
   typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

   opengm::InferenceBlackBoxTester<SumGmType> sumTester;
   sumTester.addTest(new SumGridTest(9, 9, 2, false, true, SumGridTest::POTTS, opengm::PASS, 10));
   sumTester.addTest(new SumFullTest(8,    2, false,    3, SumFullTest::POTTS, opengm::PASS, 10));
   sumTester.addTest(new SumGridTest(9, 9, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 10));
   sumTester.addTest(new SumFullTest(8,    2, false,    3, SumFullTest::RANDOM, opengm::PASS, 10));
   sumTester.addTest(new SumFullTest(100,  2, false,    3, SumFullTest::RANDOM, opengm::PASS, 1));
 



   std::cout << "LSA-TR  Tests ..." << std::endl;
   {
      std::cout << "  * Minimization/Adder  ..." << std::endl;
      typedef opengm::LSA_TR<SumGmType, opengm::Minimizer> LSATR;
      LSATR::Parameter para;
      sumTester.test<LSATR>(para);
      std::cout << " OK!"<<std::endl;
   } 
 
}



