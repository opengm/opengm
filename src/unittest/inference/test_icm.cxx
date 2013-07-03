#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/inference/icm.hxx>

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
   typedef opengm::BlackBoxTestGrid<SumGmType2> SumGridTest2;
   typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
   typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
   typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;

   opengm::InferenceBlackBoxTester<SumGmType> sumTester;
   sumTester.addTest(new SumGridTest(3, 3, 2, false, true, SumGridTest::POTTS, opengm::PASS, 1));
   sumTester.addTest(new SumFullTest(4,    3, false,    3, SumFullTest::POTTS, opengm::PASS, 1));
 
   opengm::InferenceBlackBoxTester<SumGmType2> sumTester2;
   sumTester2.addTest(new SumGridTest2(3, 3, 2, false, true, SumGridTest2::POTTS, opengm::PASS, 1));

   opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
   prodTester.addTest(new ProdGridTest(3, 3, 2, false, true, ProdGridTest::POTTS, opengm::PASS, 1));
   prodTester.addTest(new ProdFullTest(4,    3, false,    3, ProdFullTest::POTTS, opengm::PASS, 1));


   std::cout << "ICM  Tests ..." << std::endl;
   {
      std::cout << "  * Minimization/Adder  ..." << std::endl;
      typedef opengm::ICM<SumGmType, opengm::Minimizer> ICM;
      ICM::Parameter para;
      sumTester.test<ICM>(para);
      std::cout << " OK!"<<std::endl;
   } 
   {
      std::cout << "  * Minimization/Adder  (unsigned int) ..." << std::endl;
      typedef opengm::ICM<SumGmType2, opengm::Minimizer> ICM;
      ICM::Parameter para;
      sumTester2.test<ICM>(para);
      std::cout << " OK!"<<std::endl;
   }
   {
      std::cout << "  * Maximization/Multiplier  ..." << std::endl;
      typedef opengm::ICM<ProdGmType, opengm::Maximizer> ICM;
      ICM::Parameter para;
      prodTester.test<ICM>(para);
      std::cout << " OK!"<<std::endl;
   }
}



