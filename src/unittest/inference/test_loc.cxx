#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/inference/loc.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

int main() {

   typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;
   typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
   typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;
   typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
   typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
   typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;

   opengm::InferenceBlackBoxTester<SumGmType> sumTester;
   sumTester.addTest(new SumGridTest(5, 5, 2, false, true, SumGridTest::POTTS, opengm::PASS, 5));
   //sumTester.addTest(new SumFullTest(4,    3, false,    3, SumFullTest::POTTS, opengm::PASS, 5));

   //opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
   // prodTester.addTest(new ProdGridTest(3,4 , 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 5));
   //prodTester.addTest(new ProdFullTest(4,    3, false,    3, ProdFullTest::RANDOM, opengm::PASS, 5));
   
   //const size_t ad3Threshold=4;
   std::cout << "LOC -AD3 Tests ..." << std::endl;
   {
      std::cout << "  * Maximization/Adder  ..." << std::endl;
      typedef opengm::LOC<SumGmType, opengm::Maximizer> LOC;
      LOC::Parameter para("ad3",0.5,5,0);
      sumTester.test<LOC>(para);
      std::cout << " OK!"<<std::endl;
   }
   {
      std::cout << "  * Minimization/Adder  ..." << std::endl;
      typedef opengm::LOC<SumGmType, opengm::Minimizer> LOC;
      LOC::Parameter para("ad3",0.5,10,200);
      sumTester.test<LOC>(para);
      std::cout << " OK!"<<std::endl;
   }
   std::cout << "LOC -ASTAR Tests ..." << std::endl;
   {
      std::cout << "  * Maximization/Adder  ..." << std::endl;
      typedef opengm::LOC<SumGmType, opengm::Maximizer> LOC;
      LOC::Parameter para("dp",0.5,5,0);
      sumTester.test<LOC>(para);
      std::cout << " OK!"<<std::endl;
   }
   {
      std::cout << "  * Minimization/Adder  ..." << std::endl;
      typedef opengm::LOC<SumGmType, opengm::Minimizer> LOC;
      LOC::Parameter para("dp",0.5,10,200);
      sumTester.test<LOC>(para);
      std::cout << " OK!"<<std::endl;
   }
}



