#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/dynamicprogramming.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

int main() {
   {
      typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
      typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;
      typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
      typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
      typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;

      opengm::InferenceBlackBoxTester<SumGmType> sumTester;    
      sumTester.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
      sumTester.addTest(new SumStarTest(4,    3, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 10));
      sumTester.addTest(new SumStarTest(1000,  10, false, true, SumStarTest::RANDOM, opengm::PASS, 10));
     
      opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
      prodTester.addTest(new ProdStarTest(6,   4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 20));
      prodTester.addTest(new ProdStarTest(4,    3, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 10));
      prodTester.addTest(new ProdStarTest(1000,  10, false, true, ProdStarTest::RANDOM, opengm::PASS, 10));
     
      std::cout << "Dynamic Programming Tests"<<std::endl;
      {
         std::cout << "  * Minimization/Adder ..."<<std::endl;
         typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
         typedef opengm::DynamicProgramming<GraphicalModelType, opengm::Minimizer> BP;
         BP::Parameter para;
         sumTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Adder ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder> GraphicalModelType;
         typedef opengm::DynamicProgramming<GraphicalModelType, opengm::Maximizer>            BP;
         BP::Parameter para;
         sumTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Multiplier ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
         typedef opengm::DynamicProgramming<GraphicalModelType, opengm::Maximizer>  BP;
         BP::Parameter para;
         prodTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
       }
       std::cout << "done!"<<std::endl;
   }
};
