#ifdef WITH_GUROBI
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>


#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>


#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/inference/lpgurobi.hxx>
#endif
#include <iostream>

int main(){
#ifdef WITH_GUROBI
   {
      typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

      opengm::InferenceBlackBoxTester<SumGmType> sumTester;
      sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 5));
      sumTester.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::PASS, 5));
      sumTester.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::PASS, 20));
      sumTester.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::PASS, 5));
 
      opengm::InferenceBlackBoxTester<SumGmType> sumTesterOpt;
      sumTesterOpt.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::OPTIMAL, 5));
      sumTesterOpt.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::OPTIMAL, 5));
      sumTesterOpt.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
      sumTesterOpt.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::OPTIMAL, 5));


      std::cout << "Gurobi Tests"<<std::endl;
      {
         std::cout << "  * Minimization/Adder LP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LPGurobi<GmType, opengm::Minimizer>    INF;
         INF::Parameter para;
         para.integerConstraint_ = false;
         sumTester.test<INF>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Minimization/Adder ILP ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GmType;
         typedef opengm::LPGurobi<GmType, opengm::Minimizer>    INF;
         INF::Parameter para;
         para.integerConstraint_ = true;
         sumTesterOpt.test<INF>(para);
         std::cout << " OK!"<<std::endl;
      }
      std::cout << "done!"<<std::endl;
   }
#else
   std::cout << "LpGurobiTest test is disabled (compiled without LpGurobi) "<< std::endl;
#endif
   return 0;
}



