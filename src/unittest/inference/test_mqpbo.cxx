#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>


#include <opengm/inference/mqpbo.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

int main() {
   typedef opengm::GraphicalModel<double, opengm::Adder > AdderGmType;
   typedef opengm::BlackBoxTestGrid<AdderGmType> AdderGridTest;
   typedef opengm::BlackBoxTestFull<AdderGmType> AdderFullTest;
   typedef opengm::BlackBoxTestStar<AdderGmType> AdderStarTest;
   std::cout << "MQPBO Tests" << std::endl;
   {
      opengm::InferenceBlackBoxTester<AdderGmType> adderTester;
      adderTester.addTest(new AdderFullTest(1, 90, false, 1, AdderFullTest::RANDOM, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderGridTest(5, 4, 5, false, true, AdderGridTest::RANDOM, opengm::PASS, 2));
      adderTester.addTest(new AdderGridTest(4, 4, 5, false, false, AdderGridTest::POTTS, opengm::PASS, 2));
      adderTester.addTest(new AdderStarTest(10, 2, false, true, AdderStarTest::RANDOM, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderStarTest(5, 5, false, true, AdderStarTest::L1, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderStarTest(10, 10, false, true, AdderStarTest::L1, opengm::PASS, 2));
      adderTester.addTest(new AdderGridTest(3, 3, 3, false, true, AdderGridTest::L1, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderGridTest(5, 5, 4, false, true, AdderGridTest::L1, opengm::PASS, 2));
      adderTester.addTest(new AdderStarTest(10, 4, false, true, AdderStarTest::RANDOM, opengm::PASS, 20));
      adderTester.addTest(new AdderStarTest(10, 2, false, false, AdderStarTest::POTTS, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderStarTest(10, 4, false, true, AdderStarTest::POTTS, opengm::PASS, 20));
      adderTester.addTest(new AdderStarTest(6, 4, false, true, AdderStarTest::RANDOM, opengm::PASS, 4));
      adderTester.addTest(new AdderFullTest(5, 3, false, 3, AdderFullTest::POTTS, opengm::PASS, 20));
      //adderTester.addTest(new AdderGridTest(400, 400, 5, false, false, AdderGridTest::POTTS, opengm::PASS, 1));
    
      opengm::InferenceBlackBoxTester<AdderGmType> adderTester2;
      adderTester2.addTest(new AdderGridTest(4, 4, 5, false, false, AdderGridTest::POTTS, opengm::PASS, 2));
      adderTester2.addTest(new AdderStarTest(10, 2, false, false, AdderStarTest::POTTS, opengm::PASS, 20));
      adderTester2.addTest(new AdderStarTest(10, 4, false, true, AdderStarTest::POTTS, opengm::PASS, 20));
      adderTester2.addTest(new AdderFullTest(5, 3, false, 3, AdderFullTest::POTTS, opengm::PASS, 20));
      //adderTester.addTest(new AdderGridTest(400, 400, 5, false, false, AdderGridTest::POTTS, opengm::PASS, 1));
   
      {
         std::cout << "  * Minimization/Adder ..." << std::endl;
         typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
         typedef opengm::MQPBO<GmType,opengm::Minimizer> MQPBOType;
         MQPBOType::Parameter para;
         para.useKovtunsMethod_=false;
         para.rounds_=1;
         std::cout << "... without probing ..."<<std::endl;
         adderTester.test<MQPBOType > (para);
         //para.probing_=true;  
         //std::cout << "... with probing ..."<<std::endl;
         //adderTester.test<MQPBOType > (para); 
         std::cout << " OK!" << std::endl;
         para.useKovtunsMethod_=true;
         para.rounds_=0;
         //para.probing_=false;
         std::cout << "... with Kovtuns method ..."<<std::endl;
         adderTester2.test<MQPBOType > (para);   
      }

    }
   return 0;
}



