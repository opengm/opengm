#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/inference/pbp.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

int main() {





   typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;

   std::vector<size_t> confusingSet(100,256);

   SumGmType gm(SumGmType::SpaceType(confusingSet.begin(),confusingSet.end()));



   opengm::detail_pbp::Priority<SumGmType> dpq(gm);


   confusingSet[10]=2;
   dpq.changeConfusingSetSize(10,256,2);
   OPENGM_TEST_EQUAL( dpq.highestPriorityVi(),10);

   confusingSet[55]=1;
   dpq.changeConfusingSetSize(55,256,1);
   OPENGM_TEST_EQUAL( dpq.highestPriorityVi(),55);


   confusingSet[55]=3;
   dpq.changeConfusingSetSize(56,256,3);
   OPENGM_TEST_EQUAL( dpq.highestPriorityVi(),55);

   OPENGM_TEST_EQUAL( dpq.getAndRemoveHighestPriorityVi(),55);
   OPENGM_TEST_EQUAL( dpq.getAndRemoveHighestPriorityVi(),10);
   OPENGM_TEST_EQUAL( dpq.getAndRemoveHighestPriorityVi(),56);




   
   typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;
   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   opengm::InferenceBlackBoxTester<SumGmType> sumTester;
   sumTester.addTest(new SumGridTest(10, 10, 1000, false, true, SumGridTest::RANDOM, opengm::PASS, 1));


   
   std::cout << "ICM  Tests ..." << std::endl;
   {
      std::cout << "  * Minimization/Adder  ..." << std::endl;
      typedef opengm::PBP<SumGmType, opengm::Minimizer> PBP;
      PBP::Parameter para;
      sumTester.test<PBP>(para);
      std::cout << " OK!"<<std::endl;
   } 
   
}



