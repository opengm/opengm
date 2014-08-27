#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>

#include <opengm/inference/self_fusion.hxx>


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


      opengm::InferenceBlackBoxTester<SumGmType> sumTester;
      sumTester.addTest(new SumGridTest(20, 20, 6, false, true, SumGridTest::RANDOM, opengm::PASS, 10));
      sumTester.addTest(new SumGridTest(4, 4, 3, false, false,SumGridTest::RANDOM, opengm::PASS, 2));
      sumTester.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 2));
      sumTester.addTest(new SumFullTest(20,    3, false, 3,    SumFullTest::RANDOM, opengm::PASS, 2));
  
      //opengm::InferenceBlackBoxTester<SumGmType> sumTester2;
      //sumTester2.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));


      std::cout << "Self Fusion Tests"<<std::endl;

      {

         std::cout << "  * Self Fusion  Belief Propagation  Minimization/Adder with damping..."<<std::endl;
         typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Minimizer> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance> InfType;
         
         typedef opengm::SelfFusion<InfType> SelfFusionInf;



         InfType::Parameter infParam;
         SelfFusionInf::Parameter selfFuseInfParam(1,SelfFusionInf::LazyFlipperFusion,infParam);
         sumTester.test<SelfFusionInf>(selfFuseInfParam);
         std::cout << " OK!"<<std::endl;

      }



      std::cout << "done!"<<std::endl;
   }

}

