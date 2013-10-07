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

#include <opengm/inference/block_icm.hxx>


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
      sumTester.addTest(new SumGridTest(100, 100, 6, false, true, SumGridTest::RANDOM, opengm::PASS, 1));

      //opengm::InferenceBlackBoxTester<SumGmType> sumTester2;
      //sumTester2.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));


      std::cout << "Blocked ICM"<<std::endl;

      {

         std::cout << "  * Blocked Belief Propagation  Minimization/Adder with damping..."<<std::endl;
         typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
         typedef opengm::BlockedSolverHelper<GraphicalModelType,opengm::Minimizer> Helper;

         typedef Helper::SubGmType SubGmType;
         typedef opengm::BeliefPropagationUpdateRules<SubGmType,opengm::Minimizer> UpdateRulesType;
         typedef opengm::MessagePassing<SubGmType, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance> InfType;
         
         typedef opengm::BlockedSolver<GraphicalModelType,InfType> BlockedInfType;



         InfType::Parameter infParam(2,0,0.5);
         BlockedInfType::Parameter param;
         param.bisectionsLevels_=7;
         param.blockInfParam_=infParam;
         sumTester.test<BlockedInfType>(param);
         //std::cout << " OK!"<<std::endl;

      }



      std::cout << "done!"<<std::endl;
   }

}

