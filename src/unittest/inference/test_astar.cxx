
#include <vector>
#include <opengm/opengm.hxx>

#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/astar.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

int main() {
   {
      typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
      typedef opengm::GraphicalModel<float, opengm::Adder, 
              opengm::ExplicitFunction<float,unsigned short, unsigned char>, 
              opengm::DiscreteSpace<unsigned short, unsigned char> > SumGmType2;
      typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest; 
      typedef opengm::BlackBoxTestGrid<SumGmType2> SumGridTest2;
    
      typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
      typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
      typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;

      opengm::InferenceBlackBoxTester<SumGmType> sumTester;
      sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::OPTIMAL, 1));
      sumTester.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::OPTIMAL, 1));
      sumTester.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 1));
      sumTester.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::OPTIMAL, 20));
  
      opengm::InferenceBlackBoxTester<SumGmType2> sumTester2;
      sumTester2.addTest(new SumGridTest2(4, 4, 2, false, true, SumGridTest2::RANDOM, opengm::OPTIMAL, 1));
   
      opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
      prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::OPTIMAL, 1));
      prodTester.addTest(new ProdGridTest(4, 4, 2, false, false,ProdGridTest::RANDOM, opengm::OPTIMAL, 1));
      prodTester.addTest(new ProdStarTest(6,    4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 1));
      prodTester.addTest(new ProdFullTest(5,    2, false, 3,    ProdFullTest::RANDOM, opengm::OPTIMAL, 1));

      std::cout << "AStar Tests"<<std::endl;
      {
         std::cout << "  * Minimization/Adder with fast heuristic ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GraphicalModelType;
         typedef opengm::AStar<GraphicalModelType, opengm::Minimizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.FASTHEURISTIC;
         sumTester.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
      } 
      {
         std::cout << "  * Minimization/Adder with fast heuristic ... (float,uint16,uint8)"<<std::endl;
         typedef opengm::AStar<SumGmType2, opengm::Minimizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.FASTHEURISTIC;
         sumTester2.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Adder with fast heuristic ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GraphicalModelType;
         typedef opengm::AStar<GraphicalModelType, opengm::Maximizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.FASTHEURISTIC;
         sumTester.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Multiplier with fast heuristic ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
         typedef opengm::AStar<GraphicalModelType, opengm::Maximizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.FASTHEURISTIC;
         prodTester.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
       }
       {
         std::cout << "  * Minimization/Multiplier with fast heuristic ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
         typedef opengm::AStar<GraphicalModelType, opengm::Minimizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.FASTHEURISTIC;
         prodTester.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
       }
       std::cout << "  TEST ..."<<std::endl;
       {
         std::cout << "  * Minimization/Adder with standart heuristic ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GraphicalModelType;
         typedef opengm::AStar<GraphicalModelType, opengm::Minimizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.STANDARDHEURISTIC;
         sumTester.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Adder with standart heuristic ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder > GraphicalModelType;
         typedef opengm::AStar<GraphicalModelType, opengm::Maximizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.STANDARDHEURISTIC;
         sumTester.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Multiplier with standart heuristic ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
         typedef opengm::AStar<GraphicalModelType, opengm::Maximizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.STANDARDHEURISTIC;
         prodTester.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
       }
       {
         std::cout << "  * Minimization/Multiplier with standart heuristic ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
         typedef opengm::AStar<GraphicalModelType, opengm::Minimizer>            ASTAR;
         ASTAR::Parameter para;
         para.heuristic_ =  para.STANDARDHEURISTIC;
         prodTester.test<ASTAR>(para);
         std::cout << " OK!"<<std::endl;
       }
       std::cout << "done!"<<std::endl;
   }
}


