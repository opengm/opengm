
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#ifdef WITH_CPLEX
#include <opengm/inference/multicut.hxx>

struct MulticutTest
{
  template<class ACC, class OP>
  void testException(){
    {
      typedef opengm::GraphicalModel<float, OP> GmType;
      typedef opengm::BlackBoxTestGrid<GmType> GridTest;
      opengm::InferenceBlackBoxTester<GmType> tester;
      tester.addTest(new GridTest(4, 4, 3, false, true, GridTest::RANDOM, opengm::FAIL, 1));
      tester.addTest(new GridTest(4, 4, 2, true, true, GridTest::POTTS, opengm::FAIL, 1));

      typedef opengm::Multicut<GmType, ACC> Multicut;
      typename Multicut::Parameter para;
      tester.template test<Multicut>(para);
    }
  }
   template<class ACC, class OP>
   void testException2(){
    {
      typedef opengm::GraphicalModel<float, OP> GmType;
      typedef opengm::BlackBoxTestGrid<GmType> GridTest;
      opengm::InferenceBlackBoxTester<GmType> tester;
      tester.addTest(new GridTest(4, 4, 8, false, true, GridTest::RANDOM, opengm::FAIL, 1));
     
      typedef opengm::Multicut<GmType, ACC> Multicut;
      typename Multicut::Parameter para;
      tester.template test<Multicut>(para);
    }
  }

  template<class ACC, class OP>
  void testSupervisedCase(){
    {
      typedef opengm::GraphicalModel<float,OP> GmType; 
      typedef opengm::BlackBoxTestGrid<GmType> GridTest;
      typedef opengm::BlackBoxTestFull<GmType> FullTest;
      typedef opengm::BlackBoxTestStar<GmType> StarTest;
      opengm::InferenceBlackBoxTester<GmType> tester;
      tester.addTest(new GridTest(4, 4, 2, false, true, GridTest::RANDOM, opengm::OPTIMAL, 1));
      tester.addTest(new GridTest(3, 3, 3, false, true,  GridTest::POTTS, opengm::OPTIMAL, 5));
      tester.addTest(new GridTest(3, 3, 3, false, false, GridTest::POTTS, opengm::OPTIMAL, 5));
      tester.addTest(new FullTest(5,    3, false, 3,     FullTest::POTTS, opengm::OPTIMAL, 5));
      tester.addTest(new FullTest(5,    3, false, 2,     FullTest::POTTS, opengm::OPTIMAL, 5));

      typedef opengm::Multicut<GmType, ACC> Multicut;
      typename Multicut::Parameter para;
      tester.template test<Multicut>(para);
    }
  }

  template<class ACC,class OP>
  void testUnsupervisedCase(){
    {
      typedef opengm::GraphicalModel<float, OP> GmType;
      typedef opengm::BlackBoxTestGrid<GmType> GridTest;
      typedef opengm::BlackBoxTestFull<GmType> FullTest;
      typedef opengm::BlackBoxTestStar<GmType> StarTest;
      typedef opengm::GraphicalModel<float, opengm::Adder, 
              opengm::ExplicitFunction<float,unsigned short, unsigned char>, 
              opengm::DiscreteSpace<unsigned short, unsigned char> > GmType2;
      typedef opengm::BlackBoxTestGrid<GmType2> GridTest2;
      typedef opengm::BlackBoxTestFull<GmType2> FullTest2;
      typedef opengm::BlackBoxTestStar<GmType2> StarTest2;
     
      opengm::InferenceBlackBoxTester<GmType> tester; 
      tester.addTest(new GridTest(4, 4, 2, false, false,  GridTest::RANDOM, opengm::OPTIMAL, 5));
      tester.addTest(new GridTest(2, 2, 4, false, false,  GridTest::POTTS, opengm::OPTIMAL, 5));
      tester.addTest(new GridTest(3, 3, 9, false, false,  GridTest::POTTS, opengm::PASS, 5));
      tester.addTest(new FullTest(5,    5, false, 2,     FullTest::POTTS, opengm::OPTIMAL, 5)); 

      opengm::InferenceBlackBoxTester<GmType2> tester2; 
      tester2.addTest(new GridTest2(2, 2, 4, false, false,  GridTest2::POTTS, opengm::OPTIMAL, 5));
      tester2.addTest(new GridTest2(3, 3, 9, false, false,  GridTest2::POTTS, opengm::PASS, 5));
      tester2.addTest(new FullTest2(5,    5, false, 2,     FullTest2::POTTS, opengm::OPTIMAL, 5));

      typedef opengm::Multicut<GmType, ACC> Multicut;
      typename Multicut::Parameter para;
      tester.template test<Multicut>(para); 

      typedef opengm::Multicut<GmType2, ACC> Multicut2;
      typename Multicut2::Parameter para2;
      tester2.template test<Multicut2>(para2);
    }
  }
   

  void run(){
    std::cout <<std::endl;

    std::cout << "  * Start Black-Box Tests for Min-Sum (Multiwaycut)..."<<std::endl;
    testSupervisedCase<opengm::Minimizer,opengm::Adder>();
  
    std::cout << "  * Start Black-Box Tests for Min-Sum (Multicut)..."<<std::endl;
    testUnsupervisedCase<opengm::Minimizer,opengm::Adder>();

    std::cout << "  * Start Black-Box Tests for Min-Sum..."<<std::endl;
    testException2<opengm::Minimizer,opengm::Adder>();

    std::cout << "  * Start Black-Box Tests for Max-Sum..."<<std::endl;
    testException<opengm::Maximizer,opengm::Adder>();

    std::cout << "  * Start Black-Box Tests for Sum-Sum..."<<std::endl;
    testException<opengm::Integrator,opengm::Adder>();

    std::cout << "  * Start Black-Box Tests for Min-Prod..."<<std::endl;
    testException<opengm::Minimizer,opengm::Multiplier>();

    std::cout << "  * Start Black-Box Tests for Max-Prod..."<<std::endl;
    testException<opengm::Maximizer,opengm::Multiplier>();

    std::cout << "  * Start Black-Box Tests for Sum-Prod..."<<std::endl;
    testException<opengm::Integrator,opengm::Multiplier>();
  }
};
#endif

int main(){
#ifdef WITH_CPLEX
   MulticutTest t; 
   t.run();
   return 0;
#endif
   std::cout << "Multicut test is disabled (compiled without LpCplex) "<< std::endl;
   return 0;
}



