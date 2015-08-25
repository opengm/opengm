
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


#include <opengm/functions/potts.hxx>
#include <opengm/inference/cgc.hxx>

struct CGCTest
{
  template<class ACC,class OP>
  void testUnsupervisedCase(){
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
      tester.addTest(new GridTest(2, 2, 4, false, false,  GridTest::POTTS, opengm::PASS, 5));
      tester.addTest(new GridTest(3, 3, 9, false, false,  GridTest::POTTS, opengm::PASS, 5));
      tester.addTest(new GridTest(6, 6, 36, false, false,  GridTest::POTTS, opengm::PASS, 5));
     
      opengm::InferenceBlackBoxTester<GmType2> tester2; 
      tester2.addTest(new GridTest2(2, 2, 4, false, false,  GridTest2::POTTS, opengm::PASS, 5));
      tester2.addTest(new GridTest2(3, 3, 9, false, false,  GridTest2::POTTS, opengm::PASS, 5));
      tester2.addTest(new GridTest2(6, 6, 36, false, false,  GridTest2::POTTS, opengm::PASS, 5));
   
      typedef opengm::CGC<GmType, ACC> CGC;
      typename CGC::Parameter para;
      tester.template test<CGC>(para); 
      para.planar_=true;
      tester.template test<CGC>(para); 

      typedef opengm::CGC<GmType2, ACC> CGC2;
      typename CGC2::Parameter para2;
      tester2.template test<CGC2>(para2); 
      para2.planar_=true;
      tester2.template test<CGC2>(para2); 
  }      

  void run(){
    std::cout <<std::endl;  
    std::cout << "  * Start Black-Box Tests for Min-Sum (Multicut)..."<<std::endl;
    testUnsupervisedCase<opengm::Minimizer,opengm::Adder>();
  }
};

int main(){
   CGCTest t; 
   t.run();
   return 0;
}



