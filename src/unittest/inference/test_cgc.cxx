
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
#include <opengm/inference/bruteforce.hxx>

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
      tester.addTest(new GridTest(2, 2, 4, false, false,  GridTest::POTTS, opengm::OPTIMAL, 5));
      tester.addTest(new GridTest(2, 2, 4, false, false,  GridTest::RPOTTS, opengm::OPTIMAL, 5));
      tester.addTest(new GridTest(2, 1, 2, false, false,  GridTest::RPOTTS, opengm::OPTIMAL, 5));
      tester.addTest(new GridTest(3, 3, 9, false, false,  GridTest::POTTS, opengm::PASS, 5));
      tester.addTest(new GridTest(6, 6, 36, false, false,  GridTest::POTTS, opengm::PASS, 5));
     
      opengm::InferenceBlackBoxTester<GmType2> tester2; 
      tester2.addTest(new GridTest2(2, 2, 4, false, false,  GridTest2::POTTS, opengm::OPTIMAL, 5));
      tester2.addTest(new GridTest2(2, 2, 4, false, false,  GridTest2::RPOTTS, opengm::OPTIMAL, 5));
      tester2.addTest(new GridTest2(2, 1, 2, false, false,  GridTest2::RPOTTS, opengm::OPTIMAL, 5));
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


  void testSmall(){ 
     typedef double                                                                 ValueType;          // type used for values
     typedef size_t                                                                 IndexType;          // type used for indexing nodes and factors (default : size_t)
     typedef size_t                                                                 LabelType;          // type used for labels (default : size_t)
     typedef opengm::Adder                                                          OpType;             // operation used to combine terms
     typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                ExplicitFunction;   // shortcut for explicite function 
     typedef opengm::PottsFunction<ValueType,IndexType,LabelType>                   PottsFunction;      // shortcut for Potts function
     typedef opengm::meta::TypeListGenerator<ExplicitFunction,PottsFunction>::type  FunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
     typedef opengm::DiscreteSpace<IndexType, LabelType>                            SpaceType;          // type used to define the feasible statespace
     typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>    Model;              // type of the model
     typedef Model::FunctionIdentifier                                              FunctionIdentifier; // type of the function identifier  

     IndexType N = 2;
     IndexType M = 2; 
     LabelType numLabel = N*M;
     std::vector<LabelType> numbersOfLabels(N*M,numLabel);
     Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
   
     IndexType vars[]  = {0,1}; 
     for(IndexType n=0; n<N;++n){
        for(IndexType m=0; m<M;++m){
           vars[0] = n + m*N;
           if(n+1<N){ //check for right neighbor
              vars[1] =  (n+1) + (m  )*N;
              OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
              PottsFunction potts(numLabel, numLabel, 0.0, (rand()%200) - 100.0);
              gm.addFactor( gm.addFunction(potts), vars, vars + 2);
           } 
           if(m+1<M){ //check for lower neighbor
              vars[1] =  (n  ) + (m+1)*N; 
              OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered! 
              PottsFunction potts(numLabel, numLabel, 0.0, (rand()%200) - 100.0);
              gm.addFactor( gm.addFunction(potts), vars, vars + 2);
           }
        }
     }  

     typedef opengm::CGC<Model, opengm::Minimizer> CGC;
     typedef opengm::Bruteforce<Model, opengm::Minimizer> BF;
     typename CGC::Parameter para;
     para.planar_=true;

     CGC cgc(gm,para);
     cgc.infer();
     std::vector<LabelType> l;
     cgc.arg(l);

     BF bf(gm);
     bf.infer();
     std::vector<LabelType> l2;
     bf.arg(l2);
     OPENGM_TEST(gm.evaluate(l) ==  gm.evaluate(l2));
  }  

  void run(){
    std::cout <<std::endl;  
    std::cout << "  * Start Black-Box Tests for Min-Sum (CGC)..."<<std::endl;
    testUnsupervisedCase<opengm::Minimizer,opengm::Adder>(); 
    std::cout << "  * Start Black-Box Tests for small model..."<<std::endl;
    for(size_t i=0; i<20; ++i){
       testSmall();
    } 
    std::cout << "PASS!"<<std::endl;
  }
};

int main(){
   CGCTest t; 
   t.run();
   return 0;
}



