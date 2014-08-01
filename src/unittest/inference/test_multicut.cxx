
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
   
   void testAsymetricMultiwayCut(){
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
      typedef opengm::Multicut<Model, opengm::Minimizer>                             Multicut;

      // Build empty Model
      LabelType numLabel = 2;
      size_t N=8,M=8;
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      
      // Add 1st order functions and factors to the model
      for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
         // construct 1st order function
         const LabelType shape[] = {gm.numberOfLabels(variable)};
         ExplicitFunction f(shape, shape + 1);
         f(0) = (rand()%10000) / 10000.0 - 0.5;
         f(1) = (rand()%10000) / 10000.0 - 0.5;
         // add function
         FunctionIdentifier id = gm.addFunction(f);
         // add factor
         IndexType variableIndex[] = {variable};
         gm.addFactor(id, variableIndex, variableIndex + 1);
      }
      // add 2nd order functions for all variables neighbored on the grid
      {
         // add a potts function to the model  
         IndexType vars[]  = {0,1}; 
         for(IndexType n=0; n<N;++n){
            for(IndexType m=0; m<M;++m){
               vars[0] = n + m*N;
               if(n+1<N){ //check for right neighbor
                  vars[1] =  (n+1) + (m  )*N;
                  OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered! 
                  double b = (rand()%10000) / 10000.0 - 0.5;
                  PottsFunction potts(numLabel, numLabel, 0.0, b);
                  FunctionIdentifier pottsid = gm.addFunction(potts);
                  gm.addFactor(pottsid, vars, vars + 2);
               } 
               if(m+1<M){ //check for lower neighbor
                  vars[1] =  (n  ) + (m+1)*N; 
                  OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
                  double b = (rand()%10000) / 10000.0 - 0.5;
                  PottsFunction potts(numLabel, numLabel, 0.0, b);
                  FunctionIdentifier pottsid = gm.addFunction(potts);
                  gm.addFactor(pottsid, vars, vars + 2);
               }
            }
         }
      }

      Multicut::Parameter para;
      para.allowCutsWithin_.resize(numLabel, false);
      para.allowCutsWithin_[0] = true;
      Multicut mc(gm,para);  
      mc.infer();
      std::vector<size_t> arg;
      std::vector<size_t> seg = mc.getSegmentation();
      mc.arg(arg);
   
      std::cout <<std::endl;
      for(size_t n=0; n<N; ++n){
         for(size_t m=0; m<M; ++m){
            std::cout <<arg[n+m*N]<< " ";
         }
         std::cout <<std::endl;
      }
      std::cout <<std::endl;
      for(size_t n=0; n<N; ++n){
         for(size_t m=0; m<M; ++m){
            std::cout <<seg[n+m*N]<< " ";
         }
         std::cout <<std::endl;
      }


      std::vector<size_t> map(N*M);
      for(size_t n=0; n<N; ++n){
         for(size_t m=0; m<M; ++m){
            map[seg[n+m*N]] = arg[n+m*N];
         }
      }
      for(size_t n=0; n<N; ++n){
         for(size_t m=0; m<M; ++m){
            OPENGM_ASSERT(map[seg[n+m*N]] == arg[n+m*N]);
         }
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
 
    std::cout << "  * Start Test for Asymetric Multiway Cut ..."<<std::endl;
    testAsymetricMultiwayCut();
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



