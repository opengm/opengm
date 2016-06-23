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

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>


void testOperations() {
   size_t shape[] ={2,2};
   marray::Marray<double> M(shape,shape+2);
   marray::Marray<double> M2(shape,shape+2);
   marray::Marray<double> M3(shape,shape+2);
   M(0,0)=2;
   M(1,0)=4;
   M(0,1)=3;
   M(1,1)=5;
   opengm::messagepassingOperations::normalize<opengm::Adder,opengm::Minimizer>(M);
   OPENGM_ASSERT(M(0,0)==0);
   OPENGM_ASSERT(M(1,0)==2);
   OPENGM_ASSERT(M(0,1)==1);
   OPENGM_ASSERT(M(1,1)==3);
   opengm::messagepassingOperations::clean<opengm::Adder>(M);
   OPENGM_ASSERT(M(0,0)==0);
   OPENGM_ASSERT(M(1,0)==0);
   OPENGM_ASSERT(M(0,1)==0);
   OPENGM_ASSERT(M(1,1)==0);

   M(0,0)=1;
   M(1,0)=0;
   M(0,1)=10;
   M(1,1)=1;

   M2(0,0)=0;
   M2(1,0)=1;
   M2(0,1)=1;
   M2(1,1)=1; 
   opengm::messagepassingOperations::weightedMean<opengm::Adder>(M,M2,0.25,M3);
   OPENGM_ASSERT(M3(0,0)==0.25);
   OPENGM_ASSERT(M3(1,0)==0.75);
   OPENGM_TEST_EQUAL_TOLERANCE(M3(0,1),3.25,0.00001);
   OPENGM_ASSERT(M3(1,1)==1);
}


void testSumProd(){
   typedef opengm::GraphicalModel<double,opengm::Multiplier,opengm::ExplicitFunction<double, size_t, size_t>,opengm::DiscreteSpace<size_t, size_t> > Model;
   typedef Model::IndependentFactorType IndependentFactor;

   size_t n_var=3;
   int n_stats[]={2,2,2};

   opengm::DiscreteSpace<size_t,size_t> space;
   for(int i=0;i<n_var;i++)
   {
      space.addVariable(n_stats[i]);   
   }
   Model model(space);

   opengm::ExplicitFunction<double> f1(n_stats,n_stats+2,0);
   opengm::ExplicitFunction<double> f2(n_stats,n_stats+2,0);
   opengm::ExplicitFunction<double> f3(n_stats,n_stats+2,0);
   f1(0,0)=0.2;
   f1(0,1)=0.1;
   f1(1,0)=0.3;
   f1(1,1)=0.4;
   f2(0,0)=0.22;
   f2(0,1)=0.25;
   f2(1,0)=0.35;
   f2(1,1)=0.18;
   f3(0,0)=0.32;
   f3(0,1)=0.28;
   f3(1,0)=0.14;
   f3(1,1)=0.26;


   size_t vars1[]={0,1};
   size_t vars2[]={0,2};
   size_t vars3[]={1,2};
   Model::FunctionIdentifier fid1=model.addFunction(f1);
   Model::FunctionIdentifier fid2=model.addFunction(f2);
   Model::FunctionIdentifier fid3=model.addFunction(f3);
   size_t facid1=model.addFactor(fid1,vars1,vars1+2);
   size_t facid2=model.addFactor(fid2,vars2,vars2+2);
   size_t facid3=model.addFactor(fid3,vars3,vars3+2);
   size_t vars[]={0,1};
 
   typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Integrator> UpdateRules_sum_prod;
   typedef opengm::MessagePassing<Model, opengm::Integrator, UpdateRules_sum_prod, opengm::MaxDistance> BeliefPropagation_sum_prod;

   const size_t maxNumberOfIterations = 10000;
   const double convergenceBound = 1e-90;
   const double damping = 0.0;
   BeliefPropagation_sum_prod::Parameter parameter_s_p(maxNumberOfIterations, convergenceBound, damping); 
   parameter_s_p.inferSequential_ = true;
   //parameter_s_p.useNormalization_=true;
  
   BeliefPropagation_sum_prod bp_s_p(model, parameter_s_p);
 

   bp_s_p.infer();

   IndependentFactor IF;
   double Z=0;
   for(size_t i=0;i<n_var;i++)
   {
      bp_s_p.marginal(i,IF);
      Z=0;
      for(size_t j=0;j<n_stats[i];j++)
      {
         Z=IF(j)+Z;
      }
      std::cout<<"X_"<<i<<": ";
      for(size_t j=0;j<n_stats[i];j++)
      {
         std::cout<<"state: "<<j<<" value: "<<IF(j)/Z<<"; ";
      }
      std::cout<<"\n";
   }
   
}

int main() {
   {
      std::cout << "Test Operations ...";
      testOperations();
      std::cout <<" PASS!"<<std::endl<<std::endl;


      typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
      typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;
      typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
      typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
      typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;

      opengm::InferenceBlackBoxTester<SumGmType> sumTester;
      sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 5));
      sumTester.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::PASS, 5));
      sumTester.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
      sumTester.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::PASS, 5));
  
      opengm::InferenceBlackBoxTester<SumGmType> sumTester2;
      sumTester2.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
   
      opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
      prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 5));
      prodTester.addTest(new ProdGridTest(4, 4, 2, false, false,ProdGridTest::RANDOM, opengm::PASS, 5));
      prodTester.addTest(new ProdStarTest(6,    4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 20));
      prodTester.addTest(new ProdFullTest(5,    2, false, 3,    ProdFullTest::RANDOM, opengm::PASS, 5));

      std::cout << "Belief Propagation Tests"<<std::endl;
      {
         std::cout << "  * Minimization/Adder ..."<<std::endl;
         typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Minimizer> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para(size_t(10));
         sumTester.test<BP>(para); 
         std::cout << " ... parallel ... ";
         para.isAcyclic_=opengm::Tribool::False;
         sumTester2.test<BP>(para); 
         std::cout << " OK!"<<std::endl;
      } 
      {
         std::cout << "  * Minimization/Adder ..."<<std::endl;
         typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Minimizer> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para(size_t(100));
         para.isAcyclic_ = false;
         sumTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Minimization/Adder with damping..."<<std::endl;
         typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Minimizer> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para(10,0,0.5);
         sumTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Adder ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder> GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Maximizer> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Maximizer,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para(size_t(10));
         sumTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Multiplier ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Maximizer> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Maximizer,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para(size_t(10));
         prodTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
       }
      {
         std::cout << "  * Maximizer/Multiplier with damping..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier > GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Maximizer> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Maximizer,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para(10,0,0.5);
         prodTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
       }
       {
         std::cout << "  * Minimization/Multiplier ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Minimizer> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para;
         prodTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
       }
       {
         std::cout << "  * Integrator/Adder ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Adder  > GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Integrator> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Integrator,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para;
         sumTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
       }
       {
         std::cout << "  * Integrator/Multiplier ..."<<std::endl;
         typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
         typedef opengm::BeliefPropagationUpdateRules<GraphicalModelType,opengm::Integrator> UpdateRulesType;
         typedef opengm::MessagePassing<GraphicalModelType, opengm::Integrator,UpdateRulesType, opengm::MaxDistance>            BP;
         BP::Parameter para;
         prodTester.test<BP>(para);
         std::cout << " OK!"<<std::endl;
       }
       std::cout << "done!"<<std::endl;
   }
   {
   typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
   typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
   typedef opengm::GraphicalModel<
        float, opengm::Adder, 
        opengm::ExplicitFunction<float,opengm::UInt16Type, opengm::UInt8Type>, 
        opengm::DiscreteSpace<opengm::UInt16Type, opengm::UInt8Type> 
   > SumGmType2;

   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
   typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;
   typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
   typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
   typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;
   typedef opengm::BlackBoxTestGrid<SumGmType2> SumGridTest2;
  
   opengm::InferenceBlackBoxTester<SumGmType> sumTester;
   sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 5));
   sumTester.addTest(new SumGridTest(4, 4, 2, false, false,SumGridTest::RANDOM, opengm::PASS, 5));
   sumTester.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
   sumTester.addTest(new SumFullTest(5,    2, false, 3,    SumFullTest::RANDOM, opengm::PASS, 5));
 
   opengm::InferenceBlackBoxTester<SumGmType> sumTester2;
   sumTester2.addTest(new SumStarTest(6,    4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
  
   opengm::InferenceBlackBoxTester<SumGmType2> sumTester3;
   sumTester3.addTest(new SumGridTest2(4, 4, 2, false, true, SumGridTest2::RANDOM, opengm::PASS, 5));

   opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
   prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 5));
   prodTester.addTest(new ProdGridTest(4, 4, 2, false, false,ProdGridTest::RANDOM, opengm::PASS, 5));
   prodTester.addTest(new ProdStarTest(6,    4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 20));
   prodTester.addTest(new ProdFullTest(5,    2, false, 3,    ProdFullTest::RANDOM, opengm::PASS, 5));

   std::cout << "Tree Reweighted Belief Propagation Tests"<<std::endl;
   {
      std::cout << "  * Minimization/Adder ..."<<std::endl;
      typedef opengm::GraphicalModel<double,opengm::Adder > GraphicalModelType;
      typedef opengm::TrbpUpdateRules<GraphicalModelType,opengm::Minimizer> UpdateRulesType;
      typedef opengm::MessagePassing<GraphicalModelType, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance>            BP;
      BP::Parameter para(size_t(100));
      sumTester.test<BP>(para);
      std::cout << " ... parallel ... ";
      para.isAcyclic_=opengm::Tribool::False;
      sumTester2.test<BP>(para); 
      std::cout << " OK!"<<std::endl;
   }

   
   {
      std::cout << "  * Minimization/Adder with damping (float,uint16,uint8)..."<<std::endl;
      typedef opengm::TrbpUpdateRules<SumGmType2,opengm::Minimizer> UpdateRulesType;
      typedef opengm::MessagePassing<SumGmType2, opengm::Minimizer,UpdateRulesType ,opengm::MaxDistance>            BP;
      BP::Parameter para(10,0,0.5);
      sumTester3.test<BP>(para);
      std::cout << " OK!"<<std::endl;
   }
   
   
   {
      std::cout << "  * Minimization/Adder with damping..."<<std::endl;
      typedef opengm::GraphicalModel<double,opengm::Adder > GraphicalModelType;
      typedef opengm::TrbpUpdateRules<GraphicalModelType,opengm::Minimizer> UpdateRulesType;
      typedef opengm::MessagePassing<GraphicalModelType, opengm::Minimizer,UpdateRulesType ,opengm::MaxDistance>            BP;
      BP::Parameter para(10,0,0.5);
      sumTester.test<BP>(para);
      std::cout << " OK!"<<std::endl;
   }
   {
      std::cout << "  * Maximizer/Adder ..."<<std::endl;
      typedef opengm::GraphicalModel<double,opengm::Adder>   GraphicalModelType;
      typedef opengm::TrbpUpdateRules<GraphicalModelType,opengm::Maximizer> UpdateRulesType;
      typedef opengm::MessagePassing<GraphicalModelType, opengm::Maximizer, UpdateRulesType,opengm::MaxDistance>            BP;
      BP::Parameter para(size_t(10));
      sumTester.test<BP>(para);
      std::cout << " OK!"<<std::endl;
    }
    {
      std::cout << "  * Maximizer/Multiplier ..."<<std::endl;
      typedef opengm::GraphicalModel<double,opengm::Multiplier  > GraphicalModelType;
      typedef opengm::TrbpUpdateRules<GraphicalModelType,opengm::Maximizer> UpdateRulesType;
      typedef opengm::MessagePassing<GraphicalModelType, opengm::Maximizer, UpdateRulesType,opengm::MaxDistance>            BP;
      BP::Parameter para(size_t(10));
      prodTester.test<BP>(para);
      std::cout << " OK!"<<std::endl;
    }
    {
      std::cout << "  * Maximizer/Multiplier with damping..."<<std::endl;
      typedef opengm::GraphicalModel<double,opengm::Multiplier > GraphicalModelType;
      typedef opengm::TrbpUpdateRules<GraphicalModelType,opengm::Maximizer> UpdateRulesType;
      typedef opengm::MessagePassing<GraphicalModelType, opengm::Maximizer, UpdateRulesType, opengm::MaxDistance>            BP;
      BP::Parameter para(size_t(10));
      prodTester.test<BP>(para);
      std::cout << " OK!"<<std::endl;
    }
    {
      std::cout << "    - Minimization/Multiplier ..."<<std::flush;
      typedef opengm::GraphicalModel<double,opengm::Multiplier > GraphicalModelType;
      typedef opengm::TrbpUpdateRules<GraphicalModelType,opengm::Minimizer> UpdateRulesType;
      typedef opengm::MessagePassing<GraphicalModelType, opengm::Minimizer,UpdateRulesType, opengm::MaxDistance>            BP;
      BP::Parameter para(10,0,0.5);
      prodTester.test<BP>(para);
      std::cout << " OK!"<<std::endl;
    }
    {
      std::cout << "  * Integrator/Adder ..."<<std::endl;
      typedef opengm::GraphicalModel<double,opengm::Adder > GraphicalModelType;
      typedef opengm::TrbpUpdateRules<GraphicalModelType,opengm::Integrator> UpdateRulesType;
      typedef opengm::MessagePassing<GraphicalModelType, opengm::Integrator,UpdateRulesType, opengm::MaxDistance>            BP;
      BP::Parameter para(size_t(10));
      sumTester.test<BP>(para);
      std::cout << " OK!"<<std::endl;
    }
    {
      std::cout << "  * Integrator/Multiplier ..."<<std::endl;
      typedef opengm::GraphicalModel<double,opengm::Multiplier > GraphicalModelType;
      typedef opengm::TrbpUpdateRules<GraphicalModelType,opengm::Integrator> UpdateRulesType;
      typedef opengm::MessagePassing<GraphicalModelType, opengm::Integrator,UpdateRulesType, opengm::MaxDistance>            BP;
      BP::Parameter para(size_t(10));
      prodTester.test<BP>(para);
      std::cout << " OK!"<<std::endl;
    }

    std::cout << "done!"<<std::endl;
   }

   std::cout << "Test sum prod ..."<<std::endl;
   testSumProd();
   std::cout << "done!"<<std::endl;
}

