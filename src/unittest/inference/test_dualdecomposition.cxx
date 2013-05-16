#include <stdlib.h>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_bundle.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
//#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/dynamicprogramming.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>


template <class DD>
int test() {
   typedef typename DD::AccumulationType    AccType;
   typedef typename DD::OperatorType        OperatorType;
   typedef typename DD::ValueType           ValueType;
   typedef typename DD::GraphicalModelType  GraphicalModelType; 
   typedef opengm::GraphicalModel<float, opengm::Adder, opengm::ExplicitFunction<float,unsigned int, unsigned char>, opengm::DiscreteSpace<unsigned int, unsigned char> >  GraphicalModelType2;

   typedef DD                               DualDecompositionType;
 
   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
   typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
   typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType2> GridTest2;
 

   opengm::InferenceBlackBoxTester<GraphicalModelType> tester;
   tester.addTest(new GridTest(10, 10, 16, false, true,  GridTest::RANDOM, opengm::PASS, 1));
   tester.addTest(new FullTest(5,       4, false, 3,     FullTest::RANDOM, opengm::PASS, 1));
   tester.addTest(new FullTest(20,      4, false, 2,     FullTest::RANDOM, opengm::PASS, 1));
   tester.addTest(new GridTest(6, 6,    5, false, true,  GridTest::RANDOM, opengm::PASS, 1));
   tester.addTest(new GridTest(6, 6,    5, false, false, GridTest::RANDOM, opengm::PASS, 1));
   tester.addTest(new StarTest(4,       3, false, true,  StarTest::RANDOM, opengm::OPTIMAL, 1));

   {
      std::cout << "  * Tree-Decomposition ... " << std::endl;
      typename DualDecompositionType::Parameter para;
      para.decompositionId_= DualDecompositionType::Parameter::TREE;
      tester.template test<DualDecompositionType>(para);  
   }
   {
      std::cout << "  *  SpanningTrees-Decomposition ... " << std::endl;
      typename DualDecompositionType::Parameter para;  
      para.decompositionId_= DualDecompositionType::Parameter::SPANNINGTREES;
      tester.template test<DualDecompositionType>(para);
   } 
   return 0;
}

int main() {
   typedef float ValueType; 
   typedef opengm::Adder OperatorType;
   typedef opengm::Minimizer AccType;
   typedef opengm::GraphicalModel<ValueType, OperatorType, opengm::ExplicitFunction<ValueType,unsigned int, unsigned char>, opengm::DiscreteSpace<unsigned int, unsigned char> > GraphicalModelType;  
 
   typedef opengm::DDDualVariableBlock<marray::Marray<ValueType> >       DualBlockType; 
   typedef opengm::DDDualVariableBlock2<marray::View<ValueType,false> >  DualBlockType2;
  
   typedef opengm::DualDecompositionBase<GraphicalModelType,DualBlockType>::SubGmType                                      SubGmType;
   typedef opengm::MessagePassing<SubGmType, AccType, opengm::BeliefPropagationUpdateRules<SubGmType, AccType>, opengm::MaxDistance>   InfType;
   //typedef opengm::LPCplex<SubGmType, AccType>   InfTypeX; 
   typedef opengm::DynamicProgramming<SubGmType, AccType>   InfTypeY;
   typedef opengm::DualDecompositionBase<GraphicalModelType,DualBlockType2>::SubGmType                                     SubGmType2;
   typedef opengm::MessagePassing<SubGmType2, AccType, opengm::BeliefPropagationUpdateRules<SubGmType2, AccType>, opengm::MaxDistance> InfType2;
   //typedef opengm::LPCplex<SubGmType2, AccType> InfType2X;
   typedef opengm::DynamicProgramming<SubGmType2, AccType> InfType2Y;
 
   typedef opengm::DualDecompositionSubGradient<GraphicalModelType,InfTypeY,DualBlockType>  DualDecompositionSubGradient;
   std::cout << "  * Test with Min-Sum-MArray and Subgradient-Method" << std::endl;
   test<DualDecompositionSubGradient>(); 
   typedef opengm::DualDecompositionSubGradient<GraphicalModelType,InfType2Y,DualBlockType2>  DualDecompositionSubGradient2;
   std::cout << "  * Test with Min-Sum-VIEW and Subgradient-Method" << std::endl;
   test<DualDecompositionSubGradient2>();

#ifdef WITH_CONICBUNDLE
   typedef opengm::DualDecompositionBundle<GraphicalModelType,InfType2Y,DualBlockType2>     DDBundle;
   std::cout << "  * Test with Min-Sum-VIEW and Bundle-Method" << std::endl;
   test<DDBundle>(); 
#endif

/* 
   typedef marray::View<double,false> DualVarType2;
   typedef opengm::DualDecompositionHelper<GraphicalModelType, DualVarType2>::SubGmType  SubGmType2;
   typedef opengm::MessagePassing<SubGmType2, AccType, opengm::BpUpdateRules<SubGmType2, AccType>, opengm::MaxDistance> BPType2;
   typedef opengm::DualDecompositionSubGradient<BPType2, DualVarType2> DualSolverType2; 

   typedef marray::View<double,false> DualVarType3;
   typedef opengm::DualDecompositionHelper<GraphicalModelType, DualVarType3>::SubGmType  SubGmType3;
   typedef opengm::MessagePassing<SubGmType3, AccType, opengm::BpUpdateRules<SubGmType3, AccType>, opengm::MaxDistance> BPType3;
   typedef opengm::DDBundle<BPType3, DualVarType3> DualSolverType3;
 
   std::cout << "**Test with Min-Sum-View" << std::endl;
   test<DualSolverType2>(); 
   std::cout << "**Test with Min-Sum-Bundle" << std::endl;
   test<DualSolverType3>();
*/
   return 0;
}



