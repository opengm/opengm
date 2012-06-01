#include <iostream>
//#include <stdlib.h>
//#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/unittests/inferencetester.hxx>
#include <opengm/unittests/inferencetests/test_empty.hxx>
//#include <opengm/unittests/inferencetests/test_grid2.hxx>
#include <opengm/unittests/inferencetests/test_blackbox.hxx>
#include <opengm/unittests/inferencetests/test_multimode.hxx>
#include <opengm/unittests/inferencetests/test_functions.hxx>

#include <opengm/inference/bruteforce.hxx>

int main(){ 
   typedef double ValueType;
   typedef opengm::meta::TypeListGenerator
   <
   opengm::PottsFunction<ValueType>,
   opengm::PottsNFunction<ValueType>,
   opengm::PottsGFunction<ValueType>,
   opengm::AbsoluteDifferenceFunction<ValueType>,
   opengm::SquaredDifferenceFunction<ValueType>,
   opengm::TruncatedAbsoluteDifferenceFunction<ValueType>,
   opengm::TruncatedSquaredDifferenceFunction<ValueType>
   >::type FunctionTypeList;

   typedef opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList>  GmType;
//   typedef opengm::GraphicalModel<ValueType,opengm::Adder>  GmType;

   typedef opengm::Bruteforce<GmType,opengm::Minimizer> InfType;

   typedef opengm::test::TestBlackBox<InfType> TestBlackBox;
   typedef opengm::test::TestMultiMode<InfType> TestMultiMode;
   typedef opengm::test::TestFunctions<InfType> TestFunctions;
   
   opengm::test::InferenceTester<InfType> tester;
   tester.addTest(new opengm::test::TestEmpty<InfType>());
   tester.addTest(new TestMultiMode(3,4));
   tester.addTest(new TestFunctions());
   tester.addTest(new TestBlackBox(opengm::test::GRID,9,3,3,opengm::test::POS_POTTS,opengm::test::OPTIMAL,"",6,0));
   tester.addTest(new TestBlackBox(opengm::test::FULL,5,3,3,opengm::test::POS_POTTS,opengm::test::OPTIMAL,"",2,0)); 
   tester.addTest(new TestBlackBox(opengm::test::STAR,5,3,3,opengm::test::POS_POTTS,opengm::test::OPTIMAL,"",2,0)); 
  
   InfType::Parameter para;
   tester.test(para);

   return 0;
};
