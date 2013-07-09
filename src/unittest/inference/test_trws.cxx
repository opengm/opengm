
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/external/trws.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

struct TRWSTest {
   typedef opengm::GraphicalModel<float, opengm::Adder > GraphicalModelType;
   typedef GraphicalModelType::ValueType ValueType;
   typedef opengm::ExplicitFunction<float>  ExplicitFunctionType; //explicit Factorfunction(=dense marray)
   //typedef GraphicalModelType::SparseFunctionType SparseFunctionType ;	//sparse Factorfunction (=sparse marray)
   //typedef GraphicalModelType::ImplicitFunctionType ImplicitFunctionType ;	// implicit Factorfunction (=hard coded function)
   typedef GraphicalModelType::IndependentFactorType IndependentFactorType; //independet Factor (detached from the graphical model)
   typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;

   void run() {
      typedef opengm::GraphicalModel<float, opengm::Adder > SumGmType;
      typedef opengm::GraphicalModel<float, opengm::Multiplier > ProdGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

      typedef opengm::GraphicalModel<float, opengm::Adder > GraphicalModelType;
      typedef opengm::external::TRWS<GraphicalModelType> TRWS;

      opengm::InferenceBlackBoxTester<GraphicalModelType> sumTester;
      sumTester.addTest(new SumStarTest(6,    2, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
      sumTester.addTest(new SumFullTest(5, 2, false, 3, SumFullTest::POTTS, opengm::OPTIMAL, 5));

      sumTester.addTest(new SumStarTest(6, 5, true, true, SumStarTest::RANDOM, opengm::OPTIMAL, 20));
      sumTester.addTest(new SumStarTest(7, 3, true, false, SumStarTest::RANDOM, opengm::OPTIMAL, 20));

      TRWS::Parameter para;
      para.energyType_ = TRWS::Parameter::TABLES;
      sumTester.test<TRWS>(para);
      para.energyType_ = TRWS::Parameter::VIEW;
      sumTester.test<TRWS>(para);
   };
};
int main() {
   #ifdef WITH_TRWS
   std::cout << "TRWS  Tests ..." << std::endl;
   {
      TRWSTest t; t.run();
   }
   #else
   std::cout << "Compiled withput TRWS ,TRWS  Tests  is disabled..." << std::endl;
   #endif
   return 0;
}
