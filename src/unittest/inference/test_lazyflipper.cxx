#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>

#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/inference/lazyflipper.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

void additionalTest();

int main() {
   typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;
   typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
   typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;
   typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
   typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
   typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;

   opengm::InferenceBlackBoxTester<SumGmType> sumTester;
   sumTester.addTest(new SumGridTest(3, 3, 2, false, true, SumGridTest::POTTS, opengm::PASS, 1));
   sumTester.addTest(new SumFullTest(4,    3, false,    3, SumFullTest::POTTS, opengm::PASS, 1));

   opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
   prodTester.addTest(new ProdGridTest(3, 3, 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 1));
   prodTester.addTest(new ProdFullTest(4,    3, false,    3, ProdFullTest::RANDOM, opengm::PASS, 1));

   std::cout << "LazyFlipper  Tests ..." << std::endl;
   {
      std::cout << "  * Minimization/Adder  ..." << std::endl;
      typedef opengm::LazyFlipper<SumGmType, opengm::Minimizer> LF;
      LF::Parameter para;
      sumTester.test<LF>(para);
      std::cout << " OK!"<<std::endl;
   }
   {
      std::cout << "  * Maximization/Multiplier  ..." << std::endl;
      typedef opengm::LazyFlipper<ProdGmType, opengm::Maximizer> LF;
      LF::Parameter para;
      prodTester.test<LF>(para);
      std::cout << " OK!"<<std::endl;
   }

   additionalTest();
}

void additionalTest() {
   typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
   typedef float Value;
   typedef opengm::ExplicitFunction<Value> ExplicitFunction;
   typedef opengm::GraphicalModel<Value, opengm::Adder, opengm::meta::TypeList<ExplicitFunction, opengm::meta::ListEnd>, Space> GraphicalModel;
   typedef GraphicalModel::FunctionIdentifier FunctionIdentifier;
   typedef opengm::LazyFlipper<GraphicalModel, opengm::Minimizer> LazyFlipper;

   const size_t numberOfVariables = 50;
   Space space(numberOfVariables, 3);
   GraphicalModel model(space);

   srand(0);
   for(size_t j=0; j<numberOfVariables-3; ++j) {
      const size_t shape[] = {3, 3, 3};
      ExplicitFunction f(shape, shape + 3);
      for(size_t k=0; k<27; ++k) {
         f(k) = rand() % 20;
      }
      FunctionIdentifier fid = model.addFunction(f);

      const size_t variableIndices[] = {j, j+1, j+2};
      model.addFactor(fid, variableIndices, variableIndices + 3);
   }

   {
      LazyFlipper::Parameter parameter(6);
      LazyFlipper lazyFlipper(model, parameter);
      lazyFlipper.infer();

      std::vector<size_t> label;
      lazyFlipper.arg(label);

      OPENGM_TEST_EQUAL(lazyFlipper.value(), model.evaluate(label.begin()));
   }
}
