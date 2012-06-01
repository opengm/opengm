#include "opengm/unittests/test.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/graphicalmodel/space/simplediscretespace.hxx"

void testDiscreteSpace() {
   {
      opengm::DiscreteSpace<> space;
      OPENGM_TEST(space.numberOfVariables() == 0);

      size_t numbersOfLabels[] = {3, 2, 4};
      space.assign(numbersOfLabels, numbersOfLabels + 3);
      OPENGM_TEST(space.numberOfVariables() == 3);
      OPENGM_TEST(space.numberOfLabels(0) == 3);
      OPENGM_TEST(space.numberOfLabels(1) == 2);
      OPENGM_TEST(space.numberOfLabels(2) == 4);

      opengm::DiscreteSpace<>::IndexType index = space.addVariable(5);
      OPENGM_TEST(index == 3);
      OPENGM_TEST(space.numberOfVariables() == 4);
      OPENGM_TEST(space.numberOfLabels(0) == 3);
      OPENGM_TEST(space.numberOfLabels(1) == 2);
      OPENGM_TEST(space.numberOfLabels(2) == 4);
      OPENGM_TEST(space.numberOfLabels(3) == 5);

      numbersOfLabels[0] = 2;
      numbersOfLabels[1] = 3;
      space.assign(numbersOfLabels, numbersOfLabels + 2);
      OPENGM_TEST(space.numberOfVariables() == 2);
      OPENGM_TEST(space.numberOfLabels(0) == 2);
      OPENGM_TEST(space.numberOfLabels(1) == 3);
   }
   {
      size_t numbersOfLabels[] = {3, 2, 4};
      opengm::DiscreteSpace<> space(
         numbersOfLabels, numbersOfLabels + 3);
      OPENGM_TEST(space.numberOfVariables() == 3);
      OPENGM_TEST(space.numberOfLabels(0) == 3);
      OPENGM_TEST(space.numberOfLabels(1) == 2);
      OPENGM_TEST(space.numberOfLabels(2) == 4);
   }
}

void testSimpleDiscreteSpace() {
   { 
      opengm::SimpleDiscreteSpace<> space; 
      OPENGM_TEST(space.numberOfVariables() == 0);

      size_t numberOfVariables = 3;
      size_t numberOfLabels = 5;
      space.assign(numberOfVariables, numberOfLabels);
      OPENGM_TEST(space.numberOfVariables() == 3);
      OPENGM_TEST(space.numberOfLabels(0) == 5);
      OPENGM_TEST(space.numberOfLabels(1) == 5);
      OPENGM_TEST(space.numberOfLabels(2) == 5);

      opengm::SimpleDiscreteSpace<>::IndexType index = space.addVariable(5);
      OPENGM_TEST(index == 3);
      OPENGM_TEST(space.numberOfVariables() == 4);
      OPENGM_TEST(space.numberOfLabels(0) == 5);
      OPENGM_TEST(space.numberOfLabels(1) == 5);
      OPENGM_TEST(space.numberOfLabels(2) == 5);
      OPENGM_TEST(space.numberOfLabels(3) == 5);
      try {
         space.addVariable(2);
         OPENGM_TEST(0 == 1);
      }
      catch(opengm::RuntimeError& e) {
      }

      numberOfVariables = 2;
      numberOfLabels = 4;
      space.assign(numberOfVariables, numberOfLabels);
      OPENGM_TEST(space.numberOfVariables() == 2);
      OPENGM_TEST(space.numberOfLabels(0) == 4);
      OPENGM_TEST(space.numberOfLabels(1) == 4);
   }
   {
      size_t numberOfVariables = 3;
      size_t numberOfLabels = 5;
      opengm::SimpleDiscreteSpace<> space(
         numberOfVariables, numberOfLabels);
      OPENGM_TEST(space.numberOfVariables() == 3);
      OPENGM_TEST(space.numberOfLabels(0) == 5);
      OPENGM_TEST(space.numberOfLabels(1) == 5);
      OPENGM_TEST(space.numberOfLabels(2) == 5);
   }
}

int main() {
   return 0;
}
