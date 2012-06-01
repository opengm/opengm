#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/utilities/random.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/datastructures/marray/marray.hxx>

int main(int argc, char** argv) {
   opengm::RandomUniform<float> rf1(0,1);
   opengm::RandomUniform<float> rf2(1,2);
   opengm::RandomUniform<size_t> ri1(0,10);
   opengm::RandomUniform<size_t> ri2(5,10);
   std::cout << "starting test random..." << std::endl;
   for(size_t i = 0; i< 100000;++i) {
      OPENGM_TEST(rf1() >= 0);
      OPENGM_TEST(rf2() >= 1);
      OPENGM_TEST(ri1() >= 0);
      OPENGM_TEST(ri2() >= 5);

      OPENGM_TEST(rf1() < 1);
      OPENGM_TEST(rf2() < 2);
      OPENGM_TEST(ri1() < 10);
      OPENGM_TEST(ri2() < 10);
   }
   std::cout << "test random finished successfully" << std::endl;
   return 0;
}
