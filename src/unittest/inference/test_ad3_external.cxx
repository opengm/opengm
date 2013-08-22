#include <iostream>
#ifdef WITH_AD3
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/external/ad3.hxx>
#include <opengm/inference/bruteforce.hxx>


#include <opengm/operations/adder.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

struct AD3Test
{
  typedef opengm::GraphicalModel<double,opengm::Adder  > GraphicalModelType;
  typedef GraphicalModelType::ValueType ValueType;
  typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType ;	//explicit Factorfunction(=dense marray)
  typedef GraphicalModelType::IndependentFactorType IndependentFactorType ;	//independet Factor (detached from the graphical model)
  typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;

  template<class AD3>
  void testOpt(typename AD3::Parameter para) {
   typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;
   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
   typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

   opengm::InferenceBlackBoxTester<SumGmType> sumTesterOpt;
   sumTesterOpt.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::OPTIMAL, 10));
   sumTesterOpt.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::POTTS, opengm::OPTIMAL, 10));
   sumTesterOpt.addTest(new SumGridTest(2, 2, 3, false, true, SumGridTest::POTTS, opengm::OPTIMAL, 10));
   sumTesterOpt.template test<AD3>(para);

  }

  template<class AD3>
  void test(typename AD3::Parameter para) {
   typedef opengm::GraphicalModel<double, opengm::Adder> SumGmType;
   typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
   typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
   typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

   opengm::InferenceBlackBoxTester<SumGmType> sumTesterOpt;
   sumTesterOpt.addTest(new SumGridTest(5, 5, 3, false, true, SumGridTest::POTTS, opengm::PASS, 10));
   sumTesterOpt.addTest(new SumGridTest(6, 6, 5, false, true, SumGridTest::POTTS, opengm::PASS, 10));
   sumTesterOpt.template test<AD3>(para);

  }

 void run()
  {
    std::cout << std::endl;
    std::cout << "  * Start Black-Box Tests ...\n"<<std::endl;

    {
      typedef opengm::GraphicalModel<double,opengm::Adder  > GraphicalModelType;
      typedef opengm::external::AD3Inf<GraphicalModelType,opengm::Minimizer>  AD3;
      AD3::Parameter para;

      para.solverType_= AD3::AD3_ILP;
      std::cout << "    - Minimization/Adder AD3_ILP...\n"<<std::flush;
      this->testOpt<AD3>(para);

      para.solverType_= AD3::AD3_LP;
      std::cout << "    - Minimization/Adder AD3_LP...\n"<<std::flush;
      this->test<AD3>(para);

      para.solverType_= AD3::PSDD_LP;
      para.eta_=0.9;
      std::cout << "    - Minimization/Adder PSDD_LP...\n"<<std::flush;
      this->test<AD3>(para);
    }
    {
      typedef opengm::GraphicalModel<double,opengm::Adder  > GraphicalModelType;
      typedef opengm::external::AD3Inf<GraphicalModelType,opengm::Maximizer>  AD3;
      AD3::Parameter para;

      para.solverType_= AD3::AD3_ILP;
      std::cout << "    - Maximization/Adder AD3_ILP...\n"<<std::flush;
      this->testOpt<AD3>(para);

      para.solverType_= AD3::AD3_LP;
      std::cout << "    - Maximization/Adder AD3_LP...\n"<<std::flush;
      this->test<AD3>(para);

      para.solverType_= AD3::PSDD_LP;
      para.eta_=0.9;
      std::cout << "    - Maximization/Adder PSDD_LP...\n"<<std::flush;
      this->test<AD3>(para);
    }
  };
};


#endif
int main() {
   #ifdef WITH_AD3
   std::cout << "AD3  Tests ..." << std::endl;
   {
      AD3Test t; t.run();
   }
   #else
   std::cout << "Compiled without AD3 ,AD3  Tests  is disabled..." << std::endl;
   #endif
   return 0;
}
