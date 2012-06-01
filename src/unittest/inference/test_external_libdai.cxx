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

#include <opengm/inference/external/libdai/exact.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>
#include <opengm/inference/external/libdai/bp.hxx>
#include "opengm/inference/external/libdai/tree_reweighted_bp.hxx"
#include "opengm/inference/external/libdai/fractional_bp.hxx"
#include "opengm/inference/external/libdai/tree_expectation_propagation.hxx"

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>



int main() {
   {
      typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
      typedef opengm::GraphicalModel<double, opengm::Multiplier > ProdGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;
      typedef opengm::BlackBoxTestGrid<ProdGmType> ProdGridTest;
      typedef opengm::BlackBoxTestFull<ProdGmType> ProdFullTest;
      typedef opengm::BlackBoxTestStar<ProdGmType> ProdStarTest;
      
      std::cout << "LibDai Interface Exact Tests" << std::endl;
      {
         opengm::InferenceBlackBoxTester<SumGmType> sumTester;
         sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::OPTIMAL, 4));
         sumTester.addTest(new SumGridTest(4, 4, 2, false, false, SumGridTest::RANDOM, opengm::OPTIMAL, 4));
         sumTester.addTest(new SumStarTest(6, 4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 4));
         sumTester.addTest(new SumFullTest(3, 3, false, 3, SumFullTest::RANDOM, opengm::OPTIMAL, 4));

         opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::OPTIMAL, 4));
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, false, ProdGridTest::RANDOM, opengm::OPTIMAL, 4));
         prodTester.addTest(new ProdStarTest(6, 4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 4));
         prodTester.addTest(new ProdFullTest(3, 3, false, 3, ProdFullTest::RANDOM, opengm::OPTIMAL, 4));
      
         {
            std::cout << "  * Minimization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::Exact<GmType,opengm::Minimizer> DaiExactType;
            DaiExactType::Parameter para(0);
            sumTester.test<DaiExactType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::Exact<GmType,opengm::Maximizer> DaiExactType;
            DaiExactType::Parameter para(0);
            sumTester.test<DaiExactType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::Exact<GmType,opengm::Maximizer> DaiExactType;
            DaiExactType::Parameter para(0);
            prodTester.test<DaiExactType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Minimization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::Exact<GmType,opengm::Minimizer> DaiExactType;
            DaiExactType::Parameter para(0);
            prodTester.test<DaiExactType > (para);
            std::cout << " OK!" << std::endl;
         }
      }
      std::cout << "LibDai Interface Junction Tree Tests" << std::endl;
      {
         opengm::InferenceBlackBoxTester<SumGmType> sumTester;
         sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::OPTIMAL, 4));
         sumTester.addTest(new SumGridTest(4, 4, 2, false, false, SumGridTest::RANDOM, opengm::OPTIMAL, 4));
         sumTester.addTest(new SumStarTest(6, 4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 4));
         sumTester.addTest(new SumFullTest(3, 3, false, 3, SumFullTest::RANDOM, opengm::OPTIMAL, 4));

         opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::OPTIMAL, 4));
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, false, ProdGridTest::RANDOM, opengm::OPTIMAL, 4));
         prodTester.addTest(new ProdStarTest(6, 4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 4));
         prodTester.addTest(new ProdFullTest(3, 3, false, 3, ProdFullTest::RANDOM, opengm::OPTIMAL, 4));
      
         {
            std::cout << "  * Minimization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::JunctionTree<GmType,opengm::Minimizer> DaiExactType;
            DaiExactType::Parameter para;
            sumTester.test<DaiExactType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::JunctionTree<GmType,opengm::Maximizer> DaiExactType;
            DaiExactType::Parameter para;
            sumTester.test<DaiExactType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::JunctionTree<GmType,opengm::Maximizer> DaiExactType;
            DaiExactType::Parameter para;
            prodTester.test<DaiExactType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Minimization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::JunctionTree<GmType,opengm::Minimizer> DaiExactType;
            DaiExactType::Parameter para;
            prodTester.test<DaiExactType > (para);
            std::cout << " OK!" << std::endl;
         }
      }
      std::cout << "LibDai Interface BP Tests" << std::endl;
      {
         opengm::InferenceBlackBoxTester<SumGmType> sumTester;
         sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 2));
         sumTester.addTest(new SumGridTest(4, 4, 2, false, false, SumGridTest::RANDOM, opengm::PASS, 2));
         sumTester.addTest(new SumStarTest(6, 4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 10));
         sumTester.addTest(new SumFullTest(5, 3, false, 3, SumFullTest::RANDOM, opengm::PASS, 2));

         opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 2));
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, false, ProdGridTest::RANDOM, opengm::PASS, 2));
         prodTester.addTest(new ProdStarTest(6, 4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 10));
         prodTester.addTest(new ProdFullTest(5, 3, false, 3, ProdFullTest::RANDOM, opengm::PASS, 2));
      
         {
            std::cout << "  * Minimization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::Bp<GmType,opengm::Minimizer> DaiBpType;
            
            DaiBpType::Parameter para(10);
            sumTester.test<DaiBpType > (para);
            
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::Bp<GmType,opengm::Maximizer> DaiBpType;
            DaiBpType::Parameter para(10);
            sumTester.test<DaiBpType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::Bp<GmType,opengm::Maximizer> DaiBpType;
            DaiBpType::Parameter para(10);
            prodTester.test<DaiBpType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Minimization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::Bp<GmType,opengm::Minimizer> DaiBpType;
            DaiBpType::Parameter para(10);
            prodTester.test<DaiBpType > (para);
            std::cout << " OK!" << std::endl;
         }
      }
      std::cout << "LibDai Interface TR-BP Tests" << std::endl;
      {
         opengm::InferenceBlackBoxTester<SumGmType> sumTester;
         sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 2));
         sumTester.addTest(new SumGridTest(4, 4, 2, false, false, SumGridTest::RANDOM, opengm::PASS, 2));
         sumTester.addTest(new SumStarTest(6, 4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 10));
         sumTester.addTest(new SumFullTest(5, 3, false, 3, SumFullTest::RANDOM, opengm::PASS, 2));

         opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 2));
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, false, ProdGridTest::RANDOM, opengm::PASS, 2));
         prodTester.addTest(new ProdStarTest(6, 4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 10));
         prodTester.addTest(new ProdFullTest(5, 3, false, 3, ProdFullTest::RANDOM, opengm::PASS, 2));
         {
            std::cout << "  * Minimization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::TreeReweightedBp<GmType,opengm::Minimizer> DaiTrbpType;
            DaiTrbpType::Parameter para(10);
            sumTester.test<DaiTrbpType > (para);
            
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::TreeReweightedBp<GmType,opengm::Maximizer> DaiTrbpType;
            DaiTrbpType::Parameter para(10);
            sumTester.test<DaiTrbpType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::TreeReweightedBp<GmType,opengm::Maximizer> DaiTrbpType;
            DaiTrbpType::Parameter para(10);
            prodTester.test<DaiTrbpType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Minimization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::TreeReweightedBp<GmType,opengm::Minimizer> DaiTrbpType;
            DaiTrbpType::Parameter para(10);
            prodTester.test<DaiTrbpType > (para);
            std::cout << " OK!" << std::endl;
         }
      }
      std::cout << "LibDai Interface Fractional-BP Tests" << std::endl;
      {
         opengm::InferenceBlackBoxTester<SumGmType> sumTester;
         sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 1));
         sumTester.addTest(new SumGridTest(4, 4, 2, false, false, SumGridTest::RANDOM, opengm::PASS, 1));
         sumTester.addTest(new SumStarTest(6, 4, false, true, SumStarTest::RANDOM, opengm::OPTIMAL, 2));
         sumTester.addTest(new SumFullTest(5, 3, false, 3, SumFullTest::RANDOM, opengm::PASS, 1));

         opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 1));
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, false, ProdGridTest::RANDOM, opengm::PASS, 1));
         prodTester.addTest(new ProdStarTest(6, 4, false, true, ProdStarTest::RANDOM, opengm::OPTIMAL, 2));
         prodTester.addTest(new ProdFullTest(5, 3, false, 3, ProdFullTest::RANDOM, opengm::PASS, 1));
         {
            std::cout << "  * Minimization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::FractionalBp<GmType,opengm::Minimizer> DaiType;
            DaiType::Parameter para(10);
            sumTester.test<DaiType > (para);
            
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::FractionalBp<GmType,opengm::Maximizer> DaiType;
            DaiType::Parameter para(10);
            sumTester.test<DaiType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::FractionalBp<GmType,opengm::Maximizer> DaiType;
            DaiType::Parameter para(10);
            prodTester.test<DaiType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Minimization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::FractionalBp<GmType,opengm::Minimizer> DaiType;
            DaiType::Parameter para(10);
            prodTester.test<DaiType > (para);
            std::cout << " OK!" << std::endl;
         }
      }
      
      std::cout << "LibDai Interface Tree-Expectation-Propagation Tests" << std::endl;
      {
         opengm::InferenceBlackBoxTester<SumGmType> sumTester;
         sumTester.addTest(new SumGridTest(4, 4, 2, false, true, SumGridTest::RANDOM, opengm::PASS, 2));
         sumTester.addTest(new SumGridTest(4, 4, 2, false, false, SumGridTest::RANDOM, opengm::PASS, 2));
         sumTester.addTest(new SumStarTest(6, 4, false, true, SumStarTest::RANDOM, opengm::PASS, 2));
         sumTester.addTest(new SumFullTest(5, 3, false, 3, SumFullTest::RANDOM, opengm::PASS, 2));

         opengm::InferenceBlackBoxTester<ProdGmType> prodTester;
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, true, ProdGridTest::RANDOM, opengm::PASS, 2));
         prodTester.addTest(new ProdGridTest(4, 4, 2, false, false, ProdGridTest::RANDOM, opengm::PASS, 2));
         prodTester.addTest(new ProdStarTest(6, 4, false, true, ProdStarTest::RANDOM, opengm::PASS, 2));
         prodTester.addTest(new ProdFullTest(5, 3, false, 3, ProdFullTest::RANDOM, opengm::PASS, 2));
         {
            std::cout << "  * Minimization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::TreeExpectationPropagation<GmType,opengm::Minimizer> DaiType;
            DaiType::Parameter para(DaiType::ORG,200);
            sumTester.test<DaiType > (para);
            
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Adder ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
            typedef opengm::external::libdai::TreeExpectationPropagation<GmType,opengm::Maximizer> DaiType;
            DaiType::Parameter para(DaiType::ORG,200);
            sumTester.test<DaiType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Maximization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::TreeExpectationPropagation<GmType,opengm::Maximizer> DaiType;
            DaiType::Parameter para(DaiType::ORG,200);
            prodTester.test<DaiType > (para);
            std::cout << " OK!" << std::endl;
         }
         {
            std::cout << "  * Minimization/Multiplier ..." << std::endl;
            typedef opengm::GraphicalModel<double, opengm::Multiplier> GmType;
            typedef opengm::external::libdai::TreeExpectationPropagation<GmType,opengm::Minimizer> DaiType;
            DaiType::Parameter para(DaiType::ORG,200);
            prodTester.test<DaiType > (para);
            std::cout << " OK!" << std::endl;
         }
      }

      std::cout << "done!" << std::endl;
   }
};



