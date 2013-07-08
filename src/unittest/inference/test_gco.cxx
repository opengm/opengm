#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#include <opengm/inference/external/gco.hxx>

int main() {
   typedef opengm::GraphicalModel<float, opengm::Adder > GraphicalModelType;
   typedef opengm::GraphicalModel<float, opengm::Adder, opengm::ExplicitFunction<float,unsigned int, unsigned char>, opengm::DiscreteSpace<unsigned int, unsigned char> > GraphicalModelType2;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
   typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
   typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType2> GridTest2;

   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester;
   minTester.addTest(new GridTest(4, 4, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 1));
   minTester.addTest(new GridTest(3, 3, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new GridTest(3, 3, 2, false, false,GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new StarTest(5,    2, false, true, StarTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new FullTest(5,    2, false, 3,    FullTest::POTTS, opengm::OPTIMAL, 3));
   minTester.addTest(new GridTest(4, 4, 9, false, true, GridTest::POTTS, opengm::PASS,   10));
   minTester.addTest(new GridTest(4, 4, 9, false, false,GridTest::POTTS, opengm::PASS,   10));
   minTester.addTest(new FullTest(6,    4, false, 3,    FullTest::POTTS, opengm::PASS,   10));

   opengm::InferenceBlackBoxTester<GraphicalModelType2> minTester2;
   minTester2.addTest(new GridTest2(4, 4, 2, false, true, GridTest2::POTTS, opengm::OPTIMAL, 1));

   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester3;
   minTester3.addTest(new GridTest(4, 4, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 1));
   minTester3.addTest(new GridTest(3, 3, 2, false, true, GridTest::POTTS, opengm::OPTIMAL, 3));
   minTester3.addTest(new GridTest(3, 3, 2, false, false,GridTest::POTTS, opengm::OPTIMAL, 3));

   std::cout << "Test Expansion ..." << std::endl;

   typedef opengm::external::GCOLIB<GraphicalModelType> GCO;
   typedef opengm::external::GCOLIB<GraphicalModelType2> GCO2;
   GCO::Parameter para;
   GCO2::Parameter para2;

   para.doNotUseGrid_ = true;
   para2.doNotUseGrid_ = true;
   std::cout << "Test Expansion VIEW no Grid structure..." << std::endl;
   minTester.test<GCO>(para);
   std::cout << "Test Expansion VIEW no Grid structure(float, uint16,uint8)..." << std::endl;
   minTester2.test<GCO2>(para2);

   para.doNotUseGrid_ = false;
   para2.doNotUseGrid_ = false;
   std::cout << "Test Expansion VIEW Grid structure..." << std::endl;
   minTester3.test<GCO>(para);
   std::cout << "Test Expansion VIEW Grid structure(float, uint16,uint8)..." << std::endl;
   minTester2.test<GCO2>(para2);

   para.energyType_ = GCO::Parameter::TABLES;
   para2.energyType_ = GCO2::Parameter::TABLES;

   std::cout << "Test Expansion TABLES..." << std::endl;
   minTester3.test<GCO>(para);
   std::cout << "Test Expansion TABLES (float, uint16,uint8)..." << std::endl;
   minTester2.test<GCO2>(para2);

   std::cout << "Test Swap ..." << std::endl;

   para.energyType_ = GCO::Parameter::VIEW;
   para2.energyType_ = GCO2::Parameter::VIEW;

   para.doNotUseGrid_ = true;
   para.inferenceType_ = GCO::Parameter::SWAP;
   para2.doNotUseGrid_ = true;
   para2.inferenceType_ = GCO2::Parameter::SWAP;
   std::cout << "Test Swap VIEW no Grid structure..." << std::endl;
   minTester.test<GCO>(para);
   std::cout << "Test Swap VIEW no Grid structure(float, uint16,uint8)..." << std::endl;
   minTester2.test<GCO2>(para2);

   para.doNotUseGrid_ = false;
   para2.doNotUseGrid_ = false;
   std::cout << "Test Swap VIEW Grid structure..." << std::endl;
   minTester3.test<GCO>(para);
   std::cout << "Test Swap VIEW Grid structure(float, uint16,uint8)..." << std::endl;
   minTester2.test<GCO2>(para2);

   para.energyType_ = GCO::Parameter::TABLES;
   para2.energyType_ = GCO2::Parameter::TABLES;

   std::cout << "Test Swap TABLES..." << std::endl;
   minTester3.test<GCO>(para);
   std::cout << "Test Swap TABLES (float, uint16,uint8)..." << std::endl;
   minTester2.test<GCO2>(para2);

   return 0;
}
