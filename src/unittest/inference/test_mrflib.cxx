#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/external/mrflib.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>

int main() {
   typedef opengm::GraphicalModel<float, opengm::Adder > GraphicalModelType;
   typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
  
   opengm::InferenceBlackBoxTester<GraphicalModelType> minTester;
   minTester.addTest(new GridTest(4, 4, 2, false, true, GridTest::POTTS, opengm::PASS, 1));
   minTester.addTest(new GridTest(3, 3, 2, false, true, GridTest::POTTS, opengm::PASS, 3));
   minTester.addTest(new GridTest(3, 3, 2, false, false,GridTest::POTTS, opengm::PASS, 3));
  
   std::cout << "Test MRFLIB-ICM ..." << std::endl;
   typedef opengm::external::MRFLIB<GraphicalModelType> MRFLIB;
   MRFLIB::Parameter para;
   para.inferenceType_ = MRFLIB::Parameter::ICM;
   para.energyType_    = MRFLIB::Parameter::VIEW;
   para.numberOfIterations_ = 10;
   minTester.test<MRFLIB>(para);
 
   std::cout << "Test MRFLIB-TRWS ..." << std::endl;
   para.inferenceType_ = MRFLIB::Parameter::TRWS;
   minTester.test<MRFLIB>(para);

   return 0;
}

