#include <vector>
#include <iostream>

#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/utilities/metaprogramming.hxx>

template<class T>
class TestOpenGmHdf5 {
public:

   void run() {
      typedef T ValueType;
      typedef opengm::GraphicalModel<
         ValueType, // value type (should be float double or long double)
         opengm::Multiplier, // operator (something like Adder or Multiplier)
         typename opengm::meta::TypeListGenerator<
            opengm::ExplicitFunction<ValueType>,
            opengm::PottsNFunction<ValueType>
         >::type // implicit function functor
      > GraphicalModelType;
      typedef opengm::GraphicalModel<
         ValueType, // value type (should be float double or long double)
         opengm::Multiplier, // operator (something like Adder or Multiplier)
         typename opengm::meta::TypeListGenerator<
            opengm::ExplicitFunction<ValueType>
         >::type // implicit function functor
      > GraphicalModelTypeOnlyExplicit;
      typedef opengm::GraphicalModel<
         int, // value type (should be float double or long double)
         opengm::Multiplier, // operator (something like Adder or Multiplier)
         typename opengm::meta::TypeListGenerator<
            opengm::ExplicitFunction<int>,
            opengm::PottsNFunction<int>
         >::type // implicit function functor
      > GmTypeInt;
      // 2 ) factor types:
      // type of the factors and factor functions:
      typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
      typedef typename opengm::ExplicitFunction<ValueType> ExplicitFunctionType; //explicit Factorfunction(=dense marray)
      size_t nos3[] = {3, 3, 3};
      GraphicalModelType gmExplicit(opengm::DiscreteSpace<size_t, size_t > (nos3, nos3 + 3));
      GraphicalModelType gmMixed(opengm::DiscreteSpace<size_t, size_t > (nos3, nos3 + 3));
      size_t vi[] = {0, 1, 2};
      //        size_t vib[] = {1,2};
      //        size_t vic[] = {0,2};
      //pair potentials functions
      {
         std::cout << "    - pair potentials functions" << std::endl;
         opengm::PottsNFunction<ValueType> fi(nos3, nos3 + 3, 1, 6);
         ExplicitFunctionType fe(nos3, nos3 + 3, 6);
         fe(0, 0, 0) = 1;
         fe(1, 1, 1) = 1;
         fe(2, 2, 2) = 1;
         ExplicitFunctionType fs(nos3, nos3 + 3, 6);
         fs(0, 0, 0) = 1;
         fs(1, 1, 1) = 1;
         fs(2, 2, 2) = 1;
         FunctionIdentifier iE = gmExplicit.addFunction(fe);
         //FunctionIdentifier iI=gmMixed.addFunction(fi);
         //FunctionIdentifier iS=gmMixed.addFunction(fs);
         FunctionIdentifier iE_ = gmMixed.addFunction(fe);
         FunctionIdentifier iI_ = gmMixed.addFunction(fi);
         FunctionIdentifier iS_ = gmMixed.addFunction(fs);
         gmExplicit.addFactor(iE, vi, vi + 3);
         gmExplicit.addFactor(iE, vi, vi + 3);
         gmExplicit.addFactor(iE, vi, vi + 3);
         gmMixed.addFactor(iE_, vi, vi + 3);
         gmMixed.addFactor(iS_, vi, vi + 3);
         gmMixed.addFactor(iI_, vi, vi + 3);
      }
      //Single Side Factors
      {
         std::cout << "    - Single Side Factors" << std::endl;
         ExplicitFunctionType fe1(nos3, nos3 + 1);
         fe1(0) = 2.4;
         fe1(1) = 6;
         fe1(2) = 6;
         ExplicitFunctionType fs1(nos3, nos3 + 1, 6);
         fs1(0) = 2.4;
         //fs1(1)=6;
         //fs1(2)=6;
         ExplicitFunctionType fe2(nos3 + 1, nos3 + 2);
         fe2(0) = 6;
         fe2(1) = 1.5;
         fe2(2) = 6;
         ExplicitFunctionType fs2(nos3 + 1, nos3 + 2, 6);
         //fs1(0)=6;
         fs2(1) = 1.5;
         //fs1(2)=6;
         ExplicitFunctionType fe3(nos3 + 2, nos3 + 3);
         fe3(0) = 6;
         fe3(1) = 6;
         fe3(2) = 2;
         ExplicitFunctionType fs3(nos3 + 2, nos3 + 3, 6);
         //fs1(0)=6;
         //fs1(1)=6;
         fs3(2) = 2;
         FunctionIdentifier iE1 = gmExplicit.addFunction(fe1);
         FunctionIdentifier iE2 = gmExplicit.addFunction(fe2);
         FunctionIdentifier iE3 = gmExplicit.addFunction(fe3);
         FunctionIdentifier iE1_ = gmMixed.addFunction(fe1);
         //FunctionIdentifier iE2_=gmMixed.addFunction(fe2);
         //FunctionIdentifier iE3_=gmMixed.addFunction(fe3);
         //FunctionIdentifier iS1_=gmMixed.addFunction(fs1);
         FunctionIdentifier iS2_ = gmMixed.addFunction(fs2);
         FunctionIdentifier iS3_ = gmMixed.addFunction(fs3);
         gmExplicit.addFactor(iE1, vi, vi + 1);
         gmExplicit.addFactor(iE2, vi + 1, vi + 2);
         gmExplicit.addFactor(iE3, vi + 2, vi + 3);
         gmMixed.addFactor(iE1_, vi, vi + 1);
         gmMixed.addFactor(iS2_, vi + 1, vi + 2);
         gmMixed.addFactor(iS3_, vi + 2, vi + 3);
      }

      {
         //save gm
         {
            opengm::hdf5::save(gmExplicit, "saveGmTestGmExplicit.h5", "explicit");
            opengm::hdf5::save(gmMixed, "saveGmTestGmMixed.h5", "mixed");
         }
         //load
         {
            GraphicalModelType loadedGmExplicit;
            GraphicalModelType loadedGmMixed;
            GraphicalModelTypeOnlyExplicit loadOnlyExplicit;
            opengm::hdf5::load(loadedGmExplicit, "saveGmTestGmExplicit.h5", "explicit");
            opengm::hdf5::load(loadedGmMixed, "saveGmTestGmMixed.h5", "mixed");
            opengm::hdf5::load(loadOnlyExplicit,"saveGmTestGmExplicit.h5", "explicit");

            GraphicalModelEqualityTest<GraphicalModelType, GraphicalModelType> testEqualGm;
            testEqualGm(gmExplicit, loadedGmExplicit);
            //testEqualGm(gmMixed,loadedGmMixed);
         }
         //load
         {
            GmTypeInt loadedGmExplicit;
            GmTypeInt loadedGmMixed;
            opengm::hdf5::load(loadedGmExplicit, "saveGmTestGmExplicit.h5", "explicit");
            opengm::hdf5::load(loadedGmMixed, "saveGmTestGmMixed.h5", "mixed");
         }
         //load
         {
            typedef T ValueType;
            typedef opengm::GraphicalModel
               <
               ValueType, //value type (should be float double or long double)
               opengm::Multiplier, //operator (something like Adder or Multiplier)
               typename opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType>,opengm::PottsNFunction<ValueType>, opengm::PottsFunction<ValueType> >::type
               >
               GmType2;
            GmType2 loadedGmExplicit;
            GmType2 loadedGmMixed;
            opengm::hdf5::load(loadedGmExplicit, "saveGmTestGmExplicit.h5", "explicit");
            opengm::hdf5::load(loadedGmMixed, "saveGmTestGmMixed.h5", "mixed");
         }
         //load
         {
            typedef T ValueType;
            typedef opengm::GraphicalModel
               <
               ValueType, //value type (should be float double or long double)
               opengm::Multiplier, //operator (something like Adder or Multiplier)
               typename opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType>,opengm::PottsFunction<ValueType>, opengm::PottsNFunction<ValueType> >::type
               >
               GmType2;
            GmType2 loadedGmExplicit;
            GmType2 loadedGmMixed;
            opengm::hdf5::load(loadedGmExplicit, "saveGmTestGmExplicit.h5", "explicit");
            opengm::hdf5::load(loadedGmMixed, "saveGmTestGmMixed.h5", "mixed");
         }
      }
   }
};

void testNumberOfFactors() {
   typedef opengm::GraphicalModel<int, opengm::Multiplier> GraphicalModel;
   typedef opengm::ExplicitFunction<int> Function;
   typedef GraphicalModel::FunctionIdentifier FID;

   size_t numbersOfStates[] = {2, 2, 2, 2};
   Function f1(numbersOfStates, numbersOfStates + 2);
   GraphicalModel gm(opengm::DiscreteSpace<size_t, size_t > (numbersOfStates, numbersOfStates + 4));
   FID fid1 = gm.addFunction(f1);
   size_t vi[] = {0, 1, 2, 3, 0, 3};
   gm.addFactor(fid1, vi + 0, vi + 2);
   gm.addFactor(fid1, vi + 1, vi + 3);
   gm.addFactor(fid1, vi + 2, vi + 4);
   gm.addFactor(fid1, vi + 4, vi + 6);
   OPENGM_ASSERT(gm.numberOfFactors(0) == 2);
   opengm::hdf5::save(gm, "saveGmTest.h5", "gm");
   GraphicalModel gm2;
   opengm::hdf5::load(gm2, "saveGmTest.h5", "gm");
   OPENGM_ASSERT(gm2.numberOfFactors(0) == 2);
}

int main() {

   testNumberOfFactors();
   std::cout << "Test hdf5 i/o  " << std::endl;
   {
      std::cout << "  * FLOAT" << std::endl;
      TestOpenGmHdf5<float> t;
      t.run();
   }
   {
      std::cout << "  * DOUBLE" << std::endl;
      TestOpenGmHdf5<double> t;
      t.run();
   }
   std::cout << "done.." << std::endl;
   return 0;
}

