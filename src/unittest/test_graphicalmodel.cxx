#include <vector>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/utilities/metaprogramming.hxx>

template<class T, class I, class L>
struct GraphicalModelTest {
   typedef T ValueType;
   typedef opengm::GraphicalModel
   <
      ValueType, //value type (should be float double or long double)
      opengm::Multiplier, //operator (something like Adder or Multiplier)
      typename opengm::meta::TypeListGenerator<
         opengm::ExplicitFunction<ValueType,I,L>, 
         opengm::PottsNFunction<ValueType,I,L> 
      >::type, //implicit function functor
      opengm::DiscreteSpace<I, L>
   >  GraphicalModelType;
   typedef opengm::ExplicitFunction<ValueType,I,L> ExplicitFunctionType;
   //typedef typename GraphicalModelType::ImplicitFunctionType ImplicitFunctionType;
   typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;

   void testFunctionAccess() {
     GraphicalModelType gm;
     size_t shape[] = {1};
     ExplicitFunctionType f1(shape, shape + 1);
     f1(0) = 13;
     ExplicitFunctionType f2(shape, shape + 1);
     f2(0) = 26;
     
     FunctionIdentifier fid1 = gm.addFunction(f1);
     FunctionIdentifier fid2 = gm.addFunction(f2);
     ExplicitFunctionType g2 = gm.template getFunction<ExplicitFunctionType>(fid2);
     ExplicitFunctionType g1 = gm.template getFunction<ExplicitFunctionType>(fid1);
     
     OPENGM_TEST_EQUAL(g1(0), 13);
     OPENGM_TEST_EQUAL(g2(0), 26);
   };

   void testFunctionTypeList() {
      typedef typename opengm::meta::TypeListGenerator
         <
         opengm::ExplicitFunction<int,I,L>,
         opengm::PottsFunction<int,I,L>,
         opengm::PottsNFunction<int,I,L>
         >::type FunctionTypeList;
      typedef opengm::GraphicalModel<int, opengm::Minimizer, FunctionTypeList, opengm::DiscreteSpace<I, L> > GmTypeA;
      typedef opengm::GraphicalModel<int, opengm::Minimizer, opengm::ExplicitFunction<int,I,L>, opengm::DiscreteSpace<I, L> > GmTypeB;
      typedef typename GmTypeA::FunctionIdentifier FIA;
      typedef typename GmTypeB::FunctionIdentifier FIB;

      typedef opengm::ExplicitFunction<int,I,L> EF;

      size_t nos[] = {2, 2, 3};
      GmTypeA gmA(opengm::DiscreteSpace<I, L > (nos, nos + 3));
      GmTypeB gmB(opengm::DiscreteSpace<I, L > (nos, nos + 3));

      opengm::PottsNFunction<int,I,L> fp1(nos + 1, nos + 3, 0, 1);
      opengm::PottsFunction<int,I,L> fp2(2, 2, 0, 1);

      EF fe1(nos + 1, nos + 3, 1);
      fe1(0, 0) = 0;
      fe1(1, 1) = 0;

      EF fe2(nos, nos + 2, 1);
      fe2(0, 0) = 0;
      fe2(1, 1) = 0;

      FIA ia = gmA.addFunction(fp1);
      FIA ib = gmA.addFunction(fp2);

      FIB iae = gmB.addFunction(fe1);
      FIB ibe = gmB.addFunction(fe2);
      FIB iaes = gmB.addSharedFunction(fe1);
      FIB ibes = gmB.addSharedFunction(fe2);
      OPENGM_ASSERT(iae==iaes);
      OPENGM_ASSERT(ibe==ibes);
      size_t vi[] = {0, 1, 2};
      gmA.addFactor(ia, vi + 1, vi + 3);
      gmA.addFactor(ib, vi, vi + 2);

      gmB.addFactor(iae, vi + 1, vi + 3);
      gmB.addFactor(ibe, vi, vi + 2);
      
      gmB.addFactor(iaes, vi + 1, vi + 3);
      gmB.addFactor(ibes, vi, vi + 2);
      
      OPENGM_ASSERT(gmB[0].functionIndex()==gmB[2].functionIndex());
      OPENGM_ASSERT(gmB[0].functionType()==gmB[2].functionType());
      OPENGM_ASSERT(gmB[1].functionIndex()==gmB[3].functionIndex());
      OPENGM_ASSERT(gmB[1].functionType()==gmB[3].functionType());
      size_t c[] = {0, 0};
      for (size_t i = 0; i < 3; ++i)
         for (size_t j = 0; j < 2; ++j) {
            c[0] = j;
            c[1] = i;
            OPENGM_TEST(gmA[0](c) == gmB[0](c));
         }

      for (size_t i = 0; i < 2; ++i)
         for (size_t j = 0; j < 2; ++j) {
            c[0] = j;
            c[1] = i;
            OPENGM_TEST(gmA[1](c) == gmB[1](c));
         }
   };

   void testGenerateModels(GraphicalModelType & explicitGm, GraphicalModelType & mixedGm) {
      size_t nos3[] = {3, 3, 3};
      GraphicalModelType gmExplicit(opengm::DiscreteSpace<I, L > (nos3, nos3 + 3));
      GraphicalModelType gmMixed(opengm::DiscreteSpace<I, L > (nos3, nos3 + 3));
      size_t vi[] = {0, 1, 2};
      //size_t via[] = {0, 1};
      //        size_t vib[] = {1,2};
      //        size_t vic[] = {0,2};
      //pair potentials functions
      {
         //pair potentials functions
         opengm::PottsNFunction<ValueType,I,L> fi(nos3, nos3 + 3, 1, 6);
         ExplicitFunctionType fe(nos3, nos3 + 3, 6);
         fe(0, 0, 0) = 1;
         fe(1, 1, 1) = 1;
         fe(2, 2, 2) = 1;
         ExplicitFunctionType fs(nos3, nos3 + 3, 6);
         fs(0, 0, 0) = 1;
         fs(1, 1, 1) = 1;
         fs(2, 2, 2) = 1;
         FunctionIdentifier iE = gmExplicit.addFunction(fe);
         //FunctionIdentifier iI = gmMixed.addFunction(fi);
         //FunctionIdentifier iS = gmMixed.addFunction(fs);
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
         //FunctionIdentifier iE2_ = gmMixed.addFunction(fe2);
         //FunctionIdentifier iE3_ = gmMixed.addFunction(fe3);
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
      mixedGm = gmMixed;
      explicitGm = gmExplicit;
      testEqualGm(mixedGm, gmMixed);
      testEqualGm(explicitGm, gmExplicit);
   }

   void testConstructionAndAssigment() {
      GraphicalModelType c;
      //MutableGmType m;
      OPENGM_TEST(c.numberOfVariables() == 0);
      OPENGM_TEST(c.numberOfFactors() == 0);
      OPENGM_TEST(c.factorOrder() == 0);
      OPENGM_TEST(c.isAcyclic() == 1);
      //OPENGM_TEST(GraphicalModelType::IsMutable==0);
      //OPENGM_TEST(MutableGmType::IsMutable==1);
      //testEqualGm(c,m);

      {
         GraphicalModelType c1 = c, c2 = c;
         //MutableGmType m1=c,m2=m;
         testEqualGm(c, c1);
         testEqualGm(c, c2);
         //testEqualGm(c,m1);
         //testEqualGm(c,m2);
      }
      {
         GraphicalModelType c1, c2;
         c1 = c;
         c2 = c;
         //MutableGmType m1,m2;m1=c;m2=m;
         testEqualGm(c, c1);
         testEqualGm(c, c2);
         //testEqualGm(c,m1);
         //testEqualGm(c,m2);
      }
      {
         GraphicalModelType ce, cm;
         this->testGenerateModels(ce, cm);
         OPENGM_TEST(ce.numberOfVariables() == 3);
         OPENGM_TEST(ce.numberOfFactors() == 6);
         OPENGM_TEST(ce.factorOrder() == 3)


         OPENGM_TEST(cm.numberOfVariables() == 3);
         OPENGM_TEST(cm.numberOfFactors() == 6);
         OPENGM_TEST(cm.factorOrder() == 3);
         //MutableGmType me=ce,mm=cm;
         //testEqualGm(ce,me);
         //testEqualGm(cm,mm);
         {
            GraphicalModelType c1 = ce, c2 = ce;
            //MutableGmType m1=ce,m2=me;
            testEqualGm(ce, c1);
            testEqualGm(ce, c2);
            //testEqualGm(ce,m1);
            //testEqualGm(ce,m2);
         }
         {
            GraphicalModelType c1, c2;
            c1 = ce;
            c2 = ce;
            //MutableGmType m1,m2;m1=ce;m2=me;
            testEqualGm(ce, c1);
            testEqualGm(ce, c2);
            //testEqualGm(ce,m1);
            //testEqualGm(ce,m2);
         }
         {
            GraphicalModelType c1 = cm, c2 = cm;
            //MutableGmType m1=cm,m2=mm;
            testEqualGm(cm, c1);
            testEqualGm(cm, c2);
            //testEqualGm(cm,m1);
            //testEqualGm(cm,m2);
         }
         {
            GraphicalModelType c1, c2;
            c1 = cm;
            c2 = cm;
            //MutableGmType m1,m2;m1=cm;m2=mm;
            testEqualGm(cm, c1);
            testEqualGm(cm, c2);
            //testEqualGm(cm,m1);
            //testEqualGm(cm,m2);
         }
      }
   }



   void testIsAcyclic() {
      size_t nos[] = {2, 2, 2, 2, 2};
      GraphicalModelType gm;
      GraphicalModelType gm2;
      gm.assign(opengm::DiscreteSpace<I, L > (nos, nos + 5));
      gm2.assign(opengm::DiscreteSpace<I, L > (nos, nos + 5));
      opengm::PottsNFunction<ValueType,I,L> i1(nos, nos + 2, 0, 1);
      ExplicitFunctionType e1(nos, nos + 1, 1);
      {
         FunctionIdentifier ide1 = gm.addFunction(e1);
         FunctionIdentifier idi1 = gm.addFunction(i1);
         for (size_t i = 0; i < 5; ++i)
            gm.addFactor(ide1, &i, &i + 1);
         {
            size_t v[] = {0, 2};
            gm.addFactor(idi1, v, v + 2);
         }
         {
            size_t v[] = {1, 2};
            gm.addFactor(idi1, v, v + 2);
         }
         {
            size_t v[] = {2, 3};
            gm.addFactor(idi1, v, v + 2);
         }
         {
            size_t v[] = {2, 4};
            gm.addFactor(idi1, v, v + 2);
         }
      }
      {
         for (size_t i = 0; i < 5; ++i) {
            FunctionIdentifier ide1 = gm2.addFunction(e1);
            gm2.addFactor(ide1, &i, &i + 1);
         }
         {
            size_t v[] = {0, 2};
            FunctionIdentifier idi1 = gm2.addFunction(i1);
            gm2.addFactor(idi1, v, v + 2);
         }
         {
            size_t v[] = {1, 2};
            FunctionIdentifier idi1 = gm2.addFunction(i1);
            gm2.addFactor(idi1, v, v + 2);
         }
         {
            size_t v[] = {2, 3};
            FunctionIdentifier idi1 = gm2.addFunction(i1);
            gm2.addFactor(idi1, v, v + 2);
         }
         {
            size_t v[] = {2, 4};
            FunctionIdentifier idi1 = gm2.addFunction(i1);
            gm2.addFactor(idi1, v, v + 2);
         }
      }
      OPENGM_ASSERT(gm2.isAcyclic());
      OPENGM_ASSERT(gm.isAcyclic());
   }

   void run() {
      //a lot of gm functions are constructed implicitly within
      //testConstructionAndAssigment()
      this->testFunctionAccess();
      this->testFunctionTypeList();
      this->testConstructionAndAssigment();
      //test isAcyclic
      this->testIsAcyclic();
   }
};

int main() {
   {
      std::cout << "GraphicalModel test... " << std::flush;
      {
         GraphicalModelTest<float, unsigned int, unsigned short> t;
         t.run();
      }
      {
         GraphicalModelTest<double, unsigned long, unsigned short> t;
         t.run();
      }
      {
         GraphicalModelTest<double, unsigned short, unsigned short> t;
         t.run();
      }
      {
         GraphicalModelTest<double, size_t, size_t> t;
         t.run();
      }
      std::cout << "done." << std::endl;
   }

   return 0;
}
