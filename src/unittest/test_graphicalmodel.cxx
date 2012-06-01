#include <vector>

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
      typename opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType>, opengm::PottsNFunction<ValueType> >::type, //implicit function functor
      opengm::DiscreteSpace<I, L>,
      false
      >
      GraphicalModelType;
   typedef opengm::GraphicalModel
      <
      ValueType, //value type (should be float double or long double)
      opengm::Multiplier, //operator (something like Adder or Multiplier)
      typename opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType>, opengm::PottsNFunction<ValueType> >::type, //implicit function functor
      opengm::DiscreteSpace<I, L>,
      true
      >
      MGmType;
   typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
   //typedef typename GraphicalModelType::ImplicitFunctionType ImplicitFunctionType;
   typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;

   void testFunctionTypeList() {
      typedef opengm::meta::TypeListGenerator
         <
         opengm::ExplicitFunction<int>,
         opengm::PottsFunction<int>,
         opengm::PottsNFunction<int>
         >::type FunctionTypeList;
      typedef opengm::GraphicalModel<int, opengm::Minimizer, FunctionTypeList, opengm::DiscreteSpace<I, L> > GmTypeA;
      typedef opengm::GraphicalModel<int, opengm::Minimizer, opengm::ExplicitFunction<int>, opengm::DiscreteSpace<I, L> > GmTypeB;
      typedef typename GmTypeA::FunctionIdentifier FIA;
      typedef typename GmTypeB::FunctionIdentifier FIB;

      typedef opengm::ExplicitFunction<int> EF;

      size_t nos[] = {2, 2, 3};
      GmTypeA gmA(opengm::DiscreteSpace<I, L > (nos, nos + 3));
      GmTypeB gmB(opengm::DiscreteSpace<I, L > (nos, nos + 3));

      opengm::PottsNFunction<int> fp1(nos + 1, nos + 3, 0, 1);
      opengm::PottsFunction<int> fp2(2, 2, 0, 1);

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
      size_t via[] = {0, 1};
      //        size_t vib[] = {1,2};
      //        size_t vic[] = {0,2};
      //pair potentials functions
      {
         //pair potentials functions
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
         FunctionIdentifier iI = gmMixed.addFunction(fi);
         FunctionIdentifier iS = gmMixed.addFunction(fs);
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
         FunctionIdentifier iE2_ = gmMixed.addFunction(fe2);
         FunctionIdentifier iE3_ = gmMixed.addFunction(fe3);
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

   void testIntroduceEvidence() {
      size_t nos[] = {3, 2, 4, 5};
      GraphicalModelType gm;
      gm.assign(opengm::DiscreteSpace<I, L > (opengm::DiscreteSpace<I, L > (nos, nos + 4)));
      size_t vi[] = {0, 1, 2, 3};
      //vi=0,1,2 shape= 3,2,4
      {
         opengm::PottsNFunction<ValueType> i1(nos, nos + 3, 0, 1);
         ExplicitFunctionType e1(nos, nos + 3, 1);
         ExplicitFunctionType s1(nos, nos + 3, 1);
         for (size_t k = 0; k < 4; ++k)
            for (size_t j = 0; j < 2; ++j)
               for (size_t i = 0; i < 3; ++i) {
                  if (k == j && k == i) {
                     e1(i, j, k) = 0;
                     s1(i, j, k) = 0;
                  }
               }
         FunctionIdentifier ide = gm.addFunction(e1);
         FunctionIdentifier ids = gm.addFunction(s1);
         FunctionIdentifier idi = gm.addFunction(i1);
         gm.addFactor(ide, vi, vi + 3);
         gm.addFactor(ids, vi, vi + 3);
         gm.addFactor(idi, vi, vi + 3);
      }
      //vi=2,3 shape= 4,5
      {
         opengm::PottsNFunction<ValueType> i1(nos + 2, nos + 4, 0, 1);
         ExplicitFunctionType e1(nos + 2, nos + 4, 1);
         ExplicitFunctionType s1(nos + 2, nos + 4, 1);
         for (size_t j = 0; j < 5; ++j)
            for (size_t i = 0; i < 4; ++i) {
               if (i == j) {
                  e1(i, j) = 0;
                  s1(i, j) = 0;
               }
            }
         FunctionIdentifier ide = gm.addFunction(e1);
         FunctionIdentifier ids = gm.addFunction(s1);
         FunctionIdentifier idi = gm.addFunction(i1);
         gm.addFactor(ide, vi + 2, vi + 4);
         gm.addFactor(ids, vi + 2, vi + 4);
         gm.addFactor(idi, vi + 2, vi + 4);
      }
      //vi=0,shape=3
      {
         ExplicitFunctionType e1(nos, nos + 1);
         e1(0) = 0;
         e1(1) = 1;
         e1(2) = 2;
         ExplicitFunctionType s1(nos, nos + 1, 0);
         //s1(0)=0;
         s1(1) = 1;
         s1(2) = 2;
         FunctionIdentifier ide = gm.addFunction(e1);
         FunctionIdentifier ids = gm.addFunction(s1);
         gm.addFactor(ide, vi, vi + 1);
         gm.addFactor(ids, vi, vi + 1);
      }
      //vi=1,shape 2
      {
         ExplicitFunctionType e1(nos + 1, nos + 2);
         e1(0) = 0;
         e1(1) = 1;
         ExplicitFunctionType s1(nos + 1, nos + 2, 0);
         //s1(0)=0;
         s1(1) = 1;
         FunctionIdentifier ide = gm.addFunction(e1);
         FunctionIdentifier ids = gm.addFunction(s1);
         gm.addFactor(ide, vi + 1, vi + 2);
         gm.addFactor(ids, vi + 1, vi + 2);
      }
      //vi=2 ,shape=4
      {
         ExplicitFunctionType e1(nos + 2, nos + 3);
         e1(0) = 0;
         e1(1) = 1;
         e1(2) = 2;
         e1(3) = 3;
         ExplicitFunctionType s1(nos + 2, nos + 3, 0);
         //s1(0)=0;
         s1(1) = 1;
         s1(2) = 2;
         s1(3) = 3;
         FunctionIdentifier ide = gm.addFunction(e1);
         FunctionIdentifier ids = gm.addFunction(s1);
         gm.addFactor(ide, vi + 2, vi + 3);
         gm.addFactor(ids, vi + 2, vi + 3);
      }
      //vi=3 ,shape=5
      {
         ExplicitFunctionType e1(nos + 3, nos + 4);
         e1(0) = 0;
         e1(1) = 1;
         e1(2) = 2;
         e1(3) = 3;
         e1(4) = 4;
         ExplicitFunctionType s1(nos + 3, nos + 4, 0);
         //s1(0)=0;
         s1(1) = 1;
         s1(2) = 2;
         s1(3) = 3;
         s1(4) = 4;
         FunctionIdentifier ide = gm.addFunction(e1);
         FunctionIdentifier ids = gm.addFunction(s1);
         gm.addFactor(ide, vi + 3, vi + 4);
         gm.addFactor(ids, vi + 3, vi + 4);
      }
      //convert into a mutable graphical model
      MGmType gmm = gm;
      testEqualGm(gm, gmm);

      //evidence sequence:
      size_t viEvidence[] = {1, 3};
      size_t statesEvidence[] = {0, 1};
      gmm.introduceEvidence(viEvidence, viEvidence + 2, statesEvidence);

      //vi=0,1,2 shape= 3,2,4     =>vi 0,2  shape 3,4
      //0	e
      //1	s
      //2	i
      OPENGM_TEST_EQUAL(gmm[0].numberOfVariables(), 2);
      OPENGM_TEST_EQUAL(gmm[0].variableIndex(0), 0);
      OPENGM_TEST_EQUAL(gmm[0].variableIndex(1), 2);
      OPENGM_TEST_EQUAL(gmm[0].numberOfLabels(0), 3);
      OPENGM_TEST_EQUAL(gmm[0].numberOfLabels(1), 4);

      OPENGM_TEST_EQUAL(gmm[1].numberOfVariables(), 2);
      OPENGM_TEST_EQUAL(gmm[1].variableIndex(0), 0);
      OPENGM_TEST_EQUAL(gmm[1].variableIndex(1), 2);
      OPENGM_TEST_EQUAL(gmm[1].numberOfLabels(0), 3);
      OPENGM_TEST_EQUAL(gmm[1].numberOfLabels(1), 4);

      OPENGM_TEST_EQUAL(gmm[2].numberOfVariables(), 2);
      OPENGM_TEST_EQUAL(gmm[2].variableIndex(0), 0);
      OPENGM_TEST_EQUAL(gmm[2].variableIndex(1), 2);
      OPENGM_TEST_EQUAL(gmm[2].numberOfLabels(0), 3);
      OPENGM_TEST_EQUAL(gmm[2].numberOfLabels(1), 4);
      //vi=2,3 shape= 4,5			=>vi 2  shape 4
      //3	e
      //4	s
      //5	i
      OPENGM_TEST_EQUAL(gmm[3].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[3].variableIndex(0), 2);
      OPENGM_TEST_EQUAL(gmm[3].numberOfLabels(0), 4);

      OPENGM_TEST_EQUAL(gmm[4].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[4].variableIndex(0), 2);
      OPENGM_TEST_EQUAL(gmm[4].numberOfLabels(0), 4);

      OPENGM_TEST_EQUAL(gmm[5].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[5].variableIndex(0), 2);
      OPENGM_TEST_EQUAL(gmm[5].numberOfLabels(0), 4);
      //vi=0,shape=3				=>vi 0  shape 3
      //6 e
      //7 s
      OPENGM_TEST_EQUAL(gmm[6].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[6].variableIndex(0), 0);
      OPENGM_TEST_EQUAL(gmm[6].numberOfLabels(0), 3);

      OPENGM_TEST_EQUAL(gmm[7].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[7].variableIndex(0), 0);
      OPENGM_TEST_EQUAL(gmm[7].numberOfLabels(0), 3);
      //vi=1,shape=2				=>vi {}  shape {??}
      //8 e
      //9 s
      OPENGM_TEST_EQUAL(gmm[8].numberOfVariables(), 0);
      OPENGM_TEST_EQUAL(gmm[9].numberOfVariables(), 0);
      //vi=2,shape=4				=>vi 2  shape 4
      //10 e
      //11 s
      OPENGM_TEST_EQUAL(gmm[10].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[10].variableIndex(0), 2);
      OPENGM_TEST_EQUAL(gmm[10].numberOfLabels(0), 4);

      OPENGM_TEST_EQUAL(gmm[11].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[11].variableIndex(0), 2);
      OPENGM_TEST_EQUAL(gmm[11].numberOfLabels(0), 4);
      //vi=3,shape=4				=>vi {}  shape {??}
      //12 e
      //13 s
      OPENGM_TEST_EQUAL(gmm[12].numberOfVariables(), 0);
      OPENGM_TEST_EQUAL(gmm[13].numberOfVariables(), 0);
   }

   void testIsolateFactor() {
      size_t nos[] = {3, 2, 4, 5};
      MGmType gm;
      gm.assign(opengm::DiscreteSpace<I, L > (nos, nos + 4));
      size_t vi[] = {0, 1, 2, 3};
      //vi=0,1,2 shape= 3,2,4
      {
         opengm::PottsNFunction<ValueType> i1(nos, nos + 3, 0, 15);
         opengm::PottsNFunction<ValueType> i2(nos, nos + 3, 2, 3);
         opengm::PottsNFunction<ValueType> i3(nos, nos + 3, 4, 5);
         ExplicitFunctionType e1(nos, nos + 3, 7);
         ExplicitFunctionType e2(nos, nos + 3, 9);
         ExplicitFunctionType e3(nos, nos + 3, 11);
         for (size_t k = 0; k < 4; ++k)
            for (size_t j = 0; j < 2; ++j)
               for (size_t i = 0; i < 3; ++i) {
                  if (k == j && k == i) {
                     e1(i, j, k) = 4;
                     e2(i, j, k) = 5;
                     e2(i, j, k) = 6;
                  }
               }
         FunctionIdentifier ide1 = gm.addFunction(e1);
         FunctionIdentifier ide2 = gm.addFunction(e2);
         FunctionIdentifier ide3 = gm.addFunction(e3);
         FunctionIdentifier idi1 = gm.addFunction(i1);
         FunctionIdentifier idi2 = gm.addFunction(i2);
         FunctionIdentifier idi3 = gm.addFunction(i3);
         gm.addFactor(ide1, vi, vi + 3);
         gm.addFactor(ide1, vi, vi + 3);
         gm.addFactor(ide1, vi, vi + 3);

         gm.addFactor(idi1, vi, vi + 3);
         gm.addFactor(idi1, vi, vi + 3);
         gm.addFactor(idi1, vi, vi + 3);

         gm.addFactor(ide2, vi, vi + 3);
         gm.addFactor(ide2, vi, vi + 3);
         gm.addFactor(ide2, vi, vi + 3);

         gm.addFactor(idi2, vi, vi + 3);
         gm.addFactor(idi2, vi, vi + 3);
         gm.addFactor(idi2, vi, vi + 3);

         gm.addFactor(ide2, vi, vi + 3);
         gm.addFactor(ide2, vi, vi + 3);
         gm.addFactor(ide2, vi, vi + 3);

         gm.addFactor(idi2, vi, vi + 3);
         gm.addFactor(idi2, vi, vi + 3);
         gm.addFactor(idi2, vi, vi + 3);
         for (size_t k = 0; k < 4; ++k)
            for (size_t j = 0; j < 2; ++j)
               for (size_t i = 0; i < 3; ++i) {
                  size_t coordinate[] = {i, j, k};
                  if (k == j && k == i) {
                     OPENGM_TEST_EQUAL(e1(i, j, k), gm[2](coordinate));
                     OPENGM_TEST_EQUAL(0, gm[3](coordinate));
                  } else {
                     OPENGM_TEST_EQUAL(15, gm[3](coordinate));
                  }
               }
         //OPENGM_TEST_EQUAL(gm[2].functionType(), opengm::meta::GetIndexInTypeList<>);
         //OPENGM_TEST_EQUAL(gm[3].functionType(), 1);
         gm.isolateFactor(2);
         gm.isolateFactor(3);

         //OPENGM_TEST_EQUAL(gm[2].functionType(), 0);
         //OPENGM_TEST_EQUAL(gm[3].functionType(), 0);

         for (size_t k = 0; k < 4; ++k)
            for (size_t j = 0; j < 2; ++j)
               for (size_t i = 0; i < 3; ++i) {
                  size_t coordinate[] = {i, j, k};
                  if (k == j && k == i) {
                     OPENGM_TEST_EQUAL(e1(i, j, k), gm[2](coordinate));
                     OPENGM_TEST_EQUAL(0, gm[3](coordinate));
                  } else {
                     OPENGM_TEST_EQUAL(15, gm[3](coordinate));
                  }
               }
      }
   }

   void testReplaceFactor() {
      size_t nos[] = {3, 2, 4, 5};
      MGmType gm;
      gm.assign(opengm::DiscreteSpace<I, L > (nos, nos + 4));
      size_t vi[] = {0, 1, 2, 3};
      opengm::PottsNFunction<ValueType> i1(nos, nos + 3, 0, 1);
      ExplicitFunctionType e1(nos, nos + 3, 3);

      for (size_t k = 0; k < 4; ++k)
         for (size_t j = 0; j < 2; ++j)
            for (size_t i = 0; i < 3; ++i) {
               if (k == j && k == i) {
                  e1(i, j, k) = 2;
               }
            }
      FunctionIdentifier ide1 = gm.addFunction(e1);
      FunctionIdentifier idi1 = gm.addFunction(i1);
      //std::cout<<"\n \n add factors \n";
      gm.addFactor(ide1, vi, vi + 3);
      gm.addFactor(ide1, vi, vi + 3);
      gm.addFactor(ide1, vi, vi + 3);

      gm.addFactor(idi1, vi, vi + 3);
      gm.addFactor(idi1, vi, vi + 3);
      gm.addFactor(idi1, vi, vi + 3);


      //OPENGM_TEST_EQUAL(gm[2].functionType(), 0);
      //OPENGM_TEST_EQUAL(gm[3].functionType(), 1);
      gm.isolateFactor(2);
      gm.isolateFactor(3);
      //OPENGM_TEST_EQUAL(gm[2].functionType(), 0);
      //OPENGM_TEST_EQUAL(gm[3].functionType(), 0);
      //std::cout << "iosolating done \n";
      for (size_t k = 0; k < 4; ++k)
         for (size_t j = 0; j < 2; ++j)
            for (size_t i = 0; i < 3; ++i) {
               size_t coordinate[] = {i, j, k};
               if (k == j && k == i) {
                  OPENGM_TEST_EQUAL(e1(i, j, k), gm[2](coordinate));
                  OPENGM_TEST_EQUAL(0, gm[3](coordinate));
               } else {
                  OPENGM_TEST_EQUAL(1, gm[3](coordinate));
               }
            }
      gm.replaceFactor(3, 0, vi, vi + 3);
      for (size_t k = 0; k < 4; ++k)
         for (size_t j = 0; j < 2; ++j)
            for (size_t i = 0; i < 3; ++i) {
               size_t coordinate[] = {i, j, k};
               if (k == j && k == i) {
                  OPENGM_TEST_EQUAL(e1(i, j, k), gm[2](coordinate));
                  OPENGM_TEST_EQUAL(e1(i, j, k), gm[3](coordinate));
               }
               OPENGM_TEST_EQUAL(e1(i, j, k), gm[3](coordinate));
            }
   }

   void testIsAcyclic() {
      size_t nos[] = {2, 2, 2, 2, 2};
      GraphicalModelType gm;
      GraphicalModelType gm2;
      gm.assign(opengm::DiscreteSpace<I, L > (nos, nos + 5));
      gm2.assign(opengm::DiscreteSpace<I, L > (nos, nos + 5));
      opengm::PottsNFunction<ValueType> i1(nos, nos + 2, 0, 1);
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
      this->testFunctionTypeList();
      this->testConstructionAndAssigment();
      //test introduce evidence
      this->testIntroduceEvidence();
      //test isolate factor
      this->testIsolateFactor();
      //test replace factor
      this->testReplaceFactor();
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
