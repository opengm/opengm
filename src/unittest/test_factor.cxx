#include <vector>

#include <opengm/functions/pottsn.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>

template<class T>
struct FactorTest {
   typedef T ValueType;
   typedef opengm::GraphicalModel
      <
      ValueType, //value type (should be float double or long double)
      opengm::Multiplier, //operator (something like Adder or Multiplier)
      typename opengm::meta::TypeListGenerator<
         opengm::PottsNFunction<ValueType,size_t,size_t>, //implicit function functor
         opengm::ExplicitFunction<ValueType,size_t,size_t>
      >::type ,
      opengm::DiscreteSpace<size_t,size_t>
      >
      GraphicalModelType;


   typedef typename opengm::ExplicitFunction<ValueType,size_t,size_t> ExplicitFunctionType;
   //typedef typename  GraphicalModelType::ImplicitFunctionType ImplicitFunctionType ;
   typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
   typedef typename GraphicalModelType::FactorType ConstFactorType;
   typedef typename GraphicalModelType::FactorType MutableFactorType;

   void testFactorValues() {
      size_t nos3[] = {3, 3, 3};
      GraphicalModelType gmExplicit(opengm::DiscreteSpace<size_t,size_t>(nos3, nos3 + 3));
      GraphicalModelType gmMixed(opengm::DiscreteSpace<size_t,size_t>(nos3, nos3 + 3));
      size_t vi[] = {0, 1, 2};
      //            size_t via[] = {0,1};
      //            size_t vib[] = {1,2};
      //            size_t vic[] = {0,2};
      //pair potentials functions
      {
         //pair potentials functions
         opengm::PottsNFunction<ValueType> fi(nos3, nos3 + 3, 0, 1);
         ExplicitFunctionType fe(nos3, nos3 + 3, 1);
         fe(0, 0, 0) = 0;
         fe(1, 1, 1) = 0;
         fe(2, 2, 2) = 0;
         ExplicitFunctionType fs(nos3, nos3 + 3, 1);
         fs(0, 0, 0) = 0;
         fs(1, 1, 1) = 0;
         fs(2, 2, 2) = 0;
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
         for(size_t f = 0; f < 3; ++f)
            for(size_t k = 0; k < 3; ++k)
               for(size_t j = 0; j < 3; ++j)
                  for(size_t i = 0; i < 3; ++i) {
                     size_t coordinate[] = {i, j, k};
                     if(k == j && k == i) {
                        OPENGM_TEST_EQUAL(gmExplicit[f](coordinate), 0);
                        OPENGM_TEST_EQUAL(gmMixed[f](coordinate), 0);
                     }
                     else {
                        OPENGM_TEST_EQUAL(gmExplicit[f](coordinate), 1);
                        OPENGM_TEST_EQUAL(gmMixed[f](coordinate), 1);
                     }
                  }
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
         FunctionIdentifier iS2_ = gmMixed.addFunction(fs2);
         FunctionIdentifier iS3_ = gmMixed.addFunction(fs3);
         gmExplicit.addFactor(iE1, vi, vi + 1);
         gmExplicit.addFactor(iE2, vi + 1, vi + 2);
         gmExplicit.addFactor(iE3, vi + 2, vi + 3);
         gmMixed.addFactor(iE1_, vi, vi + 1);
         gmMixed.addFactor(iS2_, vi + 1, vi + 2);
         gmMixed.addFactor(iS3_, vi + 2, vi + 3);
         {
            size_t coordinate[] = {0};
            OPENGM_TEST_EQUAL_TOLERANCE(gmExplicit[3](coordinate), 2.4, 0.0001);
            OPENGM_TEST_EQUAL_TOLERANCE(gmMixed[3](coordinate), 2.4, 0.0001);
            coordinate[0] = 1;
            OPENGM_TEST_EQUAL(gmExplicit[3](coordinate), 6);
            OPENGM_TEST_EQUAL(gmMixed[3](coordinate), 6);
            coordinate[0] = 2;
            OPENGM_TEST_EQUAL(gmExplicit[3](coordinate), 6);
            OPENGM_TEST_EQUAL(gmMixed[3](coordinate), 6);
         }
         {
            size_t coordinate[] = {0};
            OPENGM_TEST_EQUAL(gmExplicit[4](coordinate), 6);
            OPENGM_TEST_EQUAL(gmMixed[4](coordinate), 6);
            coordinate[0] = 1;
            OPENGM_TEST_EQUAL(gmExplicit[4](coordinate), 1.5);
            OPENGM_TEST_EQUAL(gmMixed[4](coordinate), 1.5);
            coordinate[0] = 2;
            OPENGM_TEST_EQUAL(gmExplicit[4](coordinate), 6);
            OPENGM_TEST_EQUAL(gmMixed[4](coordinate), 6);
         }
         {
            size_t coordinate[] = {0};
            OPENGM_TEST_EQUAL(gmExplicit[5](coordinate), 6);
            OPENGM_TEST_EQUAL(gmMixed[5](coordinate), 6);
            coordinate[0] = 1;
            OPENGM_TEST_EQUAL(gmExplicit[5](coordinate), 6);
            OPENGM_TEST_EQUAL(gmMixed[5](coordinate), 6);
            coordinate[0] = 2;
            OPENGM_TEST_EQUAL(gmExplicit[5](coordinate), 2);
            OPENGM_TEST_EQUAL(gmMixed[5](coordinate), 2);
         }
      }
   }
   /*
   void testFixVariables() {
      //std::cout<<"start fix variables test"<<"\n";
      size_t nos[] = {3, 2, 4, 5};
      GraphicalModelType gm; //(nos, nos + 4);
      gm.assign(opengm::DiscreteSpace<size_t,size_t>(nos, nos + 4));
      size_t vi[] = {0, 1, 2, 3};
      //vi=0,1,2 shape= 3,2,4
      {
         opengm::PottsNFunction<ValueType> i1(nos, nos + 3, 0, 1);
         ExplicitFunctionType e1(nos, nos + 3, 1);
         ExplicitFunctionType s1(nos, nos + 3, 1);
         for(size_t k = 0; k < 4; ++k)
            for(size_t j = 0; j < 2; ++j)
               for(size_t i = 0; i < 3; ++i) {
                  size_t index[] = {i, j, k};
                  if(k == j && k == i) {
                     e1(i, j, k) = 0;
                     s1(i, j, k) = 0;
                     OPENGM_TEST_EQUAL(i1(index), 0);
                  }
                  else {
                     OPENGM_TEST_EQUAL(i1(index), 1);
                  }

                  OPENGM_TEST_EQUAL(e1(index), i1(index));
                  OPENGM_TEST_EQUAL(e1(index), s1(index));
         }
         FunctionIdentifier ide = gm.addFunction(e1);
         FunctionIdentifier ids = gm.addFunction(s1);
         FunctionIdentifier idi = gm.addFunction(i1);
         {
            gm.addFactor(ide, vi, vi + 3);
            size_t coordinate[] = {0, 0, 0};
            OPENGM_TEST_EQUAL(gm[0](coordinate), 0);
            coordinate[2] = 1;
            OPENGM_TEST_EQUAL(gm[0](coordinate), 1);
            gm.addFactor(ids, vi, vi + 3);
            gm.addFactor(idi, vi, vi + 3);
         }
         for(size_t k = 0; k < 4; ++k) {
            for(size_t j = 0; j < 2; ++j) {
               for(size_t i = 0; i < 3; ++i) {
                  size_t coordinate[] = {i, j, k};
                  OPENGM_TEST_EQUAL(gm[0](coordinate), e1(coordinate));
                  OPENGM_TEST_EQUAL(gm[1](coordinate), s1(coordinate));
                  OPENGM_TEST_EQUAL(e1(coordinate), i1(coordinate));
                  OPENGM_TEST_EQUAL(gm[2](coordinate), i1(coordinate));
                  OPENGM_TEST_EQUAL(gm[0](coordinate), gm[1](coordinate));
                  OPENGM_TEST_EQUAL(gm[0](coordinate), gm[2](coordinate));
                  if(k == j && k == i) {
                     OPENGM_TEST_EQUAL(gm[0](coordinate), 0);
                     OPENGM_TEST_EQUAL(gm[1](coordinate), 0);
                     OPENGM_TEST_EQUAL(gm[2](coordinate), 0);
                  }
                  else {
                     OPENGM_TEST_EQUAL(gm[0](coordinate), 1);
                     OPENGM_TEST_EQUAL(gm[1](coordinate), 1);
                     OPENGM_TEST_EQUAL(gm[2](coordinate), 1);
                  }
               }
            }
         }
      }
      //vi=2,3 shape= 4,5
      {
         opengm::PottsNFunction<ValueType> i1(nos + 2, nos + 4, 0, 1);
         ExplicitFunctionType e1(nos + 2, nos + 4, 1);
         ExplicitFunctionType s1(nos + 2, nos + 4, 1);
         for(size_t j = 0; j < 5; ++j)
            for(size_t i = 0; i < 4; ++i) {
               if(i == j) {
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
         s1(0) = 0;
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
         s1(0) = 0;
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
         s1(0) = 0;
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
         s1(0) = 0;
         s1(1) = 1;
         s1(2) = 2;
         s1(3) = 3;
         s1(4) = 4;
         FunctionIdentifier ide = gm.addFunction(e1);
         FunctionIdentifier ids = gm.addFunction(s1);
         gm.addFactor(ide, vi + 3, vi + 4);
         gm.addFactor(ids, vi + 3, vi + 4);
      }

      //MutableGmType gmm = gm;
      //testEqualGm(gm, gmm);

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
      for(size_t j = 0; j < 4; ++j)
         for(size_t i = 0; i < 3; ++i) {
            size_t coordinate[] = {i, j};
            if(i == 0 && j == 0) {
               OPENGM_TEST_EQUAL(gmm[0](coordinate), 0);
               OPENGM_TEST_EQUAL(gmm[1](coordinate), 0);
               OPENGM_TEST_EQUAL(gmm[2](coordinate), 0);
            }
            else {
               OPENGM_TEST_EQUAL(gmm[0](coordinate), 1);
               OPENGM_TEST_EQUAL(gmm[1](coordinate), 1);
               OPENGM_TEST_EQUAL(gmm[2](coordinate), 1);
            }
         }
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
      for(size_t i = 0; i < 4; ++i) {
         size_t coordinate[] = {i};
         if(i == 1) {
            OPENGM_TEST_EQUAL(gmm[3](coordinate), 0);
            OPENGM_TEST_EQUAL(gmm[4](coordinate), 0);
            OPENGM_TEST_EQUAL(gmm[5](coordinate), 0);
         }
         else {
            OPENGM_TEST_EQUAL(gmm[3](coordinate), 1);
            OPENGM_TEST_EQUAL(gmm[4](coordinate), 1);
            OPENGM_TEST_EQUAL(gmm[5](coordinate), 1);
         }
      }
      //vi=0,shape=3				=>vi 0  shape 3
      //6 e
      //7 s
      OPENGM_TEST_EQUAL(gmm[6].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[6].variableIndex(0), 0);
      OPENGM_TEST_EQUAL(gmm[6].numberOfLabels(0), 3);

      OPENGM_TEST_EQUAL(gmm[7].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[7].variableIndex(0), 0);
      OPENGM_TEST_EQUAL(gmm[7].numberOfLabels(0), 3);
      for(size_t i = 0; i < 3; ++i) {
         size_t coordinate[] = {i};
         OPENGM_TEST_EQUAL(gmm[6](coordinate), i);
         OPENGM_TEST_EQUAL(gmm[7](coordinate), i);
      }
      //vi=1,shape=2				=>vi {}  shape {??}
      //8 e
      //9 s
      OPENGM_TEST_EQUAL(gmm[8].numberOfVariables(), 0);
      OPENGM_TEST_EQUAL(gmm[9].numberOfVariables(), 0);
      {
         size_t coordinate[] = {0};
         OPENGM_TEST_EQUAL(gmm[8](coordinate), 0);
         OPENGM_TEST_EQUAL(gmm[9](coordinate), 0);
      }
      //vi=2,shape=4				=>vi 2  shape 4
      //10 e
      //11 s
      OPENGM_TEST_EQUAL(gmm[10].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[10].variableIndex(0), 2);
      OPENGM_TEST_EQUAL(gmm[10].numberOfLabels(0), 4);

      OPENGM_TEST_EQUAL(gmm[11].numberOfVariables(), 1);
      OPENGM_TEST_EQUAL(gmm[11].variableIndex(0), 2);
      OPENGM_TEST_EQUAL(gmm[11].numberOfLabels(0), 4);
      for(size_t i = 0; i < 4; ++i) {
         size_t coordinate[] = {i};
         OPENGM_TEST_EQUAL(gmm[10](coordinate), i);
         OPENGM_TEST_EQUAL(gmm[11](coordinate), i);
      }
      //vi=3,shape=4				=>vi {}  shape {??}
      //12 e
      //13 s
      OPENGM_TEST_EQUAL(gmm[12].numberOfVariables(), 0);
      OPENGM_TEST_EQUAL(gmm[13].numberOfVariables(), 0);
      {
         size_t coordinate[] = {0};
         OPENGM_TEST_EQUAL(gmm[12](coordinate), 1);
         OPENGM_TEST_EQUAL(gmm[13](coordinate), 1);
      }
   }
   */

   void testScalarFactor() { }

   void testFactorProperties() { 
      size_t nos3[] = {3, 3, 3};
      GraphicalModelType gm(opengm::DiscreteSpace<size_t,size_t>(nos3, nos3 + 3));
      size_t vi[] = {0, 1, 2};
      {
         //pair potentials functions
         opengm::PottsNFunction<ValueType> fi(nos3, nos3 + 3, 0, 1);
         ExplicitFunctionType fe(nos3, nos3 + 3, 1);
         fe(0, 0, 0) = 0;
         fe(1, 1, 1) = 0;
         fe(2, 2, 2) = 0;
     
         FunctionIdentifier iE = gm.addFunction(fe);
         FunctionIdentifier iI = gm.addFunction(fi);
        
         gm.addFactor(iE, vi, vi + 3);
         gm.addFactor(iI, vi, vi + 3);
         
         OPENGM_TEST(gm[0].isPotts());
         OPENGM_TEST(gm[1].isPotts());
         
         {
            
            {
               bool a=gm[0].isPotts();
               bool b=gm[0]. template binaryProperty<opengm::BinaryProperties::IsPotts>();
               OPENGM_TEST(a==b);
            }
            {
               ValueType a=gm[0].sum();
               ValueType b=gm[0]. template valueProperty<opengm::ValueProperties::Sum>();
               
               opengm::AccumulationFunctor<opengm::Adder,ValueType> functor;
               gm[0].forAllValuesInAnyOrder(functor);
               ValueType c=functor.value();
               
               OPENGM_TEST_EQUAL(a,b);
               OPENGM_TEST_EQUAL(a,c);
            }
            {
               ValueType a=gm[0].product();
               ValueType b=gm[0]. template valueProperty<opengm::ValueProperties::Product>();
               OPENGM_TEST(a==b);
            }
            /*
            a=gm[0].isGeneralizedPotts();
            bool c=gm[0].isSubmodular(); //not jet implemented for order 3
            
            ValueType b=gm[0].sum();
            b=gm[0].product();
            b=gm[0].min();
            b=gm[0].max();
            
             a=gm[0].isSquaredDifference(); 
             a=gm[0].isTruncatedSquaredDifference();
             a=gm[0].isAbsoluteDifference();
             a=gm[0].isTruncatedAbsoluteDifference();
             * */

         } 
      }
   }

   void run() {
      this->testFactorValues();
      //this->testFixVariables(); 
      this->testFactorProperties();
   }
};

int
main() {
   std::cout << "Factor test...  " << std::endl;
   {
      FactorTest<float >t;
      t.run();
   }
   {
      FactorTest<double >t;
      t.run();
   }
   std::cout << "done.." << std::endl;
   return 0;
}
