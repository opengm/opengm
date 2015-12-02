#include <vector>

#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/absolute_difference.hxx"
#include "opengm/functions/constant.hxx"
#include "opengm/functions/modelviewfunction.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/pottsn.hxx"
#include "opengm/functions/pottsg.hxx"
#include "opengm/functions/scaled_view.hxx"
#include "opengm/functions/squared_difference.hxx"
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "opengm/functions/view.hxx"
#include "opengm/functions/singlesitefunction.hxx"
#include "opengm/functions/view_fix_variables_function.hxx"
#include "opengm/functions/fieldofexperts.hxx"
#include <opengm/functions/constraint_functions/linear_constraint_function.hxx>
#include <opengm/functions/constraint_functions/label_order_function.hxx>
#include <opengm/functions/constraint_functions/num_labels_limitation_function.hxx>
#include <opengm/functions/soft_constraint_functions/sum_constraint_function.hxx>
#include <opengm/functions/soft_constraint_functions/label_cost_function.hxx>

#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/utilities/random.hxx>

template<class T>
struct FunctionsTest {
   template<class TYPE>
   struct ComparePairs{
      ComparePairs(const TYPE& vectorIn): vector(vectorIn) {}
      const TYPE& vector;
      bool operator()(int a, int b){ return vector[a] < vector[b]; }
   };

   template<class TYPE>
   ComparePairs<TYPE> CreateComparePairs(const TYPE& vector) { return ComparePairs<TYPE>(vector); }

   // sort vector in non descending order and compute corresponding permutation vector
   template<class TYPE1, class TYPE2>
   void sortingPermutation(const TYPE1& values, TYPE2& permutation){
      permutation.clear();
      permutation.reserve(values.size());
      for(size_t i = 0; i < values.size(); i++) {
         permutation.push_back(i);
      }
      std::sort(permutation.begin(), permutation.end(), CreateComparePairs(values));
   }
   
   template<class FUNCTION>
   void testProperties(const FUNCTION & f) {
      opengm::ShapeWalker<typename FUNCTION::FunctionShapeIteratorType > walker(f.functionShapeBegin(),f.dimension());
      for(size_t i=0;i<f.dimension();++i) {
         OPENGM_TEST(f.functionShapeBegin()[i] == f.shape(i));   
      }
      
      typedef typename  FUNCTION::ValueType ValueType;
      ValueType min=f(walker.coordinateTuple().begin());
      ValueType max=f(walker.coordinateTuple().begin());
      ValueType sum=static_cast<ValueType>(0);
      ValueType prod=static_cast<ValueType>(1);
      for(size_t i=0;i<f.size();++i) {
         ValueType tmp=f(walker.coordinateTuple().begin());
         min=tmp<min?tmp:min;
         max=tmp>max?tmp:max;
         sum+=tmp;
         prod*=tmp;
         ++walker;
      }
      OPENGM_TEST_EQUAL_TOLERANCE(min,f.min(),static_cast<ValueType>(0.0001));
      OPENGM_TEST_EQUAL_TOLERANCE(max,f.max(),static_cast<ValueType>(0.0001));
      OPENGM_TEST_EQUAL_TOLERANCE(f.minMax().max(),f.max(),static_cast<ValueType>(0.0001));
      OPENGM_TEST_EQUAL_TOLERANCE(f.minMax().min(),f.min(),static_cast<ValueType>(0.0001));
      OPENGM_TEST_EQUAL_TOLERANCE(sum,f.sum(),static_cast<ValueType>(0.0001));
      OPENGM_TEST_EQUAL_TOLERANCE(prod,f.product(),static_cast<ValueType>(0.0001));
   }
   
   template<class FUNCTION>
   void testSerialization(const FUNCTION & f) {

      const size_t sizeIndices=opengm::FunctionSerialization<FUNCTION>::indexSequenceSize(f);
      const size_t sizeValues=opengm::FunctionSerialization<FUNCTION>::valueSequenceSize(f);
      std::vector<long long unsigned> indices(sizeIndices);
      std::vector<T> values(sizeValues);

      opengm::FunctionSerialization<FUNCTION>::serialize(f,indices.begin(),values.begin());
      FUNCTION f2;
      opengm::FunctionSerialization<FUNCTION>::deserialize(indices.begin(),values.begin(),f2);

      OPENGM_TEST(f.dimension()==f2.dimension());
      OPENGM_TEST(f.size() == f2.size());
      std::vector<size_t> shape(f.dimension());
      for(size_t i=0;i<f.dimension();++i) {
         shape[i]=f.shape(i);
         OPENGM_TEST(f.shape(i)==f2.shape(i));
      }
      opengm::ShapeWalker<std::vector<size_t>::const_iterator > walker(shape.begin(),f.dimension());
      for(size_t i=0;i<f.size();++i) {
         OPENGM_TEST(walker.coordinateTuple().size()==f.dimension());
         OPENGM_TEST(f(walker.coordinateTuple().begin())==f2(walker.coordinateTuple().begin()) );
         ++walker;
      }
   }

   void testAbsoluteDifference() {
      std::cout << "  * AbsoluteDifference" << std::endl;
      opengm::AbsoluteDifferenceFunction<T> f(4,4);
      OPENGM_TEST(f.shape(0)==4);
      OPENGM_TEST(f.shape(1)==4);
      OPENGM_TEST(f.dimension()==2);
      OPENGM_TEST(f.size()==16);


      size_t i[]={0,1};
      OPENGM_TEST(f(i)==1);
      i[0]=1;
      i[1]=0;
      OPENGM_TEST(f(i)==1);
      i[0]=0;
      i[1]=0;
      OPENGM_TEST(f(i)==0);
      i[0]=1;
      i[1]=1;
      OPENGM_TEST(f(i)==0);
      i[0]=2;
      i[1]=1;
      OPENGM_TEST(f(i)==1);
      i[0]=3;
      i[1]=1;
      OPENGM_TEST(f(i)==2);
      i[0]=1;
      i[1]=3;
      OPENGM_TEST(f(i)==2);
      testSerialization(f);
      testProperties(f);
   }
   void testConstant() {
      std::cout << "  * Constant" << std::endl;
      std::vector<size_t> shape1(1,4);
      std::vector<size_t> shape2(2,4);
      opengm::ConstantFunction<T> f1(shape1.begin(),shape1.end(),1);
      OPENGM_TEST(f1.shape(0)==4);
      OPENGM_TEST(f1.dimension()==1);
      OPENGM_TEST(f1.size()==4);
      opengm::ConstantFunction<T> f2(shape2.begin(),shape2.end(),2);
      OPENGM_TEST(f2.shape(0)==4);
      OPENGM_TEST(f2.shape(1)==4);
      OPENGM_TEST(f2.dimension()==2);
      OPENGM_TEST(f2.size()==16);
      size_t i[] = {0, 1};
      i[0]=0;
      i[1]=0;
      OPENGM_TEST(f1(i)==1);
      i[0]=1;
      i[1]=0;
      OPENGM_TEST(f1(i)==1);
      i[0]=0;
      i[1]=0;
      OPENGM_TEST(f2(i)==2);
      i[0]=1;
      i[1]=0;
      OPENGM_TEST(f2(i)==2);
      testSerialization(f1);
      testProperties(f1);
      testSerialization(f2);
      testProperties(f2);
   }
   void testModelViewFunction() {
      std::cout << "  * ModelView" << std::endl;

      typedef T ValueType;
      typedef opengm::GraphicalModel
      <
         ValueType, //value type (should be float double or long double)
         opengm::Multiplier //operator (something like Adder or Multiplier)
      >
      GraphicalModelType;
      typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
      typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
      size_t nos[]={2,2,2};
      GraphicalModelType gm(opengm::DiscreteSpace<size_t,size_t>(nos,nos+3));
      ExplicitFunctionType f(nos,nos+2);
      f(0,0)=0;
      f(0,1)=1;
      f(1,0)=2;
      f(1,1)=3;

      FunctionIdentifier id=gm.addFunction(f);


      marray::Marray<T> offset(nos,nos+2);
      offset(0,0)=4;
      offset(0,1)=5;
      offset(1,0)=6;
      offset(1,1)=7;

      size_t vi[]={0,1};
      gm.addFactor(id,vi,vi+2);

      OPENGM_ASSERT(gm[0].numberOfVariables() == offset.dimension());
      opengm::ModelViewFunction<GraphicalModelType, marray::Marray<T> > fv(gm,0,2.0,&offset);
      OPENGM_TEST(fv.shape(0)==2);
      OPENGM_TEST(fv.shape(1)==2);
      OPENGM_TEST(fv.dimension()==2);
      OPENGM_TEST(fv.size()==4);

      size_t i[] = {0, 0 };
      OPENGM_TEST(fv(i)==0+4);
      i[0]=0;
      i[1]=1;
      OPENGM_TEST(fv(i)==2+5);
      i[0]=1;
      i[1]=0;
      OPENGM_TEST(fv(i)==4+6);
      i[0]=1;
      i[1]=1;
      OPENGM_TEST(fv(i)==6+7);
      testProperties(fv);
   }
   void testPotts() {
      std::cout << "  * Potts" << std::endl;
      //size_t shape[]={4,4};
      opengm::PottsFunction<T> f(4,4,0,1);
      OPENGM_TEST(f.isGeneralizedPotts());
      OPENGM_TEST(f.isPotts()==true);
      OPENGM_TEST(f.shape(0)==4);
      OPENGM_TEST(f.shape(1)==4);
      OPENGM_TEST(f.dimension()==2);
      OPENGM_TEST(f.size()==16);
      size_t i[] = {0, 1};
      i[0]=0;
      i[1]=0;
      OPENGM_TEST(f(i)==0);
      i[0]=1;
      i[1]=0;
      OPENGM_TEST(f(i)==1);
      i[0]=1;
      i[1]=1;
      OPENGM_TEST(f(i)==0);
      i[0]=0;
      i[1]=1;
      OPENGM_TEST(f(i)==1);
      i[0]=2;
      i[1]=1;
      OPENGM_TEST(f(i)==1);
      i[0]=3;
      i[1]=1;
      OPENGM_TEST(f(i)==1);
      i[0]=3;
      i[1]=3;
      OPENGM_TEST(f(i)==0);

      testSerialization(f);
      testProperties(f);
   }
   void testExplicitFunction() {
      std::cout << "  * ExplicitFunction "<<std::endl;
      size_t shape[]={4,4,4};
      opengm::ExplicitFunction<T> f(shape,shape+3);
      OPENGM_TEST(f.dimension()==3);
      OPENGM_TEST(f.size()==64);
      OPENGM_TEST(f.shape(0)==4);
      OPENGM_TEST(f.shape(1)==4);
      OPENGM_TEST(f.shape(2)==4);
      for(size_t i=0;i<f.size();++i) {
         f(i)=i;
      }

      testSerialization(f);
      testProperties(f);

   }
   void testPottsn() {
      std::cout << "  * PottsN" << std::endl;
      size_t myshape[]={4,4,4};
      opengm::PottsNFunction<T> f(myshape,myshape+3,0,1);
      OPENGM_TEST(f.isGeneralizedPotts());
      OPENGM_TEST(f.isPotts());
      OPENGM_TEST(f.shape(0)==4);
      OPENGM_TEST(f.shape(1)==4);
      OPENGM_TEST(f.shape(2)==4);
      OPENGM_TEST(f.dimension()==3);
      OPENGM_TEST(f.size()==64);
      size_t i[] = {0, 0, 0 };
      OPENGM_TEST(f(i)==0);
      i[0]=1;
      i[1]=0;
      i[2]=0;
      OPENGM_TEST(f(i)==1);
      i[0]=1;
      i[1]=1;
      i[2]=1;
      OPENGM_TEST(f(i)==0);
      i[0]=0;
      i[1]=1;
      i[2]=0;
      OPENGM_TEST(f(i)==1);
      i[0]=2;
      i[1]=1;
      i[2]=3;
      OPENGM_TEST(f(i)==1);
      i[0] = 3;
      i[1] = 1;
      i[2] = 0;
      OPENGM_TEST(f(i)==1);
      i[0]=3;
      i[1]=3;
      i[2]=3;
      OPENGM_TEST(f(i)==0);

      testSerialization(f);
      testProperties(f);
   }
   void testPottsg() {
      std::cout << "  * PottsG" << std::endl;

      //3rd order
      {
         size_t shape[]={4,4,4};
         size_t values[5]={10,11,12,13,14};
         opengm::PottsGFunction<T> f(shape,shape+3,values);
         OPENGM_TEST(f.isGeneralizedPotts());
         OPENGM_TEST(!f.isPotts());
         OPENGM_TEST(f.shape(0)==4);
         OPENGM_TEST(f.shape(1)==4);
         OPENGM_TEST(f.shape(2)==4);
         OPENGM_TEST(f.dimension()==3);
         OPENGM_TEST(f.size()==64);
         {
            size_t i[] = {0, 1, 2 };
            OPENGM_TEST(f(i)==10);
         }
         {
            size_t i[] = {0, 0, 1 };
            OPENGM_TEST(f(i)==11);
         }
         {
            size_t i[] = {1, 2, 1 };
            OPENGM_TEST(f(i)==12);
         }
         {
            size_t i[] = {2, 1, 1 };
            OPENGM_TEST(f(i)==13);
         }
         {
            size_t i[] = {0, 0, 0 };
            OPENGM_TEST(f(i)==14);
         }
         
         testSerialization(f);
         testProperties(f);
      }
      //4th order
      {
         size_t shape[]={4,4,4,4};
         size_t values[]={10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
         opengm::PottsGFunction<T> f(shape,shape+4,values);
         OPENGM_TEST(f.isGeneralizedPotts());
         OPENGM_TEST(!f.isPotts());
         OPENGM_TEST(f.shape(0)==4);
         OPENGM_TEST(f.shape(1)==4);
         OPENGM_TEST(f.shape(2)==4);
         OPENGM_TEST(f.shape(3)==4);
         OPENGM_TEST(f.dimension()==4);
         OPENGM_TEST(f.size()==256);
         {
            size_t i[] = {0, 1, 2, 3 };
            OPENGM_TEST(f(i)==10);
         }
         {
            size_t i[] = {0, 0, 2, 3 };
            OPENGM_TEST(f(i)==11);
         } 
         {
            size_t i[] = {0, 1, 0, 3 };
            OPENGM_TEST(f(i)==12);
         } 
         {
            size_t i[] = {0, 1, 1, 3 };
            OPENGM_TEST(f(i)==13);
         }
         {
            size_t i[] = {0, 0, 0, 1 };
            OPENGM_TEST(f(i)==14);
         } 
         {
            size_t i[] = {0, 1, 2, 0 };
            OPENGM_TEST(f(i)==15);
         }
         {
            size_t i[] = {0, 1, 1, 0 };
            OPENGM_TEST(f(i)==16);
         } 
         {
            size_t i[] = {1, 0, 2, 0 };
            OPENGM_TEST(f(i)==17);
         } 
         {
            size_t i[] = {1, 0, 1, 0 };
            OPENGM_TEST(f(i)==18);
         } 
         {
            size_t i[] = {0, 0, 1, 0 };
            OPENGM_TEST(f(i)==19);
         } 
         {
            size_t i[] = {1, 2, 0, 0 };
            OPENGM_TEST(f(i)==20);
         }
         {
            size_t i[] = {1, 1, 0, 0 };
            OPENGM_TEST(f(i)==21);
         }
         {
            size_t i[] = {0, 1, 0, 0 };
            OPENGM_TEST(f(i)==22);
         } 
         {
            size_t i[] = {1, 0, 0, 0 };
            OPENGM_TEST(f(i)==23);
         } 
         {
            size_t i[] = {0, 0, 0, 0 };
            OPENGM_TEST(f(i)==24);
         }
      }

   }
   void testSingleSiteFunction() {
      {
         opengm::StaticSingleSiteFunction<size_t,5,opengm::HeapStorage> f;
         OPENGM_TEST( f.dimension() == 1);
         OPENGM_TEST( f.size() == 5);
         OPENGM_TEST( f.shape(0) == 5);
         for(size_t i=0;i<5;++i) {
            size_t c[]={i};
            f(c)=i;
         }
         for(size_t i=0;i<5;++i) {
            size_t c[]={i};
            OPENGM_TEST( f(c) == i);
         }
         testSerialization(f);
         testProperties(f);
      }
      {
         opengm::StaticSingleSiteFunction<size_t,5,opengm::StackStorage> f;
         OPENGM_TEST( f.dimension() == 1);
         OPENGM_TEST( f.size() == 5);
         OPENGM_TEST( f.shape(0) == 5);
         for(size_t i=0;i<5;++i) {
            size_t c[]={i};
            f(c)=i;
         }
         for(size_t i=0;i<5;++i) {
            size_t c[]={i};
            OPENGM_TEST( f(c) == i);
         }
         testSerialization(f);
         testProperties(f);
      }
   }
   void testScaledView() {
      typedef T ValueType;
      typedef opengm::GraphicalModel
      <
         ValueType, //value type (should be float double or long double)
         opengm::Multiplier //operator (something like Adder or Multiplier)
      >
      GraphicalModelType;
      typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
      typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
      size_t nos[]={2,2,2};
      GraphicalModelType gm(opengm::DiscreteSpace<size_t,size_t>(nos,nos+3));
      ExplicitFunctionType f(nos,nos+2);
      f(0,0)=0;
      f(0,1)=1;
      f(1,0)=2;
      f(1,1)=3;
      FunctionIdentifier id=gm.addFunction(f);

      size_t vi[]={0,1};
      gm.addFactor(id,vi,vi+2);

      opengm::ScaledViewFunction<GraphicalModelType> fv(gm,0,2);
      OPENGM_TEST(fv.shape(0)==2);
      OPENGM_TEST(fv.shape(1)==2);
      OPENGM_TEST(fv.dimension()==2);
      OPENGM_TEST(fv.size()==4);
      size_t i[] = {0, 0 };
      OPENGM_TEST(fv(i)==0);
      i[0]=0;
      i[1]=1;
      OPENGM_TEST(fv(i)==2);
      i[0]=1;
      i[1]=0;
      OPENGM_TEST(fv(i)==4);
      i[0]=1;
      i[1]=1;
      OPENGM_TEST(fv(i)==6);
   }
   void testSquaredDifference() {
      std::cout << "  * SquaredDiffrence" << std::endl;
      opengm::SquaredDifferenceFunction<T> f(4,4);
      OPENGM_TEST(f.shape(0)==4);
      OPENGM_TEST(f.shape(1)==4);
      OPENGM_TEST(f.dimension()==2);
      OPENGM_TEST(f.size()==16);
      size_t i[] = {0, 1 };
      OPENGM_TEST(f(i)==1);
      i[0]=1;
      i[1]=0;
      OPENGM_TEST(f(i)==1);
      i[0]=0;
      i[1]=0;
      OPENGM_TEST(f(i)==0);
      i[0]=0;
      i[1]=1;
      OPENGM_TEST(f(i)==1);
      i[0]=2;
      i[1]=1;
      OPENGM_TEST(f(i)==1);
      i[0]=3;
      i[1]=1;
      OPENGM_TEST(f(i)==4);
      i[0]=1;
      i[1]=3;
      OPENGM_TEST(f(i)==4);

      testSerialization(f);
      testProperties(f);
   }
   void testTruncatedAbsoluteDifference() {
      std::cout << "  * TruncatedAbsoluteDifference" << std::endl;
      opengm::TruncatedAbsoluteDifferenceFunction<T> f(5,6,2,3);
      OPENGM_TEST(f.shape(0)==5);
      OPENGM_TEST(f.shape(1)==6);
      OPENGM_TEST(f.dimension()==2);
      OPENGM_TEST(f.size()==30);
      size_t i[] = {0, 5 };
      OPENGM_TEST(f(i)==6);
      i[0]=1;
      i[1]=0;
      OPENGM_TEST(f(i)==3);

      testSerialization(f);
      testProperties(f);
   }
   void testTruncatedSquaredDifference() {
      opengm::TruncatedSquaredDifferenceFunction<T> f(6,5,4,3);
      OPENGM_TEST(f.shape(0)==6);
      OPENGM_TEST(f.shape(1)==5);
      OPENGM_TEST(f.dimension()==2);
      OPENGM_TEST(f.size()==30);
      size_t i[] = {0, 5 };
      OPENGM_TEST(f(i)==12);
      i[0]=2;
      i[1]=0;
      OPENGM_TEST(f(i)==4*3);

      testSerialization(f);
      testProperties(f);
   }

   void testFoE(){
      std::cout << "  * FoE" << std::endl;
      double alpha[3] = {0.586612685392731, 1.157638405566669, 0.846059486257292};
      double experts[12] = {
         -0.0582774013402734, 0.0339010363051084, -0.0501593018104054, 0.0745568557931712,
         0.0492112815304123, -0.0307820846538285, -0.123247230948424, 0.104812330861557,
         0.0562633568728865, 0.0152832583489560, -0.0576215592718086, -0.0139673758425540
      };
      double expert[3][4] = {
         {-0.0582774013402734, 0.0339010363051084, -0.0501593018104054, 0.0745568557931712},
         {0.0492112815304123, -0.0307820846538285, -0.123247230948424, 0.104812330861557},
         {0.0562633568728865, 0.0152832583489560, -0.0576215592718086, -0.0139673758425540}
      };

      opengm::FoEFunction<T> foe(experts,alpha,256,4,3);
      std::vector<size_t> x(4,0);

      for(size_t it=1;it<200;++it){
         double energy = 0.0;
         for (size_t i = 0; i < 3; ++i) {
            double dot = 0.0;
            for (size_t j = 0; j < 4; ++j) {
               dot += expert[i][j] * double(x[j]);
            }
            energy += alpha[i] * std::log(1 + 0.5 * dot * dot);
         };
         //std::cout <<x[0] <<" " <<x[1] <<" " <<x[2] <<" " <<x[3] <<std::endl;
         OPENGM_TEST_EQUAL_TOLERANCE(energy,foe(x.begin()),0.000001);
         x[it%4] =  (x[it%4]+it)%256;
      }

      const size_t sizeIndices=opengm::FunctionSerialization<opengm::FoEFunction<T> >::indexSequenceSize(foe);
      const size_t sizeValues=opengm::FunctionSerialization< opengm::FoEFunction<T> >::valueSequenceSize(foe);
      std::vector<long long unsigned> indices(sizeIndices);
      std::vector<T> values(sizeValues);

      opengm::FunctionSerialization< opengm::FoEFunction<T> >::serialize(foe,indices.begin(),values.begin());
      opengm::FoEFunction<T> f2;
      opengm::FunctionSerialization< opengm::FoEFunction<T> >::deserialize(indices.begin(),values.begin(),f2);

      for(size_t it=0;it<200;++it){
         double energy = 0.0;
         for (size_t i = 0; i < 3; ++i) {
            double dot = 0.0;
            for (size_t j = 0; j < 4; ++j) {
               dot += expert[i][j] * double(x[j]);
            }
            energy += alpha[i] * std::log(1 + 0.5 * dot * dot);
         }
         //std::cout <<x[0] <<" " <<x[1] <<" " <<x[2] <<" " <<x[3] <<std::endl;
         OPENGM_TEST_EQUAL_TOLERANCE(energy,f2(x.begin()),0.000001);
         x[it%4] =  (x[it%4]+it)%256;
      }

      //testProperties(foe);
   }

   void testView() {
      typedef T ValueType;
      typedef opengm::GraphicalModel
      <
         ValueType, //value type (should be float double or long double)
         opengm::Multiplier //operator (something like Adder or Multiplier)
      >
      GraphicalModelType;
      typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
      typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
      size_t nos[]={2,2,2};
      GraphicalModelType gm(opengm::DiscreteSpace<size_t,size_t>(nos,nos+3));
      ExplicitFunctionType f(nos,nos+2);
      f(0,0)=0;
      f(0,1)=1;
      f(1,0)=2;
      f(1,1)=3;
      FunctionIdentifier id=gm.addFunction(f);

      size_t vi[]={0,1};
      gm.addFactor(id,vi,vi+2);

      opengm::ViewFunction<GraphicalModelType> fv(gm[0]);
      OPENGM_TEST(f.shape(0)==2);
      OPENGM_TEST(f.shape(1)==2);
      OPENGM_TEST(f.dimension()==2);
      OPENGM_TEST(f.size()==4);
      OPENGM_TEST(fv.shape(0)==2);
      OPENGM_TEST(fv.shape(1)==2);
      OPENGM_TEST(fv.dimension()==2);
      OPENGM_TEST(fv.size()==4);
      size_t i[] = {0, 0 };
      OPENGM_TEST_EQUAL(gm[0](i),int(0));
      OPENGM_TEST_EQUAL(fv(i),int(0));
      i[0]=0;
      i[1]=1;
      OPENGM_TEST_EQUAL(gm[0](i),int(1));
      OPENGM_TEST_EQUAL(fv(i),int(1));
      i[0]=1;
      i[1]=0;
      OPENGM_TEST_EQUAL(gm[0](i),int(2));
      OPENGM_TEST_EQUAL(fv(i),int(2));
      i[0]=1;
      i[1]=1;
      OPENGM_TEST_EQUAL(gm[0](i),int(3));
      OPENGM_TEST_EQUAL(fv(i),int(3));
      testProperties(fv);

   }
   void testViewAndFixVariables() {
      {
         typedef T ValueType;
         typedef opengm::GraphicalModel
         <
            ValueType, //value type (should be float double or long double)
            opengm::Multiplier //operator (something like Adder or Multiplier)
         >
         GraphicalModelType;
         typedef opengm::PositionAndLabel<typename GraphicalModelType::IndexType, typename GraphicalModelType::LabelType > PandL;
         typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
         typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
         size_t nos[] = {2, 3, 4, 5, 6 ,7, 8};
         GraphicalModelType gm(opengm::DiscreteSpace<size_t, size_t > (nos, nos + 7));
         ExplicitFunctionType f(nos, nos + 7,0);
         //f(0, 0) = 0;
         //f(0, 1) = 1;
         //f(1, 0) = 2;
         //f(1, 1) = 3;
         FunctionIdentifier id = gm.addFunction(f);

         size_t vi[] = {0, 1 ,2, 3, 4, 5, 6, 7};
         gm.addFactor(id, vi, vi + 7);

         std::vector<PandL> pAndL;
         pAndL.push_back(PandL(1,1) );
         pAndL.push_back(PandL(3,3) );
         pAndL.push_back(PandL(5,5) );
         opengm::ViewFixVariablesFunction<GraphicalModelType> fv(gm[0],pAndL);
         OPENGM_TEST(fv.dimension() == 4);
         OPENGM_TEST(fv.dimension() == 4);
         OPENGM_TEST(fv.shape(0) == 2);
         OPENGM_TEST(fv.shape(1) == 4);
         OPENGM_TEST(fv.shape(2) == 6);
         OPENGM_TEST(fv.shape(3) == 8);
         OPENGM_TEST(fv.shape(0) == 2);
         OPENGM_TEST(fv.shape(1) == 4);
         OPENGM_TEST(fv.shape(2) == 6);
         OPENGM_TEST(fv.shape(3) == 8);
         OPENGM_TEST_EQUAL(fv.functionShapeBegin()[0] , 2);
         //OPENGM_TEST_EQUAL(fv.functionShapeBegin()[1] , 4);
         OPENGM_TEST_EQUAL(fv.functionShapeBegin()[2] , 6);
         OPENGM_TEST_EQUAL(fv.functionShapeBegin()[3] , 8);
         OPENGM_TEST_EQUAL(fv.size() , 2*4*6*8);
         testProperties(fv);
      }
      {
         typedef T ValueType;
         typedef opengm::GraphicalModel
         <
            ValueType, //value type (should be float double or long double)
            opengm::Multiplier //operator (something like Adder or Multiplier)
         >
         GraphicalModelType;
         typedef opengm::PositionAndLabel<typename GraphicalModelType::IndexType, typename GraphicalModelType::LabelType > PandL;
         typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
         typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;
         size_t nos[] = {2, 5, 3,7};
         GraphicalModelType gm(opengm::DiscreteSpace<size_t, size_t > (nos, nos + 4));
         ExplicitFunctionType f(nos, nos + 2,0);
         //f(0, 0) = 0;
         //f(0, 1) = 1;
         //f(1, 0) = 2;
         //f(1, 1) = 3;
         FunctionIdentifier id = gm.addFunction(f);

         size_t vi[] = {0, 1};
         gm.addFactor(id, vi, vi + 2);
         
         {
         std::vector<PandL> pAndL;
         pAndL.push_back(PandL(0,1) );
         opengm::ViewFixVariablesFunction<GraphicalModelType> fv(gm[0],pAndL);
         OPENGM_TEST(fv.dimension() == 1);
         OPENGM_TEST(fv.shape(0) == 5);
         OPENGM_TEST_EQUAL(gm[0].size() , 10);
         OPENGM_TEST_EQUAL(fv.size() , 5);
         testProperties(fv);
         }
         {
         std::vector<PandL> pAndL;
         pAndL.push_back(PandL(1,1) );
         opengm::ViewFixVariablesFunction<GraphicalModelType> fv(gm[0],pAndL);
         OPENGM_TEST(fv.dimension() == 1);
         OPENGM_TEST(fv.shape(0) == 2);
         OPENGM_TEST_EQUAL(gm[0].size() , 10);
         OPENGM_TEST_EQUAL(fv.size() , 2);
         testProperties(fv);
         }
      }


   }

   void testLinearConstraintFunction() {
      std::cout << "  * LinearConstraintFunction" << std::endl;

      typedef T ValueType;
      typedef size_t IndexType;
      typedef size_t LabelType;
      typedef opengm::Adder OperatorType;
      typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

      const LabelType maxNumLabels = 6;
      const IndexType maxNumVariables = 6;
      const LabelType maxNumConstraints = 3;
      const size_t numTestIterations = 50;
      const size_t numEvaluationsPerTest = 100;
      const ValueType minCoefficientsValue = -10.0;
      const ValueType maxCoefficientsValue = 10.0;
      const ValueType validValue = -1.0;
      const ValueType invalidValue = 1.0;

      typedef opengm::ExplicitFunction<ValueType, IndexType, LabelType>          ExplicitFunction;
      typedef opengm::LinearConstraintFunction<ValueType, IndexType, LabelType>  LinearConstraintFunction;

      typedef typename opengm::meta::TypeListGenerator< ExplicitFunction, LinearConstraintFunction>::type FunctionTypeList;

      typedef opengm::GraphicalModel<ValueType, OperatorType, FunctionTypeList, SpaceType> GmType;

      typedef typename LinearConstraintFunction::LinearConstraintsContainerType  LinearConstraintsContainerType;
      typedef typename LinearConstraintFunction::LinearConstraintType            LinearConstraintType;
      typedef typename LinearConstraintType::IndicatorVariablesContainerType     IndicatorVariablesContainerType;
      typedef typename LinearConstraintType::IndicatorVariableType               IndicatorVariableType;
      typedef typename LinearConstraintType::CoefficientsContainerType           CoefficientsContainerType;

      typedef typename LinearConstraintFunction::LinearConstraintsIteratorType         LinearConstraintsIteratorType;
      typedef typename LinearConstraintFunction::IndicatorVariablesIteratorType        IndicatorVariablesIteratorType;
      typedef typename LinearConstraintType::CoefficientsIteratorType                  CoefficientsIteratorType;
      typedef typename LinearConstraintType::VariableLabelPairsIteratorType            VariableLabelPairsIteratorType;
      typedef typename LinearConstraintFunction::ViolatedLinearConstraintsIteratorType ViolatedLinearConstraintsIteratorType;
      typedef typename LinearConstraintFunction::ViolatedLinearConstraintsWeightsIteratorType  ViolatedLinearConstraintsWeightsIteratorType;

      typedef std::vector<LabelType> ShapeType;

      typedef opengm::RandomUniformInteger<LabelType> RandomUniformLabelType;
      RandomUniformLabelType labelGenerator(0, maxNumLabels);
      typedef opengm::RandomUniformInteger<LabelType> RandomUniformLabelType;
      RandomUniformLabelType indicatorVariableLogicalOeratorGenerator(0, 3);
      typedef opengm::RandomUniformInteger<IndexType> RandomUniformIndexType;
      RandomUniformIndexType indexGenerator(0, maxNumVariables);
      typedef opengm::RandomUniformFloatingPoint<double> RandomUniformValueType;
      RandomUniformValueType valueGenerator(minCoefficientsValue, maxCoefficientsValue);
      typedef opengm::RandomUniformFloatingPoint<double> RandomUniformDoubleType;
      RandomUniformDoubleType boxGenerator(0.0, 1.0);

      static const typename IndicatorVariableType::LogicalOperatorType possibleIndicatorVariableLogicalOperatorTypes[] = {IndicatorVariableType::And, IndicatorVariableType::Or, IndicatorVariableType::Not };

      // test property
      LabelType smallShape[] = {2, 2};
      ExplicitFunction ef(smallShape, smallShape + 2);
      LinearConstraintFunction lcf(smallShape, smallShape + 2, LinearConstraintsContainerType(), 20.0, 5.0);
      OPENGM_TEST(!ef.isLinearConstraint());
      OPENGM_TEST(lcf.isLinearConstraint());

      // test min
      OPENGM_TEST_EQUAL(lcf.min(), 5.0);

      // test max
      OPENGM_TEST_EQUAL(lcf.max(), 20);

      // test minmax
      opengm::MinMaxFunctor<ValueType> minmax = lcf.minMax();
      OPENGM_TEST_EQUAL(minmax.min(), 5.0);
      OPENGM_TEST_EQUAL(minmax.max(), 20);

      // test shape, dimension, size, constraint access and evaluation (operator())
      for(size_t testIter = 0; testIter < numTestIterations; testIter++) {
         // create shape
         IndexType numVariables = indexGenerator() + 1;
         ShapeType shape(numVariables);
         for(size_t i = 0; i < numVariables; i++) {
            shape[i] = labelGenerator() + 1;
         }

         // number of constraints
         IndexType numConstraints = (indexGenerator() % maxNumConstraints) + 1;

         // create constraints
         LinearConstraintsContainerType constraints(numConstraints);
         std::vector<ValueType> bounds(numConstraints);
         std::vector<std::vector<ValueType> > coefficients(numConstraints);
         std::vector<std::vector<IndicatorVariableType> > variables(numConstraints);
         for(IndexType i = 0; i < numConstraints; i++) {
            // create constraint parts
            RandomUniformIndexType currentIndexGenerator(0, numVariables);
            IndexType numConstraintParts = currentIndexGenerator() + 1;
            std::vector<IndexType> numVariablesPerConstraintPart(numConstraintParts);
            for(IndexType j = 0; j < numConstraintParts; j++) {
               numVariablesPerConstraintPart[j] = currentIndexGenerator() + 1;
            }

            // create variables
            IndicatorVariablesContainerType currentVariables;
            for(IndexType j = 0; j < numConstraintParts; j++) {
               IndicatorVariableType variable;
               variable.setLogicalOperatorType(possibleIndicatorVariableLogicalOperatorTypes[indicatorVariableLogicalOeratorGenerator()]);
               std::vector<IndexType> indices(numVariables);
               for(size_t k = 0; k < indices.size(); k++) {
                  indices[k] = k;
               }
               std::random_shuffle(indices.begin(), indices.end());
               for(IndexType k = 0; k < numVariablesPerConstraintPart[j]; k++) {
                  IndexType currentIndex = indices[k];
                  RandomUniformLabelType currentLabelGenerator(0, shape[currentIndex]);
                  LabelType currentLabel = currentLabelGenerator();

                  variable.add(currentIndex, currentLabel);
               }
               variables[i].push_back(variable);
               currentVariables.push_back(variable);
            }

            // create coefficients
            CoefficientsContainerType currentCoefficients;
            ValueType sumCurrentCoefficients = 0.0;
            for(IndexType j = 0; j < numConstraintParts; j++) {
               const ValueType coefficient = valueGenerator();
               sumCurrentCoefficients += coefficient;
               coefficients[i].push_back(coefficient);
               currentCoefficients.push_back(coefficient);
            }

            // add variables and coefficients
            constraints[i].add(currentVariables, currentCoefficients);

            // create bound
            const ValueType bound = sumCurrentCoefficients / static_cast<ValueType>(numConstraintParts);
            bounds[i] = bound;
            constraints[i].setBound(bound);
         }

         // create functions
         for(size_t i = 0; i < constraints.size(); ++i) {
            constraints[i].setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::LessEqual);
         }
         LinearConstraintFunction lessEqualFunction(shape.begin(), shape.end(), constraints, validValue, invalidValue);

         for(size_t i = 0; i < constraints.size(); ++i) {
            constraints[i].setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::Equal);
         }
         LinearConstraintFunction equalFunction(shape.begin(), shape.end(), constraints, validValue, invalidValue);

         for(size_t i = 0; i < constraints.size(); ++i) {
            constraints[i].setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::GreaterEqual);
         }
         LinearConstraintFunction greaterEqualFunction(shape.begin(), shape.end(), constraints, validValue, invalidValue);

         // test dimension
         OPENGM_TEST_EQUAL(lessEqualFunction.dimension(), numVariables);
         OPENGM_TEST_EQUAL(equalFunction.dimension(), numVariables);
         OPENGM_TEST_EQUAL(greaterEqualFunction.dimension(), numVariables);

         // test shape
         for(size_t i = 0; i < numVariables; i++) {
            OPENGM_TEST_EQUAL(lessEqualFunction.shape(i), shape[i]);
            OPENGM_TEST_EQUAL(equalFunction.shape(i), shape[i]);
            OPENGM_TEST_EQUAL(greaterEqualFunction.shape(i), shape[i]);
         }

         // test size
         size_t expectedSize = 1;
         for(size_t i = 0; i < numVariables; i++) {
            expectedSize *= shape[i];
         }
         OPENGM_TEST_EQUAL(lessEqualFunction.size(), expectedSize);
         OPENGM_TEST_EQUAL(equalFunction.size(), expectedSize);
         OPENGM_TEST_EQUAL(greaterEqualFunction.size(), expectedSize);

         // test serialization
         testSerialization(lessEqualFunction);
         testSerialization(equalFunction);
         testSerialization(greaterEqualFunction);

         // test indicator variable order
         for(IndicatorVariablesIteratorType variablesIter = lessEqualFunction.indicatorVariablesOrderBegin(); variablesIter != lessEqualFunction.indicatorVariablesOrderEnd(); ++variablesIter) {
            bool variableFound = false;
            for(size_t i = 0; i < variables.size(); ++i) {
               if(std::find(variables[i].begin(), variables[i].end(), *variablesIter) != variables[i].end()) {
                  variableFound = true;
                  break;
               }
            }
            OPENGM_TEST(variableFound);
         }
         for(size_t i = 0; i < variables.size(); ++i) {
            for(size_t j = 0; j < variables[i].size(); ++j) {
               OPENGM_TEST(std::find(lessEqualFunction.indicatorVariablesOrderBegin(), lessEqualFunction.indicatorVariablesOrderEnd(), variables[i][j]) != lessEqualFunction.indicatorVariablesOrderEnd());
            }
         }

         for(IndicatorVariablesIteratorType variablesIter = equalFunction.indicatorVariablesOrderBegin(); variablesIter != equalFunction.indicatorVariablesOrderEnd(); ++variablesIter) {
            bool variableFound = false;
            for(size_t i = 0; i < variables.size(); ++i) {
               if(std::find(variables[i].begin(), variables[i].end(), *variablesIter) != variables[i].end()) {
                  variableFound = true;
                  break;
               }
            }
            OPENGM_TEST(variableFound);
         }
         for(size_t i = 0; i < variables.size(); ++i) {
            for(size_t j = 0; j < variables[i].size(); ++j) {
               OPENGM_TEST(std::find(equalFunction.indicatorVariablesOrderBegin(), equalFunction.indicatorVariablesOrderEnd(), variables[i][j]) != equalFunction.indicatorVariablesOrderEnd());
            }
         }

         for(IndicatorVariablesIteratorType variablesIter = greaterEqualFunction.indicatorVariablesOrderBegin(); variablesIter != greaterEqualFunction.indicatorVariablesOrderEnd(); ++variablesIter) {
            bool variableFound = false;
            for(size_t i = 0; i < variables.size(); ++i) {
               if(std::find(variables[i].begin(), variables[i].end(), *variablesIter) != variables[i].end()) {
                  variableFound = true;
                  break;
               }
            }
            OPENGM_TEST(variableFound);
         }
         for(size_t i = 0; i < variables.size(); ++i) {
            for(size_t j = 0; j < variables[i].size(); ++j) {
               OPENGM_TEST(std::find(greaterEqualFunction.indicatorVariablesOrderBegin(), greaterEqualFunction.indicatorVariablesOrderEnd(), variables[i][j]) != greaterEqualFunction.indicatorVariablesOrderEnd());
            }
         }

         // test constraint access
         OPENGM_TEST_EQUAL(std::distance(lessEqualFunction.linearConstraintsBegin(), lessEqualFunction.linearConstraintsEnd()), numConstraints);
         OPENGM_TEST_EQUAL(std::distance(equalFunction.linearConstraintsBegin(), equalFunction.linearConstraintsEnd()), numConstraints);
         OPENGM_TEST_EQUAL(std::distance(greaterEqualFunction.linearConstraintsBegin(), greaterEqualFunction.linearConstraintsEnd()), numConstraints);

         // constraint operator type
         for(LinearConstraintsIteratorType constraintsIter = lessEqualFunction.linearConstraintsBegin(); constraintsIter != lessEqualFunction.linearConstraintsEnd(); ++constraintsIter) {
            OPENGM_TEST_EQUAL(constraintsIter->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::LessEqual);
         }
         for(LinearConstraintsIteratorType constraintsIter = equalFunction.linearConstraintsBegin(); constraintsIter != equalFunction.linearConstraintsEnd(); ++constraintsIter) {
            OPENGM_TEST_EQUAL(constraintsIter->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::Equal);
         }
         for(LinearConstraintsIteratorType constraintsIter = greaterEqualFunction.linearConstraintsBegin(); constraintsIter != greaterEqualFunction.linearConstraintsEnd(); ++constraintsIter) {
            OPENGM_TEST_EQUAL(constraintsIter->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::GreaterEqual);
         }

         // bounds
         typename std::vector<ValueType>::const_iterator boundsIter = bounds.begin();
         for(LinearConstraintsIteratorType constraintsIter = lessEqualFunction.linearConstraintsBegin(); constraintsIter != lessEqualFunction.linearConstraintsEnd(); ++constraintsIter) {
            OPENGM_TEST_EQUAL_TOLERANCE(constraintsIter->getBound(), *boundsIter, OPENGM_FLOAT_TOL);
            ++boundsIter;
         }
         OPENGM_TEST(boundsIter == bounds.end());
         boundsIter = bounds.begin();
         for(LinearConstraintsIteratorType constraintsIter = equalFunction.linearConstraintsBegin(); constraintsIter != equalFunction.linearConstraintsEnd(); ++constraintsIter) {
            OPENGM_TEST_EQUAL_TOLERANCE(constraintsIter->getBound(), *boundsIter, OPENGM_FLOAT_TOL);
            ++boundsIter;
         }
         OPENGM_TEST(boundsIter == bounds.end());
         boundsIter = bounds.begin();
         for(LinearConstraintsIteratorType constraintsIter = greaterEqualFunction.linearConstraintsBegin(); constraintsIter != greaterEqualFunction.linearConstraintsEnd(); ++constraintsIter) {
            OPENGM_TEST_EQUAL_TOLERANCE(constraintsIter->getBound(), *boundsIter, OPENGM_FLOAT_TOL);
            ++boundsIter;
         }
         OPENGM_TEST(boundsIter == bounds.end());

         // indicator variables
         {
            size_t i = 0;
            for(LinearConstraintsIteratorType constraintsIter = lessEqualFunction.linearConstraintsBegin(); constraintsIter != lessEqualFunction.linearConstraintsEnd(); ++constraintsIter) {
               for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
                  OPENGM_TEST(std::find(variables[i].begin(), variables[i].end(), *variablesIter) != variables[i].end());
               }
               for(size_t j = 0; j < variables[i].size(); ++j) {
                  OPENGM_TEST(std::find(constraintsIter->indicatorVariablesBegin(), constraintsIter->indicatorVariablesEnd(), variables[i][j]) != constraintsIter->indicatorVariablesEnd());
               }
               ++i;
            }

            i = 0;
            for(LinearConstraintsIteratorType constraintsIter = equalFunction.linearConstraintsBegin(); constraintsIter != equalFunction.linearConstraintsEnd(); ++constraintsIter) {
               for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
                  OPENGM_TEST(std::find(variables[i].begin(), variables[i].end(), *variablesIter) != variables[i].end());
               }
               for(size_t j = 0; j < variables[i].size(); ++j) {
                  OPENGM_TEST(std::find(constraintsIter->indicatorVariablesBegin(), constraintsIter->indicatorVariablesEnd(), variables[i][j]) != constraintsIter->indicatorVariablesEnd());
               }
               ++i;
            }

            i = 0;
            for(LinearConstraintsIteratorType constraintsIter = greaterEqualFunction.linearConstraintsBegin(); constraintsIter != greaterEqualFunction.linearConstraintsEnd(); ++constraintsIter) {
               for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
                  OPENGM_TEST(std::find(variables[i].begin(), variables[i].end(), *variablesIter) != variables[i].end());
               }
               for(size_t j = 0; j < variables[i].size(); ++j) {
                  OPENGM_TEST(std::find(constraintsIter->indicatorVariablesBegin(), constraintsIter->indicatorVariablesEnd(), variables[i][j]) != constraintsIter->indicatorVariablesEnd());
               }
               ++i;
            }
         }


         // coefficients
         {
            size_t i = 0;
            for(LinearConstraintsIteratorType constraintsIter = lessEqualFunction.linearConstraintsBegin(); constraintsIter != lessEqualFunction.linearConstraintsEnd(); ++constraintsIter) {
               OPENGM_TEST_EQUAL(std::distance(constraintsIter->coefficientsBegin(), constraintsIter->coefficientsEnd()), coefficients[i].size());
               OPENGM_TEST_EQUAL_SEQUENCE(constraintsIter->coefficientsBegin(), constraintsIter->coefficientsEnd(), coefficients[i].begin());
               ++i;
            }
            i = 0;
            for(LinearConstraintsIteratorType constraintsIter = equalFunction.linearConstraintsBegin(); constraintsIter != equalFunction.linearConstraintsEnd(); ++constraintsIter) {
               OPENGM_TEST_EQUAL(std::distance(constraintsIter->coefficientsBegin(), constraintsIter->coefficientsEnd()), coefficients[i].size());
               OPENGM_TEST_EQUAL_SEQUENCE(constraintsIter->coefficientsBegin(), constraintsIter->coefficientsEnd(), coefficients[i].begin());
               ++i;
            }
            i = 0;
            for(LinearConstraintsIteratorType constraintsIter = greaterEqualFunction.linearConstraintsBegin(); constraintsIter != greaterEqualFunction.linearConstraintsEnd(); ++constraintsIter) {
               OPENGM_TEST_EQUAL(std::distance(constraintsIter->coefficientsBegin(), constraintsIter->coefficientsEnd()), coefficients[i].size());
               OPENGM_TEST_EQUAL_SEQUENCE(constraintsIter->coefficientsBegin(), constraintsIter->coefficientsEnd(), coefficients[i].begin());
               ++i;
            }
         }

         // test evaluation and challange
         for(size_t evaluationIter = 0; evaluationIter < numEvaluationsPerTest; evaluationIter++) {
            // create evaluation vector
            std::vector<LabelType> evalVec(numVariables);
            for(size_t i = 0; i < numVariables; i++) {
               RandomUniformLabelType stateGenerator(0, shape[i]);
               LabelType currentState = stateGenerator();
               evalVec[i] = currentState;
            }

            // compute expected values
            std::vector<ValueType> expectedValues(numConstraints);
            for(IndexType i = 0; i < numConstraints; i++) {
               expectedValues[i] = 0;
               CoefficientsIteratorType coeffIter = constraints[i].coefficientsBegin();
               for(IndicatorVariablesIteratorType varIter = constraints[i].indicatorVariablesBegin(); varIter != constraints[i].indicatorVariablesEnd(); ++varIter) {
                  OPENGM_TEST(varIter->getLogicalOperatorType() == IndicatorVariableType::And || varIter->getLogicalOperatorType() == IndicatorVariableType::Or || varIter->getLogicalOperatorType() == IndicatorVariableType::Not);
                  bool validConstraintPart = true;
                  if(varIter->getLogicalOperatorType() == IndicatorVariableType::And) {
                     for(VariableLabelPairsIteratorType indicatorIter = varIter->begin(); indicatorIter != varIter->end(); ++indicatorIter) {
                        if(evalVec[indicatorIter->first] != indicatorIter->second) {
                           validConstraintPart = false;
                           break;
                        }
                     }
                  } else if(varIter->getLogicalOperatorType() == IndicatorVariableType::Or) {
                     validConstraintPart = false;
                     for(VariableLabelPairsIteratorType indicatorIter = varIter->begin(); indicatorIter != varIter->end(); ++indicatorIter) {
                        if(evalVec[indicatorIter->first] == indicatorIter->second) {
                           validConstraintPart = true;
                           break;
                        }
                     }
                  } else {
                     // if(varIter->getLogicalOperatorType() == IndicatorVariableType::Not)
                     for(VariableLabelPairsIteratorType indicatorIter = varIter->begin(); indicatorIter != varIter->end(); ++indicatorIter) {
                        if(evalVec[indicatorIter->first] == indicatorIter->second) {
                           validConstraintPart = false;
                           break;
                        }
                     }
                  }
                  if(validConstraintPart) {
                     expectedValues[i] += *coeffIter;
                  }
                  ++coeffIter;
               }
            }

            // check results
            bool expectedEqual = true;
            bool expectedLessEqual = true;
            bool expectedGreaterEqual = true;

            for(IndexType i = 0; i < numConstraints; i++) {
               if(expectedValues[i] < bounds[i]) {
                  expectedEqual = false;
                  expectedGreaterEqual = false;
               } else if(expectedValues[i] == bounds[i]) {

               } else {
                  expectedEqual = false;
                  expectedLessEqual = false;
               }
            }

            if(expectedLessEqual) {
               OPENGM_TEST_EQUAL(lessEqualFunction(evalVec.begin()), validValue);
               // test challenge function
               ViolatedLinearConstraintsIteratorType violatedConstraintsBegin;
               ViolatedLinearConstraintsIteratorType violatedConstraintsEnd;
               ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsBegin;
               lessEqualFunction.challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, evalVec.begin());
               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin, violatedConstraintsEnd), 0);
            } else {
               OPENGM_TEST_EQUAL(lessEqualFunction(evalVec.begin()), invalidValue);
               // test challenge function
               ViolatedLinearConstraintsIteratorType violatedConstraintsBegin;
               ViolatedLinearConstraintsIteratorType violatedConstraintsEnd;
               ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsBegin;
               lessEqualFunction.challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, evalVec.begin());
               OPENGM_TEST(std::distance(violatedConstraintsBegin, violatedConstraintsEnd) > 0);
               for(size_t i = 0; i < numConstraints; i++) {
                  if(expectedValues[i] > bounds[i]) {
                     OPENGM_TEST_EQUAL_TOLERANCE(expectedValues[i] - bounds[i], *violatedConstraintsWeightsBegin, OPENGM_FLOAT_TOL);
                     OPENGM_TEST_EQUAL_TOLERANCE(bounds[i], violatedConstraintsBegin->getBound(), OPENGM_FLOAT_TOL);
                     OPENGM_TEST_EQUAL_SEQUENCE(constraints[i].indicatorVariablesBegin(), constraints[i].indicatorVariablesEnd(), violatedConstraintsBegin->indicatorVariablesBegin());
                     ++violatedConstraintsBegin;
                     ++violatedConstraintsWeightsBegin;
                  }
               }
               OPENGM_TEST(violatedConstraintsBegin == violatedConstraintsEnd);
            }
            if(expectedEqual) {
               OPENGM_TEST_EQUAL(equalFunction(evalVec.begin()), validValue);
               // test challenge function
               ViolatedLinearConstraintsIteratorType violatedConstraintsBegin;
               ViolatedLinearConstraintsIteratorType violatedConstraintsEnd;
               ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsBegin;
               equalFunction.challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, evalVec.begin());
               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin, violatedConstraintsEnd), 0);
            } else {
               OPENGM_TEST_EQUAL(equalFunction(evalVec.begin()), invalidValue);
               // test challenge function
               ViolatedLinearConstraintsIteratorType violatedConstraintsBegin;
               ViolatedLinearConstraintsIteratorType violatedConstraintsEnd;
               ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsBegin;
               equalFunction.challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, evalVec.begin());
               OPENGM_TEST(std::distance(violatedConstraintsBegin, violatedConstraintsEnd) > 0);
               for(size_t i = 0; i < numConstraints; i++) {
                  if(expectedValues[i] != bounds[i]) {
                     OPENGM_TEST_EQUAL_TOLERANCE(std::abs(expectedValues[i] - bounds[i]), *violatedConstraintsWeightsBegin, OPENGM_FLOAT_TOL);
                     OPENGM_TEST_EQUAL_TOLERANCE(bounds[i], violatedConstraintsBegin->getBound(), OPENGM_FLOAT_TOL);
                     OPENGM_TEST_EQUAL_SEQUENCE(constraints[i].indicatorVariablesBegin(), constraints[i].indicatorVariablesEnd(), violatedConstraintsBegin->indicatorVariablesBegin());
                     ++violatedConstraintsBegin;
                     ++violatedConstraintsWeightsBegin;
                  }
               }
               OPENGM_TEST(violatedConstraintsBegin == violatedConstraintsEnd);
            }
            if(expectedGreaterEqual) {
               OPENGM_TEST_EQUAL(greaterEqualFunction(evalVec.begin()), validValue);
               // test challenge function
               ViolatedLinearConstraintsIteratorType violatedConstraintsBegin;
               ViolatedLinearConstraintsIteratorType violatedConstraintsEnd;
               ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsBegin;
               greaterEqualFunction.challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, evalVec.begin());
               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin, violatedConstraintsEnd), 0);
            } else {
               OPENGM_TEST_EQUAL(greaterEqualFunction(evalVec.begin()), invalidValue);
               // test challenge function
               ViolatedLinearConstraintsIteratorType violatedConstraintsBegin;
               ViolatedLinearConstraintsIteratorType violatedConstraintsEnd;
               ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsBegin;
               greaterEqualFunction.challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, evalVec.begin());
               OPENGM_TEST(std::distance(violatedConstraintsBegin, violatedConstraintsEnd) > 0);
               for(size_t i = 0; i < numConstraints; i++) {
                  if(expectedValues[i] < bounds[i]) {
                     OPENGM_TEST_EQUAL_TOLERANCE(bounds[i] - expectedValues[i], *violatedConstraintsWeightsBegin, OPENGM_FLOAT_TOL);
                     OPENGM_TEST_EQUAL_TOLERANCE(bounds[i], violatedConstraintsBegin->getBound(), OPENGM_FLOAT_TOL);
                     OPENGM_TEST_EQUAL_SEQUENCE(constraints[i].indicatorVariablesBegin(), constraints[i].indicatorVariablesEnd(), violatedConstraintsBegin->indicatorVariablesBegin());
                     ++violatedConstraintsBegin;
                     ++violatedConstraintsWeightsBegin;
                  }
               }
               OPENGM_TEST(violatedConstraintsBegin == violatedConstraintsEnd);
            }
         }

         // test relaxed challenge
         for(size_t evaluationIter = 0; evaluationIter < numEvaluationsPerTest; evaluationIter++) {
            // create relaxed evaluation vector
            const size_t numIndicatorVariables = std::distance(lessEqualFunction.indicatorVariablesOrderBegin(), lessEqualFunction.indicatorVariablesOrderEnd());
            std::vector<double> evalVecRelaxed(numIndicatorVariables);
            for(size_t i = 0; i < numIndicatorVariables; i++) {
               const double currentState = boxGenerator();
               evalVecRelaxed[i] = currentState;
            }

            // compute expected values
            std::vector<double> expectedValues(numConstraints);
            for(IndexType i = 0; i < numConstraints; i++) {
               expectedValues[i] = 0.0;
               CoefficientsIteratorType coeffIter = constraints[i].coefficientsBegin();
               for(IndicatorVariablesIteratorType varIter = constraints[i].indicatorVariablesBegin(); varIter != constraints[i].indicatorVariablesEnd(); ++varIter) {
                  IndicatorVariablesIteratorType varPosition = std::find(lessEqualFunction.indicatorVariablesOrderBegin(), lessEqualFunction.indicatorVariablesOrderEnd(), *varIter);
                  OPENGM_TEST(varPosition != lessEqualFunction.indicatorVariablesOrderEnd());
                  const size_t evalVecRelaxedPosition = std::distance(lessEqualFunction.indicatorVariablesOrderBegin(), varPosition);
                  const size_t coeffIterPosition = std::distance(constraints[i].indicatorVariablesBegin(), varIter);
                  expectedValues[i] += coeffIter[coeffIterPosition] * evalVecRelaxed[evalVecRelaxedPosition];
               }
            }

            ViolatedLinearConstraintsIteratorType violatedConstraintsLessEqualFunctionBegin;
            ViolatedLinearConstraintsIteratorType violatedConstraintsLessEqualFunctionEnd;
            ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsLessEqualFunctionBegin;
            lessEqualFunction.challengeRelaxed(violatedConstraintsLessEqualFunctionBegin, violatedConstraintsLessEqualFunctionEnd, violatedConstraintsWeightsLessEqualFunctionBegin, evalVecRelaxed.begin());

            ViolatedLinearConstraintsIteratorType violatedConstraintsEqualFunctionBegin;
            ViolatedLinearConstraintsIteratorType violatedConstraintsEqualFunctionEnd;
            ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsEqualFunctionBegin;
            equalFunction.challengeRelaxed(violatedConstraintsEqualFunctionBegin, violatedConstraintsEqualFunctionEnd, violatedConstraintsWeightsEqualFunctionBegin, evalVecRelaxed.begin());

            ViolatedLinearConstraintsIteratorType violatedConstraintsGreaterEqualFunctionBegin;
            ViolatedLinearConstraintsIteratorType violatedConstraintsGreaterEqualFunctionEnd;
            ViolatedLinearConstraintsWeightsIteratorType  violatedConstraintsWeightsGreaterEqualFunctionBegin;
            greaterEqualFunction.challengeRelaxed(violatedConstraintsGreaterEqualFunctionBegin, violatedConstraintsGreaterEqualFunctionEnd, violatedConstraintsWeightsGreaterEqualFunctionBegin, evalVecRelaxed.begin());

            for(size_t i = 0; i < numConstraints; i++) {
               if(expectedValues[i] < bounds[i]) {  
                  OPENGM_TEST_EQUAL_TOLERANCE(bounds[i] - expectedValues[i], *violatedConstraintsWeightsEqualFunctionBegin, OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL_TOLERANCE(bounds[i], violatedConstraintsEqualFunctionBegin->getBound(), OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL_SEQUENCE(constraints[i].indicatorVariablesBegin(), constraints[i].indicatorVariablesEnd(), violatedConstraintsEqualFunctionBegin->indicatorVariablesBegin());
                  ++violatedConstraintsEqualFunctionBegin;
                  ++violatedConstraintsWeightsEqualFunctionBegin;

                  OPENGM_TEST_EQUAL_TOLERANCE(bounds[i] - expectedValues[i], *violatedConstraintsWeightsGreaterEqualFunctionBegin, OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL_TOLERANCE(bounds[i], violatedConstraintsGreaterEqualFunctionBegin->getBound(), OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL_SEQUENCE(constraints[i].indicatorVariablesBegin(), constraints[i].indicatorVariablesEnd(), violatedConstraintsGreaterEqualFunctionBegin->indicatorVariablesBegin());
                  ++violatedConstraintsGreaterEqualFunctionBegin;
                  ++violatedConstraintsWeightsGreaterEqualFunctionBegin;
               } else if(expectedValues[i] > bounds[i]) {
                  OPENGM_TEST_EQUAL_TOLERANCE(expectedValues[i] - bounds[i], *violatedConstraintsWeightsEqualFunctionBegin, OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL_TOLERANCE(bounds[i], violatedConstraintsEqualFunctionBegin->getBound(), OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL_SEQUENCE(constraints[i].indicatorVariablesBegin(), constraints[i].indicatorVariablesEnd(), violatedConstraintsEqualFunctionBegin->indicatorVariablesBegin());
                  ++violatedConstraintsEqualFunctionBegin;
                  ++violatedConstraintsWeightsEqualFunctionBegin;

                  OPENGM_TEST_EQUAL_TOLERANCE(expectedValues[i] - bounds[i], *violatedConstraintsWeightsLessEqualFunctionBegin, OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL_TOLERANCE(bounds[i], violatedConstraintsLessEqualFunctionBegin->getBound(), OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL_SEQUENCE(constraints[i].indicatorVariablesBegin(), constraints[i].indicatorVariablesEnd(), violatedConstraintsLessEqualFunctionBegin->indicatorVariablesBegin());
                  ++violatedConstraintsLessEqualFunctionBegin;
                  ++violatedConstraintsWeightsLessEqualFunctionBegin;
               }
            }
            OPENGM_TEST(violatedConstraintsLessEqualFunctionBegin == violatedConstraintsLessEqualFunctionEnd);
            OPENGM_TEST(violatedConstraintsEqualFunctionBegin == violatedConstraintsEqualFunctionEnd);
            OPENGM_TEST(violatedConstraintsGreaterEqualFunctionBegin == violatedConstraintsGreaterEqualFunctionEnd);
         }
      }

      // test model with linear constraint
      // create unary function
      ExplicitFunction upperHalfUnaryBase(smallShape, smallShape + 1);
      ExplicitFunction lowerHalfUnaryBase(smallShape, smallShape + 1);
      LabelType index[] = {0};
      upperHalfUnaryBase(index) = -1.0;
      lowerHalfUnaryBase(index) = 1.0;
      index[0] = 1;
      upperHalfUnaryBase(index) = 1.0;
      lowerHalfUnaryBase(index) = -1.0;

      // create potts function as constraint function
      LinearConstraintsContainerType linearConstraints(2);
      IndicatorVariableType indicatorVar1;
      indicatorVar1.add(IndexType(0), LabelType(0));
      indicatorVar1.add(IndexType(1), LabelType(1));
      IndicatorVariableType indicatorVar2;
      indicatorVar2.add(IndexType(0), LabelType(1));
      indicatorVar2.add(IndexType(1), LabelType(0));

      linearConstraints[0].add(indicatorVar1, 1.0);
      linearConstraints[1].add(indicatorVar2, 1.0);

      linearConstraints[0].setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      linearConstraints[1].setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::LessEqual);

      linearConstraints[0].setBound(0.0);
      linearConstraints[1].setBound(0.0);

      LinearConstraintFunction pottsConstraint(smallShape, smallShape + 2, linearConstraints.begin(), linearConstraints.end());

      OPENGM_TEST(pottsConstraint.isPotts());

      IndexType gridWidth = 4;
      IndexType gridHeight = 4;
      IndexType numVariables = gridHeight * gridWidth;
      std::vector<LabelType> shape(16, 2);

      GmType model(typename GmType::SpaceType(shape.begin(), shape.end()));

      // add unary
      typename GmType::FunctionIdentifier upperHalfUnaryBaseID = model.addFunction(upperHalfUnaryBase);
      typename GmType::FunctionIdentifier lowerHalfUnaryBaseID = model.addFunction(lowerHalfUnaryBase);

      for(size_t i = 0; i < numVariables/2; i++) {
         IndexType index[] = {i};
         model.addFactor(upperHalfUnaryBaseID, index, index + 1);
      }
      for(size_t i = numVariables/2; i < numVariables; i++) {
         IndexType index[] = {i};
         model.addFactor(lowerHalfUnaryBaseID, index, index + 1);
      }

      // add constraint function
      typename GmType::FunctionIdentifier pottsConstraintID = model.addFunction(pottsConstraint);
      IndexType variables[2];
      for(size_t i = 0; i < gridHeight; ++i) {
         for(size_t j = 0; j < gridWidth; ++j) {
            size_t variable0 = i + gridHeight * j;
            if(i + 1 < gridHeight) {
               variables[0] = variable0;
               variables[1] = i + 1 + gridHeight * j;
               model.addFactor(pottsConstraintID, variables, variables + 2);
            }
            if(j + 1 < gridWidth) {
               variables[0] = variable0;
               variables[1] = i + gridHeight * (j + 1);
               model.addFactor(pottsConstraintID, variables, variables + 2);
            }
         }
      }

      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(model.isGrid(gridIDs));

      // solve grid
      std::vector<LabelType> expectedStates(numVariables, 0);
      for(size_t i = numVariables/2; i < numVariables; i++) {
         expectedStates[i] = 1;
      }

      opengm::Bruteforce<GmType, opengm::Minimizer> solver(model);
      solver.infer();
      std::vector<LabelType> computedStates;
      solver.arg(computedStates);
      OPENGM_TEST_EQUAL_SEQUENCE(computedStates.begin(), computedStates.end(), expectedStates.begin());
   }

   void testLabelOrderFunction() {
      std::cout << "  * LabelOrderFunction" << std::endl;

      typedef T      ValueType;
      typedef size_t IndexType;
      typedef size_t LabelType;

      const LabelType maxNumLabels = 6;
      const size_t numTestIterations = 50;
      const size_t numEvaluationsPerTest = 100;
      const ValueType minCoefficientsValue = -2.0;
      const ValueType maxCoefficientsValue = 2.0;
      const ValueType validValue = 0.0;
      const ValueType invalidValue = 1.0;

      typedef opengm::ExplicitFunction<ValueType, IndexType, LabelType>   ExplicitFunction;
      typedef opengm::LabelOrderFunction<ValueType, IndexType, LabelType> LabelOrderFunction;

      typedef typename LabelOrderFunction::LabelOrderType                               LabelOrderType;
      typedef typename LabelOrderFunction::LinearConstraintType                         LinearConstraintType;
      typedef typename LabelOrderFunction::IndicatorVariablesIteratorType               VariablesIteratorType;
      typedef typename LinearConstraintType::CoefficientsIteratorType                   CoefficientsIteratorType;
      typedef typename LabelOrderFunction::ViolatedLinearConstraintsIteratorType        ViolatedLinearConstraintsIteratorType;
      typedef typename LabelOrderFunction::ViolatedLinearConstraintsWeightsIteratorType LinearConstraintsWeightsIteratorType;

      typedef opengm::RandomUniformInteger<LabelType> RandomUniformLabelType;
      RandomUniformLabelType labelGenerator(0, maxNumLabels);
      typedef opengm::RandomUniformInteger<IndexType> RandomUniformIndexType;
      RandomUniformIndexType indexGenerator(0, 2);
      typedef opengm::RandomUniformFloatingPoint<double> RandomUniformValueType;
      RandomUniformValueType valueGenerator(minCoefficientsValue, maxCoefficientsValue);
      typedef opengm::RandomUniformFloatingPoint<double> RandomUniformDoubleType;
      RandomUniformDoubleType boxGenerator(0.0, 1.0);

      // test property
      LabelType smallShape[] = {2, 2};
      ExplicitFunction ef(smallShape, smallShape + 2);
      LabelOrderFunction lof(2, 2, LabelOrderType(2), 20.0, 5.0);
      OPENGM_TEST(!ef.isLinearConstraint());
      OPENGM_TEST(lof.isLinearConstraint());

      // test min
      OPENGM_TEST_EQUAL(lof.min(), 5.0);

      // test max
      OPENGM_TEST_EQUAL(lof.max(), 20.0);

      // test minmax
      opengm::MinMaxFunctor<ValueType> minmax = lof.minMax();
      OPENGM_TEST_EQUAL(minmax.min(), 5.0);
      OPENGM_TEST_EQUAL(minmax.max(), 20.0);

      // test shape, dimension, size and evaluation (operator())
      for(size_t testIter = 0; testIter < numTestIterations; testIter++) {
         // create shape
         LabelType numLabelsVar1 = labelGenerator() + 1;
         LabelType numLabelsVar2 = labelGenerator() + 1;

         LabelType maxNumLabels = numLabelsVar1 > numLabelsVar2 ? numLabelsVar1 : numLabelsVar2;

         // create label order
         LabelOrderType labelOrder(maxNumLabels);
         for(LabelType i = 0; i < maxNumLabels; i++) {
            labelOrder[i] = valueGenerator();
         }

         // create functions
         LabelOrderFunction labelOrderFunction(numLabelsVar1, numLabelsVar2, labelOrder, validValue, invalidValue);

         // test dimension
         OPENGM_TEST_EQUAL(labelOrderFunction.dimension(), 2);

         // test shape
         OPENGM_TEST_EQUAL(labelOrderFunction.shape(0), numLabelsVar1);
         OPENGM_TEST_EQUAL(labelOrderFunction.shape(1), numLabelsVar2);

         // test size
         size_t expectedSize = numLabelsVar1 * numLabelsVar2;
         OPENGM_TEST_EQUAL(labelOrderFunction.size(), expectedSize);

         // test serialization
         testSerialization(labelOrderFunction);

         // test evaluation
         // compute sorted label order
         std::vector<LabelType> sortedLabelOrder;
         sortingPermutation(labelOrder, sortedLabelOrder);

         for(size_t evaluationIter = 0; evaluationIter < numEvaluationsPerTest; evaluationIter++) {
            // create evaluation vector
            std::vector<LabelType> evalVec(2);
            for(size_t i = 0; i < 2; i++) {
               RandomUniformLabelType stateGenerator(0, labelOrderFunction.shape(i));
               LabelType currentState = stateGenerator();
               evalVec[i] = currentState;
            }

            // compute expected value

            ValueType expectedResult = 2 * invalidValue;
            for(size_t i = 0; i < sortedLabelOrder.size(); i++) {
               if(labelOrder[evalVec[0]] == labelOrder[evalVec[1]]) {
                  expectedResult = validValue;
                  break;
               } else if(sortedLabelOrder[i] == evalVec[0]) {
                  expectedResult = validValue;
                  break;
               } else if(sortedLabelOrder[i] == evalVec[1]) {
                  expectedResult = invalidValue;
                  break;
               }
            }

            OPENGM_TEST(expectedResult != 2 * invalidValue);

            // check results
            OPENGM_TEST_EQUAL(labelOrderFunction(evalVec.begin()), expectedResult);

            // check challenge function
            ViolatedLinearConstraintsIteratorType violatedConstraintsLabelOrderFunctionBegin;
            ViolatedLinearConstraintsIteratorType violatedConstraintsLabelOrderFunctionEnd;
            LinearConstraintsWeightsIteratorType  violatedConstraintsWeightsLabelOrderFunctionBegin;
            labelOrderFunction.challenge(violatedConstraintsLabelOrderFunctionBegin, violatedConstraintsLabelOrderFunctionEnd, violatedConstraintsWeightsLabelOrderFunctionBegin, evalVec.begin());
            if(expectedResult == validValue) {
               OPENGM_TEST(violatedConstraintsLabelOrderFunctionBegin == violatedConstraintsLabelOrderFunctionEnd);
            } else {
               if(LabelOrderFunction::useSingleConstraint_) {
                  if(LabelOrderFunction::useMultipleConstraints_) {
                     OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin, violatedConstraintsLabelOrderFunctionEnd), 2);
                  } else {
                     OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin, violatedConstraintsLabelOrderFunctionEnd), 1);
                  }
               } else {
                  if(LabelOrderFunction::useMultipleConstraints_) {
                     OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin, violatedConstraintsLabelOrderFunctionEnd), 1);
                  } else {
                     throw opengm::RuntimeError("Unsupported configuration for label order function. At least one of LabelOrderFunction::useSingleConstraint_ and LabelOrderFunction::useMultipleConstraints_ has to be set to true.");
                  }
               }

               if(LabelOrderFunction::useSingleConstraint_) {
                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin(), violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesEnd()), numLabelsVar1 + numLabelsVar2);
                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin(), violatedConstraintsLabelOrderFunctionBegin->coefficientsEnd()), numLabelsVar1 + numLabelsVar2);

                  for(LabelType i = 0; i < numLabelsVar1; ++i) {
                     OPENGM_TEST_EQUAL(std::distance((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i)->begin(), (violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i)->end()), 1);
                  }
                  for(LabelType i = 0; i < numLabelsVar2; ++i) {
                     OPENGM_TEST_EQUAL(std::distance((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + numLabelsVar1)->begin(), (violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + numLabelsVar1)->end()), 1);
                  }

                  for(LabelType i = 0; i < numLabelsVar1; ++i) {
                     OPENGM_TEST_EQUAL(std::distance((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i)->begin(), (violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i)->end()), 1);
                     OPENGM_TEST_EQUAL((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i)->begin()->first, 0);
                     OPENGM_TEST_EQUAL((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i)->begin()->second, i);
                     OPENGM_TEST_EQUAL(*(violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin() + i), labelOrder[i]);
                  }

                  for(LabelType i = 0; i < numLabelsVar2; ++i) {
                     OPENGM_TEST_EQUAL(std::distance((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + numLabelsVar1)->begin(), (violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + numLabelsVar1)->end()), 1);
                     OPENGM_TEST_EQUAL((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + numLabelsVar1)->begin()->first, 1);
                     OPENGM_TEST_EQUAL((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + numLabelsVar1)->begin()->second, i);
                     OPENGM_TEST_EQUAL(*(violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin() + i + numLabelsVar1), -labelOrder[i]);
                  }

                  OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionBegin->getBound(), 0.0);
                  OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionBegin->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::LessEqual);

                  ++violatedConstraintsLabelOrderFunctionBegin;
               }

               if(LabelOrderFunction::useMultipleConstraints_) {
                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin(), violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesEnd()), numLabelsVar2 + 1);
                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin(), violatedConstraintsLabelOrderFunctionBegin->coefficientsEnd()), numLabelsVar2 + 1);

                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin()->begin(), violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin()->end()), 1);
                  for(LabelType i = 0; i < numLabelsVar2; ++i) {
                     OPENGM_TEST_EQUAL(std::distance((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + 1)->begin(), (violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + 1)->end()), 1);
                  }

                  OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin()->begin()->first, 0);
                  OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin()->begin()->second, evalVec[0]);

                  for(LabelType i = 0; i < numLabelsVar2; ++i) {
                     OPENGM_TEST_EQUAL((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + 1)->begin()->first, 1);
                     OPENGM_TEST_EQUAL((violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin() + i + 1)->begin()->second, i);
                  }

                  OPENGM_TEST_EQUAL(*(violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin()), labelOrder[evalVec[0]]);
                  for(LabelType i = 0; i < numLabelsVar2; ++i) {
                     OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin()[i + 1], -labelOrder[i]);
                  }

                  OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionBegin->getBound(), 0.0);
                  OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionBegin->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::LessEqual);
               }
            }

            // test relaxed challenge
            // create relaxed evaluation vector
            const size_t numIndicatorVariables = std::distance(labelOrderFunction.indicatorVariablesOrderBegin(), labelOrderFunction.indicatorVariablesOrderEnd());
            const size_t numConstraints = std::distance(labelOrderFunction.linearConstraintsBegin(), labelOrderFunction.linearConstraintsEnd());
            if(LabelOrderFunction::useSingleConstraint_) {
               if(LabelOrderFunction::useMultipleConstraints_) {
                  OPENGM_TEST_EQUAL(numConstraints, 1 + numLabelsVar1);
               } else {
                  OPENGM_TEST_EQUAL(numConstraints, 1);
               }
            } else {
               if(LabelOrderFunction::useMultipleConstraints_) {
                  OPENGM_TEST_EQUAL(numConstraints, numLabelsVar1);
               } else {
                  throw opengm::RuntimeError("Unsupported configuration for label order function. At least one of LabelOrderFunction::useSingleConstraint_ and LabelOrderFunction::useMultipleConstraints_ has to be set to true.");
               }
            }
            std::vector<double> evalVecRelaxed(numIndicatorVariables);
            for(size_t i = 0; i < numIndicatorVariables; i++) {
               const double currentState = boxGenerator();
               evalVecRelaxed[i] = currentState;
            }

            // compute expected values
            std::vector<double> expectedValues(numConstraints);
            for(IndexType i = 0; i < numConstraints; i++) {
               const LinearConstraintType& currentConstraint = labelOrderFunction.linearConstraintsBegin()[i];
               expectedValues[i] = 0.0;
               CoefficientsIteratorType coeffIter = currentConstraint.coefficientsBegin();
               for(VariablesIteratorType varIter = currentConstraint.indicatorVariablesBegin(); varIter != currentConstraint.indicatorVariablesEnd(); ++varIter) {
                  VariablesIteratorType varPosition = std::find(labelOrderFunction.indicatorVariablesOrderBegin(), labelOrderFunction.indicatorVariablesOrderEnd(), *varIter);
                  OPENGM_TEST(varPosition != labelOrderFunction.indicatorVariablesOrderEnd());
                  const size_t evalVecRelaxedPosition = std::distance(labelOrderFunction.indicatorVariablesOrderBegin(), varPosition);
                  const size_t coeffIterPosition = std::distance(currentConstraint.indicatorVariablesBegin(), varIter);
                  expectedValues[i] += coeffIter[coeffIterPosition] * evalVecRelaxed[evalVecRelaxedPosition];
               }
            }

            IndexType numViolatedConstraints = 0;
            for(IndexType i = 0; i < numConstraints; i++) {
               if(expectedValues[i] > 0.0) {
                  ++numViolatedConstraints;
               }
            }

            labelOrderFunction.challengeRelaxed(violatedConstraintsLabelOrderFunctionBegin, violatedConstraintsLabelOrderFunctionEnd, violatedConstraintsWeightsLabelOrderFunctionBegin, evalVecRelaxed.begin());
            OPENGM_TEST_EQUAL(numViolatedConstraints, std::distance(violatedConstraintsLabelOrderFunctionBegin, violatedConstraintsLabelOrderFunctionEnd));

            if(LabelOrderFunction::useSingleConstraint_) {
               if(expectedValues[0] > 0.0) {
                  --violatedConstraintsLabelOrderFunctionEnd;
                  OPENGM_TEST_EQUAL_TOLERANCE(expectedValues[0], violatedConstraintsWeightsLabelOrderFunctionBegin[numViolatedConstraints - 1], OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL(0.0, violatedConstraintsLabelOrderFunctionEnd->getBound());
                  OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionEnd->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::LessEqual);
                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionEnd->indicatorVariablesBegin(), violatedConstraintsLabelOrderFunctionEnd->indicatorVariablesEnd()), numLabelsVar1 + numLabelsVar2);
                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionEnd->coefficientsBegin(), violatedConstraintsLabelOrderFunctionEnd->coefficientsEnd()), numLabelsVar1 + numLabelsVar2);

                  typename LinearConstraintType::IndicatorVariablesContainerType expectedVariables;
                  for(LabelType i = 0; i < numLabelsVar1; ++i) {
                     expectedVariables.push_back(typename LinearConstraintType::IndicatorVariableType(IndexType(0), i));
                  }
                  for(LabelType i = 0; i < numLabelsVar2; ++i) {
                     expectedVariables.push_back(typename LinearConstraintType::IndicatorVariableType(IndexType(1), i));
                  }

                  OPENGM_TEST_EQUAL_SEQUENCE(expectedVariables.begin(), expectedVariables.end(), violatedConstraintsLabelOrderFunctionEnd->indicatorVariablesBegin());

                  for(LabelType i = 0; i < numLabelsVar1; ++i) {
                     OPENGM_TEST_EQUAL_TOLERANCE(labelOrder[i], violatedConstraintsLabelOrderFunctionEnd->coefficientsBegin()[i], OPENGM_FLOAT_TOL);
                  }
                  for(LabelType i = 0; i < numLabelsVar2; ++i) {
                     OPENGM_TEST_EQUAL_TOLERANCE(-labelOrder[i], violatedConstraintsLabelOrderFunctionEnd->coefficientsBegin()[i + numLabelsVar1], OPENGM_FLOAT_TOL);
                  }
               }
            }
            for(size_t i = 0 + (LabelOrderFunction::useSingleConstraint_ ? 1 : 0); i < numConstraints; ++i) {
               if(expectedValues[i] > 0.0) {
                  OPENGM_TEST_EQUAL_TOLERANCE(expectedValues[i], *violatedConstraintsWeightsLabelOrderFunctionBegin, OPENGM_FLOAT_TOL);
                  OPENGM_TEST_EQUAL(0.0, violatedConstraintsLabelOrderFunctionBegin->getBound());
                  OPENGM_TEST_EQUAL(violatedConstraintsLabelOrderFunctionBegin->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::LessEqual);
                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin(), violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesEnd()), numLabelsVar2 + 1);
                  OPENGM_TEST_EQUAL(std::distance(violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin(), violatedConstraintsLabelOrderFunctionBegin->coefficientsEnd()), numLabelsVar2 + 1);

                  const LabelType labelVar1 = i - (LabelOrderFunction::useSingleConstraint_ ? 1 : 0);

                  typename LinearConstraintType::IndicatorVariablesContainerType expectedVariables;
                  expectedVariables.push_back(typename LinearConstraintType::IndicatorVariableType(IndexType(0), labelVar1));
                  for(LabelType j = 0; j < numLabelsVar2; ++j) {
                     expectedVariables.push_back(typename LinearConstraintType::IndicatorVariableType(IndexType(1), j));
                  }

                  OPENGM_TEST_EQUAL_SEQUENCE(expectedVariables.begin(), expectedVariables.end(), violatedConstraintsLabelOrderFunctionBegin->indicatorVariablesBegin());
                  OPENGM_TEST_EQUAL_TOLERANCE(labelOrder[labelVar1], violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin()[0], OPENGM_FLOAT_TOL);
                  for(LabelType j = 0; j < numLabelsVar2; ++j) {
                     OPENGM_TEST_EQUAL_TOLERANCE(-labelOrder[j], violatedConstraintsLabelOrderFunctionBegin->coefficientsBegin()[j + 1], OPENGM_FLOAT_TOL);
                  }

                  ++violatedConstraintsLabelOrderFunctionBegin;
                  ++violatedConstraintsWeightsLabelOrderFunctionBegin;
               }
            }
            OPENGM_TEST(violatedConstraintsLabelOrderFunctionBegin == violatedConstraintsLabelOrderFunctionEnd);
         }
      }
   }
   
   void testNumLabelsLimitationFunction() {
      std::cout << "  * NumLabelsLimitationFunction" << std::endl;

      typedef T      ValueType;
      typedef size_t IndexType;
      typedef size_t LabelType;

      const IndexType minNumVariables = 1;
      const IndexType maxNumVariables = 6;
      const LabelType minNumLabels = 1;
      const LabelType maxNumLabels = 5;
      const size_t numTestIterations = 50;
      const size_t numEvaluationsPerTest = 100;
      const ValueType validValue = static_cast<ValueType>(1.11);
      const ValueType invalidValue = static_cast<ValueType>(3.14);

      typedef opengm::NumLabelsLimitationFunction<ValueType, IndexType, LabelType> NumLabelsLimitFunction;

      typedef typename NumLabelsLimitFunction::LinearConstraintType   LinearConstraintType;
      typedef typename LinearConstraintType::IndicatorVariableType    IndicatorVariableType;

      typedef typename NumLabelsLimitFunction::LinearConstraintsIteratorType                LinearConstraintsIteratorType;
      typedef typename NumLabelsLimitFunction::IndicatorVariablesIteratorType               VariablesIteratorType;
      typedef typename NumLabelsLimitFunction::ViolatedLinearConstraintsIteratorType        ViolatedLinearConstraintsIteratorType;
      typedef typename NumLabelsLimitFunction::ViolatedLinearConstraintsWeightsIteratorType LinearConstraintsWeightsIteratorType;

      typedef opengm::RandomUniformInteger<IndexType> RandomUniformIndexType;
      RandomUniformIndexType numVariablesGenerator(minNumVariables, maxNumVariables + 1);
      typedef opengm::RandomUniformInteger<LabelType> RandomUniformLabelType;
      RandomUniformLabelType labelGenerator(minNumLabels, maxNumLabels + 1);
      typedef opengm::RandomUniformFloatingPoint<double> RandomUniformDoubleType;
      RandomUniformDoubleType boxGenerator(0.0, 1.0);

      RandomUniformLabelType boolGenerator(0, 2);

      // test shape, dimension, size and evaluation (operator())
      for(size_t testIter = 0; testIter < numTestIterations; testIter++) {
         const bool useSameNumLabels = static_cast<bool>(boolGenerator());
         const IndexType numVariables = numVariablesGenerator();
         std::vector<LabelType> shape(numVariables);
         for(IndexType i = 0; i < numVariables; ++i) {
            shape[i] = labelGenerator();
         }

         const LabelType currentMaxNumLabels = useSameNumLabels ? shape[0] : *(std::max_element(shape.begin(), shape.end()));
         RandomUniformLabelType maxNumDifferentLabelsGenerator(0, currentMaxNumLabels + 1);
         const LabelType maxNumUsedLabels = maxNumDifferentLabelsGenerator();

         // create function
         NumLabelsLimitFunction* numLabelsLimitFunction = NULL;
         if(useSameNumLabels) {
            numLabelsLimitFunction = new NumLabelsLimitFunction(numVariables, currentMaxNumLabels, maxNumUsedLabels, validValue, invalidValue);
         } else {
            numLabelsLimitFunction = new NumLabelsLimitFunction(shape.begin(), shape.end(), maxNumUsedLabels, validValue, invalidValue);
         }

         // test dimension
         OPENGM_TEST_EQUAL(numLabelsLimitFunction->dimension(), numVariables);

         // test shape
         for(IndexType i = 0; i < numVariables; ++i) {
            if(useSameNumLabels) {
               OPENGM_TEST_EQUAL(numLabelsLimitFunction->shape(i), currentMaxNumLabels);
            } else {
               OPENGM_TEST_EQUAL(numLabelsLimitFunction->shape(i), shape[i]);
            }
         }

         // test size
         size_t expectedSize = 1.0;
         for(IndexType i = 0; i < numVariables; ++i) {
            if(useSameNumLabels) {
               expectedSize *= currentMaxNumLabels;
            } else {
               expectedSize *= shape[i];
            }
         }
         OPENGM_TEST_EQUAL(numLabelsLimitFunction->size(), expectedSize);

         // test min
         OPENGM_TEST_EQUAL(numLabelsLimitFunction->min(), std::min(validValue, invalidValue));

         // test max
         OPENGM_TEST_EQUAL(numLabelsLimitFunction->max(), std::max(validValue, invalidValue));

         // test minmax
         opengm::MinMaxFunctor<ValueType> minmax = numLabelsLimitFunction->minMax();
         OPENGM_TEST_EQUAL(minmax.min(), std::min(validValue, invalidValue));
         OPENGM_TEST_EQUAL(minmax.max(), std::max(validValue, invalidValue));

         // check variable order
         const VariablesIteratorType variablesOrderBegin = numLabelsLimitFunction->indicatorVariablesOrderBegin();
         const VariablesIteratorType variablesOrderEnd = numLabelsLimitFunction->indicatorVariablesOrderEnd();
         OPENGM_TEST_EQUAL(std::distance(variablesOrderBegin, variablesOrderEnd), currentMaxNumLabels);
         for(LabelType i = 0; i < currentMaxNumLabels; ++i) {
            OPENGM_TEST(variablesOrderBegin[i].getLogicalOperatorType() == IndicatorVariableType::Or);
            IndexType expectedVariableLength = 0;
            if(useSameNumLabels) {
               expectedVariableLength = numVariables;
            } else {
               for(IndexType j = 0; j < numVariables; ++j) {
                  if(shape[j] > i) {
                     ++expectedVariableLength;
                  }
               }
            }
            OPENGM_TEST_EQUAL(std::distance(variablesOrderBegin[i].begin(), variablesOrderBegin[i].end()), expectedVariableLength);
            for(IndexType j = 0; j < expectedVariableLength; j++) {
               OPENGM_TEST(variablesOrderBegin[i].begin()[j].first < numVariables);
               if(!useSameNumLabels) {
                  OPENGM_TEST(shape[variablesOrderBegin[i].begin()[j].first] > i);
               }

               OPENGM_TEST_EQUAL(variablesOrderBegin[i].begin()[j].second, i);
            }
         }

         // check linear constraints
         const LinearConstraintsIteratorType linearConstraintsBegin = numLabelsLimitFunction->linearConstraintsBegin();
         const LinearConstraintsIteratorType linearConstraintsEnd = numLabelsLimitFunction->linearConstraintsEnd();
         OPENGM_TEST_EQUAL(std::distance(linearConstraintsBegin, linearConstraintsEnd), 1);
         OPENGM_TEST_EQUAL(linearConstraintsBegin->getBound(), maxNumUsedLabels);
         OPENGM_TEST_EQUAL(linearConstraintsBegin->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::LessEqual);
         OPENGM_TEST_EQUAL(std::distance(linearConstraintsBegin->indicatorVariablesBegin(), linearConstraintsBegin->indicatorVariablesEnd()), currentMaxNumLabels);
         OPENGM_TEST_EQUAL(std::distance(linearConstraintsBegin->coefficientsBegin(), linearConstraintsBegin->coefficientsEnd()), currentMaxNumLabels);
         for(LabelType i = 0; i < currentMaxNumLabels; ++i) {
            OPENGM_TEST(linearConstraintsBegin->indicatorVariablesBegin()[i] == variablesOrderBegin[i]);
            OPENGM_TEST_EQUAL(linearConstraintsBegin->coefficientsBegin()[i], 1.0);
         }


         // test evaluation
         for(size_t evalIter = 0; evalIter < numEvaluationsPerTest; ++evalIter) {
            std::vector<LabelType> evalVec(numVariables);
            for(size_t i = 0; i < numVariables; ++i) {
               if(useSameNumLabels) {
                  RandomUniformLabelType stateGenerator(0, currentMaxNumLabels);
                  evalVec[i] = stateGenerator();
               } else {
                  RandomUniformLabelType stateGenerator(0, shape[i]);
                  evalVec[i] = stateGenerator();
               }
            }
            LabelType currentNumUsedLabels = 0;
            for(LabelType i = 0; i < currentMaxNumLabels; ++i) {
               if(std::find(evalVec.begin(), evalVec.end(), i) != evalVec.end()) {
                  ++currentNumUsedLabels;
               }
            }
            const ValueType expectedResult = currentNumUsedLabels > maxNumUsedLabels ? invalidValue : validValue;

            // operator()
            const ValueType result = numLabelsLimitFunction->operator()(evalVec.begin());
            OPENGM_TEST_EQUAL_TOLERANCE(result, expectedResult, OPENGM_FLOAT_TOL);

            // challenge function
            ViolatedLinearConstraintsIteratorType violatedConstraintsBegin;
            ViolatedLinearConstraintsIteratorType violatedConstraintsEnd;
            LinearConstraintsWeightsIteratorType  violatedConstraintsWeightsBegin;
            numLabelsLimitFunction->challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, evalVec.begin());
            if(expectedResult == validValue) {
               OPENGM_TEST(violatedConstraintsBegin == violatedConstraintsEnd);
            } else {
               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin, violatedConstraintsEnd), 1);
               const double expectedWeight = static_cast<double>(currentNumUsedLabels) - static_cast<double>(maxNumUsedLabels);
               OPENGM_TEST_EQUAL_TOLERANCE(*violatedConstraintsWeightsBegin, expectedWeight, OPENGM_FLOAT_TOL);
               OPENGM_TEST_EQUAL_TOLERANCE(violatedConstraintsBegin->getBound(), maxNumUsedLabels, OPENGM_FLOAT_TOL);
               OPENGM_TEST_EQUAL(violatedConstraintsBegin->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::LessEqual);

               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin->indicatorVariablesBegin(), violatedConstraintsBegin->indicatorVariablesEnd()), currentMaxNumLabels);
               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin->coefficientsBegin(), violatedConstraintsBegin->coefficientsEnd()), currentMaxNumLabels);

               for(LabelType i = 0; i < currentMaxNumLabels; ++i) {
                  OPENGM_TEST_EQUAL(violatedConstraintsBegin->coefficientsBegin()[i], 1.0);
                  OPENGM_TEST(violatedConstraintsBegin->indicatorVariablesBegin()[i] == variablesOrderBegin[i]);
               }
            }

            // test relaxed challenge
            // create relaxed evaluation vector
            std::vector<double> evalVecRelaxed(currentMaxNumLabels);
            for(size_t i = 0; i < currentMaxNumLabels; i++) {
               const double currentState = boxGenerator();
               evalVecRelaxed[i] = currentState;
            }

            // compute expected values
            double expectedRelaxedValue = std::accumulate(evalVecRelaxed.begin(), evalVecRelaxed.end(), 0.0);

            // challenge relaxed
            numLabelsLimitFunction->challengeRelaxed(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, evalVecRelaxed.begin());

            // check results
            if(expectedRelaxedValue <= maxNumUsedLabels) {
               OPENGM_TEST(violatedConstraintsBegin == violatedConstraintsEnd);
            } else {
               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin, violatedConstraintsEnd), 1);
               const double expectedWeight = expectedRelaxedValue - static_cast<double>(maxNumUsedLabels);
               OPENGM_TEST_EQUAL_TOLERANCE(*violatedConstraintsWeightsBegin, expectedWeight, OPENGM_FLOAT_TOL);
               OPENGM_TEST_EQUAL(violatedConstraintsBegin->getBound(), maxNumUsedLabels);
               OPENGM_TEST_EQUAL(violatedConstraintsBegin->getConstraintOperator(), LinearConstraintType::LinearConstraintOperatorType::LessEqual);

               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin->indicatorVariablesBegin(), violatedConstraintsBegin->indicatorVariablesEnd()), currentMaxNumLabels);
               OPENGM_TEST_EQUAL(std::distance(violatedConstraintsBegin->coefficientsBegin(), violatedConstraintsBegin->coefficientsEnd()), currentMaxNumLabels);

               for(LabelType i = 0; i < currentMaxNumLabels; ++i) {
                  OPENGM_TEST_EQUAL(violatedConstraintsBegin->coefficientsBegin()[i], 1.0);
                  OPENGM_TEST(violatedConstraintsBegin->indicatorVariablesBegin()[i] == variablesOrderBegin[i]);
               }
            }
         }

         // test serialization
         testSerialization(*numLabelsLimitFunction);

         // cleanup
         delete numLabelsLimitFunction;
      }
   }

   void testSumConstraintFunction() {
      std::cout << "  * SumConstraintFunction" << std::endl;

      typedef T      ValueType;
      typedef size_t IndexType;
      typedef size_t LabelType;

      const IndexType minNumVariables = 1;
      const IndexType maxNumVariables = 10;
      const LabelType minNumLabels = 1;
      const LabelType maxNumLabels = 6;
      const size_t numTestIterations = 10;
      const size_t numEvaluationsPerTest = 20;
      const ValueType minCoefficientsValue = -2.0;
      const ValueType maxCoefficientsValue = 2.0;
      const ValueType minLambda = 1.0;
      const ValueType maxLambda = 2.0;

      typedef opengm::SumConstraintFunction<ValueType, IndexType, LabelType> SumConstraintFunction;

      typedef opengm::RandomUniformInteger<IndexType> RandomUniformIndexType;
      RandomUniformIndexType numVariablesGenerator(minNumVariables, maxNumVariables + 1);
      typedef opengm::RandomUniformInteger<LabelType> RandomUniformLabelType;
      RandomUniformLabelType labelGenerator(minNumLabels, maxNumLabels + 1);

      typedef opengm::RandomUniformFloatingPoint<double> RandomUniformValueType;
      RandomUniformValueType coefficientsGenerator(minCoefficientsValue, maxCoefficientsValue);
      RandomUniformValueType lambdaGenerator(minLambda, maxLambda);

      // test shape, dimension, size and evaluation (operator())
      for(size_t testIter = 0; testIter < numTestIterations; testIter++) {
         // create shape
         IndexType numVariables = numVariablesGenerator();
         std::vector<LabelType> shape(numVariables);
         LabelType currentMaxNumLabels = 0;
         for(IndexType i = 0; i < numVariables; ++i) {
            shape[i] = labelGenerator();
            if(shape[i] > currentMaxNumLabels) {
               currentMaxNumLabels = shape[i];
            }
         }

         // create coefficients
         std::vector<ValueType> coefficients;
         size_t numCoefficients = 0;
         for(IndexType i = 0; i < numVariables; ++i) {
            numCoefficients += shape[i];
         }

         if(numCoefficients >= numVariables * shape[0]) {
            coefficients.reserve(numCoefficients);
            for(size_t i = 0; i < numCoefficients; ++i) {
               coefficients.push_back(coefficientsGenerator());
            }
         } else {
            coefficients.reserve(numVariables * shape[0]);
            for(size_t i = 0; i < numVariables * shape[0]; ++i) {
               coefficients.push_back(coefficientsGenerator());
            }
         }


         // create lambda
         ValueType lambda = lambdaGenerator();

         // create bound
         ValueType bound = ((maxCoefficientsValue - minCoefficientsValue) / 2) * numVariables;

         // create function
         SumConstraintFunction sumConstraintFunction(shape.begin(), shape.end(), coefficients.begin(), coefficients.begin() + numCoefficients, false, lambda, bound);
         SumConstraintFunction sumConstraintFunctionSharedCoefficients(shape.begin(), shape.end(), coefficients.begin(), coefficients.begin() + currentMaxNumLabels, true, lambda, bound);
         SumConstraintFunction sumConstraintFunctionSameNumLabels(numVariables, shape[0], coefficients.begin(), coefficients.begin() + (numVariables * shape[0]), false, lambda, bound);
         SumConstraintFunction sumConstraintFunctionSameNumLabelsSharedCoefficients(numVariables, shape[0], coefficients.begin(), coefficients.begin() + shape[0], true, lambda, bound);

         // test dimension
         OPENGM_TEST_EQUAL(sumConstraintFunction.dimension(), numVariables);
         OPENGM_TEST_EQUAL(sumConstraintFunctionSharedCoefficients.dimension(), numVariables);
         OPENGM_TEST_EQUAL(sumConstraintFunctionSameNumLabels.dimension(), numVariables);
         OPENGM_TEST_EQUAL(sumConstraintFunctionSameNumLabelsSharedCoefficients.dimension(), numVariables);

         // test shape
         for(IndexType i = 0; i < numVariables; ++i) {
            OPENGM_TEST_EQUAL(sumConstraintFunction.shape(i), shape[i]);
            OPENGM_TEST_EQUAL(sumConstraintFunctionSharedCoefficients.shape(i), shape[i]);
            OPENGM_TEST_EQUAL(sumConstraintFunctionSameNumLabels.shape(i), shape[0]);
            OPENGM_TEST_EQUAL(sumConstraintFunctionSameNumLabelsSharedCoefficients.shape(i), shape[0]);
         }

         // test size
         size_t expectedSize = 1.0;
         size_t expectedSizeSameNumLabels = 1.0;
         for(IndexType i = 0; i < numVariables; ++i) {
            expectedSize *= shape[i];
            expectedSizeSameNumLabels *= shape[0];
         }
         OPENGM_TEST_EQUAL(sumConstraintFunction.size(), expectedSize);
         OPENGM_TEST_EQUAL(sumConstraintFunctionSharedCoefficients.size(), expectedSize);
         OPENGM_TEST_EQUAL(sumConstraintFunctionSameNumLabels.size(), expectedSizeSameNumLabels);
         OPENGM_TEST_EQUAL(sumConstraintFunctionSameNumLabelsSharedCoefficients.size(), expectedSizeSameNumLabels);

         // test evaluation
         for(size_t evaluationIter = 0; evaluationIter < numEvaluationsPerTest; evaluationIter++) {
            // create evaluation vector
            std::vector<LabelType> evalVec(numVariables);
            std::vector<LabelType> evalVecSameNumLabels(numVariables);
            for(IndexType i = 0; i < numVariables; ++i) {
               RandomUniformLabelType stateGenerator(0, shape[i]);
               RandomUniformLabelType stateGeneratorSameNumLabels(0, shape[0]);
               const LabelType currentState = stateGenerator();
               const LabelType currentStateSameNumLabels = stateGeneratorSameNumLabels();
               evalVec[i] = currentState;
               evalVecSameNumLabels[i] = currentStateSameNumLabels;
            }

            // compute expected value
            ValueType expectedResult = -bound;
            ValueType expectedResultSharedCoefficients = -bound;
            ValueType expectedResultSameNumLabels = -bound;
            ValueType expectedResultSameNumLabelsSharedCoefficients = -bound;
            size_t currentOffset = 0;
            for(IndexType i = 0; i < numVariables; ++i) {
               expectedResult += coefficients[evalVec[i] + currentOffset];
               expectedResultSharedCoefficients += coefficients[evalVec[i]];
               expectedResultSameNumLabels += coefficients[evalVecSameNumLabels[i] + (i * shape[0])];
               expectedResultSameNumLabelsSharedCoefficients += coefficients[evalVecSameNumLabels[i]];
               currentOffset += shape[i];
            }

            expectedResult = std::abs(expectedResult) * lambda;
            expectedResultSharedCoefficients = std::abs(expectedResultSharedCoefficients) * lambda;
            expectedResultSameNumLabels = std::abs(expectedResultSameNumLabels) * lambda;
            expectedResultSameNumLabelsSharedCoefficients = std::abs(expectedResultSameNumLabelsSharedCoefficients) * lambda;

            // check results
            const ValueType computedResult = sumConstraintFunction(evalVec.begin());
            const ValueType computedResultSharedCoefficients = sumConstraintFunctionSharedCoefficients(evalVec.begin());
            const ValueType computedResultSameNumLabels = sumConstraintFunctionSameNumLabels(evalVecSameNumLabels.begin());
            const ValueType computedResultSameNumLabelsSharedCoefficients = sumConstraintFunctionSameNumLabelsSharedCoefficients(evalVecSameNumLabels.begin());
            OPENGM_TEST_EQUAL_TOLERANCE(computedResult, expectedResult, OPENGM_FLOAT_TOL);
            OPENGM_TEST_EQUAL_TOLERANCE(computedResultSharedCoefficients, expectedResultSharedCoefficients, OPENGM_FLOAT_TOL);
            OPENGM_TEST_EQUAL_TOLERANCE(computedResultSameNumLabels, expectedResultSameNumLabels, OPENGM_FLOAT_TOL);
            OPENGM_TEST_EQUAL_TOLERANCE(computedResultSameNumLabelsSharedCoefficients, expectedResultSameNumLabelsSharedCoefficients, OPENGM_FLOAT_TOL);
         }

         // test serialization
         testSerialization(sumConstraintFunction);
         testSerialization(sumConstraintFunctionSharedCoefficients);
         testSerialization(sumConstraintFunctionSameNumLabels);
         testSerialization(sumConstraintFunctionSameNumLabelsSharedCoefficients);
      }
   }

   void testLabelCostFunction() {
      std::cout << "  * labelCostFunction" << std::endl;

      typedef T      ValueType;
      typedef size_t IndexType;
      typedef size_t LabelType;

      const IndexType minNumVariables = 1;
      const IndexType maxNumVariables = 10;
      const LabelType minNumLabels = 1;
      const LabelType maxNumLabels = 6;
      const size_t numTestIterations = 10;
      const size_t numEvaluationsPerTest = 20;
      const ValueType minCostsValue = 0.0;
      const ValueType maxCostsValue = 1.0;

      typedef opengm::LabelCostFunction<ValueType, IndexType, LabelType> LabelCostFunction;

      typedef opengm::RandomUniformInteger<IndexType> RandomUniformIndexType;
      RandomUniformIndexType numVariablesGenerator(minNumVariables, maxNumVariables + 1);
      typedef opengm::RandomUniformInteger<LabelType> RandomUniformLabelType;
      RandomUniformLabelType labelGenerator(minNumLabels, maxNumLabels + 1);

      typedef opengm::RandomUniformFloatingPoint<double> RandomUniformValueType;
      RandomUniformValueType costsGenerator(minCostsValue, maxCostsValue);
      RandomUniformLabelType boolGenerator(0, 2);

      // test shape, dimension, size and evaluation (operator())
      for(size_t testIter = 0; testIter < numTestIterations; testIter++) {
         const bool useSameNumLabels = static_cast<bool>(boolGenerator());
         const bool useSingleLabel = static_cast<bool>(boolGenerator());

         const IndexType numVariables = numVariablesGenerator();
         std::vector<LabelType> shape(numVariables);
         std::vector<ValueType> costs(maxNumLabels);
         for(IndexType i = 0; i < numVariables; ++i) {
            shape[i] = labelGenerator();
         }
         for(LabelType i = 0; i < maxNumLabels; ++i) {
            costs[i] = costsGenerator();
         }

         // create function
         LabelCostFunction* labelCostFunction = NULL;
         if(useSameNumLabels) {
            if(useSingleLabel) {
               labelCostFunction = new LabelCostFunction(numVariables, shape[0], shape[0] - 1, costs[shape[0] - 1]);
            } else {
               labelCostFunction = new LabelCostFunction(numVariables, shape[0], costs.begin(), costs.end());
            }
         } else {
            if(useSingleLabel) {
               labelCostFunction = new LabelCostFunction(shape.begin(), shape.end(), shape[0] - 1, costs[shape[0] - 1]);
            } else {
               labelCostFunction = new LabelCostFunction(shape.begin(), shape.end(), costs.begin(), costs.end());
            }
         }

         // test dimension
         OPENGM_TEST_EQUAL(labelCostFunction->dimension(), numVariables);

         // test shape
         for(IndexType i = 0; i < numVariables; ++i) {
            if(useSameNumLabels) {
               OPENGM_TEST_EQUAL(labelCostFunction->shape(i), shape[0]);
            } else {
               OPENGM_TEST_EQUAL(labelCostFunction->shape(i), shape[i]);
            }
         }

         // test size
         size_t expectedSize = 1.0;
         for(IndexType i = 0; i < numVariables; ++i) {
            if(useSameNumLabels) {
               expectedSize *= shape[0];
            } else {
               expectedSize *= shape[i];
            }
         }
         OPENGM_TEST_EQUAL(labelCostFunction->size(), expectedSize);

         // test evaluation
         for(size_t evalIter = 0; evalIter < numEvaluationsPerTest; ++evalIter) {
            std::vector<LabelType> evalVec(numVariables);
            for(size_t i = 0; i < numVariables; ++i) {
               if(useSameNumLabels) {
                  RandomUniformLabelType stateGenerator(0, shape[0]);
                  evalVec[i] = stateGenerator();
               } else {
                  RandomUniformLabelType stateGenerator(0, shape[i]);
                  evalVec[i] = stateGenerator();
               }
            }

            const ValueType result = labelCostFunction->operator()(evalVec.begin());
            if(useSingleLabel) {
               ValueType expectedResultSingleLabel = 0.0;
               if(std::find(evalVec.begin(), evalVec.end(), shape[0] - 1) != evalVec.end()) {
                  expectedResultSingleLabel = costs[shape[0] - 1];
               }
               OPENGM_TEST_EQUAL_TOLERANCE(result, expectedResultSingleLabel, OPENGM_FLOAT_TOL);
            } else {
               ValueType expectedResultAllLabels = 0.0;
               for(LabelType i = 0; i < maxNumLabels; ++i) {
                  if(std::find(evalVec.begin(), evalVec.end(), i) != evalVec.end()) {
                     expectedResultAllLabels += costs[i];
                  }
               }
               OPENGM_TEST_EQUAL_TOLERANCE(result, expectedResultAllLabels, OPENGM_FLOAT_TOL);
            }
         }

         // test serialization
         testSerialization(*labelCostFunction);

         // cleanup
         delete labelCostFunction;
      }
   }

   void run() {
      testExplicitFunction();
      testAbsoluteDifference();
      testAbsoluteDifference();
      testConstant();
      testModelViewFunction();
      testPotts();
      testPottsn();
      testPottsg();
      testScaledView();
      testSquaredDifference() ;
      testTruncatedAbsoluteDifference() ;
      testTruncatedSquaredDifference() ;
      testView();
      testViewAndFixVariables();
      testSingleSiteFunction();
      testLinearConstraintFunction();
      testLabelOrderFunction();
      testNumLabelsLimitationFunction();
      testSumConstraintFunction();
      testLabelCostFunction();
   }
   void run2() {
      testFoE();
   }
};

int main() {
   std::cout << "Functions test...  " << std::endl;
   {
      FunctionsTest<int >t;
      t.run();
      FunctionsTest<double >t2;
      t2.run2();
   }
   std::cout << "done.." << std::endl;
   return 0;
}
