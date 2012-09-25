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

#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>

template<class T>
struct FunctionsTest {
   
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
   }
};

int main() {
   std::cout << "Functions test...  " << std::endl;
   {
      FunctionsTest<int >t;
      t.run();
   }
   std::cout << "done.." << std::endl;
   return 0;
}
