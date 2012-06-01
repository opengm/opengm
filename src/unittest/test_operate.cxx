#include <iostream>
#include <map>
#include <cstdlib>
#include <string>

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>

template<class T>
class TestImplicitFunction {
public:

   enum Identifier {
      T1, T2
   };
   T parameter0_;
   T parameter1_;
   template<class Iterator>
   TestImplicitFunction(Iterator, Iterator, const Identifier id);
   template<class Iterator>
   T operator()(Iterator) const;

   size_t
   shape(size_t i)const {
      return shape_[i];
   };

   size_t
   dimension()const {
      return shape_.size();
   };

   size_t
   size()const {
      return size_;
   };
private:
   size_t id_;
   std::vector<size_t> shape_;
   size_t size_;
};

template<class T>
template<class Iterator>
TestImplicitFunction<T>
::TestImplicitFunction(
                     Iterator sBegin, Iterator sEnd,
                     const typename TestImplicitFunction<T>::Identifier id
                     ) : id_(id),
shape_(sBegin, sEnd),
size_() {
   if(shape_.size() != 0) {
      size_ = std::accumulate(sBegin, sEnd, 1, std::multiplies< typename std::iterator_traits<Iterator>::value_type > ());
   }
   else size_ = 1;
}

template<class T>
template<class Iterator>
T TestImplicitFunction<T>::operator()
(
 Iterator begin
 ) const {
   Iterator end = begin + shape_.size();
   switch(id_) {
   case T1:
   {
      if(std::distance(begin, end) == 1) {
         return parameter0_;
      }
      else if(begin == end) {
         return parameter0_;
      }
      else {
         size_t tmp = *begin;
         while(begin != end) {
            if(*begin != tmp) {
               return parameter1_;
            }
            ++begin;
         }
      }
      return parameter0_;
   }
   case T2:
   {
      T tmp = *begin;
      T tmp2 = 1;
      ++begin;
      while(begin != end) {
         tmp2 *= 4;
         tmp += (*begin * tmp2);
         ++begin;
      }
      return tmp;
   }
   default:
   {
      return static_cast<T> (0);
   }
   }
}

class FooOperator {
};

template<class T>
struct TestOperate {

   void run() {
      typedef typename opengm::meta::TypeListGenerator<opengm::ExplicitFunction<T>,TestImplicitFunction<T> >::type FunctionTypeList;
      typedef opengm::GraphicalModel<T, FooOperator, FunctionTypeList > GraphicalModelType;
      typedef typename GraphicalModelType::ValueType ValueType;
      typedef  opengm::ExplicitFunction<T> ExplicitFunctionType;
      typedef TestImplicitFunction<T> ImplicitFunctionType;
      typedef typename GraphicalModelType::IndependentFactorType IndependentFactorType;
      typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;


      size_t variableStates[] = {4, 4, 4, 4, 4, 4, 4, 4,
         4, 4, 4, 4, 4, 4, 4, 4,
         4, 4, 4, 4, 4, 4, 4, 4};
      GraphicalModelType gm(opengm::DiscreteSpace<size_t,size_t>(variableStates, variableStates + 24));
      size_t shape[] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

      IndependentFactorType tmp;
      IndependentFactorType tmp2;
      IndependentFactorType tmp3;
      ExplicitFunctionType fe1a(shape, shape + 1, 0);
      ExplicitFunctionType fe2a(shape, shape + 2, 0);
      ExplicitFunctionType fe3a(shape, shape + 3, 0);
      ExplicitFunctionType fe4a(shape, shape + 4, 0);
      ExplicitFunctionType fe1b(shape, shape + 1, 0);
      ExplicitFunctionType fe2b(shape, shape + 2, 0);
      ExplicitFunctionType fe3b(shape, shape + 3, 0);
      ExplicitFunctionType fe4b(shape, shape + 4, 0);

      ExplicitFunctionType fs1a(shape, shape + 1, 0);
      ExplicitFunctionType fs2a(shape, shape + 2, 0);
      ExplicitFunctionType fs3a(shape, shape + 3, 0);
      ExplicitFunctionType fs4a(shape, shape + 4, 0);
      ExplicitFunctionType fs1b(shape, shape + 1, 0);
      ExplicitFunctionType fs2b(shape, shape + 2, 0);
      ExplicitFunctionType fs3b(shape, shape + 3, 0);
      ExplicitFunctionType fs4b(shape, shape + 4, 0);

      ImplicitFunctionType fi1a(shape, shape + 1, ImplicitFunctionType::T1);
      ImplicitFunctionType fi2a(shape, shape + 2, ImplicitFunctionType::T1);
      ImplicitFunctionType fi3a(shape, shape + 3, ImplicitFunctionType::T1);
      ImplicitFunctionType fi4a(shape, shape + 4, ImplicitFunctionType::T1);

      fi1a.parameter0_ = 1;
      fi1a.parameter1_ = 0;

      fi2a.parameter0_ = 1;
      fi2a.parameter1_ = 0;

      fi3a.parameter0_ = 1;
      fi3a.parameter1_ = 0;

      fi4a.parameter0_ = 1;
      fi4a.parameter1_ = 0;

      ImplicitFunctionType fi1b(shape, shape + 1, ImplicitFunctionType::T2);
      ImplicitFunctionType fi2b(shape, shape + 2, ImplicitFunctionType::T2);
      ImplicitFunctionType fi3b(shape, shape + 3, ImplicitFunctionType::T2);
      ImplicitFunctionType fi4b(shape, shape + 4, ImplicitFunctionType::T2);

      //fill with data;
      for(size_t i = 0; i < 4; i++) {
         fe1a(i) = 1;
         fe2a(i, i) = 1;
         fe3a(i, i, i) = 1;
         fe4a(i, i, i, i) = 1;

         fs1a(i) = 1;
         fs2a(i, i) = 1;
         fs3a(i, i, i) = 1;
         fs4a(i, i, i, i) = 1;
      }
      size_t counter = 0;
      for(size_t x0 = 0; x0 < 4; ++x0) {
         fe1b(x0) = counter;
         fs1b(x0) = counter;
         counter++;
      }
      counter = 0;
      for(size_t x1 = 0; x1 < 4; ++x1)
         for(size_t x0 = 0; x0 < 4; ++x0) {
            fe2b(x0, x1) = counter;
            fs2b(x0, x1) = counter;
            counter++;
         }

      counter = 0;
      for(size_t x2 = 0; x2 < 4; ++x2)
         for(size_t x1 = 0; x1 < 4; ++x1)
            for(size_t x0 = 0; x0 < 4; ++x0) {
               fe3b(x0, x1, x2) = counter;
               fs3b(x0, x1, x2) = counter;
               counter++;
            }
      counter = 0;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  fe4b(x0, x1, x2, x3) = counter;
                  fs4b(x0, x1, x2, x3) = counter;
                  counter++;
               }
      //add functions
      FunctionIdentifier ie1a = gm.addFunction(fe1a);
      FunctionIdentifier ie2a = gm.addFunction(fe2a);
      FunctionIdentifier ie3a = gm.addFunction(fe3a);
      FunctionIdentifier ie4a = gm.addFunction(fe4a);
      FunctionIdentifier ie1b = gm.addFunction(fe1b);
      FunctionIdentifier ie2b = gm.addFunction(fe2b);
      FunctionIdentifier ie3b = gm.addFunction(fe3b);
      FunctionIdentifier ie4b = gm.addFunction(fe4b);

      FunctionIdentifier is1a = gm.addFunction(fs1a);
      FunctionIdentifier is2a = gm.addFunction(fs2a);
      FunctionIdentifier is3a = gm.addFunction(fs3a);
      FunctionIdentifier is4a = gm.addFunction(fs4a);
      FunctionIdentifier is1b = gm.addFunction(fs1b);
      FunctionIdentifier is2b = gm.addFunction(fs2b);
      FunctionIdentifier is3b = gm.addFunction(fs3b);
      FunctionIdentifier is4b = gm.addFunction(fs4b);

      FunctionIdentifier ii1a = gm.addFunction(fi1a);
      FunctionIdentifier ii2a = gm.addFunction(fi2a);
      FunctionIdentifier ii3a = gm.addFunction(fi3a);
      FunctionIdentifier ii4a = gm.addFunction(fi4a);

      FunctionIdentifier ii1b = gm.addFunction(fi1b);
      FunctionIdentifier ii2b = gm.addFunction(fi2b);
      FunctionIdentifier ii3b = gm.addFunction(fi3b);
      FunctionIdentifier ii4b = gm.addFunction(fi4b);

      IndependentFactorType resultFactorEE;
      IndependentFactorType resultFactorES;
      IndependentFactorType resultFactorEI;
      IndependentFactorType resultFactorSS;
      IndependentFactorType resultFactorSE;
      IndependentFactorType resultFactorSI;
      IndependentFactorType resultFactorII;
      IndependentFactorType resultFactorIE;
      IndependentFactorType resultFactorIS;

      size_t vi[] = {0, 1, 2, 3, 4};
      size_t e0a = gm.addFactor(ie1a, vi, vi + 1);
      size_t e1a = gm.addFactor(ie1a, vi, vi + 1);
      size_t s0a = gm.addFactor(is1a, vi, vi + 1);
      size_t s1a = gm.addFactor(is1a, vi, vi + 1);
      size_t i0a = gm.addFactor(ii1a, vi, vi + 1);
      size_t i1a = gm.addFactor(ii1a, vi, vi + 1);

      size_t e0b = gm.addFactor(ie1b, vi, vi + 1);
      size_t e1b = gm.addFactor(ie1b, vi, vi + 1);
      size_t s0b = gm.addFactor(is1b, vi, vi + 1);
      size_t s1b = gm.addFactor(is1b, vi, vi + 1);
      size_t i0b = gm.addFactor(ii1b, vi, vi + 1);
      size_t i1b = gm.addFactor(ii1b, vi, vi + 1);

      opengm::operateBinary(gm[e0a], gm[e1a], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[s1a], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[i1a], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0a], gm[e1a], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[s1a], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[i1a], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0a], gm[e1a], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[s1a], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[i1a], resultFactorII, std::plus<ValueType > ());
      bool passed = true;
      for(size_t i = 0; i < 4 && passed == true; ++i) {
         if(
            (
             resultFactorEE(i) == 2 &&
             resultFactorES(i) == 2 &&
             resultFactorEI(i) == 2 &&
             resultFactorSE(i) == 2 &&
             resultFactorSS(i) == 2 &&
             resultFactorSI(i) == 2 &&
             resultFactorIE(i) == 2 &&
             resultFactorIS(i) == 2 &&
             resultFactorII(i) == 2
             ) == false
            ) {
            passed = false;
            break;
         }
      }
      OPENGM_TEST(passed);


      tmp = (gm[e0a] + gm[e1a]);
      testEqualFactor(resultFactorEE, tmp);
      tmp2 = gm[e0a];
      tmp = tmp2 + gm[e1a];
      testEqualFactor(resultFactorEE, tmp);
      tmp2 = gm[e1a];
      tmp = gm[e0a] + tmp2;
      testEqualFactor(resultFactorEE, tmp);

      tmp = gm[e0a] + gm[s1a];
      testEqualFactor(resultFactorES, tmp);
      tmp2 = gm[e0a];
      tmp = tmp2 + gm[s1a];
      testEqualFactor(resultFactorES, tmp);
      tmp2 = gm[s1a];
      tmp = gm[e0a] + tmp2;
      testEqualFactor(resultFactorES, tmp);

      tmp = gm[e0a] + gm[i1a];
      testEqualFactor(resultFactorEI, tmp);
      tmp2 = gm[e0a];
      tmp = tmp2 + gm[i1a];
      testEqualFactor(resultFactorEI, tmp);
      tmp2 = gm[i1a];
      tmp = gm[e0a] + tmp2;
      testEqualFactor(resultFactorEI, tmp);

      tmp = (gm[s0a] + gm[e1a]);
      testEqualFactor(resultFactorSE, tmp);
      tmp2 = gm[s0a];
      tmp = tmp2 + gm[e1a];
      testEqualFactor(resultFactorSE, tmp);
      tmp2 = gm[e1a];
      tmp = gm[s0a] + tmp2;
      testEqualFactor(resultFactorSE, tmp);

      tmp = gm[s0a] + gm[s1a];
      testEqualFactor(resultFactorSS, tmp);
      tmp2 = gm[s0a];
      tmp = tmp2 + gm[s1a];
      testEqualFactor(resultFactorSS, tmp);
      tmp2 = gm[s1a];
      tmp = gm[s0a] + tmp2;
      testEqualFactor(resultFactorSS, tmp);

      tmp = gm[s0a] + gm[i1a];
      testEqualFactor(resultFactorSI, tmp);
      tmp2 = gm[s0a];
      tmp = tmp2 + gm[i1a];
      testEqualFactor(resultFactorSI, tmp);
      tmp2 = gm[i1a];
      tmp = gm[s0a] + tmp2;
      testEqualFactor(resultFactorSI, tmp);

      tmp = (gm[i0a] + gm[e1a]);
      testEqualFactor(resultFactorSE, tmp);
      tmp2 = gm[i0a];
      tmp = tmp2 + gm[e1a];
      testEqualFactor(resultFactorSE, tmp);
      tmp2 = gm[e1a];
      tmp = gm[i0a] + tmp2;
      testEqualFactor(resultFactorSE, tmp);

      tmp = gm[i0a] + gm[s1a];
      testEqualFactor(resultFactorSS, tmp);
      tmp2 = gm[i0a];
      tmp = tmp2 + gm[s1a];
      testEqualFactor(resultFactorSS, tmp);
      tmp2 = gm[s1a];
      tmp = gm[i0a] + tmp2;
      testEqualFactor(resultFactorSS, tmp);

      tmp = gm[i0a] + gm[i1a];
      testEqualFactor(resultFactorII, tmp);
      tmp2 = gm[i0a];
      tmp = tmp2 + gm[i1a];
      testEqualFactor(resultFactorII, tmp);
      tmp2 = gm[i1a];
      tmp = gm[i0a] + tmp2;
      testEqualFactor(resultFactorII, tmp);
      tmp3 = gm[i0a];
      tmp = tmp3 + tmp2;
      testEqualFactor(resultFactorII, tmp);
      tmp = gm[i0a];
      tmp += gm[i1a];
      testEqualFactor(resultFactorII, tmp);

      opengm::operateBinary(gm[e0b], gm[e1b], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[s1b], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[i1b], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0b], gm[e1b], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[s1b], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[i1b], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0b], gm[e1b], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[s1b], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[i1b], resultFactorII, std::plus<ValueType > ());

      for(size_t i = 0; i < 4 && passed == true; ++i) {
         if(
            (
             resultFactorEE(i) == 2 * i &&
             resultFactorES(i) == 2 * i &&
             resultFactorEI(i) == 2 * i &&
             resultFactorSE(i) == 2 * i &&
             resultFactorSS(i) == 2 * i &&
             resultFactorSI(i) == 2 * i &&
             resultFactorIE(i) == 2 * i &&
             resultFactorIS(i) == 2 * i &&
             resultFactorII(i) == 2 * i
             ) == false
            ) {
            passed = false;
            break;
         }
      }
      OPENGM_TEST(passed);
      tmp = (gm[e0b] + gm[e1b]);
      testEqualFactor(resultFactorEE, tmp);
      tmp2 = gm[e0b];
      tmp = tmp2 + gm[e1b];
      testEqualFactor(resultFactorEE, tmp);
      tmp2 = gm[e1b];
      tmp = gm[e0b] + tmp2;
      testEqualFactor(resultFactorEE, tmp);

      tmp = gm[e0b] + gm[s1b];
      testEqualFactor(resultFactorES, tmp);
      tmp2 = gm[e0b];
      tmp = tmp2 + gm[s1b];
      testEqualFactor(resultFactorES, tmp);
      tmp2 = gm[s1b];
      tmp = gm[e0b] + tmp2;
      testEqualFactor(resultFactorES, tmp);

      tmp = gm[e0b] + gm[i1b];
      testEqualFactor(resultFactorEI, tmp);
      tmp2 = gm[e0b];
      tmp = tmp2 + gm[i1b];
      testEqualFactor(resultFactorEI, tmp);
      tmp2 = gm[i1b];
      tmp = gm[e0b] + tmp2;
      testEqualFactor(resultFactorEI, tmp);

      tmp = (gm[s0b] + gm[e1b]);
      testEqualFactor(resultFactorSE, tmp);
      tmp2 = gm[s0b];
      tmp = tmp2 + gm[e1b];
      testEqualFactor(resultFactorSE, tmp);
      tmp2 = gm[e1b];
      tmp = gm[s0b] + tmp2;
      testEqualFactor(resultFactorSE, tmp);

      tmp = gm[s0b] + gm[s1b];
      testEqualFactor(resultFactorSS, tmp);
      tmp2 = gm[s0b];
      tmp = tmp2 + gm[s1b];
      testEqualFactor(resultFactorSS, tmp);
      tmp2 = gm[s1b];
      tmp = gm[s0b] + tmp2;
      testEqualFactor(resultFactorSS, tmp);

      tmp = gm[s0b] + gm[i1b];
      testEqualFactor(resultFactorSI, tmp);
      tmp2 = gm[s0b];
      tmp = tmp2 + gm[i1b];
      testEqualFactor(resultFactorSI, tmp);
      tmp2 = gm[i1b];
      tmp = gm[s0b] + tmp2;
      testEqualFactor(resultFactorSI, tmp);

      tmp = (gm[i0b] + gm[e1b]);
      testEqualFactor(resultFactorSE, tmp);
      tmp2 = gm[i0b];
      tmp = tmp2 + gm[e1b];
      testEqualFactor(resultFactorSE, tmp);
      tmp2 = gm[e1b];
      tmp = gm[i0b] + tmp2;
      testEqualFactor(resultFactorSE, tmp);

      tmp = gm[i0b] + gm[s1b];
      testEqualFactor(resultFactorSS, tmp);
      tmp2 = gm[i0b];
      tmp = tmp2 + gm[s1b];
      testEqualFactor(resultFactorSS, tmp);
      tmp2 = gm[s1b];
      tmp = gm[i0b] + tmp2;
      testEqualFactor(resultFactorSS, tmp);

      tmp = gm[i0b] + gm[i1b];
      testEqualFactor(resultFactorII, tmp);
      tmp2 = gm[i0b];
      tmp = tmp2 + gm[i1b];
      testEqualFactor(resultFactorII, tmp);
      tmp2 = gm[i1b];
      tmp = gm[i0b] + tmp2;
      testEqualFactor(resultFactorII, tmp);
      tmp = gm[i0b];
      tmp += gm[i1b];
      testEqualFactor(resultFactorII, tmp);
      //2d
      //---
      //3d
      //---
      //4d
      e0a = gm.addFactor(ie4a, vi, vi + 4);
      e1a = gm.addFactor(ie4a, vi, vi + 4);
      s0a = gm.addFactor(is4a, vi, vi + 4);
      s1a = gm.addFactor(is4a, vi, vi + 4);
      i0a = gm.addFactor(ii4a, vi, vi + 4);
      i1a = gm.addFactor(ii4a, vi, vi + 4);

      e0b = gm.addFactor(ie4b, vi, vi + 4);
      e1b = gm.addFactor(ie4b, vi, vi + 4);
      s0b = gm.addFactor(is4b, vi, vi + 4);
      s1b = gm.addFactor(is4b, vi, vi + 4);
      i0b = gm.addFactor(ii4b, vi, vi + 4);
      i1b = gm.addFactor(ii4b, vi, vi + 4);

      IndependentFactorType t1, t2, t4, t5, t6, t7, t8, t9;
      t1 = gm[e0a] + static_cast<ValueType> (1);
      t2 = static_cast<ValueType> (1) + gm[e0a];
      t4 = gm[s0a] + static_cast<ValueType> (1);
      t5 = gm[i0a] + static_cast<ValueType> (1);
      t6 = gm[e0a];
      t6 += static_cast<ValueType> (1);
      t7 = gm[e0a];
      t8 = t7 + static_cast<ValueType> (1);
      t9 = static_cast<ValueType> (1) + t7;

      for(size_t l = 0; l < 4; ++l)
         for(size_t k = 0; k < 4; ++k)
            for(size_t j = 0; j < 4; ++j)
               for(size_t i = 0; i < 4; ++i) {
                  size_t coordiante[] = {i, j, k, l};
                  OPENGM_TEST_EQUAL_TOLERANCE(t1(coordiante), gm[e0a](coordiante) + 1, 0.00001);
                  OPENGM_TEST_EQUAL_TOLERANCE(t2(coordiante), gm[e0a](coordiante) + 1, 0.00001);
                  OPENGM_TEST_EQUAL_TOLERANCE(t4(coordiante), gm[s0a](coordiante) + 1, 0.00001);
                  OPENGM_TEST_EQUAL_TOLERANCE(t5(coordiante), gm[i0a](coordiante) + 1, 0.00001);
                  OPENGM_TEST_EQUAL_TOLERANCE(t6(coordiante), gm[e0a](coordiante) + 1, 0.00001);
                  OPENGM_TEST_EQUAL_TOLERANCE(t8(coordiante), gm[e0a](coordiante) + 1, 0.00001);
                  OPENGM_TEST_EQUAL_TOLERANCE(t9(coordiante), gm[e0a](coordiante) + 1, 0.00001);
               }

      opengm::operateBinary(gm[e0a], gm[e1a], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[s1a], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[i1a], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0a], gm[e1a], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[s1a], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[i1a], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0a], gm[e1a], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[s1a], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[i1a], resultFactorII, std::plus<ValueType > ());
      passed = true;
      for(size_t x3 = 0; x3 < 4 && passed == true; ++x3)
         for(size_t x2 = 0; x2 < 4 && passed == true; ++x2)
            for(size_t x1 = 0; x1 < 4 && passed == true; ++x1)
               for(size_t x0 = 0; x0 < 4 && passed == true; ++x0) {
                  if(x3 == x2 && x2 == x1 && x1 == x0) {
                     if(
                        (
                         resultFactorEE(x0, x1, x2, x3) == 2 &&
                         resultFactorES(x0, x1, x2, x3) == 2 &&
                         resultFactorEI(x0, x1, x2, x3) == 2 &&
                         resultFactorSE(x0, x1, x2, x3) == 2 &&
                         resultFactorSS(x0, x1, x2, x3) == 2 &&
                         resultFactorSI(x0, x1, x2, x3) == 2 &&
                         resultFactorIE(x0, x1, x2, x3) == 2 &&
                         resultFactorIS(x0, x1, x2, x3) == 2 &&
                         resultFactorII(x0, x1, x2, x3) == 2
                         ) == false
                        ) {
                        passed = false;
                        break;
                     }
                  }
                  else {
                     if(
                        (
                         resultFactorEE(x0, x1, x2, x3) == 0 &&
                         resultFactorES(x0, x1, x2, x3) == 0 &&
                         resultFactorEI(x0, x1, x2, x3) == 0 &&
                         resultFactorSE(x0, x1, x2, x3) == 0 &&
                         resultFactorSS(x0, x1, x2, x3) == 0 &&
                         resultFactorSI(x0, x1, x2, x3) == 0 &&
                         resultFactorIE(x0, x1, x2, x3) == 0 &&
                         resultFactorIS(x0, x1, x2, x3) == 0 &&
                         resultFactorII(x0, x1, x2, x3) == 0
                         ) == false
                        ) {
                        passed = false;
                        break;
                     }
                  }
               }
      OPENGM_TEST(passed);

      opengm::operateBinary(gm[e0b], gm[e1b], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[s1b], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[i1b], resultFactorEI, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[e1b], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[s1b], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[i1b], resultFactorSI, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[e1b], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[s1b], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[i1b], resultFactorII, std::plus<ValueType > ());
      for(size_t i = 0; i < 16 && passed == true; ++i) {
         if(
            (
             resultFactorEE(i) == 2 * i &&
             resultFactorES(i) == 2 * i &&
             resultFactorEI(i) == 2 * i &&
             resultFactorSE(i) == 2 * i &&
             resultFactorSS(i) == 2 * i &&
             resultFactorSI(i) == 2 * i &&
             resultFactorIE(i) == 2 * i &&
             resultFactorIS(i) == 2 * i &&
             resultFactorII(i) == 2 * i
             ) == false
            ) {
            passed = false;
            break;
         }
      }
      OPENGM_TEST(passed);
      //dimD==dimA dimD!=dimB
      //dimD=4 dimA=A4  dimB=1
      //a factors
      e0a = gm.addFactor(ie4a, vi, vi + 4);
      s0a = gm.addFactor(is4a, vi, vi + 4);
      i0a = gm.addFactor(ii4a, vi, vi + 4);

      e0b = gm.addFactor(ie4b, vi, vi + 4);
      s0b = gm.addFactor(is4b, vi, vi + 4);
      i0b = gm.addFactor(ii4b, vi, vi + 4);
      //b factors
      size_t viB[] = {1, 2, 3};
      e1a = gm.addFactor(ie1a, viB, viB + 1);
      s1a = gm.addFactor(is1a, viB, viB + 1);
      i1a = gm.addFactor(ii1a, viB, viB + 1);

      e1b = gm.addFactor(ie1b, viB, viB + 1);
      s1b = gm.addFactor(is1b, viB, viB + 1);
      i1b = gm.addFactor(ii1b, viB, viB + 1);
      opengm::operateBinary(gm[e0a], gm[e1a], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[s1a], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[i1a], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0a], gm[e1a], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[s1a], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[i1a], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0a], gm[e1a], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[s1a], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[i1a], resultFactorII, std::plus<ValueType > ());

      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  if(x3 == x2 && x2 == x1 && x1 == x0) {
                     if((
                         resultFactorEE(x0, x1, x2, x3) == 2 &&
                         resultFactorES(x0, x1, x2, x3) == 2 &&
                         resultFactorEI(x0, x1, x2, x3) == 2 &&
                         resultFactorSE(x0, x1, x2, x3) == 2 &&
                         resultFactorSS(x0, x1, x2, x3) == 2 &&
                         resultFactorSI(x0, x1, x2, x3) == 2 &&
                         resultFactorIE(x0, x1, x2, x3) == 2 &&
                         resultFactorIS(x0, x1, x2, x3) == 2 &&
                         resultFactorII(x0, x1, x2, x3) == 2
                         ) == false) {
                        passed = false;
                     }
                  }
                  else {
                     if((
                         resultFactorEE(x0, x1, x2, x3) == 1 &&
                         resultFactorES(x0, x1, x2, x3) == 1 &&
                         resultFactorEI(x0, x1, x2, x3) == 1 &&
                         resultFactorSE(x0, x1, x2, x3) == 1 &&
                         resultFactorSS(x0, x1, x2, x3) == 1 &&
                         resultFactorSI(x0, x1, x2, x3) == 1 &&
                         resultFactorIE(x0, x1, x2, x3) == 1 &&
                         resultFactorIS(x0, x1, x2, x3) == 1 &&
                         resultFactorII(x0, x1, x2, x3) == 1
                         ) == false) {
                        passed = false;
                     }
                  }
               }
      OPENGM_TEST(passed);

      opengm::operateBinary(gm[e0b], gm[e1b], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[s1b], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[i1b], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0b], gm[e1b], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[s1b], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[i1b], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0b], gm[e1b], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[s1b], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[i1b], resultFactorII, std::plus<ValueType > ());

      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  ValueType val = x0 + 4 * x1 + 16 * x2 + 64 * x3 + x1;
                  //std::cout<<" x0:"<<x0<<"x1:"<<x1<<"x2:"<<x2<<"x3:"<<x3<<"val:"<<resultFactorSS(x0,x1,x2,x3)<<" e-val:"<<val<<std::endl;
                  if((
                      resultFactorEE(x0, x1, x2, x3) == val &&
                      resultFactorES(x0, x1, x2, x3) == val &&
                      resultFactorEI(x0, x1, x2, x3) == val &&
                      resultFactorSE(x0, x1, x2, x3) == val &&
                      resultFactorSS(x0, x1, x2, x3) == val &&
                      resultFactorSI(x0, x1, x2, x3) == val &&
                      resultFactorIE(x0, x1, x2, x3) == val &&
                      resultFactorIS(x0, x1, x2, x3) == val &&
                      resultFactorII(x0, x1, x2, x3) == val
                      ) == false) {
                     passed = false;
                  }
               }
      OPENGM_TEST(passed);

      //dimD=4 dimA=A4  dimB=2
      //a factors
      e0a = gm.addFactor(ie4a, vi, vi + 4);
      s0a = gm.addFactor(is4a, vi, vi + 4);
      i0a = gm.addFactor(ii4a, vi, vi + 4);

      e0b = gm.addFactor(ie4b, vi, vi + 4);
      s0b = gm.addFactor(is4b, vi, vi + 4);
      i0b = gm.addFactor(ii4b, vi, vi + 4);
      //b factors
      e1a = gm.addFactor(ie2a, viB, viB + 2);
      s1a = gm.addFactor(is2a, viB, viB + 2);
      i1a = gm.addFactor(ii2a, viB, viB + 2);

      e1b = gm.addFactor(ie2b, viB, viB + 2);
      s1b = gm.addFactor(is2b, viB, viB + 2);
      i1b = gm.addFactor(ii2b, viB, viB + 2);

      opengm::operateBinary(gm[e0a], gm[e1a], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[s1a], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[i1a], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0a], gm[e1a], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[s1a], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[i1a], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0a], gm[e1a], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[s1a], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[i1a], resultFactorII, std::plus<ValueType > ());

      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  if(x3 == x2 && x2 == x1 && x1 == x0) {
                     if((
                         resultFactorEE(x0, x1, x2, x3) == 2 &&
                         resultFactorES(x0, x1, x2, x3) == 2 &&
                         resultFactorEI(x0, x1, x2, x3) == 2 &&
                         resultFactorSE(x0, x1, x2, x3) == 2 &&
                         resultFactorSS(x0, x1, x2, x3) == 2 &&
                         resultFactorSI(x0, x1, x2, x3) == 2 &&
                         resultFactorIE(x0, x1, x2, x3) == 2 &&
                         resultFactorIS(x0, x1, x2, x3) == 2 &&
                         resultFactorII(x0, x1, x2, x3) == 2
                         ) == false) {
                        passed = false;
                     }
                  }
                  else if(x2 == x1 && x3 != x2 && x1 != x3) {
                     if((
                         resultFactorEE(x0, x1, x2, x3) == 1 &&
                         resultFactorES(x0, x1, x2, x3) == 1 &&
                         resultFactorEI(x0, x1, x2, x3) == 1 &&
                         resultFactorSE(x0, x1, x2, x3) == 1 &&
                         resultFactorSS(x0, x1, x2, x3) == 1 &&
                         resultFactorSI(x0, x1, x2, x3) == 1 &&
                         resultFactorIE(x0, x1, x2, x3) == 1 &&
                         resultFactorIS(x0, x1, x2, x3) == 1 &&
                         resultFactorII(x0, x1, x2, x3) == 1
                         ) == false) {
                        passed = false;
                     }
                  }
               }

      OPENGM_TEST(passed);
      opengm::operateBinary(gm[e0b], gm[e1b], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[s1b], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[i1b], resultFactorEI, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[e1b], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[s1b], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[i1b], resultFactorSI, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[e1b], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[s1b], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[i1b], resultFactorII, std::plus<ValueType > ());
      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  ValueType val = x0 + 4 * x1 + 16 * x2 + 64 * x3 + (x1 + 4 * x2);
                  if((
                      resultFactorEE(x0, x1, x2, x3) == val &&
                      resultFactorES(x0, x1, x2, x3) == val &&
                      resultFactorEI(x0, x1, x2, x3) == val &&
                      resultFactorSE(x0, x1, x2, x3) == val &&
                      resultFactorSS(x0, x1, x2, x3) == val &&
                      resultFactorSI(x0, x1, x2, x3) == val &&
                      resultFactorIE(x0, x1, x2, x3) == val &&
                      resultFactorIS(x0, x1, x2, x3) == val &&
                      resultFactorII(x0, x1, x2, x3) == val
                      ) == false) {
                     passed = false;
                  }
               }
      OPENGM_TEST(passed);

      //dimD!=dimA dimD==dimB
      //dimD=4 dimA=A4  dimB=1
      //a factors
      e1a = gm.addFactor(ie4a, vi, vi + 4);
      s1a = gm.addFactor(is4a, vi, vi + 4);
      i1a = gm.addFactor(ii4a, vi, vi + 4);

      e1b = gm.addFactor(ie4b, vi, vi + 4);
      s1b = gm.addFactor(is4b, vi, vi + 4);
      i1b = gm.addFactor(ii4b, vi, vi + 4);
      //b factors
      e0a = gm.addFactor(ie1a, viB, viB + 1);
      s0a = gm.addFactor(is1a, viB, viB + 1);
      i0a = gm.addFactor(ii1a, viB, viB + 1);

      e0b = gm.addFactor(ie1b, viB, viB + 1);
      s0b = gm.addFactor(is1b, viB, viB + 1);
      i0b = gm.addFactor(ii1b, viB, viB + 1);

      opengm::operateBinary(gm[e0a], gm[e1a], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[s1a], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[i1a], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0a], gm[e1a], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[s1a], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[i1a], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0a], gm[e1a], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[s1a], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[i1a], resultFactorII, std::plus<ValueType > ());

      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  if(x3 == x2 && x2 == x1 && x1 == x0) {
                     if((
                         resultFactorEE(x0, x1, x2, x3) == 2 &&
                         resultFactorES(x0, x1, x2, x3) == 2 &&
                         resultFactorEI(x0, x1, x2, x3) == 2 &&
                         resultFactorSE(x0, x1, x2, x3) == 2 &&
                         resultFactorSS(x0, x1, x2, x3) == 2 &&
                         resultFactorSI(x0, x1, x2, x3) == 2 &&
                         resultFactorIE(x0, x1, x2, x3) == 2 &&
                         resultFactorIS(x0, x1, x2, x3) == 2 &&
                         resultFactorII(x0, x1, x2, x3) == 2
                         ) == false) {
                        passed = false;
                     }
                  }
                  else {
                     if((
                         resultFactorEE(x0, x1, x2, x3) == 1 &&
                         resultFactorES(x0, x1, x2, x3) == 1 &&
                         resultFactorEI(x0, x1, x2, x3) == 1 &&
                         resultFactorSE(x0, x1, x2, x3) == 1 &&
                         resultFactorSS(x0, x1, x2, x3) == 1 &&
                         resultFactorSI(x0, x1, x2, x3) == 1 &&
                         resultFactorIE(x0, x1, x2, x3) == 1 &&
                         resultFactorIS(x0, x1, x2, x3) == 1 &&
                         resultFactorII(x0, x1, x2, x3) == 1
                         ) == false) {
                        passed = false;
                     }
                  }
               }

      OPENGM_TEST(passed);

      opengm::operateBinary(gm[e0b], gm[e1b], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[s1b], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[i1b], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0b], gm[e1b], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[s1b], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[i1b], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0b], gm[e1b], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[s1b], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[i1b], resultFactorII, std::plus<ValueType > ());

      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  ValueType val = x0 + 4 * x1 + 16 * x2 + 64 * x3 + x1;
                  //std::cout<<" x0:"<<x0<<"x1:"<<x1<<"x2:"<<x2<<"x3:"<<x3<<"val:"<<resultFactorSS(x0,x1,x2,x3)<<" e-val:"<<val<<std::endl;
                  if((
                      resultFactorEE(x0, x1, x2, x3) == val &&
                      resultFactorES(x0, x1, x2, x3) == val &&
                      resultFactorEI(x0, x1, x2, x3) == val &&
                      resultFactorSE(x0, x1, x2, x3) == val &&
                      resultFactorSS(x0, x1, x2, x3) == val &&
                      resultFactorSI(x0, x1, x2, x3) == val &&
                      resultFactorIE(x0, x1, x2, x3) == val &&
                      resultFactorIS(x0, x1, x2, x3) == val &&
                      resultFactorII(x0, x1, x2, x3) == val
                      ) == false) {
                     passed = false;
                  }
               }
      OPENGM_TEST(passed);

      //dimD=4 dimA=A4  dimB=2
      //a factors
      e1a = gm.addFactor(ie4a, vi, vi + 4);
      s1a = gm.addFactor(is4a, vi, vi + 4);
      i1a = gm.addFactor(ii4a, vi, vi + 4);

      e1b = gm.addFactor(ie4b, vi, vi + 4);
      s1b = gm.addFactor(is4b, vi, vi + 4);
      i1b = gm.addFactor(ii4b, vi, vi + 4);
      //b factors
      e0a = gm.addFactor(ie2a, viB, viB + 2);
      s0a = gm.addFactor(is2a, viB, viB + 2);
      i0a = gm.addFactor(ii2a, viB, viB + 2);

      e0b = gm.addFactor(ie2b, viB, viB + 2);
      s0b = gm.addFactor(is2b, viB, viB + 2);
      i0b = gm.addFactor(ii2b, viB, viB + 2);


      opengm::operateBinary(gm[e0a], gm[e1a], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[s1a], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[i1a], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0a], gm[e1a], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[s1a], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[i1a], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0a], gm[e1a], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[s1a], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[i1a], resultFactorII, std::plus<ValueType > ());

      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  if(x3 == x2 && x2 == x1 && x1 == x0) {
                     if((
                         resultFactorEE(x0, x1, x2, x3) == 2 &&
                         resultFactorES(x0, x1, x2, x3) == 2 &&
                         resultFactorEI(x0, x1, x2, x3) == 2 &&
                         resultFactorSE(x0, x1, x2, x3) == 2 &&
                         resultFactorSS(x0, x1, x2, x3) == 2 &&
                         resultFactorSI(x0, x1, x2, x3) == 2 &&
                         resultFactorIE(x0, x1, x2, x3) == 2 &&
                         resultFactorIS(x0, x1, x2, x3) == 2 &&
                         resultFactorII(x0, x1, x2, x3) == 2
                         ) == false) {
                        passed = false;
                     }
                  }
                  else if(x2 == x1 && x3 != x2 && x1 != x3) {
                     if((
                         resultFactorEE(x0, x1, x2, x3) == 1 &&
                         resultFactorES(x0, x1, x2, x3) == 1 &&
                         resultFactorEI(x0, x1, x2, x3) == 1 &&
                         resultFactorSE(x0, x1, x2, x3) == 1 &&
                         resultFactorSS(x0, x1, x2, x3) == 1 &&
                         resultFactorSI(x0, x1, x2, x3) == 1 &&
                         resultFactorIE(x0, x1, x2, x3) == 1 &&
                         resultFactorIS(x0, x1, x2, x3) == 1 &&
                         resultFactorII(x0, x1, x2, x3) == 1
                         ) == false) {
                        passed = false;
                     }
                  }
               }

      OPENGM_TEST(passed);
      opengm::operateBinary(gm[e0b], gm[e1b], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[s1b], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[i1b], resultFactorEI, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[e1b], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[s1b], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[i1b], resultFactorSI, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[e1b], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[s1b], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[i1b], resultFactorII, std::plus<ValueType > ());
      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  ValueType val = x0 + 4 * x1 + 16 * x2 + 64 * x3 + (x1 + 4 * x2);
                  if((
                      resultFactorEE(x0, x1, x2, x3) == val &&
                      resultFactorES(x0, x1, x2, x3) == val &&
                      resultFactorEI(x0, x1, x2, x3) == val &&
                      resultFactorSE(x0, x1, x2, x3) == val &&
                      resultFactorSS(x0, x1, x2, x3) == val &&
                      resultFactorSI(x0, x1, x2, x3) == val &&
                      resultFactorIE(x0, x1, x2, x3) == val &&
                      resultFactorIS(x0, x1, x2, x3) == val &&
                      resultFactorII(x0, x1, x2, x3) == val
                      ) == false) {
                     passed = false;
                  }
               }
      OPENGM_TEST(passed);

      //dimD!=dimA dimD!=dimB  dimA==DimB;
      //a factors
      e1a = gm.addFactor(ie2a, vi, vi + 2);
      s1a = gm.addFactor(is2a, vi, vi + 2);
      i1a = gm.addFactor(ii2a, vi, vi + 2);

      e1b = gm.addFactor(ie2b, vi, vi + 2);
      s1b = gm.addFactor(is2b, vi, vi + 2);
      i1b = gm.addFactor(ii2b, vi, vi + 2);
      //b factors
      e0a = gm.addFactor(ie2a, vi + 2, vi + 4);
      s0a = gm.addFactor(is2a, vi + 2, vi + 4);
      i0a = gm.addFactor(ii2a, vi + 2, vi + 4);

      e0b = gm.addFactor(ie2b, vi + 2, vi + 4);
      s0b = gm.addFactor(is2b, vi + 2, vi + 4);
      i0b = gm.addFactor(ii2b, vi + 2, vi + 4);

      opengm::operateBinary(gm[e0a], gm[e1a], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[s1a], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[i1a], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0a], gm[e1a], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[s1a], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[i1a], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0a], gm[e1a], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[s1a], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[i1a], resultFactorII, std::plus<ValueType > ());

      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  ValueType val = static_cast<ValueType> (x0 == x1) + static_cast<ValueType> (x2 == x3);
                  if((
                      resultFactorEE(x0, x1, x2, x3) == val &&
                      resultFactorES(x0, x1, x2, x3) == val &&
                      resultFactorEI(x0, x1, x2, x3) == val &&
                      resultFactorSE(x0, x1, x2, x3) == val &&
                      resultFactorSS(x0, x1, x2, x3) == val &&
                      resultFactorSI(x0, x1, x2, x3) == val &&
                      resultFactorIE(x0, x1, x2, x3) == val &&
                      resultFactorIS(x0, x1, x2, x3) == val &&
                      resultFactorII(x0, x1, x2, x3) == val
                      ) == false) {
                     passed = false;
                  }

               }
      OPENGM_TEST(passed);
      opengm::operateBinary(gm[e0b], gm[e1b], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[s1b], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[i1b], resultFactorEI, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[e1b], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[s1b], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[i1b], resultFactorSI, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[e1b], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[s1b], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[i1b], resultFactorII, std::plus<ValueType > ());
      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  ValueType val = (x0 + 4 * x1) +(x2 + 4 * x3);
                  if((
                      resultFactorEE(x0, x1, x2, x3) == val &&
                      resultFactorES(x0, x1, x2, x3) == val &&
                      resultFactorEI(x0, x1, x2, x3) == val &&
                      resultFactorSE(x0, x1, x2, x3) == val &&
                      resultFactorSS(x0, x1, x2, x3) == val &&
                      resultFactorSI(x0, x1, x2, x3) == val &&
                      resultFactorIE(x0, x1, x2, x3) == val &&
                      resultFactorIS(x0, x1, x2, x3) == val &&
                      resultFactorII(x0, x1, x2, x3) == val
                      ) == false) {
                     passed = false;
                  }
               }
      OPENGM_TEST(passed);

      //dimD!=dimA dimD!=dimB
      //a=3 b=3 d=4
      //a factors
      e0a = gm.addFactor(ie3a, vi, vi + 3);
      s0a = gm.addFactor(is3a, vi, vi + 3);
      i0a = gm.addFactor(ii3a, vi, vi + 3);

      e0b = gm.addFactor(ie3b, vi, vi + 3);
      s0b = gm.addFactor(is3b, vi, vi + 3);
      i0b = gm.addFactor(ii3b, vi, vi + 3);
      //b factors
      e1a = gm.addFactor(ie3a, viB, viB + 3);
      s1a = gm.addFactor(is3a, viB, viB + 3);
      i1a = gm.addFactor(ii3a, viB, viB + 3);

      e1b = gm.addFactor(ie3b, viB, viB + 3);
      s1b = gm.addFactor(is3b, viB, viB + 3);
      i1b = gm.addFactor(ii3b, viB, viB + 3);

      opengm::operateBinary(gm[e0a], gm[e1a], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[s1a], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0a], gm[i1a], resultFactorEI, std::plus<ValueType > ());

      opengm::operateBinary(gm[s0a], gm[e1a], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[s1a], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0a], gm[i1a], resultFactorSI, std::plus<ValueType > ());

      opengm::operateBinary(gm[i0a], gm[e1a], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[s1a], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0a], gm[i1a], resultFactorII, std::plus<ValueType > ());

      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {

                  ValueType val = static_cast<ValueType> (x0 == x1 && x1 == x2) + static_cast<ValueType> (x2 == x3 && x2 == x1);
                  if((
                      resultFactorEE(x0, x1, x2, x3) == val &&
                      resultFactorES(x0, x1, x2, x3) == val &&
                      resultFactorEI(x0, x1, x2, x3) == val &&
                      resultFactorSE(x0, x1, x2, x3) == val &&
                      resultFactorSS(x0, x1, x2, x3) == val &&
                      resultFactorSI(x0, x1, x2, x3) == val &&
                      resultFactorIE(x0, x1, x2, x3) == val &&
                      resultFactorIS(x0, x1, x2, x3) == val &&
                      resultFactorII(x0, x1, x2, x3) == val
                      ) == false) {
                     passed = false;
                  }
               }

      OPENGM_TEST(passed);
      opengm::operateBinary(gm[e0b], gm[e1b], resultFactorEE, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[s1b], resultFactorES, std::plus<ValueType > ());
      opengm::operateBinary(gm[e0b], gm[i1b], resultFactorEI, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[e1b], resultFactorSE, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[s1b], resultFactorSS, std::plus<ValueType > ());
      opengm::operateBinary(gm[s0b], gm[i1b], resultFactorSI, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[e1b], resultFactorIE, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[s1b], resultFactorIS, std::plus<ValueType > ());
      opengm::operateBinary(gm[i0b], gm[i1b], resultFactorII, std::plus<ValueType > ());
      passed = true;
      for(size_t x3 = 0; x3 < 4; ++x3)
         for(size_t x2 = 0; x2 < 4; ++x2)
            for(size_t x1 = 0; x1 < 4; ++x1)
               for(size_t x0 = 0; x0 < 4; ++x0) {
                  ValueType val = (x0 + 4 * x1 + 16 * x2) +(x1 + 4 * x2 + 16 * x3);
                  if((
                      resultFactorEE(x0, x1, x2, x3) == val &&
                      resultFactorES(x0, x1, x2, x3) == val &&
                      resultFactorEI(x0, x1, x2, x3) == val &&
                      resultFactorSE(x0, x1, x2, x3) == val &&
                      resultFactorSS(x0, x1, x2, x3) == val &&
                      resultFactorSI(x0, x1, x2, x3) == val &&
                      resultFactorIE(x0, x1, x2, x3) == val &&
                      resultFactorIS(x0, x1, x2, x3) == val &&
                      resultFactorII(x0, x1, x2, x3) == val
                      ) == false) {
                     passed = false;
                  }
               }
      OPENGM_TEST(passed);
   }
};

int
main() {
   std::cout << "Test Operate Unary and Binary...  " << std::endl;
   {
      TestOperate<float >t;
      t.run();
   }
   {
      TestOperate<double >t;
      t.run();
   }
   std::cout << "done.." << std::endl;
   return 0;
}

