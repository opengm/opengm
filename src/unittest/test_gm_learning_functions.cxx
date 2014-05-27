#include <vector>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/l_potts.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/utilities/metaprogramming.hxx>


struct TestFunctor{

   template<class FUNCTION>
   void operator()(const FUNCTION & function){
      const size_t shape0 = function.shape(0);
   }
};


struct TestViFunctor{

   template<class VI_ITER,class FUNCTION>
   void operator()(
         VI_ITER viBegin,
         VI_ITER viEnd,
         const FUNCTION & function
      ){
      const size_t shape0 = function.shape(0);
   }
};


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


   void test1() {
      typedef typename opengm::meta::TypeListGenerator
         <
         opengm::ExplicitFunction<T,I,L>,
         opengm::LPottsFunction<T,I,L>
         >::type FunctionTypeList;
      typedef opengm::GraphicalModel<T, opengm::Minimizer, FunctionTypeList, opengm::DiscreteSpace<I, L> > GmType;
      typedef typename GmType::FunctionIdentifier Fid;


      typedef opengm::ExplicitFunction<T,I,L> EF;
      typedef opengm::LPottsFunction<T,I,L> LPF;


      // graphical model
      size_t nos[] = {2,2,2,2,2};
      GmType gmA(opengm::DiscreteSpace<I, L > (nos, nos + 3));

      // parameter
      const size_t numparam = 1;
      opengm::Parameters<T,I> param(numparam);
      param.setParameter(0,5.0);
      LPF lPotts(2,2,param,0);


      I labels00[2]={0,0};
      I labels01[2]={0,1};
      I labels10[2]={1,0};
      I labels11[2]={1,1};

      OPENGM_ASSERT_OP(lPotts(labels01),>,4.99);
      OPENGM_ASSERT_OP(lPotts(labels01),<,5.01);
      OPENGM_ASSERT_OP(lPotts(labels10),>,4.99);
      OPENGM_ASSERT_OP(lPotts(labels10),<,5.01);


      param.setParameter(0,3.0);

      OPENGM_ASSERT_OP(lPotts(labels01),>,2.99);
      OPENGM_ASSERT_OP(lPotts(labels01),<,3.01);
      OPENGM_ASSERT_OP(lPotts(labels10),>,2.99);
      OPENGM_ASSERT_OP(lPotts(labels10),<,3.01);
   }

   void run() {
      this->test1();
   }
};

int main() {
   {
      std::cout << "GraphicalModel test2... " << std::flush;
      {
         GraphicalModelTest<float, size_t, size_t> t;
         t.run();
      }
      std::cout << "done." << std::endl;
   }

   return 0;
}
