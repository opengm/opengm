#include <vector>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/functions/learnable/lpotts.hxx>

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
         opengm::functions::learnable::LPotts<T,I,L>
         >::type FunctionTypeList;
      typedef opengm::GraphicalModel<T, opengm::Minimizer, FunctionTypeList, opengm::DiscreteSpace<I, L> > GmType;
      typedef typename GmType::FunctionIdentifier Fid;


      typedef opengm::ExplicitFunction<T,I,L> EF;
      typedef opengm::functions::learnable::LPotts<T,I,L> LPF;


      // graphical model
      size_t nos[] = {2,2,2,2,2};
      GmType gmA(opengm::DiscreteSpace<I, L > (nos, nos + 3));

      // parameter
      const size_t numweights = 1;
      opengm::learning::Weights<T> weights(numweights);
      weights.setWeight(0,5.0);
      std::vector<size_t> weightIds(1, 0);
      std::vector<T> features(1, 1.0);
      LPF lPotts(weights,2,weightIds, features);


      I labels00[2]={0,0};
      I labels01[2]={0,1};
      I labels10[2]={1,0};
      I labels11[2]={1,1};

      OPENGM_ASSERT_OP(lPotts(labels01),>,4.99);
      OPENGM_ASSERT_OP(lPotts(labels01),<,5.01);
      OPENGM_ASSERT_OP(lPotts(labels10),>,4.99);
      OPENGM_ASSERT_OP(lPotts(labels10),<,5.01);


      weights.setWeight(0,3.0);

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
