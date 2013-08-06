#pragma once
#ifndef OPENGM_ACCUMULATIONWRAPPER_HXX
#define OPENGM_ACCUMULATIONWRAPPER_HXX

#include "opengm/functions/operations/accumulator.hxx"
#include "opengm/utilities/metaprogramming.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

namespace functionwrapper {

   struct FactorFlag;
   struct IndependentFactorFlag;
   struct IndependentFactorOrFactorFlag;
   struct ScalarFlag;
   struct ErrorFlag;

   namespace executor {

      template<class A, class B, class ACC, size_t IX, size_t DX, bool END>
      class AccumulateAllExecutor;

      template<class A, class B, class ACC, size_t IX, size_t DX>
      class AccumulateAllExecutor<A, B, ACC, IX, DX, false> {
      public:
         static inline void op
         (
            const A& a,
            B& b,
            const size_t rtia
         ) {
            if(rtia==IX) {
               typedef typename meta::GetFunction<A, IX>::FunctionType FunctionTypeA;
               const FunctionTypeA& fa = meta::GetFunction<A, IX>::get(a);
               typedef opengm::AccumulateAllImpl<FunctionTypeA, B, ACC> AccumulationType;
               AccumulationType::op(fa, b);
            }
            else {
               typedef AccumulateAllExecutor<A, B, ACC, IX+1, DX, meta::Bool<IX+1==DX>::value > NewExecutorType;
               NewExecutorType::op(a, b, rtia);
            }
         };
         
         static inline void op
         (
            const A & a,
            B & b,
            std::vector<typename A::LabelType> & state,
            const size_t rtia
         ) {
            if(rtia==IX) {
               typedef typename meta::GetFunction<A, IX>::FunctionType FunctionTypeA;
               const FunctionTypeA& fa = meta::GetFunction<A, IX>::get(a);
               typedef opengm::AccumulateAllImpl<FunctionTypeA, B, ACC> AccumulationType;
               AccumulationType::op(fa, b,state);
            }
            else {
               typedef AccumulateAllExecutor<A, B, ACC, IX+1, DX, meta::Bool<IX+1==DX>::value> NewExecutorType;
               NewExecutorType::op(a, b, state, rtia);
            }
         }
      };

      template<class A, class B, class ACC, size_t IX, size_t DX>
      class AccumulateAllExecutor<A, B, ACC, IX, DX, true> {
      public:
         typedef std::vector<size_t> ViType;

         static inline void op
         (
            const A & a,
            B & b,
            const size_t rtia
         ) {
            throw RuntimeError("wrong function id");
         };
         template<class INDEX_TYPE>
         static inline void op
         (
            const A & a,
            B & b,
            std::vector<INDEX_TYPE> & state,
            const size_t rtia
         ) {
            throw RuntimeError("wrong function id");
         }
      };

      template<class A, class B, class ACC, size_t IX, size_t DX, bool END>
      class AccumulateSomeExecutor;

      template<class A, class B, class ACC, size_t IX, size_t DX>
      class AccumulateSomeExecutor<A, B, ACC, IX, DX, false>
      {
      public:
         typedef typename A::VisContainerType ViTypeA;
         typedef typename B::VisContainerType ViTypeB;

         template<class ViAccIter>
         static void op
         (
            const A & a,
            ViAccIter beginViAcc,
            ViAccIter endViAcc,
            B & b,
            const size_t rtia
         ) {
            if(rtia==IX) {
               typedef typename meta::GetFunction<A, IX>::FunctionType FunctionTypeA;
               typedef typename meta::GetFunction<B, 0>::FunctionType FunctionTypeB;
               const FunctionTypeA & fa=meta::GetFunction<A, IX>::get(a);
               FunctionTypeB & fb=meta::GetFunction<B, 0>::get(b);
               const ViTypeA & viA=a.variableIndexSequence();
               ViTypeB  & viB=b.variableIndexSequence();
               typedef opengm::AccumulateSomeImpl<FunctionTypeA, FunctionTypeB, ACC> AccumulationType;
               AccumulationType::op(fa, viA, beginViAcc, endViAcc, fb, viB);
            }
            else{
               typedef AccumulateSomeExecutor<A, B, ACC, IX+1, DX, meta::Bool<IX+1==DX>::value > NewExecutorType;
               NewExecutorType::op(a, beginViAcc, endViAcc, b, rtia);
            }
         }
      };

      template<class A, class B, class ACC, size_t IX, size_t DX>
      class AccumulateSomeExecutor<A, B, ACC, IX, DX, true> {
      public:
         //typedef std::vector<size_t> ViType;
         typedef typename A::VisContainerType ViTypeA;
         typedef typename B::VisContainerType ViTypeB;
         template<class ViAccIter>
         static void op
         (
            const A & a,
            ViAccIter beginViAcc ,
            ViAccIter endViAcc,
            B & b,
            const size_t rtia
         ) {
            throw RuntimeError("wrong function id");
         }
      };

   } // namespace executor

   template<class A, class B, class ACC>
   class AccumulateAllWrapper {
   public:
      static void op
      (
         const A & a,
         B & b
      ) {
         typedef typename meta::EvalIf
         <
            meta::IsIndependentFactor<A>::value,
            meta::Self<meta::SizeT<1> >,
            meta::Self< meta::SizeT<A::NrOfFunctionTypes> >
         >::type NFA;
         typedef executor::AccumulateAllExecutor<A, B, ACC, 0, NFA::value, NFA::value==0> ExecutorType;
         ExecutorType::op(a, b, opengm::meta::GetFunctionTypeIndex<A>::get(a));
      }
      
      template<class LABEL_TYPE>
      static void op
      (
         const A & a,
         B & b,
         std::vector<LABEL_TYPE> & state
      ) {
         typedef typename meta::EvalIf
         <
            meta::IsIndependentFactor<A>::value,
            meta::Self<meta::SizeT<1> >,
            meta::Self< meta::SizeT<A::NrOfFunctionTypes> >
         >::type NFA;
         typedef executor::AccumulateAllExecutor<A, B, ACC, 0, NFA::value, NFA::value==0> ExecutorType;
         ExecutorType::op
         (a, b, state, opengm::meta::GetFunctionTypeIndex<A>::get(a));
      }
   };

   template<class A, class B, class ACC>
   class AccumulateSomeWrapper {
      typedef typename A::VisContainerType ViTypeA;
      typedef typename B::VisContainerType ViTypeB;

   public:
      template<class ViAccIter>
      static void op(
         const A& a,
         ViAccIter beginViAcc,
         ViAccIter endViAcc,
         B& b
      ) {
         //const ViType & viA=a.variableIndexSequence();
         //ViType & viB = b.variableIndexSequence(); // initialize references
         typedef typename meta::EvalIf
         <
            meta::IsIndependentFactor<A>::value,
            meta::Self<meta::SizeT<1> >,
            meta::Self< meta::SizeT<A::NrOfFunctionTypes> >
         >::type NFA;
         typedef executor::AccumulateSomeExecutor<A, B, ACC, 0, NFA::value, NFA::value==0> ExecutorType;
         ExecutorType::op (a, beginViAcc, endViAcc, b, opengm::meta::GetFunctionTypeIndex<A>::get(a));
      }
   };

} // namespace functionwrapper

template<class ACC, class A, class B>
inline void accumulate
(
   const A & a,
   B & b
) {
   functionwrapper::AccumulateAllWrapper<A, B, ACC>::op(a, b);
}

template<class ACC, class A, class B,class INDEX_TYPE>
inline void accumulate
(
   const A & a,
   B & b,
   std::vector<INDEX_TYPE> & state
) {
   functionwrapper::AccumulateAllWrapper<A, B, ACC>::op(a, b, state);
}

template<class ACC, class A, class ViAccIterator, class B>
inline void accumulate
(
   const A & a,
   ViAccIterator viAccBegin,
   ViAccIterator viAccEnd,
   B & b
) {
   functionwrapper::AccumulateSomeWrapper<A, B, ACC>::op(a, viAccBegin, viAccEnd, b);
}

template<class ACC, class A, class ViAccIterator>
inline void accumulate
(
   A & a,
   ViAccIterator viAccBegin,
   ViAccIterator viAccEnd
) {
   opengm::AccumulateSomeInplaceImpl<typename A::FunctionType, ACC>::op(opengm::meta::GetFunction<A,0>::get(a), a.variableIndexSequence(), viAccBegin, viAccEnd);
}

/// \endcond

} // end namespace opengm

#endif // #ifndef OPENGM_ACCUMULATIONWRAPPER_HXX
