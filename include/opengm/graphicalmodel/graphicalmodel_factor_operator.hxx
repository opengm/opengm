#pragma once
#ifndef OPENGM_OPERATIONWRAPPER_HXX
#define OPENGM_OPERATIONWRAPPER_HXX

#include "opengm/utilities/functors.hxx"
#include "opengm/functions/operations/operator.hxx"
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

      namespace unary {

         template<class A, class B, class OP, size_t IX, size_t DX, bool END>
         class OperationExecutor;

         template<class A, class B, class OP, size_t IX, size_t DX>
         class OperationExecutor<A, B, OP, IX, DX, false>
         {
         public:
            typedef std::vector<typename A::IndexType> ViType;

            static void op
            (
               const A& a,
               B& b,
               OP op,
               const size_t rtia
            ) {
               if(rtia==IX) {
                  typedef typename meta::GetFunction<A, IX>::FunctionType FunctionTypeA;
                  typedef typename meta::GetFunction<B, 0>::FunctionType FunctionTypeB;
                  const FunctionTypeA& fa=meta::GetFunction<A, IX>::get(a);
                  FunctionTypeB& fb=meta::GetFunction<B, 0>::get(b);
                  b.variableIndexSequence().assign(a.variableIndexSequence().begin(),a.variableIndexSequence().end());
                  typedef opengm::UnaryOperationImpl<FunctionTypeA, FunctionTypeB, OP> UnaryOperationType;
                  UnaryOperationType::op(fa, fb, op);
               }
               else {
                  typedef OperationExecutor
                  <
                     A, B, OP,
                     IX+1,
                     DX,
                     meta::Bool<IX+1==DX>::value
                  > NewExecutorType;
                  NewExecutorType::op(a, b, op, rtia);
               }
            }
         };

         template<class A, class B, class OP, size_t IX, size_t DX>
         class OperationExecutor<A, B, OP, IX, DX, true>
         {
         public:
            static void op
            (
               const A& a,
               B& b,
               OP op,
               const size_t rtia
            ) {
               throw RuntimeError("Incorrect function type id.");
            }
         };

      } // namespace unary

      namespace binary {

         template<class A, class B, class C, class OP, size_t IX, size_t IY, size_t DX, size_t DY, bool  END>
         class OperationExecutor;

         template<class A, class B, class C, class OP, size_t IX, size_t IY, size_t DX, size_t DY>
         class OperationExecutor<A, B, C, OP, IX, IY, DX, DY, false>
         {
         public:
            typedef typename  A::VisContainerType ViTypeA;
            typedef typename  B::VisContainerType ViTypeB;
            typedef typename  C::VisContainerType ViTypeC;
            static void op
            (
               const A& a,
               const B& b,
               C& c,
               OP op,
               const ViTypeA& via,
               const ViTypeB& vib,
               ViTypeC& vic,
               const size_t rtia,
               const size_t rtib
            ) {
               if(rtia==IX && rtib==IY) {
                  typedef typename meta::GetFunction<A, IX>::FunctionType FunctionTypeA;
                  typedef typename meta::GetFunction<B, IY>::FunctionType FunctionTypeB;
                  typedef typename meta::GetFunction<C, 0>::FunctionType FunctionTypeC;
                  const FunctionTypeA& fa = meta::GetFunction<A, IX>::get(a);
                  const FunctionTypeB& fb = meta::GetFunction<B, IY>::get(b);
                  FunctionTypeC& fc = meta::GetFunction<C, 0>::get(c);
                  typedef opengm::BinaryOperationImpl<FunctionTypeA, FunctionTypeB, FunctionTypeC, OP> BinaryOperationType;
                  BinaryOperationType::op(fa, fb, fc, via, vib, vic, op);
               }
               else {
                  typedef typename meta::If
                  <
                     IX==DX-1,
                     meta::SizeT<0>,
                     meta::SizeT<IX+1>
                  >::type IXNewType;
                  typedef typename meta::If
                  <
                     IX==DX-1,
                     meta::SizeT<IY+1>,
                     meta::SizeT<IY>
                  >::type IYNewType;
                  typedef OperationExecutor
                  <
                     A, B, C, OP,
                     IXNewType::value,
                     IYNewType::value,
                     DX, DY,
                     meta::Bool<    meta::And<(IX==DX-1) , (IY==DY-1)>::value >::value
                  > NewExecutorType;
                  NewExecutorType::op(a, b, c, op, via, vib, vic, rtia, rtib);
               }
            }
         };

         template<class A, class B, class C, class OP, size_t IX, size_t IY, size_t DX, size_t DY>
         class OperationExecutor<A, B, C, OP, IX, IY, DX, DY, true>
         {
         public:
            typedef typename  A::VisContainerType ViTypeA;
            typedef typename  B::VisContainerType ViTypeB;
            typedef typename  C::VisContainerType ViTypeC;
            static void op
            (
               const A& a,
               const B& b,
               C& c,
               OP op,
               const ViTypeA& via,
               const ViTypeB& vib,
               ViTypeC& vic,
               const size_t rtia,
               const size_t rtib
            ) {
               throw RuntimeError("Incorrect function type id.");
            }
         };

         template<class A, class B, class OP, size_t IY, size_t DY, bool END>
         class InplaceOperationExecutor;

         template<class A, class B, class OP, size_t IY, size_t DY>
         class InplaceOperationExecutor<A, B, OP, IY, DY, false>
         {
         public:
            typedef typename  A::VisContainerType ViTypeA;
            typedef typename  B::VisContainerType ViTypeB;
            static void op
            (
               A& a,
               const B& b,
               OP op,
               const size_t rtib
            ) {
               if(rtib==IY) {
                  typedef typename meta::GetFunction<A, 0>::FunctionType FunctionTypeA;
                  typedef typename meta::GetFunction<B, IY>::FunctionType FunctionTypeB;
                  typedef typename FunctionTypeA::IndexType IndexType;
                  FunctionTypeA& fa=meta::GetFunction<A, 0>::get(a);
                  const FunctionTypeB& fb=meta::GetFunction<B, IY>::get(b);
                  ViTypeA & via=a.variableIndexSequence();
                  const ViTypeB & vib=b.variableIndexSequence();
                  typedef opengm::BinaryOperationInplaceImpl<FunctionTypeA, FunctionTypeB, OP> BinaryOperationType;
                  BinaryOperationType::op(fa, fb, via, vib, op);
               }
               else {
                  typedef InplaceOperationExecutor
                  <
                     A, B, OP,
                     IY+1,
                     DY,
                     meta::Bool<IY+1==DY>::value
                  > NewExecutorType;
                  NewExecutorType::op(a, b, op, rtib);
               }
            }
         };

         template<class A, class B, class OP, size_t IY, size_t DY>
         class InplaceOperationExecutor<A, B, OP, IY, DY, true>
         {
         public:
            typedef std::vector<typename A::IndexType> ViType;
            static void op
            (
               A& a,
               const B& b,
               OP op,
               const size_t rtib
            ) {
               throw RuntimeError("Incorrect function type id.");
            }
         };

      } // namespace binary

   } // namespace executor

   namespace unary {

      template<class A, class B, class OP, class FlagA, class FlagB>
      class OperationWrapper;

      // A is Independent factor
      template<class A, class OP, class FlagA>
      class InplaceOperationWrapper;

      // A factor or independet facotr
      // B is independent factor
      template<class A, class B, class OP>
      class OperationWrapper<A, B, OP, IndependentFactorOrFactorFlag, IndependentFactorFlag>
      {
      public:
         static void op
         (
            const A& a,
            B& b,
            OP op
         ) {
            typedef typename meta::EvalIf
            <
               meta::IsIndependentFactor<A>::value,
               meta::Self<meta::SizeT<1> >,
               meta::Self< meta::SizeT<A::NrOfFunctionTypes> >
            >::type NFA;
            typedef executor::unary::OperationExecutor<A, B, OP, 0, NFA::value, NFA::value==0> ExecutorType;
            ExecutorType::op(a, b, op, opengm::meta::GetFunctionTypeIndex<A>::get(a));
         }
      };

      // A is scalar
      // B is scalar
      template<class A, class B, class OP>
      class OperationWrapper<A, B, OP, ScalarFlag, ScalarFlag>
      {
      public:
         static void op
         (
            const A& a,
            B& b,
            OP op
         ) {
            b = op(a);
         }
      };

      template<class A, class B, class OP>
      class OperationWrapperSelector
      {
         typedef meta::Bool <opengm::meta::IsFundamental<A>::value> IsAScalarType;
         typedef meta::Bool <opengm::meta::IsFundamental<B>::value> IsBScalarType;
         typedef meta::Bool <opengm::meta::IsIndependentFactor<A>::value> IsAIndependentFactorType;
         typedef meta::Bool <opengm::meta::IsIndependentFactor<B>::value> IsBIndependentFactorType;
         typedef meta::Bool <opengm::meta::IsFactor<A>::value> IsAFactorType;
         typedef meta::Bool <opengm::meta::IsFactor<B>::value> IsBFactorType;
         // meta switch
         typedef typename meta::TypeListGenerator
         <
            meta::SwitchCase<IsAScalarType::value, ScalarFlag>,
            meta::SwitchCase< meta::Or<IsAIndependentFactorType::value , IsAFactorType::value>::value , IndependentFactorOrFactorFlag>
         >::type CaseListA;
         typedef typename meta::Switch<CaseListA, ErrorFlag>::type FlagA;
         typedef typename meta::TypeListGenerator
         <
            meta::SwitchCase<IsBScalarType::value, ScalarFlag>,
            meta::SwitchCase<IsBIndependentFactorType::value, IndependentFactorFlag>
         >::type CaseListB;
         typedef typename meta::Switch<CaseListB, ErrorFlag>::type FlagB;

      public:
         static void op
         (
            const A& a,
            B& b,
            OP op
         ) {
            unary::OperationWrapper<A, B, OP, FlagA, FlagB>::op(a, b, op);
         }
      };

      // A is Independent factor
      template<class A, class OP>
      class InplaceOperationWrapper<A, OP, IndependentFactorFlag>
      {
      public:
         static void op
         (
            A& a,
            OP op
         ) {
            typedef typename meta::GetFunction<A, 0>::FunctionType FunctionTypeA;
            FunctionTypeA& fa = meta::GetFunction<A, 0>::get(a);
            typedef typename opengm::UnaryOperationInplaceImpl<FunctionTypeA, OP> UnaryOperationType;
            UnaryOperationType::op(fa, op);
         }
      };

      //A is scalar
      template<class A, class OP>
      class InplaceOperationWrapper<A, OP, ScalarFlag>
      {
      public:
         static void op
         (
            A& a,
            OP op
         ) {
            a = op(a);
         }
      };

      template<class A, class OP>
      class InplaceOperationWrapperSelector
      {
         typedef meta::Bool <opengm::meta::IsFundamental<A>::value> IsAScalarType;
         typedef meta::Bool <opengm::meta::IsIndependentFactor<A>::value> IsAIndependentFactorType;
         //meta switch
         typedef typename meta::TypeListGenerator
         <
            meta::SwitchCase<IsAScalarType::value, ScalarFlag>,
            meta::SwitchCase<IsAIndependentFactorType::value, IndependentFactorFlag>
         >::type CaseListA;
         typedef typename meta::Switch<CaseListA, ErrorFlag>::type FlagA;
         typedef unary::InplaceOperationWrapper<A, OP, FlagA> OperationWrapperType;

      public:
         static void op
         (
            A& a,
            OP op
         ) {
            OperationWrapperType::op(a, op);
         }
      };

   } // namespace unary

   namespace binary {

      template<class A, class B, class C, class OP, class FlagA, class FlagB, class FlagC>
      class OperationWrapper;
      template<class A, class B, class OP, class FlagA, class FlagB>
      class InplaceOperationWrapper;

      template<class A, class B, class C, class OP>
      class OperationWrapper<A, B, C, OP, IndependentFactorOrFactorFlag, ScalarFlag, IndependentFactorFlag>
      {
      public:
         static void op
         (
            const A& a,
            const B& b,
            C& c,
            OP op
         ) {
            typedef typename opengm::BinaryToUnaryFunctor<B, OP, false> BTUFunctor;
            BTUFunctor btufunctor(b, op);
            opengm::functionwrapper::unary::OperationWrapperSelector<A, C, BTUFunctor>::op(a, c, btufunctor);
         }
      };

      template<class A, class B, class C, class OP>
      class OperationWrapper<A, B, C, OP, ScalarFlag, IndependentFactorOrFactorFlag, IndependentFactorFlag>
      {
      public:
         static void op
         (
            const A& a,
            const B& b,
            C& c, OP op
         ) {
            typedef opengm::SwapArgumemtFunctor<A, OP> SwapFunctorType;
            OperationWrapper<B, A, C, SwapFunctorType, IndependentFactorOrFactorFlag, ScalarFlag, IndependentFactorFlag >::op(b, a, c, SwapFunctorType(op));
         }
      };

      template<class A, class B, class C, class OP>
      class OperationWrapper<A, B, C, OP, ScalarFlag, ScalarFlag, ScalarFlag>
      {
      public:
         static void op
         (
            const A& a,
            const B& b,
            C& c,
            OP op
         ) {
            c = op(a, b);
         }
      };

      template<class A, class B, class C, class OP>
      class OperationWrapper<A, B, C, OP, IndependentFactorOrFactorFlag, IndependentFactorOrFactorFlag, IndependentFactorFlag>
      {
      public:
         typedef typename A::VisContainerType ViTypeA;
         typedef typename B::VisContainerType ViTypeB;
         typedef typename C::VisContainerType ViTypeC;
         static void op
         (
            const A& a,
            const B& b,
            C& c,
            OP op
         ) {
            const ViTypeA& viA = a.variableIndexSequence();
            const ViTypeB& viB = b.variableIndexSequence();
            ViTypeC & viC = c.variableIndexSequence();
            typedef typename meta::EvalIf
            <
               meta::IsIndependentFactor<A>::value,
               meta::Self<meta::SizeT<1> >,
               meta::Self< meta::SizeT<A::NrOfFunctionTypes> >
            >::type NFA;
            typedef typename meta::EvalIf
            <
               meta::IsIndependentFactor<B>::value,
               meta::Self<meta::SizeT<1> >,
               meta::Self< meta::SizeT<B::NrOfFunctionTypes> >
            >::type NFB;
            typedef executor::binary::OperationExecutor<A, B, C, OP, 0, 0, NFA::value, NFB::value, meta::And<NFA::value==0 , NFB::value==0 >::value > ExecutorType;
            ExecutorType::op
            (
               a, b, c,
               op,
               viA, viB, viC,
               opengm::meta::GetFunctionTypeIndex<A>::get(a),
               opengm::meta::GetFunctionTypeIndex<B>::get(b)
            );
         }
      };

      template<class A, class B, class C, class OP>
      class OperationWrapperSelector
      {
         typedef meta::Bool <opengm::meta::IsFundamental<A>::value> IsAScalarType;
         typedef meta::Bool <opengm::meta::IsFundamental<B>::value> IsBScalarType;
         typedef meta::Bool <opengm::meta::IsFundamental<C>::value> IsCScalarType;
         typedef meta::Bool <opengm::meta::IsIndependentFactor<A>::value> IsAIndependentFactorType;
         typedef meta::Bool <opengm::meta::IsIndependentFactor<B>::value> IsBIndependentFactorType;
         typedef meta::Bool <opengm::meta::IsIndependentFactor<C>::value> IsCIndependentFactorType;
         typedef meta::Bool <opengm::meta::IsFactor<A>::value> IsAFactorType;
         typedef meta::Bool <opengm::meta::IsFactor<B>::value> IsBFactorType;
         typedef meta::Bool <opengm::meta::IsFactor<C>::value> IsCFactorType;
         typedef typename meta::TypeListGenerator
         <
            meta::SwitchCase<IsAScalarType::value, ScalarFlag>,
            meta::SwitchCase<IsAIndependentFactorType::value ||IsAFactorType::value, IndependentFactorOrFactorFlag>
         >::type CaseListA;
         typedef typename meta::Switch<CaseListA, ErrorFlag>::type FlagA;
         typedef typename meta::TypeListGenerator
         <
            meta::SwitchCase<IsBScalarType::value, ScalarFlag>,
            meta::SwitchCase<IsBIndependentFactorType::value ||IsBFactorType::value, IndependentFactorOrFactorFlag>
         >::type CaseListB;
         typedef typename meta::Switch<CaseListB, ErrorFlag>::type FlagB;
         typedef typename meta::TypeListGenerator
         <
            meta::SwitchCase<IsCScalarType::value, ScalarFlag>,
            meta::SwitchCase<IsCIndependentFactorType::value, IndependentFactorFlag>
         >::type CaseListC;
         typedef typename meta::Switch<CaseListC, ErrorFlag>::type FlagC;
         typedef binary::OperationWrapper<A, B, C, OP, FlagA, FlagB, FlagC> OperationWrapperType;

      public:
         static void op
         (
            const A& a,
            const B& b,
            C& c,
            OP op
         ) {
            OperationWrapperType::op(a, b, c, op);
         }
      };

      template<class A, class B, class OP>
      class InplaceOperationWrapper<A, B, OP, ScalarFlag, ScalarFlag>
      {
      public:
         static void op
         (
            A& a,
            const B& b,
            OP op
         ) {
            a = op(a, b);
         }
      };

      template<class A, class B, class OP>
      class InplaceOperationWrapper<A, B, OP, IndependentFactorFlag, ScalarFlag>
      {
      public:
         static void op
         (
            A& a,
            const B& b,
            OP op
         ) {
            typedef typename opengm::BinaryToUnaryFunctor<B, OP, false> BTUFunctor;
            BTUFunctor btufunctor(b, op);
            opengm::UnaryOperationInplaceImpl<A, BTUFunctor>::op(a, btufunctor);
         }
      };

      template<class A, class B, class OP>
      class InplaceOperationWrapper<A, B, OP, IndependentFactorFlag, IndependentFactorOrFactorFlag>
      {
      public:
         static void op
         (
            A& a,
            const B& b,
            OP op
         ) {
            typedef typename meta::EvalIf
            <
               meta::IsIndependentFactor<B>::value,
               meta::Self<meta::SizeT<1> >,
               meta::Self< meta::SizeT<B::NrOfFunctionTypes> >
            >::type NFB;
            typedef executor::binary::InplaceOperationExecutor<A, B, OP, 0, NFB::value, meta::Bool<NFB::value==0>::value> ExecutorType;
            ExecutorType::op(a, b, op, opengm::meta::GetFunctionTypeIndex<B>::get(b));
         }
      };

      template<class A, class B, class OP>
      class InplaceOperationWrapperSelector
      {
         typedef meta::Bool <opengm::meta::IsFundamental<A>::value> IsAScalarType;
         typedef meta::Bool <opengm::meta::IsFundamental<B>::value> IsBScalarType;
         typedef meta::Bool <opengm::meta::IsIndependentFactor<A>::value> IsAIndependentFactorType;
         typedef meta::Bool <opengm::meta::IsIndependentFactor<B>::value> IsBIndependentFactorType;
         typedef meta::Bool <opengm::meta::IsFactor<A>::value> IsAFactorType;
         typedef meta::Bool <opengm::meta::IsFactor<B>::value> IsBFactorType;
         typedef typename meta::TypeListGenerator
         <
            meta::SwitchCase<IsAScalarType::value, ScalarFlag>,
            meta::SwitchCase<IsAIndependentFactorType::value , IndependentFactorFlag>
         >::type CaseListA;
         typedef typename meta::Switch<CaseListA, ErrorFlag>::type FlagA;
         typedef typename meta::TypeListGenerator
         <
            meta::SwitchCase<IsBScalarType::value, ScalarFlag>,
            meta::SwitchCase< meta::Or<IsBIndependentFactorType::value, IsBFactorType::value>::value, IndependentFactorOrFactorFlag>
         >::type CaseListB;
         typedef typename meta::Switch<CaseListB, ErrorFlag>::type FlagB;
         typedef binary::InplaceOperationWrapper<A, B, OP, FlagA, FlagB> OperationWrapperType;

      public:
         static void op
         (
            A& a,
            const B& b,
            OP op
         ) {
           OperationWrapperType::op(a, b, op);
         }
      };

   } // namespace binary

} // namespace functionwrapper

template<class A , class B , class C, class OP>
inline void operateBinary
(
   const A& a,
   const B& b ,
   C& c,
   OP op
) {
   functionwrapper::binary::OperationWrapperSelector<A, B, C, OP>::op(a, b, c, op);
}

template<class A , class B, class OP>
inline void operateBinary
(
   A& a,
   const B& b ,
   OP op
) {
   functionwrapper::binary::InplaceOperationWrapperSelector<A, B, OP>::op(a, b, op);
}

template<class A , class B , class OP>
inline void operateUnary
(
   const A& a,
   B& b,
   OP op
) {
   functionwrapper::unary::OperationWrapperSelector<A, B, OP>::op(a, b, op);
}

template<class A, class OP>
inline void operateUnary
(
   A& a ,
   OP op
) {
   functionwrapper::unary::InplaceOperationWrapperSelector<A, OP>::op(a, op);
}

/// \endcond

} // namespace opengm

#endif // OPENGM_OPERATIONWRAPPER_HXX
