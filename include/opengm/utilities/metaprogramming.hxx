#pragma once
#ifndef OPENGM_METAPROGRAMMING
#define OPENGM_METAPROGRAMMING

#include <limits>
#include <vector>

#include "opengm/datastructures/marray/marray.hxx"

/// \cond HIDDEN_SYMBOLS
#define OPENGM_TYPELIST_1(T1) \
::opengm::meta::TypeList<T1, opengm::meta::ListEnd >

#define OPENGM_TYPELIST_2(T1, T2) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_1(T2) >

#define OPENGM_TYPELIST_3(T1, T2, T3) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_2(T2, T3) >

#define OPENGM_TYPELIST_4(T1, T2, T3, T4) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_3(T2, T3, T4) >

#define OPENGM_TYPELIST_5(T1, T2, T3, T4, T5) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_4(T2, T3, T4, T5) >

#define OPENGM_TYPELIST_6(T1, T2, T3, T4, T5, T6) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_5(T2, T3, T4, T5, T6) >

#define OPENGM_TYPELIST_7(T1, T2, T3, T4, T5, T6, T7) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_6(T2, T3, T4, T5, T6, T7) >

#define OPENGM_TYPELIST_8(T1, T2, T3, T4, T5, T6, T7, T8) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_7(T2, T3, T4, T5, T6, T7, T8) >

#define OPENGM_TYPELIST_9(T1, T2, T3, T4, T5, T6, T7, T8, T9) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_8(T2, T3, T4, T5, T6, T7, T8, T9) >

#define OPENGM_TYPELIST_10(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
::opengm::meta::TypeList<T1, OPENGM_TYPELIST_9(T2, T3, T4, T5, T6, T7, T8, T9, T10) >

/// metaprogramming typelist generator  metafunction

namespace opengm {

   template<class T>
   class Factor;
   template<class T,class I,class L>
   class IndependentFactor;
   
   /// namespace for meta-programming
   namespace meta {
      /// rebind a templated class with one template argument
      template< template < typename > class TO_BIND >
      struct Bind1{
         template<class BIND_ARG_0>
         struct Bind{
            typedef TO_BIND<BIND_ARG_0> type;
         };
      };
      /// rebind a templated class with two template arguments
      template< template < typename ,typename > class TO_BIND >
      struct Bind2{
         template<class BIND_ARG_0, class BIND_ARG_1>
         struct Bind{
            typedef TO_BIND<BIND_ARG_0, BIND_ARG_1> type;
         };
      };
      /// rebind a templated class with three template arguments
      template< template < typename ,typename, typename > class TO_BIND >
      struct Bind3{
         template<class BIND_ARG_0, class BIND_ARG_1, class BIND_ARG_2>
         struct Bind{
            typedef TO_BIND<BIND_ARG_0, BIND_ARG_1, BIND_ARG_2> type;
         };
      };

		/// metaprogramming apply a metafunction
		///
		/// the metaprogramming function T is applied by
		/// ApplyMetaFunction<T>::type returning the type
		/// of the metafunction
		template<class T>
		struct ApplyMetaFunction {
			typedef typename T::type type;
		};
		/// metaprogramming identity metafunction
		///
		/// Self<T>::type is equal to T
		template<class T>
		struct Self {
			typedef T type;
		};
		/// metaprogramming Empty type
		struct EmptyType {
		};
		/// metaprogramming Null type
		struct NullType {
		};
		/// end of a typelist type
		struct ListEnd {
		};
		/// metaprogramming "true" struct
		struct True {

			enum Value {
				value = 1
			};
		};
		/// metaprogramming "false" struct
		struct False {

			enum Value {
				value = 0
			};
		};

		//// metaprogramming truecase metafunction
		struct TrueCase {

			enum Value {
				value = 1
			};
			typedef meta::True type;
		};
		//// metaprogramming falsecase metafunction
		struct FalseCase {

			enum Values {
				value = 0
			};
			typedef meta::False type;
		};
		/// metaprogramming integer
		template<int N>
		struct Int {

			enum Value {
				value = N
			};
		};
      /// metaprogramming size_t
      template<size_t N>
		struct SizeT {

			enum Value {
				value = N
			};
		};
		//// metaprogramming bool  metafunction
		template < bool T_BOOL> struct Bool;
      /// metaprogramming bool true  metafunction
		template < > struct Bool < true > : meta::TrueCase {
		};
      /// metaprogramming bool false  metafunction
		template < > struct Bool < false > : meta::FalseCase {
		};
		//// metaprogramming or  metafunction
		template<bool T_BOOL_A, bool T_BOOL_B>
		struct Or;
      /// metaprogramming or true  metafunction
		template < > struct Or < true, true > : meta::TrueCase {
		};
      /// metaprogramming or true  metafunction
		template < > struct Or < true, false > : meta::TrueCase {
		};
      /// metaprogramming or false  metafunction
		template < > struct Or < false, true > : meta::TrueCase {
		};
      /// metaprogramming or false  metafunction
		template < > struct Or < false, false > : meta::FalseCase {
		};
      //// metaprogramming not  metafunction
      template<bool T_BOOL>
		struct Not;
      /// metaprogramming not false metafunction
      template<>
		struct Not<true> : meta::FalseCase {
      };
      /// metaprogramming not true metafunction
      template<>
		struct Not<false> : meta::TrueCase {
      };
		/// metaprogramming and metafunction
		template<bool T_BOOL_A, bool T_BOOL_B>
		struct And;
      /// metaprogramming and true metafunction
		template < > struct And < true, true > : meta::TrueCase {
		};
      /// metaprogramming and true metafunction
		template < > struct And < true, false > : meta::FalseCase {
		};
      /// metaprogramming not false metafunction
		template < > struct And < false, true > : meta::FalseCase {
		};
      /// metaprogramming not false metafunction
		template < > struct And < false, false > : meta::FalseCase {
		};
		/// metaprogramming if metafunction
		template<bool T_Bool, class T_True, class T_False>
		struct If;
      /// metaprogramming if true metafunction
		template<class T_True, class T_False>
		struct If < true, T_True, T_False> {
			typedef T_True type;
		};
      /// metaprogramming if false metafunction
		template<class T_True, class T_False>
		struct If < false, T_True, T_False> {
			typedef T_False type;
		};
		/// metaprogramming evalif  metafunction
		template<bool T_Bool, class MetaFunctionTrue, class MetaFunctionFalse>
		struct EvalIf : public meta::If<T_Bool, MetaFunctionTrue, MetaFunctionFalse>::type {
		};
      /// metaprogramming decrement metafunction
      template<size_t  I>
      struct Decrement{
         typedef SizeT< I-1 > type;
         enum Values{
            value=I-1
         };
      };
      /// metaprogramming increment metafunction
      template<size_t  I>
      struct Increment{
         typedef SizeT< I+1 > type;
         enum Values{
            value=I+1
         };
      };
      /// metaprogramming metafunction generator macro
      #define OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO(OPERATOR_SYMBOL,CLASS_NAME,RETURN_CLASS_TYPE) \
      template<size_t A,size_t B> \
		struct CLASS_NAME{ \
         typedef typename Bool< (A OPERATOR_SYMBOL B) >::type type; \
         enum Values{ \
            value=RETURN_CLASS_TYPE < (A OPERATOR_SYMBOL B) >::value \
         }; \
      }

      /// \class metaprogramming plus metafunction
      OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO( +  , Plus ,                 meta::SizeT );
      /// \class metaprogramming minus metafunction
      OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO( -  , Minus ,                meta::SizeT );
      /// metaprogramming multiplies metafunction
      OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO( *  , Multiplies ,           meta::SizeT );
      /// metaprogramming equal number metafunction
      OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO( == , EqualNumber ,          meta::Bool );
      /// metaprogramming bigger number metafunction
      OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO( >  , BiggerNumber ,         meta::Bool );
      /// metaprogramming bigger or equal number metafunction
      OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO( >= , BiggerOrEqualNumber ,  meta::Bool );
      /// metaprogramming smaller number metafunction
      OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO( <  , SmallerNumber ,        meta::Bool );
      /// metaprogramming smaller or equal number metafunction
      OPENGM_METAPROGRAMMING_BINARY_OPERATOR_GENERATOR_MACRO( <= , SmallerOrEqualNumber , meta::Bool );
      /// metaprogramming minimum number metafunction
      template< size_t A,size_t B>
      struct MinimumNumber{
         enum Value{
            value= meta::If<
               SmallerNumber<A,B>::value,
               SizeT<A>,
               SizeT<B>
            >::type::value
         };
      };
      
		/// metaprogramming compare to types false metafunction
      template<class T, class U>
		struct Compare : FalseCase {
		};
      /// metaprogramming compare to types true metafunction
		template<class T>
		struct Compare<T, T> : TrueCase {
		};
      /// metaprogramming invalid type 
      template<class T>
      struct InvalidType {
         typedef T type;
      };
      /// metaprogramming is invalid type false metafunction
      template<class T>
      struct IsInvalidType: opengm::meta::FalseCase {
      };
      /// metaprogramming is invalid type true metafunction
      template<class T>
      struct IsInvalidType< InvalidType< T > > : opengm::meta::TrueCase {
      };

      /// metaprogramming is factor false metafunction
      template<class T>
      struct IsFactor : meta::FalseCase {
      };
      /// metaprogramming is factor true metafunction
      template<class T>
      struct IsFactor<opengm::Factor<T> > :  opengm::meta::TrueCase {
      };

      /// metaprogramming is independent factor false metafunction
      template<class T>
      struct IsIndependentFactor :  opengm::meta::FalseCase {
      };
       /// metaprogramming is independent factor true metafunction
      template<class T,class I,class L>
      struct IsIndependentFactor<opengm::IndependentFactor<T,I,L> > : opengm::meta::TrueCase {
      };
      /// metaprogramming is void  false metafunction
		template<class T>struct IsVoid : meta::FalseCase {
		};
		/// metaprogramming is void  true metafunction
		template< > struct IsVoid<void> : meta::TrueCase {
		};
		/// metaprogramming is reference  false metafunction
		template<class T> struct IsReference : meta::FalseCase {
		};
      /// metaprogramming is reference false metafunction
		template<class T> struct IsReference<const T &> : meta::FalseCase {
		};
      /// metaprogramming is reference true metafunction
		template<class T> struct IsReference<T&> : meta::TrueCase {
		};
		/// metaprogramming is const reference false metafunction
		template<class T> struct IsConstReference : meta::FalseCase {
		};
      /// metaprogramming is const reference false metafunction
		template<class T> struct IsConstReference< T &> : meta::FalseCase {
		};
      /// metaprogramming is const reference true metafunction
		template<class T> struct IsConstReference<const T&> : meta::TrueCase {
		};
		/// metaprogramming remove reference metafunction
		template <typename T>
		struct RemoveReference {
			typedef T type;
		};
      /// metaprogramming remove reference metafunction
		template <typename T>
		struct RemoveReference<T&> {
			typedef T type;
		};
		/// metaprogramming remove const reference metafunction
		template <typename T>
		struct RemoveConst {
			typedef T type;
		};
      /// metaprogramming remove const reference metafunction
		template <typename T>
		struct RemoveConst<const T> {
			typedef T type;
		};
		/// metaprogramming add  reference metafunction
		template<class T> struct AddReference {
			typedef typename meta::If <
				meta::Or <
				meta::IsReference<T>::value,
				meta::IsConstReference<T>::value
				>::value,
				T,
				T &
				>::type type;
		};
		/// metaprogramming add const reference metafunction
		template<class T> struct AddConstReference {
			typedef typename meta::If
				<
				meta::IsConstReference<T>::value,
				T,
				typename meta::If <
				meta::IsReference<T>::value,
				typename meta::RemoveReference<T>::type const &,
				T const &
				>::type
				>::type type;
		};
      /// metaprogramming length of typelist metafunction
		template<class T_List>
		struct LengthOfTypeList {
			typedef meta::Int < 1 + LengthOfTypeList<typename T_List::TailType>::type::value> type;
			enum {
				value = type::value
			};
		};
      /// metaprogramming length of typelist metafunction
		template< >
		struct LengthOfTypeList<meta::ListEnd> {
			typedef meta::Int < 0 > type;

			enum {
				value = 0
			};
		};
      /// metaprogramming type at typelist metafunction
		template<class T_List, unsigned int Index>
		struct TypeAtTypeList {
			typedef typename TypeAtTypeList<typename T_List::TailType, Index - 1 > ::type type;
		};
      /// metaprogramming type at typelist metafunction
		template<class T_List>
		struct TypeAtTypeList<T_List, 0 > {
			typedef typename T_List::HeadType type;
		};
      /// metaprogramming type at typelist save metafunction
		template<class T_List, unsigned int Index, class T_DefaultType>
		struct TypeAtTypeListSave
		: meta::EvalIf<
         meta::LengthOfTypeList<T_List>::value >= Index ? true : false,
         meta::TypeAtTypeList<T_List, Index>,
         meta::Self<T_DefaultType>
		>::type {
		};
      
      /// metaprogramming typelist
		template<class T_Head, class T_Tail>
		struct TypeList {
			typedef meta::ListEnd ListEnd;
			typedef T_Head HeadType;
			typedef T_Tail TailType;
		};
      /// metaprogramming typelist generator  metafunction
		template
		<
		class T1,
		class T2 = opengm::meta::ListEnd,
		class T3 = opengm::meta::ListEnd,
		class T4 = opengm::meta::ListEnd,
		class T5 = opengm::meta::ListEnd,
		class T6 = opengm::meta::ListEnd,
		class T7 = opengm::meta::ListEnd,
		class T8 = opengm::meta::ListEnd,
		class T9 = opengm::meta::ListEnd,
		class T10 = opengm::meta::ListEnd,
		class T11 = opengm::meta::ListEnd,
		class T12 = opengm::meta::ListEnd,
		class T13 = opengm::meta::ListEnd,
		class T14 = opengm::meta::ListEnd,
		class T15 = opengm::meta::ListEnd
		>
		struct TypeListGenerator {
			typedef opengm::meta::TypeList<T1, typename TypeListGenerator<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>::type > type;
		};
      /// metaprogramming typelist generator  metafunction
		template< >
		struct TypeListGenerator<opengm::meta::ListEnd> {
			typedef opengm::meta::ListEnd type;
		};
      
      /// metaprogramming switch case  metafunction
		template<bool B_IsTrue, class T_TrueType>
		struct SwitchCase {
			typedef T_TrueType type;

			struct Case : opengm::meta::Bool<B_IsTrue> {
			};
		};

      /// metaprogramming switch  metafunction
		template <class T_List, class T_DefaultCase = opengm::meta::EmptyType >
		struct Switch {
			typedef typename opengm::meta::EvalIf
				<
				opengm::meta::TypeAtTypeList<T_List, 0 > ::type::Case::value,
				typename opengm::meta::TypeAtTypeList<T_List, 0 >::type,
				Switch<typename T_List::TailType, T_DefaultCase>
				>::type type;
		};
      /// metaprogramming switch metafunction
		template<class T_DefaultCase>
		struct Switch<opengm::meta::ListEnd, T_DefaultCase> {
			typedef T_DefaultCase type;
		};
      /// metaprogramming is ptr metafunction
		template<class T> struct IsPtr : opengm::meta::FalseCase {
		};
      /// metaprogramming is ptr metafunction
		template<class T> struct IsPtr<T * const> : opengm::meta::FalseCase {
		};
      /// metaprogramming is ptr metafunction
		template<class T> struct IsPtr<T * > : opengm::meta::TrueCase {
		};
      /// metaprogramming is const ptr metafunction
		template<class T> struct IsConstPtr : opengm::meta::FalseCase {
		};
      /// metaprogramming is const ptr metafunction
		template<class T> struct IsConstPtr<T * > : opengm::meta::FalseCase {
		};
      /// metaprogramming is const ptr  etafunction
		template<class T> struct IsConstPtr<T * const > : opengm::meta::TrueCase {
		};
      /// metaprogramming is const metafunction
		template<class T> struct IsConst : opengm::meta::FalseCase {
		};
      /// metaprogramming is const metafunction
		template<class T> struct IsConst< const T> : opengm::meta::TrueCase {
		};
      /// metaprogramming is fundamental metafunction
		template<class T>
		struct IsFundamental {
			typedef typename opengm::meta::Or <
				std::numeric_limits< typename RemoveConst<T>::type >::is_specialized,
				opengm::meta::IsVoid< typename RemoveConst<T>::type >::value
				>::type type;

			enum Value{
				value = type::value
			};
		};
      /// metaprogramming is floating point metafunction
		template <class T>
		struct IsFloatingPoint :
		opengm::meta::Bool<
		opengm::meta::Compare<T, float >::value ||
		opengm::meta::Compare<T, const float >::value ||
		opengm::meta::Compare<T, double >::value ||
		opengm::meta::Compare<T, const double >::value ||
		opengm::meta::Compare<T, long double >::value ||
		opengm::meta::Compare<T, const long double >::value
		> {
		};
      /// metaprogramming Type Info
		template<class T>
		struct TypeInfo {

			struct IsFundamental : opengm::meta::IsFundamental<T> {
			};

			struct IsFloatingPoint : opengm::meta::IsFloatingPoint<T> {
			};

			struct IsConst : opengm::meta::IsConst<T> {
			};

			struct IsConstReference : opengm::meta::IsConstReference<T> {
			};

			struct IsReference : opengm::meta::IsReference<T> {
			};

			struct IsPtr : opengm::meta::IsPtr<T> {
			};

			struct IsConstPtr : opengm::meta::IsConstPtr<T> {
			};
		};
      /// metaprogramming is typelist metafunction
      template<class T>
      struct IsTypeList : meta::FalseCase{};
      /// metaprogramming is typelist metafunction
      template<class TH,class TT>
      struct IsTypeList< meta::TypeList<TH,TT> > : meta::TrueCase{};
      /// metaprogramming typelist from maby typelist metafunction
      template<class T>
      struct TypeListFromMaybeTypeList{
         typedef meta::TypeList<T,meta::ListEnd> type;
      };
      /// metaprogramming typelist from maby typelist metafunction
      template<class TH,class TT>
      struct TypeListFromMaybeTypeList< meta::TypeList<TH,TT> > {
         typedef meta::TypeList<TH,TT> type;
      };
      /// metaprogramming front insert in typelist metafunction
      template<class TL,class FrontType>
      struct FrontInsert{
         typedef meta::TypeList<FrontType,TL> type;
      };
      /// metaprogramming back insert in typelist metafunction
      template<class TL,class TYPE>
      struct BackInsert;
      /// metaprogramming back insert in typelist metafunction
      template<class THEAD,class TTAIL,class TYPE>
      struct BackInsert<TypeList<THEAD,TTAIL> ,TYPE>{
         typedef TypeList<
            THEAD, 
            typename meta::BackInsert<
               TTAIL ,
               TYPE
            >::type 
         > type;
      };
      /// metaprogramming back insert in typelist metafunction
      template<class TYPE>
      struct BackInsert<ListEnd,TYPE>{
         typedef meta::TypeList<TYPE,ListEnd> type;
      };
      /// metaprogramming get index in typelist metafunction
      template<class TL,class TypeToFindx>
      struct GetIndexInTypeList;
      /// metaprogramming get index in typelist metafunction
      template<class THEAD,class TTAIL,class TypeToFind>
      struct GetIndexInTypeList<meta::TypeList<THEAD,TTAIL>,TypeToFind>{
         enum Value{
            value=GetIndexInTypeList<TTAIL,TypeToFind >::value+1
         };
         typedef meta::SizeT<GetIndexInTypeList<TTAIL,TypeToFind >::type::value +1> type;
      };
      /// metaprogramming get index in typelist metafunction
      template<class THEAD,class TTAIL>
      struct GetIndexInTypeList<meta::TypeList<THEAD,TTAIL>,THEAD >{
         enum Value{
            value=0
         };
         typedef meta::SizeT<0> type;
      };
      /// metaprogramming get index in typelist savely metafunction
      template<class TL,class TypeToFindx,size_t NOT_FOUND_INDEX>
      struct GetIndexInTypeListSafely;
      /// metaprogramming get index in typelist savely metafunction
      template<class THEAD,class TTAIL,class TypeToFind,size_t NOT_FOUND_INDEX>
      struct GetIndexInTypeListSafely<meta::TypeList<THEAD,TTAIL>,TypeToFind,NOT_FOUND_INDEX>{
         enum Value{
            value=GetIndexInTypeListSafely<TTAIL,TypeToFind,NOT_FOUND_INDEX >::value+1
         };
         typedef meta::SizeT<GetIndexInTypeListSafely<TTAIL,TypeToFind,NOT_FOUND_INDEX >::type::value +1> type;
      };
      /// metaprogramming get index in typelist savely metafunction
      template<class THEAD,class TTAIL,size_t NOT_FOUND_INDEX>
      struct GetIndexInTypeListSafely<meta::TypeList<THEAD,TTAIL>,THEAD,NOT_FOUND_INDEX >{
         enum Value{
            value=0
         };
         typedef meta::SizeT<0> type;
      };
      /// metaprogramming get index in typelist savely metafunction
      template<class TYPE_TO_FIND,size_t NOT_FOUND_INDEX>
      struct GetIndexInTypeListSafely<meta::ListEnd,TYPE_TO_FIND,NOT_FOUND_INDEX >{
         enum Value{
            value=NOT_FOUND_INDEX
         };
         typedef meta::SizeT<NOT_FOUND_INDEX> type;
      };
      /// metaprogramming delete type in typelist metafunction
      template <class TL,class T>
      struct DeleteTypeInTypeList;
      /// metaprogramming delete type in typelist metafunction
      template <class T>
      struct DeleteTypeInTypeList<ListEnd,T> {
         typedef ListEnd type;
      };
      /// metaprogramming delete type in typelist metafunction
      template <class T,class TTail>
      struct DeleteTypeInTypeList<TypeList<T,TTail>,T> {
         typedef TTail type;
      };
      /// metaprogramming delete type in typelist metafunction
      template <class THead,class TTail, class T>
      struct DeleteTypeInTypeList<TypeList<THead,TTail>,T> {
         typedef TypeList<THead, typename DeleteTypeInTypeList<TTail,T>::type> type;
      };
      /// metaprogramming has type  in typelist metafunction     
      template<class TL,class TypeToFindx>
      struct HasTypeInTypeList;
      /// metaprogramming has type  in typelist metafunction     
      template<class THEAD,class TTAIL,class TypeToFind>
      struct HasTypeInTypeList<meta::TypeList<THEAD,TTAIL>,TypeToFind>
      {
         enum Value{
            value=HasTypeInTypeList<TTAIL,TypeToFind >::value
         };
         typedef HasTypeInTypeList< TTAIL,TypeToFind>  type;
      };
      
      /// metaprogramming find type with a certain size in typelist metafunction     
      template<class TL,class TSL,size_t SIZE,class NOT_FOUND>
      struct FindSizedType;
      /// metaprogramming find type with a certain size in typelist metafunction     
      template<class TLH,class TLT,class TSLH,class TSLT,size_t SIZE,class NOT_FOUND>
      struct FindSizedType<meta::TypeList<TLH,TLT>,meta::TypeList<TSLH,TSLT>,SIZE,NOT_FOUND>{
         typedef typename FindSizedType<TLT,TSLT,SIZE,NOT_FOUND >::type type;
      };
      /// metaprogramming find type with a certain size in typelist metafunction     
      template<class TLH ,class TLT,class TSLT,size_t SIZE,class NOT_FOUND>
      struct FindSizedType< meta::TypeList<TLH,TLT>,meta::TypeList< meta::SizeT<SIZE> ,TSLT>,SIZE,NOT_FOUND  >{
         typedef TLH type;
      };
      /// metaprogramming find type with a certain size in typelist metafunction     
      template<size_t SIZE,class NOT_FOUND>
      struct FindSizedType< meta::ListEnd,meta::ListEnd,SIZE,NOT_FOUND  >{
         typedef NOT_FOUND type;
      };
      /// metaprogramming merge  typelists metafunction     
      template<class TL,class OTHER_TL>
      struct MergeTypeLists;
      /// metaprogramming merge  typelists metafunction     
      template<class THEAD,class TTAIL,class OTHER_TL>
      struct MergeTypeLists<meta::TypeList<THEAD,TTAIL>,OTHER_TL>
		{
         typedef meta::TypeList< 
				THEAD,  
				typename MergeTypeLists<TTAIL,OTHER_TL>::type  
			>  type;
      };
      /// metaprogramming merge  typelists metafunction     
      template<class OTHER_TL>
      struct MergeTypeLists<meta::ListEnd,OTHER_TL>
		{
         typedef OTHER_TL type;
      };
      /// metaprogramming has type in typelist metafunction     
      template<class THEAD,class TTAIL>
      struct HasTypeInTypeList<meta::TypeList<THEAD,TTAIL>,THEAD > : meta::TrueCase{
      };
      /// metaprogramming has type in typelist metafunction    
      template<class TypeToFindx>
      struct HasTypeInTypeList<meta::ListEnd,TypeToFindx> : meta::FalseCase{
      };
      /// metaprogramming inserts a type in typelist or move to end metafunction   
      ///
      /// back inserts a type in a typelist. If the type has been in the typelist
      /// the type is moved to the end of the typelist
      template<class TL,class TYPE>
      struct InsertInTypeListOrMoveToEnd{
         typedef typename meta::If<
            meta::HasTypeInTypeList<
               TL,
               TYPE
            >::value,
            typename meta::BackInsert< 
               typename DeleteTypeInTypeList< TL,TYPE >::type,
               TYPE
            >::type,
            typename meta::BackInsert< 
               TL,
               TYPE
            >::type
         >::type type;
      };
      /// metaprogramming has dublicates in typelist metafunction 
      template<class TL>
      struct HasDuplicatesInTypeList;
      /// metaprogramming has dublicates in typelist metafunction 
      template<class THEAD,class TTAIL>
      struct HasDuplicatesInTypeList<meta::TypeList<THEAD,TTAIL> >{
         typedef typename meta::EvalIf<
            HasTypeInTypeList<TTAIL,THEAD>::value,
            meta::Bool<true>,
            HasDuplicatesInTypeList< TTAIL>
         >::type type;

         enum Value{
            value= HasDuplicatesInTypeList<meta::TypeList<THEAD,TTAIL> >::type::value
         };
      };
      /// metaprogramming has dublicates in typelist metafunction    
      template< >
      struct HasDuplicatesInTypeList<meta::ListEnd> : meta::FalseCase{
      };
      /// metaprogramming generate function the list for the graphical model class metafunction 
      template<class MAYBE_TYPELIST,class EXPLICIT_FUNCTION_TYPE,bool EDITABLE>
      struct GenerateFunctionTypeList{
         typedef typename meta::TypeListFromMaybeTypeList<MAYBE_TYPELIST>::type StartTypeList;
         typedef typename meta::If<
            EDITABLE,
            typename InsertInTypeListOrMoveToEnd<StartTypeList,EXPLICIT_FUNCTION_TYPE>::type,
            StartTypeList
         >::type type;
         
      };
      /// metaprogramming calltraits
		template<class T>
		struct CallTraits {
			typedef T ValueType;
         typedef T value_type;
			typedef typename opengm::meta::AddReference<T>::type reference;
			typedef typename opengm::meta::AddConstReference<T>::type const_reference;
			typedef typename opengm::meta::If <
				opengm::meta::TypeInfo<T>::IsFundamental::value,
				typename opengm::meta::RemoveConst<T>::type const,
				typename opengm::meta::AddConstReference<T>::type
				>::type
				param_type;
		};
      /// metaprogramming access values of a field
      template<class TList ,template <class> class InstanceUnitType,class TListSrc>
      class FieldHelper;
      template<class ListHead,class ListTail,template  <class> class InstanceUnitType,class TListSrc>
      class FieldHelper< opengm::meta::TypeList<ListHead,ListTail> ,InstanceUnitType,TListSrc>
      : public FieldHelper<ListHead,InstanceUnitType,TListSrc>,
        public FieldHelper<ListTail,InstanceUnitType,TListSrc>{
      };
      template< class ListTail ,template <class> class InstanceUnitType,class TListSrc>
      class FieldHelper
         : public InstanceUnitType<ListTail>{
      };
      template< template <class> class InstanceUnitType,class TListSrc>
      class FieldHelper<opengm::meta::ListEnd,InstanceUnitType,TListSrc>{
      };
      /// metaprogramming  field  (InstanceUnitType with 1 template)
      template<class TList ,template <class> class InstanceUnitType>
      class Field
      :  public FieldHelper<TList,InstanceUnitType,TList>{
      public:
         public:
         template <typename T>
         struct RebingByType{
               typedef InstanceUnitType<T> type;
           };
         template <size_t Index>
         struct RebingByIndex{
               typedef  InstanceUnitType<typename  TypeAtTypeList<TList,Index>::type > type;
           };
      };
      /// metaprogramming access values of a field2
      template<class TList ,class TYPE2,template <class ,class > class InstanceUnitType,class TListSrc>
      class Field2Helper;
      /// metaprogramming access values of a field
      template<class ListHead,class ListTail,class TYPE2,template  <class,class> class InstanceUnitType,class TListSrc>
      class Field2Helper< opengm::meta::TypeList<ListHead,ListTail> ,TYPE2,InstanceUnitType,TListSrc>
      : public Field2Helper<ListHead,TYPE2,InstanceUnitType,TListSrc>,
        public Field2Helper<ListTail,TYPE2,InstanceUnitType,TListSrc>{
      };
      /// metaprogramming access values of a field
      template< class ListTail ,class TYPE2,template <class,class> class InstanceUnitType,class TListSrc>
      class Field2Helper
      : public InstanceUnitType<ListTail,TYPE2>{
      };
      /// metaprogramming access values of a field
      template< class TYPE2,template <class,class> class InstanceUnitType,class TListSrc>
      class Field2Helper<opengm::meta::ListEnd,TYPE2,InstanceUnitType,TListSrc>{
      };
      /// metaprogramming  field2 (InstanceUnitType with 2 templates)
      template<class TList,class TYPE2 ,template <class,class> class InstanceUnitType>
      class Field2
      :
      public Field2Helper<TList,TYPE2,InstanceUnitType,TList>{
      public:
         public:
         template <typename T>
         struct RebingByType{
               typedef InstanceUnitType<T,TYPE2> type;
           };
         template <size_t Index>
         struct RebingByIndex{
               typedef  InstanceUnitType<typename  TypeAtTypeList<TList,Index>::type,TYPE2 > type;
           };
      };
      /// metaprogramming access values of a field
      struct FieldAccess{
         template<size_t Index,class IG>
         static inline typename IG::template RebingByIndex<Index>::type &
         byIndex
         (
            IG & instanceGenerator
         ) {
            return instanceGenerator;
         }

         template<size_t Index,class IG>
         static inline const typename IG::template RebingByIndex<Index>::type &
         byIndex
         (
            const IG & instanceGenerator
         ) {
            return instanceGenerator;
         }

         template<class T,class IG>
         static inline typename IG::template RebingByType<T>::type &
         byType
         (
            IG & instanceGenerator
         ) {
            return instanceGenerator;
         }

         template<class T,class IG>
         static inline const typename IG::template RebingByType<T>::type &
         byType
         (
            const IG & instanceGenerator
         ) {
            return instanceGenerator;
         }
      };
      /// metaprogramming get the function of a factor
      template<class Factor,size_t FunctionIndex>
      class GetFunctionFromFactor
      {
         typedef typename  Factor::FunctionTypeList FunctionTypeList;
      public:
         typedef typename meta::TypeAtTypeList<FunctionTypeList,FunctionIndex>::type FunctionType;
         static inline const  FunctionType & get(const Factor & factor) {
            return factor. template function<FunctionIndex>();
         }
         static inline  FunctionType & get( Factor & factor) {
            return factor. template function<FunctionIndex>();
         }
      };
      /// metaprogramming get the function of a factor
      template<class Factor,size_t FunctionIndex>
      class GetFunction;
      /// metaprogramming get the function of a factor
      template<class T,size_t FunctionIndex>
      class GetFunction<opengm::Factor<T>,FunctionIndex >{
         typedef  typename Factor<T>::FunctionTypeList FunctionTypeList;
      public:
         typedef typename meta::TypeAtTypeList<FunctionTypeList,FunctionIndex>::type FunctionType;

         static inline const FunctionType & get(const Factor<T> & factor) {
            return factor. template function<FunctionIndex>();
         };
         static inline FunctionType & get(Factor<T> & factor) {
            return factor. template function<FunctionIndex>();
         };
      };
      /// metaprogramming get the an  function of a independent factor
      template<class T,class I,class L,size_t FunctionIndex>
      class GetFunction<IndependentFactor<T,I,L>,FunctionIndex >{
      public:
         typedef typename IndependentFactor<T,I,L>::FunctionType FunctionType;
         static inline const FunctionType & get(const IndependentFactor<T,I,L> & factor) {
            return factor.template  function<0>();
         };
         static inline FunctionType & get(IndependentFactor<T,I,L> & factor) {
            return factor.template  function<0>();
         };
      };
      /// metaprogramming get index of a function type
      template<class Factor>
      class GetFunctionTypeIndex;
      /// metaprogramming get index of a function type
      template<class T>
      class GetFunctionTypeIndex<opengm::Factor<T> >{
      public:
         static inline size_t get(const opengm::Factor<T> & factor) {
            return factor.functionType();
         }
         static inline size_t get(opengm::Factor<T> & factor) {
            return factor.functionType();
         }
      };
      /// metaprogramming get index of a function type
      template<class T,class I,class L>
      class GetFunctionTypeIndex<opengm::IndependentFactor<T,I,L> >{
      public:
         static inline size_t get(const opengm::IndependentFactor<T,I,L> & factor) {
            return 0;
         }
         static inline size_t get( opengm::IndependentFactor<T,I,L> & factor) {
            return 0;
         }
      };
      /// metaprogramming is field metafunction   
      template <class T>
      class IsField  : opengm::meta::FalseCase{};
      /// metaprogramming is field metafunction   
      template<class TList ,template <class> class InstanceUnitType>
      class IsField< meta::Field<TList,InstanceUnitType> >  : opengm::meta::TrueCase{};
      /// metaprogramming error message
      struct ErrorMessage{
         struct WRONG_FUNCTION_TYPE;
      };
      /// metaprogramming meta assert failed 
      template <class MSG>
      struct OPENGM_METAPROGRAMMING_COMPILE_TIME_ASSERTION_FAILED_;
      template < >
      struct OPENGM_METAPROGRAMMING_COMPILE_TIME_ASSERTION_FAILED_<meta::EmptyType >{
      };
      /// metaprogramming  assert 
      template<bool>
      class Assert;
      template<>
      class Assert<true>{
      };
      /// metaprogramming accumulation metafunction
      template<class TLIST,size_t INITIAL_VALUE, template <size_t, size_t> class ACCUMULATOR>
      class Accumulate;
      /// metaprogramming accumulation metafunction
      template<class TLIST_HEAD,class TLIST_TAIL,size_t INITIAL_VALUE,template <size_t, size_t> class ACCUMULATOR>
      class Accumulate<meta::TypeList<TLIST_HEAD,TLIST_TAIL>,INITIAL_VALUE ,ACCUMULATOR >{
         enum Value{
            value=Accumulate<
               TLIST_TAIL,
               ACCUMULATOR <
                  INITIAL_VALUE,
                  TLIST_HEAD::value
               >::value ,
               ACCUMULATOR
            >::value
         };
         typedef SizeT<
            Accumulate<
               TLIST_TAIL ,
               ACCUMULATOR<
                  INITIAL_VALUE,
                  TLIST_HEAD::value
               >::value ,
               ACCUMULATOR
            >::value
         > type;
      };
      /// metaprogramming accumulation metafunction
      template<size_t INITIAL_VALUE,template <size_t, size_t> class ACCUMULATOR>
      class Accumulate<meta::ListEnd,INITIAL_VALUE ,ACCUMULATOR >{
         enum Value{
            value=INITIAL_VALUE
         };
         typedef SizeT<INITIAL_VALUE> type;
      };
      
      template<class T>
      struct PromoteToFloatingPoint{
         typedef typename meta::If< 
            meta::IsFloatingPoint<T>::value ,
            T,
            float
         >::type type;
      };
      
   } // namespace meta
   
   
   /// generate a typedef to a function type list
   ///
   /// Usage:
   /// \code
   /// typedef opengm::FunctionTypeListGenerator<
   ///    opengm::PottsFunction<float,size_t,size_t>,
   ///    opengm::ExplicitFunction<float,size_t,size_t>
   /// >::type FunctionTypeList;
   /// \endcode
   template
   <
   class T1,
   class T2 = meta::ListEnd,
   class T3 = meta::ListEnd,
   class T4 = meta::ListEnd,
   class T5 = meta::ListEnd,
   class T6 = meta::ListEnd,
   class T7 = meta::ListEnd,
   class T8 = meta::ListEnd,
   class T9 = meta::ListEnd,
   class T10 = meta::ListEnd,
   class T11 = meta::ListEnd,
   class T12 = meta::ListEnd,
   class T13 = meta::ListEnd,
   class T14 = meta::ListEnd,
   class T15 = meta::ListEnd
   >
   struct FunctionTypeListGenerator {
      typedef meta::TypeList<T1, typename FunctionTypeListGenerator<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>::type > type;
   };

   template< >
   struct FunctionTypeListGenerator<meta::ListEnd> {
      typedef meta::ListEnd type;
   };

/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_METAPROGRAMMING

