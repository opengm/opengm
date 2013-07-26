#pragma once
#ifndef OPENGM_GRAPHICALMODEL_FUNCTION_WRAPPER_HXX
#define OPENGM_GRAPHICALMODEL_FUNCTION_WRAPPER_HXX

#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include <numeric>
#include <map>
#include <list>
#include <set>
#include <functional>

#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/function_properties.hxx"
#include "opengm/opengm.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/sorting.hxx"
#include "opengm/utilities/functors.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/graphicalmodel/graphicalmodel_factor_operator.hxx"
#include "opengm/graphicalmodel/graphicalmodel_factor_accumulator.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

template<
   class T,
   class OPERATOR,
   class FUNCTION_TYPE_LIST ,
   class SPACE 
>
class GraphicalModel;

template<class GRAPHICAL_MODEL> class Factor;

namespace detail_graphical_model {

   #define OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( RETURN_TYPE , FUNCTION_NAME ) \
   template<size_t NUMBER_OF_FUNCTIONS> \
   template<class GM> \
   inline RETURN_TYPE \
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::FUNCTION_NAME \
   ( \
      GM const * gm, \
      const size_t functionIndex, \
      const size_t functionType \
   ) { \
     typedef typename opengm::meta::SizeT< opengm::meta::Decrement<NUMBER_OF_FUNCTIONS>::value > MaxIndex;  \
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,1>::value) { \
         return gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
      } \
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,2>::value) { \
         if(functionType==0) \
            return gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
         else \
            return gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
      } \
      if(meta::BiggerOrEqualNumber<NUMBER_OF_FUNCTIONS,3>::value) { \
         switch(functionType) { \
            case 0: \
               return gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 1: \
               return gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 2: \
               return gm->template functions<meta::MinimumNumber<2,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 3: \
               return gm->template functions<meta::MinimumNumber<3,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 4: \
               return gm->template functions<meta::MinimumNumber<4,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 5: \
               return gm->template functions<meta::MinimumNumber<5,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 6: \
               return gm->template functions<meta::MinimumNumber<6,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 7: \
               return gm->template functions<meta::MinimumNumber<7,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 8: \
               return gm->template functions<meta::MinimumNumber<8,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 9: \
               return gm->template functions<meta::MinimumNumber<9,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 10: \
               return gm->template functions<meta::MinimumNumber<10,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 11: \
               return gm->template functions<meta::MinimumNumber<11,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 12: \
               return gm->template functions<meta::MinimumNumber<12,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 13: \
               return gm->template functions<meta::MinimumNumber<13,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 14: \
               return gm->template functions<meta::MinimumNumber<14,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            case 15: \
               return gm->template functions<meta::MinimumNumber<15,MaxIndex::value >::value >()[functionIndex].FUNCTION_NAME(); \
            default: \
               return FunctionWrapperExecutor< \
                  16, \
                  NUMBER_OF_FUNCTIONS, \
                  opengm::meta::BiggerOrEqualNumber<16,NUMBER_OF_FUNCTIONS >::value \
               >::FUNCTION_NAME(gm,functionIndex,functionType); \
         } \
      } \
   } \
   template<size_t IX, size_t DX> \
   template<class GM> \
   RETURN_TYPE FunctionWrapperExecutor<IX, DX, false>::FUNCTION_NAME \
   ( \
      GM const* gm, \
      const size_t functionIndex, \
      const size_t functionType \
   ) { \
      if(functionType==IX) { \
         return gm->template functions<IX>()[functionIndex].FUNCTION_NAME(); \
      } \
      else { \
         return FunctionWrapperExecutor<IX+1, DX, meta::Bool<IX+1==DX>::value >::FUNCTION_NAME (gm, functionIndex,functionType); \
      } \
   } \
   template<size_t IX, size_t DX> \
   template<class GM> \
   RETURN_TYPE FunctionWrapperExecutor<IX, DX, true>::FUNCTION_NAME \
   ( \
      GM const* gm, \
      const size_t functionIndex, \
      const size_t functionType \
   ) { \
      throw RuntimeError("Incorrect function type id."); \
   }
    
   template<size_t IX, size_t DX, bool end>
   struct FunctionWrapperExecutor;

   template<size_t IX, size_t DX>
   struct FunctionWrapperExecutor<IX,DX,false>{
      template <class GM,class ITERATOR>
      static void  getValues(const GM *,ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class ITERATOR>
      static void  getValuesSwitchedOrder(const GM *,ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class ITERATOR>
      static typename GM::ValueType  getValue(const GM *,ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAllValuesInAnyOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAtLeastAllUniqueValues(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAllValuesInOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAllValuesInSwitchedOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,int PROPERTY>
      static bool  binaryProperty(const GM *,const typename GM::IndexType ,const size_t );
      template <class GM,int PROPERTY>
      static typename GM::ValueType  valueProperty(const GM *,const typename GM::IndexType ,const size_t );
      template <class GM>
      static size_t numberOfFunctions(const GM *,const size_t );
      template <class GM_SOURCE,class GM_DEST>
      static void assignFunctions(const GM_SOURCE & ,GM_DEST &);
      template<class GM>
      static bool isPotts(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isGeneralizedPotts(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isSubmodular(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType min(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType max(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType sum(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType product(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isSquaredDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isTruncatedSquaredDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isAbsoluteDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isTruncatedAbsoluteDifference(GM const *,const size_t ,const size_t);
   };

   template<size_t IX, size_t DX>
   struct FunctionWrapperExecutor<IX,DX,true>{
      template <class GM,class ITERATOR>
      static typename GM::ValueType  getValue(const GM *,ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class ITERATOR>
      static void getValues(const GM *,ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class ITERATOR>
      static void getValuesSwitchedOrder(const GM *,ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAllValuesInAnyOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAtLeastAllUniqueValues(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAllValuesInOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAllValuesInSwitchedOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,int PROPERTY>
      static bool  binaryProperty(const GM *,const typename GM::IndexType ,const size_t );
      template <class GM,int PROPERTY>
      static typename GM::ValueType  valueProperty(const GM *,const typename GM::IndexType ,const size_t );
      template <class GM>
      static size_t numberOfFunctions(const GM *,const size_t functionTypeIndex);
      template <class GM_SOURCE,class GM_DEST>
      static void assignFunctions(const GM_SOURCE & ,GM_DEST &);
      template<class GM>
      static bool isPotts(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isGeneralizedPotts(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isSubmodular(GM const *,const size_t,const size_t );
      template<class GM>
      static typename GM::ValueType min(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType max(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType sum(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType product(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isSquaredDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isTruncatedSquaredDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isAbsoluteDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isTruncatedAbsoluteDifference(GM const *,const size_t ,const size_t);
   };

   template<size_t NUMBER_OF_FUNCTIONS>
   struct FunctionWrapper{
       
      template <class GM,class OUT_ITERATOR>
      static void  getValues(const GM *,OUT_ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class OUT_ITERATOR>
      static void  getValuesSwitchedOrder(const GM *,OUT_ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class ITERATOR>
      static typename GM::ValueType  getValue(const GM *,ITERATOR,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAllValuesInAnyOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAtLeastAllUniqueValues(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t );
      template <class GM,class FUNCTOR>
      static void forAllValuesInOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t ); 
      template <class GM,class FUNCTOR>
      static void forAllValuesInSwitchedOrder(const GM *,FUNCTOR &,const typename GM::IndexType ,const size_t ); 
      template <class GM,int PROPERTY>
      static bool  binaryProperty(const GM *,const typename GM::IndexType ,const size_t );
      template <class GM,int PROPERTY>
      static typename GM::ValueType  valueProperty(const GM *,const typename GM::IndexType ,const size_t );
      template <class GM>
      static size_t numberOfFunctions(const GM *,const size_t functionTypeIndex);
      template <class GM_SOURCE,class GM_DEST>
      static void assignFunctions(const GM_SOURCE & ,GM_DEST &);
      template<class GM>
      static bool isPotts(GM const *,const size_t,const size_t);
      template<class GM>
      static bool isGeneralizedPotts(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isSubmodular(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType min(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType max(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType sum(GM const *,const size_t ,const size_t);
      template<class GM>
      static typename GM::ValueType product(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isSquaredDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isTruncatedSquaredDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isAbsoluteDifference(GM const *,const size_t ,const size_t);
      template<class GM>
      static bool isTruncatedAbsoluteDifference(GM const *,const size_t ,const size_t);
   };
} //namespace detail_graphical_model

// implementaion
namespace detail_graphical_model {
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( bool, isSubmodular)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( bool, isPotts)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( bool, isGeneralizedPotts)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( bool, isSquaredDifference)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( bool, isTruncatedSquaredDifference)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( bool, isAbsoluteDifference)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( bool, isTruncatedAbsoluteDifference)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( typename GM::ValueType, min)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( typename GM::ValueType, max)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( typename GM::ValueType, sum)
   OPENGM_BASIC_FUNCTION_WRAPPER_CODE_GENERATOR_MACRO( typename GM::ValueType, product)

   template<size_t IX,size_t DX>
   template<class GM,class ITERATOR>
   inline typename GM::ValueType
   FunctionWrapperExecutor<IX,DX,false>::getValue
   (
      const GM * gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         return gm-> template functions<IX>()[functionIndex](iterator);
      }
      else{
         return FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >::getValue(gm,iterator,functionIndex,functionType);
      }
   }
   
   template<size_t IX,size_t DX>
   template <class GM,class FUNCTOR>
   inline void 
   FunctionWrapperExecutor<IX,DX,false>::forAllValuesInAnyOrder
   (
      const GM * gm,
      FUNCTOR & functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         gm-> template functions<IX>()[functionIndex].forAllValuesInAnyOrder(functor);
      }
      else{
         FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >::forAllValuesInAnyOrder(gm,functor,functionIndex,functionType);
      }
   }
   
   template<size_t IX,size_t DX>
   template <class GM,class FUNCTOR>
   inline void 
   FunctionWrapperExecutor<IX,DX,false>::forAtLeastAllUniqueValues
   (
      const GM * gm,
      FUNCTOR & functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         gm-> template functions<IX>()[functionIndex].forAtLeastAllUniqueValues(functor);
      }
      else{
         FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >::forAtLeastAllUniqueValues(gm,functor,functionIndex,functionType);
      }
   }
   
   template<size_t IX,size_t DX>
   template <class GM,class FUNCTOR>
   inline void 
   FunctionWrapperExecutor<IX,DX,false>::forAllValuesInOrder
   (
      const GM * gm,
      FUNCTOR & functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         gm-> template functions<IX>()[functionIndex].forAllValuesInOrder(functor);
      }
      else{
         FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >::forAllValuesInOrder(gm,functor,functionIndex,functionType);
      }
   }
   
   template<size_t IX,size_t DX>
   template <class GM,class FUNCTOR>
   inline void 
   FunctionWrapperExecutor<IX,DX,false>::forAllValuesInSwitchedOrder
   (
      const GM * gm,
      FUNCTOR & functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         gm-> template functions<IX>()[functionIndex].forAllValuesInSwitchedOrder(functor);
      }
      else{
         FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >::forAllValuesInSwitchedOrder(gm,functor,functionIndex,functionType);
      }
   }
   
   
   template<size_t IX,size_t DX>
   template <class GM,int PROPERTY>
   inline bool
   FunctionWrapperExecutor<IX,DX,false>::binaryProperty
   (
      const GM * gm,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         typedef typename GM::FunctionTypeList FTypeList;
         typedef typename meta::TypeAtTypeList<FTypeList,IX>::type FunctionType;
         return BinaryFunctionProperties<PROPERTY, FunctionType>::op(gm-> template functions<IX>()[functionIndex]);
      }
      else{
         return FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >:: template binaryProperty<GM,PROPERTY>(gm,functionIndex,functionType);
      }
   }
   
   template<size_t IX,size_t DX>
   template <class GM,int PROPERTY>
   inline typename GM::ValueType
   FunctionWrapperExecutor<IX,DX,false>::valueProperty
   (
      const GM * gm,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         typedef typename GM::FunctionTypeList FTypeList;
         typedef typename meta::TypeAtTypeList<FTypeList,IX>::type FunctionType;
         return ValueFunctionProperties<PROPERTY, FunctionType>::op(gm-> template functions<IX>()[functionIndex]);
      }
      else{
         return FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >:: template valueProperty<GM,PROPERTY>(gm,functionIndex,functionType);
      }
   }
    
   template<size_t IX,size_t DX>
   template<class GM,class ITERATOR>
   inline void
   FunctionWrapperExecutor<IX,DX,false>::getValues
   (
      const GM * gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         // COPY FUNCTION TO ITERATR
         typedef typename GM::FunctionTypeList FTypeList;
         typedef typename meta::TypeAtTypeList<FTypeList,IX>::type FunctionType;
         typedef typename FunctionType::FunctionShapeIteratorType FunctionShapeIteratorType;
         
         const FunctionType & function = gm-> template functions<IX>()[functionIndex];
         ShapeWalker< FunctionShapeIteratorType > walker(function.functionShapeBegin(),function.dimension());
         for (size_t i = 0; i < function.size(); ++i) {
               *iterator = function(walker.coordinateTuple().begin());
               ++iterator;
               ++walker;
         }

      }
      else{
         return FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >::getValues(gm,iterator,functionIndex,functionType);
      }
   }
   
   template<size_t IX,size_t DX>
   template<class GM,class ITERATOR>
   inline void
   FunctionWrapperExecutor<IX,DX,false>::getValuesSwitchedOrder
   (
      const GM * gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      if(IX==functionType) {
         // COPY FUNCTION TO ITERATR
         typedef typename GM::FunctionTypeList FTypeList;
         typedef typename meta::TypeAtTypeList<FTypeList,IX>::type FunctionType;
         typedef typename FunctionType::FunctionShapeIteratorType FunctionShapeIteratorType;
         
         const FunctionType & function = gm-> template functions<IX>()[functionIndex];
         ShapeWalkerSwitchedOrder< FunctionShapeIteratorType > walker(function.functionShapeBegin(),function.dimension());
         for (size_t i = 0; i < function.size(); ++i) {
               *iterator = function(walker.coordinateTuple().begin());
               ++iterator;
               ++walker;
         }

      }
      else{
         return FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >::getValuesSwitchedOrder(gm,iterator,functionIndex,functionType);
      }
   }
    
   template<size_t IX,size_t DX>
   template<class GM,class ITERATOR>
   inline void
   FunctionWrapperExecutor<IX,DX,true>::getValues
   (
      const GM * gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
   }
   template<size_t IX,size_t DX>
   template<class GM,class ITERATOR>
   inline void
   FunctionWrapperExecutor<IX,DX,true>::getValuesSwitchedOrder
   (
      const GM * gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
   }
   
   template<size_t IX,size_t DX>
   template<class GM,class ITERATOR>
   inline typename GM::ValueType
   FunctionWrapperExecutor<IX,DX,true>::getValue
   (
      const GM * gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
      return typename GM::ValueType();
   }
   
    template<size_t IX,size_t DX>
   template <class GM,class FUNCTOR>
   inline void 
   FunctionWrapperExecutor<IX,DX,true>::forAllValuesInAnyOrder
   (
      const GM * gm,
      FUNCTOR & functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
   }

   template<size_t IX,size_t DX>
   template <class GM,class FUNCTOR>
   inline void  
   FunctionWrapperExecutor<IX,DX,true>::forAtLeastAllUniqueValues
   (
      const GM * gm,
      FUNCTOR & functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
   }
   
   template<size_t IX,size_t DX>
   template <class GM,class FUNCTOR>
   inline void 
   FunctionWrapperExecutor<IX,DX,true>::forAllValuesInOrder
   (
      const GM * gm,
      FUNCTOR & functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
   }
   
   template<size_t IX,size_t DX>
   template <class GM,class FUNCTOR>
   inline void 
   FunctionWrapperExecutor<IX,DX,true>::forAllValuesInSwitchedOrder
   (
      const GM * gm,
      FUNCTOR & functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
   }
   
   template<size_t IX,size_t DX>
   template <class GM,int PROPERTY>
   inline bool
   FunctionWrapperExecutor<IX,DX,true>::binaryProperty
   (
      const GM * gm,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
      return false;
   }
   
   template<size_t IX,size_t DX>
   template <class GM,int PROPERTY>
   inline typename GM::ValueType
   FunctionWrapperExecutor<IX,DX,true>::valueProperty
   (
      const GM * gm,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
      return false;
   }
   
   template<size_t IX,size_t DX>
   template<class GM>
   inline size_t
   FunctionWrapperExecutor<IX,DX,false>::numberOfFunctions
   (
      const GM * gm,
      const size_t functionType
   ) {
      if(IX==functionType) {
         return gm->template functions<IX>().size();
      }
      else{
         return FunctionWrapperExecutor<
            meta::Increment<IX>::value,
            DX,
            meta::EqualNumber<
               meta::Increment<IX>::value,
               DX
            >::value
         >::numberOfFunctions(gm,functionType);
      }
   }

   template<size_t IX,size_t DX>
   template<class GM>
   inline size_t
   FunctionWrapperExecutor<IX,DX,true>::numberOfFunctions
   (
      const GM * gm,
      const size_t functionType
   ) {
      throw RuntimeError("Incorrect function type id.");
   }

   template<size_t IX,size_t DX>
   template<class GM_SOURCE,class GM_DEST>
   inline void
   FunctionWrapperExecutor<IX,DX,false>::assignFunctions
   (
      const GM_SOURCE & gmSource,
      GM_DEST & gmDest
   ) {
      typedef typename meta::TypeAtTypeList<
         typename GM_SOURCE::FunctionTypeList ,
         IX
      >::type SourceTypeAtIX;
      typedef meta::SizeT<
         meta::GetIndexInTypeList<
            typename GM_DEST::FunctionTypeList,
            SourceTypeAtIX
         >::value
      > PositionOfSourceTypeInDestType;
      gmDest.template functions<PositionOfSourceTypeInDestType::value> () =
         gmSource.template functions<IX> ();

      //recursive call to the rest
      FunctionWrapperExecutor<
         meta::Increment<IX>::value,
         DX,
         meta::EqualNumber<
            meta::Increment<IX>::value,
            DX
         >::value
      >::assignFunctions(gmSource,gmDest);
   }

   template<size_t IX,size_t DX>
   template<class GM_SOURCE,class GM_DEST>
   inline void
   FunctionWrapperExecutor<IX,DX,true>::assignFunctions
   (
      const GM_SOURCE & gmSource,
      GM_DEST & gmDest
   ) {
   }

   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM_SOURCE,class GM_DEST>
   inline void
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::assignFunctions
   (
      const GM_SOURCE & gmSource,
      GM_DEST & gmDest
   ) {
      typedef FunctionWrapperExecutor<0, NUMBER_OF_FUNCTIONS, meta::Bool<NUMBER_OF_FUNCTIONS==0>::value> ExecutorType;
      return ExecutorType::assignFunctions(gmSource, gmDest);
   }

   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM>
   inline size_t
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::numberOfFunctions
   (
      const GM * gm,
      const size_t functionType
   ) {
      typedef FunctionWrapperExecutor<0, NUMBER_OF_FUNCTIONS, meta::Bool<NUMBER_OF_FUNCTIONS==0>::value> ExecutorType;
      return ExecutorType::numberOfFunctions(gm, functionType);
   }
   
   
   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM,class ITERATOR>
   inline void
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::getValues
   (
      const GM *  gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
        FunctionWrapperExecutor<
             0,
             NUMBER_OF_FUNCTIONS,
             opengm::meta::BiggerOrEqualNumber<0,NUMBER_OF_FUNCTIONS>::value
        >::getValues(gm,iterator,functionIndex,functionType);
   }
   
   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM,class ITERATOR>
   inline void
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::getValuesSwitchedOrder
   (
      const GM *  gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
        FunctionWrapperExecutor<
             0,
             NUMBER_OF_FUNCTIONS,
             opengm::meta::BiggerOrEqualNumber<0,NUMBER_OF_FUNCTIONS>::value
        >::getValuesSwitchedOrder(gm,iterator,functionIndex,functionType);
   }
   
   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM,class ITERATOR>
   inline typename GM::ValueType
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::getValue
   (
      const GM *  gm,
      ITERATOR iterator,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      typedef typename opengm::meta::SizeT< opengm::meta::Decrement<NUMBER_OF_FUNCTIONS>::value > MaxIndex;
      // special implementation if there is only one function typelist
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,1>::value) {
         return gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex](iterator);
      }
      // special implementation if there are only two functions in the typelist
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,2>::value) {
         if(functionType==0)
            return gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex](iterator);
         else
            return gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex](iterator);
      }
      // general case : 3 or more functions in the typelist
      if(meta::BiggerOrEqualNumber<NUMBER_OF_FUNCTIONS,3>::value) {
         switch(functionType) {
            case 0:
               return gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex](iterator);
            case 1:
               return gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex](iterator);
            case 2:
               return gm->template functions<meta::MinimumNumber<2,MaxIndex::value >::value >()[functionIndex](iterator);
            case 3:
               return gm->template functions<meta::MinimumNumber<3,MaxIndex::value >::value >()[functionIndex](iterator);
            case 4:
               return gm->template functions<meta::MinimumNumber<4,MaxIndex::value >::value >()[functionIndex](iterator);
            case 5:
               return gm->template functions<meta::MinimumNumber<5,MaxIndex::value >::value >()[functionIndex](iterator);
            case 6:
               return gm->template functions<meta::MinimumNumber<6,MaxIndex::value >::value >()[functionIndex](iterator);
            case 7:
               return gm->template functions<meta::MinimumNumber<7,MaxIndex::value >::value >()[functionIndex](iterator);
            case 8:
               return gm->template functions<meta::MinimumNumber<8,MaxIndex::value >::value >()[functionIndex](iterator);
            case 9:
               return gm->template functions<meta::MinimumNumber<9,MaxIndex::value >::value >()[functionIndex](iterator);
            case 10:
               return gm->template functions<meta::MinimumNumber<10,MaxIndex::value >::value >()[functionIndex](iterator);
            case 11:
               return gm->template functions<meta::MinimumNumber<11,MaxIndex::value >::value >()[functionIndex](iterator);
            case 12:
               return gm->template functions<meta::MinimumNumber<12,MaxIndex::value >::value >()[functionIndex](iterator);
            case 13:
               return gm->template functions<meta::MinimumNumber<13,MaxIndex::value >::value >()[functionIndex](iterator);
            case 14:
               return gm->template functions<meta::MinimumNumber<14,MaxIndex::value >::value >()[functionIndex](iterator);
            case 15:
               return gm->template functions<meta::MinimumNumber<15,MaxIndex::value >::value >()[functionIndex](iterator);
            default:
               // meta/template recursive "if-else" generation if the
               // function index is bigger than 15
               return FunctionWrapperExecutor<
                  16,
                  NUMBER_OF_FUNCTIONS,
                  opengm::meta::BiggerOrEqualNumber<16,NUMBER_OF_FUNCTIONS >::value
               >::getValue(gm,iterator,functionIndex,functionType);
         }
      }
   }
   
   
   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM,class FUNCTOR>
   inline void 
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::forAllValuesInAnyOrder
   (
      const GM *  gm,
      FUNCTOR &  functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      typedef typename opengm::meta::SizeT< opengm::meta::Decrement<NUMBER_OF_FUNCTIONS>::value > MaxIndex;
      // special implementation if there is only one function typelist
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,1>::value) {
         gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
      }
      // special implementation if there are only two functions in the typelist
      else if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,2>::value) {
         if(functionType==0)
            gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
         else
            gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
      }
      // general case : 3 or more functions in the typelist
      else if(meta::BiggerOrEqualNumber<NUMBER_OF_FUNCTIONS,3>::value) {
         switch(functionType) {
            case 0:
                gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 1:
                gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 2:
                gm->template functions<meta::MinimumNumber<2,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 3:
                gm->template functions<meta::MinimumNumber<3,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 4:
                gm->template functions<meta::MinimumNumber<4,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 5:
                gm->template functions<meta::MinimumNumber<5,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 6:
                gm->template functions<meta::MinimumNumber<6,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 7:
                gm->template functions<meta::MinimumNumber<7,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 8:
                gm->template functions<meta::MinimumNumber<8,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 9:
                gm->template functions<meta::MinimumNumber<9,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 10:
                gm->template functions<meta::MinimumNumber<10,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 11:
                gm->template functions<meta::MinimumNumber<11,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 12:
                gm->template functions<meta::MinimumNumber<12,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 13:
                gm->template functions<meta::MinimumNumber<13,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 14:
                gm->template functions<meta::MinimumNumber<14,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            case 15:
                gm->template functions<meta::MinimumNumber<15,MaxIndex::value >::value >()[functionIndex].forAllValuesInAnyOrder(functor);
                break;
            default:
               // meta/template recursive "if-else" generation if the
               // function index is bigger than 15
               FunctionWrapperExecutor<
                  16,
                  NUMBER_OF_FUNCTIONS,
                  opengm::meta::BiggerOrEqualNumber<16,NUMBER_OF_FUNCTIONS >::value
               >::forAllValuesInAnyOrder(gm,functor,functionIndex,functionType);
         }
      }
   }
   
   
   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM,class FUNCTOR>
   inline void 
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::forAtLeastAllUniqueValues
   (
      const GM *  gm,
      FUNCTOR &  functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      typedef typename opengm::meta::SizeT< opengm::meta::Decrement<NUMBER_OF_FUNCTIONS>::value > MaxIndex;
      // special implementation if there is only one function typelist
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,1>::value) {
         gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
      }
      // special implementation if there are only two functions in the typelist
      else if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,2>::value) {
         if(functionType==0)
            gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
         else
            gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
      }
      // general case : 3 or more functions in the typelist
      else if(meta::BiggerOrEqualNumber<NUMBER_OF_FUNCTIONS,3>::value) {
         switch(functionType) {
            case 0:
                gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 1:
                gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 2:
                gm->template functions<meta::MinimumNumber<2,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 3:
                gm->template functions<meta::MinimumNumber<3,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 4:
                gm->template functions<meta::MinimumNumber<4,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 5:
                gm->template functions<meta::MinimumNumber<5,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 6:
                gm->template functions<meta::MinimumNumber<6,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 7:
                gm->template functions<meta::MinimumNumber<7,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 8:
                gm->template functions<meta::MinimumNumber<8,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 9:
                gm->template functions<meta::MinimumNumber<9,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 10:
                gm->template functions<meta::MinimumNumber<10,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 11:
                gm->template functions<meta::MinimumNumber<11,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 12:
                gm->template functions<meta::MinimumNumber<12,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 13:
                gm->template functions<meta::MinimumNumber<13,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 14:
                gm->template functions<meta::MinimumNumber<14,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            case 15:
                gm->template functions<meta::MinimumNumber<15,MaxIndex::value >::value >()[functionIndex].forAtLeastAllUniqueValues(functor);
                break;
            default:
               // meta/template recursive "if-else" generation if the
               // function index is bigger than 15
               FunctionWrapperExecutor<
                  16,
                  NUMBER_OF_FUNCTIONS,
                  opengm::meta::BiggerOrEqualNumber<16,NUMBER_OF_FUNCTIONS >::value
               >::forAtLeastAllUniqueValues(gm,functor,functionIndex,functionType);
         }
      }
   }

   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM,class FUNCTOR>
   inline void 
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::forAllValuesInOrder
   (
      const GM *  gm,
      FUNCTOR &  functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      typedef typename opengm::meta::SizeT< opengm::meta::Decrement<NUMBER_OF_FUNCTIONS>::value > MaxIndex;
      // special implementation if there is only one function typelist
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,1>::value) {
         gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
      }
      // special implementation if there are only two functions in the typelist
      else if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,2>::value) {
         if(functionType==0)
            gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
         else
            gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
      }
      // general case : 3 or more functions in the typelist
      else if(meta::BiggerOrEqualNumber<NUMBER_OF_FUNCTIONS,3>::value) {
         switch(functionType) {
            case 0:
                gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 1:
                gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 2:
                gm->template functions<meta::MinimumNumber<2,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 3:
                gm->template functions<meta::MinimumNumber<3,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 4:
                gm->template functions<meta::MinimumNumber<4,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 5:
                gm->template functions<meta::MinimumNumber<5,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 6:
                gm->template functions<meta::MinimumNumber<6,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 7:
                gm->template functions<meta::MinimumNumber<7,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 8:
                gm->template functions<meta::MinimumNumber<8,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 9:
                gm->template functions<meta::MinimumNumber<9,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 10:
                gm->template functions<meta::MinimumNumber<10,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 11:
                gm->template functions<meta::MinimumNumber<11,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 12:
                gm->template functions<meta::MinimumNumber<12,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 13:
                gm->template functions<meta::MinimumNumber<13,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 14:
                gm->template functions<meta::MinimumNumber<14,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            case 15:
                gm->template functions<meta::MinimumNumber<15,MaxIndex::value >::value >()[functionIndex].forAllValuesInOrder(functor);
                break;
            default:
               // meta/template recursive "if-else" generation if the
               // function index is bigger than 15
               FunctionWrapperExecutor<
                  16,
                  NUMBER_OF_FUNCTIONS,
                  opengm::meta::BiggerOrEqualNumber<16,NUMBER_OF_FUNCTIONS >::value
               >::forAllValuesInOrder(gm,functor,functionIndex,functionType);
         }
      }
   }
   
   
   template<size_t NUMBER_OF_FUNCTIONS>
   template<class GM,class FUNCTOR>
   inline void 
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::forAllValuesInSwitchedOrder
   (
      const GM *  gm,
      FUNCTOR &  functor,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      typedef typename opengm::meta::SizeT< opengm::meta::Decrement<NUMBER_OF_FUNCTIONS>::value > MaxIndex;
      // special implementation if there is only one function typelist
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,1>::value) {
         gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
      }
      // special implementation if there are only two functions in the typelist
      else if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,2>::value) {
         if(functionType==0)
            gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
         else
            gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
      }
      // general case : 3 or more functions in the typelist
      else if(meta::BiggerOrEqualNumber<NUMBER_OF_FUNCTIONS,3>::value) {
         switch(functionType) {
            case 0:
                gm->template functions<meta::MinimumNumber<0,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 1:
                gm->template functions<meta::MinimumNumber<1,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 2:
                gm->template functions<meta::MinimumNumber<2,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 3:
                gm->template functions<meta::MinimumNumber<3,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 4:
                gm->template functions<meta::MinimumNumber<4,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 5:
                gm->template functions<meta::MinimumNumber<5,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 6:
                gm->template functions<meta::MinimumNumber<6,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 7:
                gm->template functions<meta::MinimumNumber<7,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 8:
                gm->template functions<meta::MinimumNumber<8,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 9:
                gm->template functions<meta::MinimumNumber<9,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 10:
                gm->template functions<meta::MinimumNumber<10,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 11:
                gm->template functions<meta::MinimumNumber<11,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 12:
                gm->template functions<meta::MinimumNumber<12,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 13:
                gm->template functions<meta::MinimumNumber<13,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 14:
                gm->template functions<meta::MinimumNumber<14,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            case 15:
                gm->template functions<meta::MinimumNumber<15,MaxIndex::value >::value >()[functionIndex].forAllValuesInSwitchedOrder(functor);
                break;
            default:
               // meta/template recursive "if-else" generation if the
               // function index is bigger than 15
               FunctionWrapperExecutor<
                  16,
                  NUMBER_OF_FUNCTIONS,
                  opengm::meta::BiggerOrEqualNumber<16,NUMBER_OF_FUNCTIONS >::value
               >::forAllValuesInSwitchedOrder(gm,functor,functionIndex,functionType);
         }
      }
   }
   
   
   template<size_t NUMBER_OF_FUNCTIONS>
   template <class GM,int PROPERTY>
   inline bool
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::binaryProperty
   (
      const GM *  gm,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      typedef typename opengm::meta::SizeT< opengm::meta::Decrement<NUMBER_OF_FUNCTIONS>::value > MaxIndex;
      typedef typename GM::FunctionTypeList FTypeList;
      // special implementation if there is only one function typelist
      
      
      #define OPENGM_FWRAPPER_PROPERTY_GEN_MACRO( NUMBER) typedef meta::Int< NUMBER > Number; \
         typedef meta::Int<meta::MinimumNumber<Number::value,MaxIndex::value >::value> SaveNumber; \
         typedef typename meta::TypeAtTypeList<FTypeList,SaveNumber::value>::type FunctionType; \
         return BinaryFunctionProperties<PROPERTY, FunctionType>::op(gm-> template functions<SaveNumber::value>()[functionIndex])
      
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,1>::value) {OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(0);}
      // special implementation if there are only two functions in the typelist
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,2>::value) {
         if(functionType==0){OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(1);}
         else{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(2);}
      }
      // general case : 3 or more functions in the typelist
      if(meta::BiggerOrEqualNumber<NUMBER_OF_FUNCTIONS,3>::value) {
         switch(functionType) {
            case 0 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(0);}
            case 1 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(1);}
            case 2 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(2);}
            case 3 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(3);}
            case 4 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(4);}
            case 5 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(5);}
            case 6 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(6);}
            case 7 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(7);}
            case 8 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(8);}
            case 9 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(9);}
            case 10 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(10);}
            case 11 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(11);}
            case 12 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(12);}
            case 13 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(13);}
            case 14 :{OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(14);}
            case 15 :{ OPENGM_FWRAPPER_PROPERTY_GEN_MACRO(15);}
            default:{
               //meta/template recursive "if-else" generation if the
               //function index is bigger than 15
               return FunctionWrapperExecutor<
                  16,
                  NUMBER_OF_FUNCTIONS,
                  opengm::meta::BiggerOrEqualNumber<16,NUMBER_OF_FUNCTIONS >::value
               >:: template binaryProperty  <GM,PROPERTY> (gm,functionIndex,functionType);
            }
         }
      }
      #undef OPENGM_FWRAPPER_PROPERTY_GEN_MACRO
   }
   
   template<size_t NUMBER_OF_FUNCTIONS>
   template <class GM,int PROPERTY>
   inline typename GM::ValueType
   FunctionWrapper<NUMBER_OF_FUNCTIONS>::valueProperty
   (
      const GM *  gm,
      const typename GM::IndexType functionIndex,
      const size_t functionType
   ) {
      typedef typename opengm::meta::SizeT< opengm::meta::Decrement<NUMBER_OF_FUNCTIONS>::value > MaxIndex;
      typedef typename GM::FunctionTypeList FTypeList;
      // special implementation if there is only one function typelist
      
      
      #define OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO( NUMBER) typedef meta::Int< NUMBER > Number; \
         typedef meta::Int<meta::MinimumNumber<Number::value,MaxIndex::value >::value> SaveNumber; \
         typedef typename meta::TypeAtTypeList<FTypeList,SaveNumber::value>::type FunctionType; \
         return ValueFunctionProperties<PROPERTY, FunctionType>::op(gm-> template functions<SaveNumber::value>()[functionIndex])
      
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,1>::value) {OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(0);}
      // special implementation if there are only two functions in the typelist
      if(meta::EqualNumber<NUMBER_OF_FUNCTIONS,2>::value) {
         if(functionType==0){OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(1);}
         else{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(2);}
      }
      // general case : 3 or more functions in the typelist
      if(meta::BiggerOrEqualNumber<NUMBER_OF_FUNCTIONS,3>::value) {
         switch(functionType) {
            case 0 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(0);}
            case 1 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(1);}
            case 2 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(2);}
            case 3 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(3);}
            case 4 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(4);}
            case 5 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(5);}
            case 6 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(6);}
            case 7 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(7);}
            case 8 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(8);}
            case 9 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(9);}
            case 10 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(10);}
            case 11 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(11);}
            case 12 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(12);}
            case 13 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(13);}
            case 14 :{OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(14);}
            case 15 :{ OPENGM_FWRAPPER_VALUE_PROPERTY_GEN_MACRO(15);}
            default:{
               //meta/template recursive "if-else" generation if the
               //function index is bigger than 15
               return FunctionWrapperExecutor<
                  16,
                  NUMBER_OF_FUNCTIONS,
                  opengm::meta::BiggerOrEqualNumber<16,NUMBER_OF_FUNCTIONS >::value
               >:: template valueProperty  <GM,PROPERTY> (gm,functionIndex,functionType);
            }
         }
      }
      #undef OPENGM_FWRAPPER_PROPERTY_GEN_MACRO
   }

} // namespace detail_graphical_model

/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_GRAPHICALMODEL_FUNCTION_WRAPPER_HXX
