#pragma once
#ifndef OPENGM_FUNCTION_PROPERTIES_HXX
#define OPENGM_FUNCTION_PROPERTIES_HXX

#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/utilities/shape_accessor.hxx"
#include "opengm/utilities/accessor_iterator.hxx"
#include "opengm/utilities/accumulation.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/functors.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/and.hxx"
#include "opengm/operations/or.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/integrator.hxx"
#include "opengm/operations/multiplier.hxx"

#define OPENGM_FUNCTION_TYPEDEF_MACRO typedef typename FunctionType::ValueType ValueType;\
typedef typename FunctionType::IndexType IndexType;\
typedef typename FunctionType::LabelType LabelType;\
typedef typename FunctionType::FunctionShapeIteratorType FunctionShapeIteratorType

namespace opengm {

struct BinaryProperties{
   enum Values{
       IsPotts=0,
       IsSubmodular1=1,
       IsPositive=2
   };
};

struct ValueProperties{
   enum Values{
      Sum=0,
      Product=1,
      Minimum=2,
      Maximum=3
   };
};

template<int PROPERTY_ID,class FUNCTION>
class BinaryFunctionProperties;

template<int PROPERTY_ID,class FUNCTION>
class ValueFunctionProperties;

namespace detail_properties{
   template<class FUNCTION,class ACCUMULATOR>
   class AllValuesInAnyOrderFunctionProperties;
   template<class FUNCTION,class ACCUMULATOR>
   class AtLeastAllUniqueValuesFunctionProperties;
}

// Fallback implementation(s) of binary properties
template<class FUNCTION>
class BinaryFunctionProperties<BinaryProperties::IsPotts, FUNCTION> {
   typedef FUNCTION FunctionType;
   OPENGM_FUNCTION_TYPEDEF_MACRO;
public:
   static bool op(const FunctionType & f) {
      ShapeWalker<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(), f.dimension());
      ValueType vEqual = f(shapeWalker.coordinateTuple().begin());
      ++shapeWalker;
      ValueType vNotEqual = f(shapeWalker.coordinateTuple().begin());
      ++shapeWalker;
      for (IndexType i = 2; i < f.size(); ++i, ++shapeWalker) {
         // all labels are equal
         if (isEqualValueVector(shapeWalker.coordinateTuple())) {
            if (vEqual != f(shapeWalker.coordinateTuple().begin()))
               return false;
         }               // all labels are not equal
         else {
            if (vNotEqual != f(shapeWalker.coordinateTuple().begin()))
               return false;
         }
      }
      return true;
   }
};


// Fallback implementation(s) of (real) value properties
// Some basic properties are derived from 
// "AllValuesInAnyOrderFunctionProperties" and 
// "AtLeastAllUniqueValuesFunctionProperties"
template<class FUNCTION>
class ValueFunctionProperties<ValueProperties::Product, FUNCTION> 
: public  detail_properties::AllValuesInAnyOrderFunctionProperties<FUNCTION,Multiplier>{
}; 

template<class FUNCTION>
class ValueFunctionProperties<ValueProperties::Sum, FUNCTION> 
 : public  detail_properties::AllValuesInAnyOrderFunctionProperties<FUNCTION,Adder>{
}; 

template<class FUNCTION>
class ValueFunctionProperties<ValueProperties::Minimum, FUNCTION> 
 : public detail_properties::AtLeastAllUniqueValuesFunctionProperties<FUNCTION,Minimizer>{
}; 

template<class FUNCTION>
class ValueFunctionProperties<ValueProperties::Maximum, FUNCTION> 
  : public detail_properties::AtLeastAllUniqueValuesFunctionProperties<FUNCTION,Maximizer>{
}; 


namespace detail_properties{
   template<class FUNCTION,class ACCUMULATOR>
   class AllValuesInAnyOrderFunctionProperties{
      typedef FUNCTION FunctionType;
      OPENGM_FUNCTION_TYPEDEF_MACRO;
   public:
      static ValueType op(const FunctionType & f) {
         opengm::AccumulationFunctor<ACCUMULATOR,ValueType> functor;
         f.forAllValuesInAnyOrder(functor);
         return functor.value();
      }
   };
   template<class FUNCTION,class ACCUMULATOR>
   class AtLeastAllUniqueValuesFunctionProperties{
      typedef FUNCTION FunctionType;
      OPENGM_FUNCTION_TYPEDEF_MACRO;
   public:
      static ValueType op(const FunctionType & f) {
         opengm::AccumulationFunctor<ACCUMULATOR,ValueType> functor;
         f.forAllValuesInAnyOrder(functor);
         return functor.value();
      }
   };
}

}// namespace opengm

#endif //OPENGM_FUNCTION_PROPERTIES_HXX
