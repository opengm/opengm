#pragma once
#ifndef OPENGM_POTTS_FUNCTION_HXX
#define OPENGM_POTTS_FUNCTION_HXX

#include <algorithm>
#include <vector>
#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// Potts function for two variables
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class PottsFunction
: public FunctionBase<PottsFunction<T, I, L>, T, size_t, size_t>
{
public:
   typedef T ValueType;
   typedef L LabelType;
   typedef I IndexType;

   PottsFunction(const LabelType = 2, const LabelType = 2,
                 const ValueType = ValueType(), const ValueType = ValueType());
   LabelType shape(const size_t) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> ValueType operator()(ITERATOR) const;
   bool operator==(const PottsFunction& ) const;
   ValueType valueEqual() const;
   ValueType valueNotEqual() const;
   IndexType numberOfParameters() const;
   ValueType parameter(const size_t index) const;
   ValueType& parameter(const size_t index);

   // specializations
   bool isPotts() const;
   bool isGeneralizedPotts() const;
   ValueType min() const;
   ValueType max() const;
   ValueType sum() const;
   ValueType product() const;
   MinMaxFunctor<ValueType> minMax() const;

private:
   LabelType numberOfLabels1_;
   LabelType numberOfLabels2_;
   ValueType valueEqual_;
   ValueType valueNotEqual_;

friend class FunctionSerialization<PottsFunction<T, I, L> > ;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class T, class I, class L>
struct FunctionRegistration<PottsFunction<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 6
   };
};

/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<PottsFunction<T, I, L> > {
public:
   typedef typename PottsFunction<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const PottsFunction<T, I, L>&);
   static size_t valueSequenceSize(const PottsFunction<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const PottsFunction<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, PottsFunction<T, I, L>&);
};
/// \endcond

/// constructor
/// \param numberOfLabels1 number of labels of the first variable
/// \param numberOfLabels2 number of labels of the second variable
/// \param valueEqual value if the labels of the two variables are equal
/// \param valueNotEqual value if the labels of the two variables are not equal
template <class T, class I, class L>
inline
PottsFunction<T, I, L>::PottsFunction
(
   const L numberOfLabels1,
   const L numberOfLabels2,
   const T valueEqual,
   const T valueNotEqual
)
:  numberOfLabels1_(numberOfLabels1),
   numberOfLabels2_(numberOfLabels2),
   valueEqual_(valueEqual),
   valueNotEqual_(valueNotEqual)
{}

template <class T, class I, class L>
template <class ITERATOR>
inline T
PottsFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   return (begin[0]==begin[1] ? valueEqual_ : valueNotEqual_);
}

template <class T, class I, class L>
inline T
PottsFunction<T, I, L>::valueEqual()const {
   return valueEqual_;
}

template <class T, class I, class L>
inline T
PottsFunction<T, I, L>::valueNotEqual()const {
   return valueEqual_;
}

template <class T, class I, class L>
inline L
PottsFunction<T, I, L>::shape
(
   const size_t i
) const {
   OPENGM_ASSERT(i < 2);
   return (i==0 ? numberOfLabels1_ : numberOfLabels2_);
}

template <class T, class I, class L>
inline size_t
PottsFunction<T, I, L>::dimension() const {
   return 2;
}

template <class T, class I, class L>
inline size_t
PottsFunction<T, I, L>::size() const {
   return numberOfLabels1_*numberOfLabels2_;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<PottsFunction<T, I, L> >::indexSequenceSize
(
   const PottsFunction<T, I, L> & src
) {
   return 2;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<PottsFunction<T, I, L> >::valueSequenceSize
(
   const PottsFunction<T, I, L> & src
) {
   return 2;
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<PottsFunction<T, I, L> >::serialize
(
   const PottsFunction<T, I, L> & src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   *indexOutIterator = src.shape(0);
   ++indexOutIterator;
   *indexOutIterator = src.shape(1);

   *valueOutIterator = src.valueEqual_;
   ++valueOutIterator;
   *valueOutIterator = src.valueNotEqual_;
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<PottsFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   PottsFunction<T, I, L> & dst
) {
   const size_t shape1=*indexInIterator;
   ++ indexInIterator;
   const size_t shape2=*indexInIterator;
   const ValueType param1=*valueInIterator;
   ++valueInIterator;
   const ValueType param2=*valueInIterator;
   dst=PottsFunction<T, I, L>(shape1, shape2, param1, param2);
}

template<class T, class I, class L>
inline bool
PottsFunction<T, I, L>::operator==
(
   const PottsFunction & fb
   )const{
   return  numberOfLabels1_ == fb.numberOfLabels1_ &&
      numberOfLabels2_ == fb.numberOfLabels2_ &&
      valueEqual_      == fb.valueEqual_      &&
      valueNotEqual_   == fb.valueNotEqual_;
}

template<class T, class I, class L>
inline typename PottsFunction<T, I, L>::IndexType
PottsFunction<T, I, L>::numberOfParameters() const
{
   return 2;
}

template<class T, class I, class L>
inline typename PottsFunction<T, I, L>::ValueType
PottsFunction<T, I, L>::parameter(
   const size_t index
) const
{
   OPENGM_ASSERT(index < 2);
   return index == 0 ? valueEqual_ : valueNotEqual_;
}

template<class T, class I, class L>
inline typename PottsFunction<T, I, L>::ValueType&
PottsFunction<T, I, L>::parameter(
   const size_t index
)
{
   OPENGM_ASSERT(index < 2);
   return index==0 ? valueEqual_:valueNotEqual_;
}

template<class T, class I, class L>
inline bool
PottsFunction<T, I, L>::isPotts() const
{
   return true;
}

template<class T, class I, class L>
inline bool
PottsFunction<T, I, L>::isGeneralizedPotts() const
{
   return true;
}

template<class T, class I, class L>
inline typename PottsFunction<T, I, L>::ValueType
PottsFunction<T, I, L>::min() const
{
   return valueEqual_<valueNotEqual_ ? valueEqual_ :valueNotEqual_;
}

template<class T, class I, class L>
inline typename PottsFunction<T, I, L>::ValueType
PottsFunction<T, I, L>::max() const
{
   return valueNotEqual_>valueEqual_ ? valueNotEqual_ :valueEqual_;
}

template<class T, class I, class L>
inline typename PottsFunction<T, I, L>::ValueType
PottsFunction<T, I, L>::sum() const
{
   const LabelType minLabels = std::min(numberOfLabels1_, numberOfLabels2_);
   return valueNotEqual_ * static_cast<T>(numberOfLabels1_ * numberOfLabels2_ - minLabels)
      + valueEqual_*static_cast<T>(minLabels);
}

template<class T, class I, class L>
inline typename PottsFunction<T, I, L>::ValueType
PottsFunction<T, I, L>::product() const
{
   const LabelType minLabels = std::min(numberOfLabels1_, numberOfLabels2_);
   // TODO: improve this: do not use std::pow, instead write a proper pow functor class for OpenGM
   // the call of std::pow is ambiguous for many common combinations of types. this is just a
   // work-around with possible loss of precision, e.g. if valuesNotEqual_ is a long double
   const double x1 = static_cast<double>(valueNotEqual_);
   const int n1 = static_cast<int>(numberOfLabels1_ * numberOfLabels2_ - minLabels);
   const double x2 = static_cast<double>(valueEqual_);
   const int n2 = static_cast<int>(minLabels);
   return static_cast<T>(std::pow(x1, n1) * std::pow(x2, n2));
}

template<class T, class I, class L>
inline MinMaxFunctor<typename PottsFunction<T, I, L>::ValueType>
PottsFunction<T, I, L>::minMax() const
{
   if(valueEqual_<valueNotEqual_) {
      return MinMaxFunctor<T>(valueEqual_, valueNotEqual_);
   }
   else {
      return MinMaxFunctor<T>(valueNotEqual_, valueEqual_);
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_POTTS_FUNCTION_HXX
