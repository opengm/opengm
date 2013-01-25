#pragma once
#ifndef OPENGM_TRUNCATED_ABSOLUTE_DIFFERENCE_FUNCTION_HXX
#define OPENGM_TRUNCATED_ABSOLUTE_DIFFERENCE_FUNCTION_HXX

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// truncated absolute differents between the labels of 2 variables
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class TruncatedAbsoluteDifferenceFunction
: public FunctionBase<TruncatedAbsoluteDifferenceFunction<T, I, L>, T, I, L> {
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;

   TruncatedAbsoluteDifferenceFunction(const LabelType = 2, const LabelType = 2, 
      const ValueType = ValueType(), const ValueType = ValueType());
   size_t shape(const IndexType) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> T operator()(ITERATOR) const;

private:
   size_t numberOfLabels1_;
   size_t numberOfLabels2_;
   T parameter1_;
   T parameter2_;

friend class FunctionSerialization<TruncatedAbsoluteDifferenceFunction<T, I, L> > ;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template <class T, class I, class L>
struct FunctionRegistration<TruncatedAbsoluteDifferenceFunction<T, I, L> >{
   enum ID { Id = opengm::FUNCTION_TYPE_ID_OFFSET + 3 };
};

/// FunctionSerialization
template <class T, class I, class L>
class FunctionSerialization<TruncatedAbsoluteDifferenceFunction<T, I, L> > {
public:
   typedef typename TruncatedAbsoluteDifferenceFunction<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const TruncatedAbsoluteDifferenceFunction<T, I, L>&);
   static size_t valueSequenceSize(const TruncatedAbsoluteDifferenceFunction<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
      static void serialize(const TruncatedAbsoluteDifferenceFunction<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize(INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, TruncatedAbsoluteDifferenceFunction<T, I, L>&);
};
/// \endcond

template <class T, class I, class L>
inline
TruncatedAbsoluteDifferenceFunction<T, I, L>::TruncatedAbsoluteDifferenceFunction
(
   const LabelType numberOfLabels1,
   const LabelType numberOfLabels2,
   const ValueType parameter1,
   const ValueType parameter2
)
:  numberOfLabels1_(numberOfLabels1),
   numberOfLabels2_(numberOfLabels2),
   parameter1_(parameter1),
   parameter2_(parameter2)
{}

template <class T, class I, class L>
template <class ITERATOR>
inline typename TruncatedAbsoluteDifferenceFunction<T, I, L>::ValueType
TruncatedAbsoluteDifferenceFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   T value = begin[0];
   value -= begin[1];
   return abs(value) > parameter1_ ? parameter1_ * parameter2_ : abs(value) * parameter2_;
}

/// extension a value table encoding this function would have
///
/// \param i dimension
template <class T, class I, class L>
inline size_t
TruncatedAbsoluteDifferenceFunction<T, I, L>::shape(
   const IndexType i
) const {
   OPENGM_ASSERT(i < 2);
   return (i==0 ? numberOfLabels1_ : numberOfLabels2_);
}

// order (number of variables) of the function
template <class T, class I, class L>
inline size_t
TruncatedAbsoluteDifferenceFunction<T, I, L>::dimension() const {
   return 2;
}

/// number of entries a value table encoding this function would have (used for I/O)
template <class T, class I, class L>
inline size_t
TruncatedAbsoluteDifferenceFunction<T, I, L>::size() const {
   return numberOfLabels1_ * numberOfLabels2_;
}

template <class T, class I, class L>
inline size_t
FunctionSerialization<TruncatedAbsoluteDifferenceFunction<T, I, L> >::indexSequenceSize
(
   const TruncatedAbsoluteDifferenceFunction<T, I, L>& src
) {
   return 2;
}

template <class T, class I, class L>
inline size_t
FunctionSerialization<TruncatedAbsoluteDifferenceFunction<T, I, L> >::valueSequenceSize
(
   const TruncatedAbsoluteDifferenceFunction<T, I, L>& src
) {
   return 2;
}

template <class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<TruncatedAbsoluteDifferenceFunction<T, I, L> >::serialize
(
   const TruncatedAbsoluteDifferenceFunction<T, I, L>& src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   *indexOutIterator = src.shape(0);
   ++indexOutIterator;
   *indexOutIterator = src.shape(1);

   *valueOutIterator = src.parameter1_;
   ++valueOutIterator;
   *valueOutIterator = src.parameter2_;
}

template <class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<TruncatedAbsoluteDifferenceFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   TruncatedAbsoluteDifferenceFunction<T, I, L>& dst
) {
   const size_t shape1=*indexInIterator;
   ++ indexInIterator;
   const size_t shape2=*indexInIterator;
   const ValueType param1=*valueInIterator;
   ++valueInIterator;
   const ValueType param2=*valueInIterator;
   dst = TruncatedAbsoluteDifferenceFunction<T, I, L>(shape1,shape2,param1,param2);
}

} // namespace opengm

#endif // OPENGM_TRUNCATED_ABSOLUTE_DIFFERENCE_FUNCTION_HXX
