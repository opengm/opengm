#pragma once
#ifndef OPENGM_SQUARED_DIFFERENCE_FUNCTION_HXX
#define OPENGM_SQUARED_DIFFERENCE_FUNCTION_HXX

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// squared difference of the labels of two variables
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class SquaredDifferenceFunction
: public FunctionBase<SquaredDifferenceFunction<T, I, L>, T, I, L> 
{
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;

   SquaredDifferenceFunction(const LabelType = 2, const LabelType = 2, 
      const ValueType = 1);
   size_t shape(const IndexType) const;
   size_t size() const;
   size_t dimension() const;
   T weight() const;
   template<class ITERATOR> T operator()(ITERATOR) const;

private:
   size_t numberOfLabels1_;
   size_t numberOfLabels2_;
   T weight_;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template <class T, class I, class L>
struct FunctionRegistration<SquaredDifferenceFunction<T, I, L> > {
   enum ID { Id = opengm::FUNCTION_TYPE_ID_OFFSET + 4 };
};

/// FunctionSerialization
template <class T, class I, class L>
class FunctionSerialization<SquaredDifferenceFunction<T, I, L> > {
public:
   typedef typename SquaredDifferenceFunction<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const SquaredDifferenceFunction<T, I, L>&);
   static size_t valueSequenceSize(const SquaredDifferenceFunction<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
      static void serialize(const SquaredDifferenceFunction<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR , class VALUE_INPUT_ITERATOR>
      static void deserialize(INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, SquaredDifferenceFunction<T, I, L>&);
};
/// \endcond

/// constructor
/// \param numberOfLabels1 number of labels of the first variable
/// \param numberOfLabels2 number of labels of the second variable
/// \param weight weight
template <class T, class I, class L>
inline
SquaredDifferenceFunction<T, I, L>::SquaredDifferenceFunction
(
   const LabelType numberOfStates1, 
   const LabelType numberOfStates2, 
   const ValueType weight
)
:  numberOfLabels1_(numberOfStates1), 
   numberOfLabels2_(numberOfStates2), 
   weight_(weight)
{}

template <class T, class I, class L>
template <class ITERATOR>
inline T
SquaredDifferenceFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   T value = begin[0];
   value -= begin[1];
   return value*value*weight_;
}

/// extension a value table encoding this function would have
///
/// \param i dimension
template <class T, class I, class L>
inline size_t
SquaredDifferenceFunction<T, I, L>::shape(
   const IndexType i
) const {
   OPENGM_ASSERT(i < 2);
   return (i==0 ? numberOfLabels1_ : numberOfLabels2_);
}

template <class T, class I, class L>
inline T
SquaredDifferenceFunction<T, I, L>::weight() const {
   return weight_;
}

// order (number of variables) of the function
template <class T, class I, class L>
inline size_t
SquaredDifferenceFunction<T, I, L>::dimension() const {
   return 2;
}

/// number of entries a value table encoding this function would have (used for I/O)
template <class T, class I, class L>
inline size_t
SquaredDifferenceFunction<T, I, L>::size() const {
   return numberOfLabels1_ * numberOfLabels2_;
}

template <class T, class I, class L>
inline size_t
FunctionSerialization<SquaredDifferenceFunction<T, I, L> >::indexSequenceSize
(
   const SquaredDifferenceFunction<T, I, L>& src
) {
   return 2;
}

template <class T, class I, class L>
inline size_t
FunctionSerialization<SquaredDifferenceFunction<T, I, L> >::valueSequenceSize
(
   const SquaredDifferenceFunction<T, I, L>& src
) {
   return 1;
}

template <class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<SquaredDifferenceFunction<T, I, L> >::serialize
(
   const SquaredDifferenceFunction<T, I, L>& src, 
   INDEX_OUTPUT_ITERATOR indexOutIterator, 
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   *indexOutIterator = src.shape(0);
   ++indexOutIterator;
   *indexOutIterator = src.shape(1);
   *valueOutIterator =src.weight();
}

template <class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<SquaredDifferenceFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator, 
   VALUE_INPUT_ITERATOR valueInIterator, 
   SquaredDifferenceFunction<T, I, L>& dst
) {
   const size_t shape0=*indexInIterator;
   ++indexInIterator;
   dst = SquaredDifferenceFunction<T, I, L>(shape0, *indexInIterator, *valueInIterator);
}

} // namespace opengm

#endif // OPENGM_SQUARED_DIFFERENCE_FUNCTION_HXX
