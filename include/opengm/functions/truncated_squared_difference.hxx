#pragma once
#ifndef OPENGM_TRUNCATED_SQUARED_DIFFERENCE_FUNCTION_HXX
#define OPENGM_TRUNCATED_SQUARED_DIFFERENCE_FUNCTION_HXX

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

///  truncated squared difference of the labels of two variables
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class TruncatedSquaredDifferenceFunction
: public FunctionBase<TruncatedSquaredDifferenceFunction<T, I, L>, T, I, L> {
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;

   TruncatedSquaredDifferenceFunction(const LabelType = 2, const LabelType = 2, 
      const ValueType = ValueType(), const ValueType = ValueType());
   size_t shape(const IndexType) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> T operator()(ITERATOR) const;

private:
   size_t numberOfLabels1_;
   size_t numberOfLabels2_;
   ValueType parameter1_;
   ValueType parameter2_;

friend class FunctionSerialization<TruncatedSquaredDifferenceFunction<T, I, L> > ;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template <class T, class I, class L>
struct FunctionRegistration< TruncatedSquaredDifferenceFunction<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 5
   };
};

/// FunctionSerialization
template <class T, class I, class L>
class FunctionSerialization<TruncatedSquaredDifferenceFunction<T, I, L> > {
public:
   typedef typename TruncatedSquaredDifferenceFunction<T, I, L>::ValueType ValueType;
   static size_t indexSequenceSize(const TruncatedSquaredDifferenceFunction<T, I, L>&);
   static size_t valueSequenceSize(const TruncatedSquaredDifferenceFunction<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR,class VALUE_OUTPUT_ITERATOR >
      static void serialize(const TruncatedSquaredDifferenceFunction<T, I, L>&, INDEX_OUTPUT_ITERATOR,VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR ,class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR,VALUE_INPUT_ITERATOR,TruncatedSquaredDifferenceFunction<T, I, L>&);
};
/// \endcond

/// Constructor
/// \param numberOfLabels1 number of labels of the first variable
/// \param numberOfLabels2 number of labels of the second variable
template <class T, class I, class L>
inline
TruncatedSquaredDifferenceFunction<T, I, L>::TruncatedSquaredDifferenceFunction
(
   const LabelType numberOfLabels1,
   const LabelType numberOfLabels2,
   const ValueType truncation,
   const ValueType weight
)
:  numberOfLabels1_(numberOfLabels1),
   numberOfLabels2_(numberOfLabels2),
   parameter1_(truncation),
   parameter2_(weight)
{}

template <class T, class I, class L>
template <class ITERATOR>
inline typename TruncatedSquaredDifferenceFunction<T, I, L>::ValueType
TruncatedSquaredDifferenceFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   ValueType value = begin[0];
   value -= begin[1];
   return value * value > parameter1_ ? parameter1_* parameter2_ : value * value * parameter2_;
}

/// extension a value table encoding this function would have
///
/// \param i dimension
template <class T, class I, class L>
inline size_t
TruncatedSquaredDifferenceFunction<T, I, L>::shape(
   const IndexType i
) const {
   OPENGM_ASSERT(i < 2);
   return i==0 ? numberOfLabels1_ : numberOfLabels2_;
}

// order (number of variables) of the function
template <class T, class I, class L>
inline size_t
TruncatedSquaredDifferenceFunction<T, I, L>::dimension() const {
   return 2;
}

/// number of entries a value table encoding this function would have (used for I/O)
template <class T, class I, class L>
inline size_t
TruncatedSquaredDifferenceFunction<T, I, L>::size() const {
   return numberOfLabels1_ * numberOfLabels2_;
}

template <class T, class I, class L>
inline size_t
FunctionSerialization<TruncatedSquaredDifferenceFunction<T, I, L> >::indexSequenceSize
(
   const TruncatedSquaredDifferenceFunction<T, I, L>& src
) {
   return 2;
}

template <class T, class I, class L>
inline size_t
FunctionSerialization<TruncatedSquaredDifferenceFunction<T, I, L> >::valueSequenceSize
(
   const TruncatedSquaredDifferenceFunction<T, I, L>& src
) {
   return 2;
}

template <class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<TruncatedSquaredDifferenceFunction<T, I, L> >::serialize
(
   const TruncatedSquaredDifferenceFunction<T, I, L>& src,
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
FunctionSerialization< TruncatedSquaredDifferenceFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   TruncatedSquaredDifferenceFunction<T, I, L>& dst
) {
   const size_t shape1=*indexInIterator;
   ++indexInIterator;
   const size_t shape2=*indexInIterator;
   const ValueType param1=*valueInIterator;
   ++valueInIterator;
   const ValueType param2=*valueInIterator;
   dst=TruncatedSquaredDifferenceFunction<T, I, L>(shape1,shape2,param1,param2);
}

} // namespace opengm

#endif // OPENGM_TRUNCATED_SQUARED_DIFFERENCE_FUNCTION_HXX
