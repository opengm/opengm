#pragma once
#ifndef OPENGM_ABSOLUTE_DIFFERENCE_FUNCTION_HXX
#define OPENGM_ABSOLUTE_DIFFERENCE_FUNCTION_HXX

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// Absolute difference between two labels
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class AbsoluteDifferenceFunction
: public FunctionBase<AbsoluteDifferenceFunction<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;

   AbsoluteDifferenceFunction(const LabelType = 2, const LabelType = 2, const ValueType = 1);
   size_t shape(const IndexType) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> ValueType operator()(ITERATOR) const;

private:
   LabelType numberOfLabels1_;
   LabelType numberOfLabels2_;
   ValueType scale_;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template <class T, class I, class L>
struct FunctionRegistration< AbsoluteDifferenceFunction<T, I, L> >{
   /// Id  of the AbsoluteDifferenceFunction
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 2
   };
};

/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<AbsoluteDifferenceFunction<T, I, L> > {
public:
   typedef typename AbsoluteDifferenceFunction<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const AbsoluteDifferenceFunction<T, I, L>&);
   static size_t valueSequenceSize(const AbsoluteDifferenceFunction<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
      static void serialize(const AbsoluteDifferenceFunction<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR , class VALUE_INPUT_ITERATOR>
      static void deserialize(INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, AbsoluteDifferenceFunction<T, I, L>&);
};
/// \endcond

/// Constructor
///
/// \param numberOfLabels1 number of labels of the first variable
/// \param numberOfLabels2 number of labels of the second variable
///
template <class T, class I, class L>
inline
AbsoluteDifferenceFunction<T, I, L>::AbsoluteDifferenceFunction
(
   const LabelType numberOfLabels1, 
   const LabelType numberOfLabels2,
   const ValueType scale
)
:  numberOfLabels1_(numberOfLabels1), 
   numberOfLabels2_(numberOfLabels2),
   scale_(scale)
{}

template <class T, class I, class L>
template <class ITERATOR>
inline typename AbsoluteDifferenceFunction<T, I, L>::ValueType
AbsoluteDifferenceFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   ValueType value = begin[0];
   value -= begin[1];
   return scale_*abs(value);
}

/// extension a value table encoding this function would have
///
/// \param i dimension
template <class T, class I, class L>
inline size_t
AbsoluteDifferenceFunction<T, I, L>::shape
(
   const IndexType i
) const {
   OPENGM_ASSERT(i < 2);
   return (i==0 ? numberOfLabels1_ : numberOfLabels2_);
}

// order (number of variables) of the function
template <class T, class I, class L>
inline size_t
AbsoluteDifferenceFunction<T, I, L>::dimension() const {
   return 2;
}

/// number of entries a value table encoding this function would have (used for I/O)
template <class T, class I, class L>
inline size_t
AbsoluteDifferenceFunction<T, I, L>::size() const {
   return numberOfLabels1_ * numberOfLabels2_;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<AbsoluteDifferenceFunction<T, I, L> >::indexSequenceSize
(
   const AbsoluteDifferenceFunction<T, I, L>& src
) {
   return 2;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<AbsoluteDifferenceFunction<T, I, L> >::valueSequenceSize
(
   const AbsoluteDifferenceFunction<T, I, L>& src
) {
   return 1;
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<AbsoluteDifferenceFunction<T, I, L> >::serialize
(
   const AbsoluteDifferenceFunction<T, I, L>& src, 
   INDEX_OUTPUT_ITERATOR indexOutIterator, 
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   *indexOutIterator = src.shape(0);
   ++indexOutIterator;
   *indexOutIterator = src.shape(1);
   L l[]={0,1};
   *valueOutIterator = src(l);
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization< AbsoluteDifferenceFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator, 
   VALUE_INPUT_ITERATOR valueInIterator, 
   AbsoluteDifferenceFunction<T, I, L>& dst
) {
   const size_t shape0=*indexInIterator;
   ++ indexInIterator;
   dst=AbsoluteDifferenceFunction<T, I, L>(shape0, *indexInIterator,*valueInIterator);
}

} // namespace opengm

#endif // OPENGM_ABSOLUTE_DIFFERENCE_FUNCTION_HXX
