#pragma once
#ifndef OPENGM_CONSTANT_FUNCTION_HXX
#define OPENGM_CONSTANT_FUNCTION_HXX

#include <cmath>
#include <algorithm>
#include <vector>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// Constant function
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class ConstantFunction
: public FunctionBase<ConstantFunction<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;

   ConstantFunction();
   template<class ITERATOR>
      ConstantFunction(ITERATOR, ITERATOR, const T);

   size_t shape(const IndexType) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> ValueType operator()(ITERATOR) const;

   // specializations
   bool isPotts() const { return true; }
   bool isGeneralizedPotts() const { return true; }
   ValueType min() const { return value_; }
   ValueType max() const { return value_; }
   ValueType sum() const { return value_ * static_cast<T>(size_); }
   ValueType product() const { 
      // TODO: improve this. get rid of std::pow and write a custom pow functor class for OpenGM
      const double x = static_cast<double>(value_); // possible loss of precision, e.g. if value_ is a long double
      const int n = static_cast<int>(size_);
      return static_cast<T>(std::pow(x, n)); // call of std::pow can otherwise be ambiguous, e.g. if x is int
   }
   MinMaxFunctor<ValueType> minMax() const { return MinMaxFunctor<T>(value_, value_); }

private:
   ValueType value_;
   std::vector<IndexType> shape_;
   size_t size_;

template<class > friend class FunctionSerialization;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template <class T, class I, class L>
struct FunctionRegistration< ConstantFunction<T, I, L> >{
   enum ID {
      Id=opengm::FUNCTION_TYPE_ID_OFFSET+8
   };
};

/// FunctionSerialization
template <class T, class I, class L>
class FunctionSerialization<ConstantFunction<T, I, L> >{
public:
   typedef typename ConstantFunction<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const ConstantFunction<T, I, L>&);
   static size_t valueSequenceSize(const ConstantFunction<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
      static void serialize(const ConstantFunction<T, I, L> &, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR );
   template<class INDEX_INPUT_ITERATOR , class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, ConstantFunction<T, I, L> &);
};
/// \endcond

template <class T, class I, class L>
template <class ITERATOR>
inline
ConstantFunction<T, I, L>::ConstantFunction
(
   ITERATOR shapeBegin,
   ITERATOR shapeEnd,
   const T value
)
:  value_(value),
   shape_(shapeBegin, shapeEnd),
   size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<ITERATOR>::value_type >()))
{}

template <class T, class I, class L>
inline
ConstantFunction<T, I, L>::ConstantFunction()
: value_(0), shape_(), size_(0)
{}

template <class T, class I, class L>
template <class ITERATOR>
inline typename ConstantFunction<T, I, L>::ValueType
ConstantFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   return value_;
}

/// extension a value table encoding this function would have
///
/// \param i dimension
template <class T, class I, class L>
inline size_t
ConstantFunction<T, I, L>::shape (
   const IndexType i
) const {
   OPENGM_ASSERT(i < shape_.size());
   return shape_[i];
}

// order (number of variables) of the function
template <class T, class I, class L>
inline size_t
ConstantFunction<T, I, L>::dimension() const {
   return shape_.size();
}

/// number of entries a value table encoding this function would have (used for I/O)
template <class T, class I, class L>
inline size_t
ConstantFunction<T, I, L>::size() const {
   return size_;
}

template <class T, class I, class L>
inline size_t
FunctionSerialization<ConstantFunction<T, I, L> >::indexSequenceSize
(
   const ConstantFunction<T, I, L>& src
) {
   return src.dimension() + 1;
}

template <class T, class I, class L>
inline size_t
FunctionSerialization<ConstantFunction<T, I, L> >::valueSequenceSize
(
   const ConstantFunction<T, I, L>& src
) {
   return 1;
}

template <class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<ConstantFunction<T, I, L> >::serialize
(
   const ConstantFunction<T, I, L>& src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   *valueOutIterator = src.value_;
   *indexOutIterator = src.dimension();
   for(size_t i=0; i<src.dimension(); ++i) {
      ++indexOutIterator;
      *indexOutIterator = src.shape(i);
   }
}

template <class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization< ConstantFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   ConstantFunction<T, I, L>& dst
) {
   dst.value_ = *valueInIterator;
   size_t dimension = *indexInIterator;
   dst.shape_.resize(dimension);
   for(size_t i=0; i<dimension; ++i) {
      ++indexInIterator;
      dst.shape_[i]=*indexInIterator;
   }
   dst.size_=std::accumulate(dst.shape_.begin(), dst.shape_.end(), 1, std::multiplies<size_t>());
}

} // namespace opengm

#endif // OPENGM_CONSTANT_FUNCTION_HXX
