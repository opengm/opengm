#pragma once
#ifndef OPENGM_POTTS_N_FUNCTION_HXX
#define OPENGM_POTTS_N_FUNCTION_HXX

#include <algorithm>
#include <vector>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {
   
/// \brief Potts function in N variables
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class PottsNFunction
: public FunctionBase<PottsNFunction<T, I, L>, T, I, L> {
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;
   typedef T value_type;

   PottsNFunction();
   template<class ITERATOR> PottsNFunction(ITERATOR, ITERATOR, const T, const T);
   LabelType shape(const size_t) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> ValueType operator()(ITERATOR) const;
   bool isPotts() const
      { return true; }
   bool isGeneralizedPotts() const
      { return true; }

private:
   std::vector<LabelType> shape_;
   size_t size_;
   ValueType valueEqual_;
   ValueType valueNotEqual_;

friend class FunctionSerialization<PottsNFunction<T, I, L> >;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class T, class I, class L>
struct FunctionRegistration<PottsNFunction<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 7
   };
};

/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<PottsNFunction<T, I, L> > {
public:
   typedef typename PottsNFunction<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const PottsNFunction<T, I, L> &);
   static size_t valueSequenceSize(const PottsNFunction<T, I, L> &);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
      static void serialize(const PottsNFunction<T, I, L>  &, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR );
   template<class INDEX_INPUT_ITERATOR , class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, PottsNFunction<T, I, L>  &);
};
/// \endcond

template<class T, class I, class L>
template <class ITERATOR>
inline
PottsNFunction<T, I, L>::PottsNFunction
(
   ITERATOR shapeBegin,
   ITERATOR shapeEnd,
   const T valueEqual,
   const T valueNotEqual
)
:  shape_(shapeBegin, shapeEnd),
   size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<ITERATOR>::value_type >())),
   valueEqual_(valueEqual),
   valueNotEqual_(valueNotEqual)
{
   OPENGM_ASSERT(shape_.size() != 0);
}

template<class T, class I, class L>
inline
PottsNFunction<T, I, L>::PottsNFunction()
:  shape_(),
   size_(0),
   valueEqual_(T()),
   valueNotEqual_(T())
{}

template<class T, class I, class L>
template <class ITERATOR>
inline T
PottsNFunction<T, I, L>::operator ()
(
   ITERATOR begin
) const {
   size_t tmp = static_cast<size_t> (*begin);
   for(size_t i=0;i<shape_.size(); ++i) {
      if(static_cast<size_t> (begin[i]) != tmp) {
         return valueNotEqual_;
      }
   }
   return valueEqual_;
}

template<class T, class I, class L>
inline typename PottsNFunction<T, I, L>::LabelType
PottsNFunction<T, I, L>::shape
(
   const size_t i
) const {
   OPENGM_ASSERT(i < shape_.size());
   return shape_[i];
}

template<class T, class I, class L>
inline size_t
PottsNFunction<T, I, L>::dimension() const {
   return shape_.size();
}

template<class T, class I, class L>
inline size_t
PottsNFunction<T, I, L>::size() const {
   return size_;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<PottsNFunction<T, I, L> >::indexSequenceSize
(
   const PottsNFunction<T, I, L> & src
) {
   return src.dimension()+1;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<PottsNFunction<T, I, L> >::valueSequenceSize
(
   const PottsNFunction<T, I, L> & src
) {
   return 2;
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<PottsNFunction<T, I, L> >::serialize
(
   const PottsNFunction<T, I, L> & src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   const size_t dim = src.dimension();
   *indexOutIterator = dim;
   for(size_t i=0;i<dim;++i) {
      ++indexOutIterator;
      *indexOutIterator = src.shape(i);
   }
   *valueOutIterator = src.valueEqual_;
   ++valueOutIterator;
   *valueOutIterator = src.valueNotEqual_;
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<PottsNFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   PottsNFunction<T, I, L> & dst
) {
   const size_t dim = *indexInIterator;
   ++indexInIterator;
   std::vector<size_t> shape(dim);
   for(size_t i=0; i<dim; ++i) {
      shape[i] = *indexInIterator;
      ++indexInIterator;
   }

   const ValueType param1 = *valueInIterator;
   ++valueInIterator;
   const ValueType param2 = *valueInIterator;
   dst = PottsNFunction<T, I, L>(shape.begin(), shape.end(), param1, param2);
}

} // namespace opengm

#endif // #ifndef OPENGM_POTTS_N_FUNCTION_HXX
