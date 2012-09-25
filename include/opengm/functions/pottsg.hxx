#pragma once
#ifndef OPENGM_POTTS_G_FUNCTION_HXX
#define OPENGM_POTTS_G_FUNCTION_HXX

#include <algorithm>
#include <vector>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// \brief Generalized Potts Function
///
/// A generalized Potts function is a function that is invariant under all
/// permutations of labels, e.g. f(1, 1, 3, 4) = f(3, 3, 4, 1).
///
/// Its purpose is to assign different values to different partitions of the
/// set of input variables, regardless of which labels are used to describe
/// this partition.
///
/// It generalizes the Potts function that distinguishes only between equal
/// and unequal labels.
///
/// The memory required to store a generalized Potts function depends
/// only on the order of the function, not on the number of labels.
/// Due to the trasitivity of the equality relation, the number of all
/// partitions of a set of D elements is smaller than 2^(D*(D+1)/2).
/// The exact number is given by the Bell numbers B_D, e.g.
/// B_1=1, B_2=2, B_3=5, B_4=15, B_5=52, B_6=203
///
/// \ingroup functions
template<class T, class I=size_t, class L=size_t>
class PottsGFunction
: public FunctionBase<PottsGFunction<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;

   PottsGFunction();
   template<class ITERATOR> PottsGFunction(ITERATOR, ITERATOR);
   template<class ITERATOR, class ITERATOR2> PottsGFunction(ITERATOR, ITERATOR, ITERATOR2);
   LabelType shape(const size_t) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> ValueType operator()(ITERATOR) const;
   bool isPotts() const;
   bool isGeneralizedPotts() const;
   template<class LABELITERATOR> void setByLabel(LABELITERATOR, T);
   void setByPartition(size_t, T);

   static const size_t BellNumbers_[9];
   static const size_t MaximalOrder_ = 4; // maximal order currently supported

private:
   std::vector<LabelType> shape_;
   std::vector<ValueType> values_;
   size_t size_;

friend class FunctionSerialization<PottsGFunction<T, I, L> > ;
};

template<class T, class I, class L>
const size_t PottsGFunction<T, I, L>::BellNumbers_[9] = {1, 1, 2, 5, 15, 52, 203, 877, 4140};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class T, class I, class L>
struct FunctionRegistration<PottsGFunction<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 11
   };
};

/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<PottsGFunction<T, I, L> > {
public:
   typedef typename PottsGFunction<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const PottsGFunction<T, I, L> &);
   static size_t valueSequenceSize(const PottsGFunction<T, I, L> &);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
   static void serialize(const PottsGFunction<T, I, L>  &, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR );
   template<class INDEX_INPUT_ITERATOR , class VALUE_INPUT_ITERATOR>
   static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, PottsGFunction<T, I, L>  &);
};
/// \endcond

template<class T, class I, class L>
template<class ITERATOR>
inline
PottsGFunction<T, I, L>::PottsGFunction
(
   ITERATOR shapeBegin,
   ITERATOR shapeEnd
)
:  shape_(shapeBegin, shapeEnd),
   size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<ITERATOR>::value_type >()))
{
   values_.resize(BellNumbers_[shape_.size()], 0);
   OPENGM_ASSERT(shape_.size() <= MaximalOrder_);
   OPENGM_ASSERT(BellNumbers_[shape_.size()] == values_.size());
}

template<class T, class I, class L>
template<class ITERATOR, class ITERATOR2>
inline
PottsGFunction<T, I, L>::PottsGFunction
(
   ITERATOR shapeBegin,
   ITERATOR shapeEnd,
   ITERATOR2 valuesBegin
)
:  shape_(shapeBegin, shapeEnd),
   size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<ITERATOR>::value_type >()))
{
   values_.resize(BellNumbers_[shape_.size()]);
   for(size_t i=0; i<values_.size(); ++i) {
      values_[i] = *valuesBegin;
      ++valuesBegin;
   }
   OPENGM_ASSERT(shape_.size() <= MaximalOrder_);
   OPENGM_ASSERT(BellNumbers_[shape_.size()] == values_.size());
}

template<class T, class I, class L>
inline
PottsGFunction<T, I, L>::PottsGFunction()
:  shape_(),
   size_(0)
{}

template<class T, class I, class L>
template<class ITERATOR>
inline T
PottsGFunction<T, I, L>::operator ()  (ITERATOR begin) const
{
   size_t indexer = 0;
   // Memory requirement for indexer
   // order=2  1bit
   // order=3  3bit
   // order=4  6bit
   // order=5  10bit
   // order=6  11bit

   size_t bit = 1;
   for(size_t i=1; i<shape_.size(); ++i) {
      for(size_t j=0; j<i; ++j) {
         if(*(begin+i)==*(begin+j)) {
            indexer += bit;
         }
         bit *= 2;
      }
   }

   ValueType value;
   switch (indexer) {
   case 0: value = values_[0]; break; //x_1!=x_2 && x_0!=x_2 && x_0!=x_1
   case 1: value = values_[1]; break; //x_1!=x_2 && x_0!=x_2 && x_0==x_1
   case 2: value = values_[2]; break; //x_1!=x_2 && x_0==x_2 && x_0!=x_1
      //case 3: ERROR                   x_1!=x_2 && x_0==x_2 && x_0==x_1
   case 4: value = values_[3]; break; //x_1==x_2 && x_0!=x_2 && x_0!=x_1
      //case 5: ERROR                   x_1==x_2 && x_0!=x_2 && x_0==x_1
      //case 6: ERROR                   x_1==x_2 && x_0==x_2 && x_0!=x_1
   case 7: value = values_[4]; break; //x_1==x_2 && x_0==x_2 && x_0==x_1

   case 8: value = values_[5]; break; // x_2!=x_3  && x_1!=x_3  && x_0==x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0!=x_1
   case 12: value = values_[6]; break; // x_2!=x_3  && x_1!=x_3  && x_0==x_3  &&  x_1==x_2 && x_0!=x_2 && x_0!=x_1
   case 16: value = values_[7]; break; // x_2!=x_3  && x_1==x_3  && x_0!=x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0!=x_1
   case 18: value = values_[8]; break; // x_2!=x_3  && x_1==x_3  && x_0!=x_3  &&  x_1!=x_2 && x_0==x_2 && x_0!=x_1
   case 25: value = values_[9]; break; // x_2!=x_3  && x_1==x_3  && x_0==x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0==x_1
   case 32: value = values_[10]; break; // x_2==x_3  && x_1!=x_3  && x_0!=x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0!=x_1
   case 33: value = values_[11]; break; // x_2==x_3  && x_1!=x_3  && x_0!=x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0==x_1
   case 42: value = values_[12]; break; // x_2==x_3  && x_1!=x_3  && x_0==x_3  &&  x_1!=x_2 && x_0==x_2 && x_0!=x_1
   case 52: value = values_[13]; break; // x_2==x_3  && x_1==x_3  && x_0==x_3  &&  x_1==x_2 && x_0!=x_2 && x_0!=x_1
   case 63: value = values_[14]; break; // x_2==x_3  && x_1==x_3  && x_0==x_3  &&  x_1==x_2 && x_0==x_2 && x_0==x_1
   default:  value = 0;
   }
   return value;
}

template<class T, class I, class L>
template<class LABELITERATOR>
void PottsGFunction<T, I, L>::setByLabel(LABELITERATOR it, T value)
{
   size_t indexer = 0;
   size_t bit = 1;
   for(size_t i=1; i<shape_.size(); ++i) {
      for(size_t j=0; j<i; ++j) {
         if(*(it+i)==*(it+j)) indexer += bit;
         bit *= 2;
      }
   }
   setByPartition(indexer, value);
}

template<class T, class I, class L>
void PottsGFunction<T, I, L>::setByPartition(size_t partition, T value)
{
   switch(partition) {
   case 0:  values_[0] = value; break; //x_1!=x_2 && x_0!=x_2 && x_i!=x_j
   case 1:  values_[1] = value; break; //x_1!=x_2 && x_0!=x_2 && x_0==x_1
   case 2:  values_[2] = value; break; //x_1!=x_2 && x_0!=x_2 && x_0==x_2
      //case 3: ERROR                   x_1!=x_2 && x_0==x_2 && x_0==x_1
   case 4:  values_[3] = value; break; //x_1==x_2 && x_0!=x_2 && x_0!=x_1
      //case 5: ERROR                   x_1==x_2 && x_0!=x_2 && x_0==x_1
      //case 6: ERROR                   x_1==x_2 && x_0==x_2 && x_0!=x_1
   case 7:  values_[4] = value; break; //x_1==x_2 && x_0==x_2 && x_0==x_1
   case 8:  values_[5] = value; break; // x_2!=x_3  && x_1!=x_3  && x_0==x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0!=x_1
   case 12: values_[6] = value; break; // x_2!=x_3  && x_1!=x_3  && x_0==x_3  &&  x_1==x_2 && x_0!=x_2 && x_0!=x_1
   case 16: values_[7] = value; break; // x_2!=x_3  && x_1==x_3  && x_0!=x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0!=x_1
   case 18: values_[8] = value; break; // x_2!=x_3  && x_1==x_3  && x_0!=x_3  &&  x_1!=x_2 && x_0==x_2 && x_0!=x_1
   case 25: values_[9] = value; break; // x_2!=x_3  && x_1==x_3  && x_0==x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0==x_1
   case 32: values_[10] = value; break; // x_2==x_3  && x_1!=x_3  && x_0!=x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0!=x_1
   case 33: values_[11] = value; break; // x_2==x_3  && x_1!=x_3  && x_0!=x_3  &&  x_1!=x_2 && x_0!=x_2 && x_0==x_1
   case 42: values_[12] = value; break; // x_2==x_3  && x_1!=x_3  && x_0==x_3  &&  x_1!=x_2 && x_0==x_2 && x_0!=x_1
   case 52: values_[13] = value; break; // x_2==x_3  && x_1==x_3  && x_0!=x_3  &&  x_1==x_2 && x_0!=x_2 && x_0!=x_1
   case 63: values_[14] = value; break; // x_2==x_3  && x_1==x_3  && x_0==x_3  &&  x_1==x_2 && x_0==x_2 && x_0==x_1
   default:  OPENGM_ASSERT(false);
   }
}

template<class T, class I, class L>
inline typename PottsGFunction<T, I, L>::LabelType
PottsGFunction<T, I, L>::shape
(
   const size_t i
) const
{
   OPENGM_ASSERT(i < shape_.size());
   return shape_[i];
}

template<class T, class I, class L>
inline size_t
PottsGFunction<T, I, L>::dimension() const
{
   return shape_.size();
}

template<class T, class I, class L>
inline size_t
PottsGFunction<T, I, L>::size() const {
   return size_;
}

template<class T, class I, class L>
inline bool
PottsGFunction<T, I, L>::isPotts() const
{
   bool t = true;
   for(size_t i=1; i<values_.size()-1; ++i)
      t &= values_[0] == values_[i];
   return t;
}

template<class T, class I, class L>
inline bool
PottsGFunction<T, I, L>::isGeneralizedPotts() const
{
   return true;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<PottsGFunction<T, I, L> >::indexSequenceSize
(
   const PottsGFunction<T, I, L> & src
) {
   return src.dimension()+1;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<PottsGFunction<T, I, L> >::valueSequenceSize
(
   const PottsGFunction<T, I, L> & src
) {
   return src.values_.size();
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<PottsGFunction<T, I, L> >::serialize
(
   const PottsGFunction<T, I, L> & src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   const size_t dim = src.dimension();
   *indexOutIterator = dim;
   ++indexOutIterator;
   for(size_t i=0; i<dim; ++i) {
      *indexOutIterator = src.shape(i);
      ++indexOutIterator;
   }
   for(size_t i=0; i<src.values_.size(); ++i) {
      *valueOutIterator = src.values_[i];
      ++valueOutIterator;
   }
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<PottsGFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   PottsGFunction<T, I, L> & dst
) {
   const size_t dim=*indexInIterator;
   ++indexInIterator;
   std::vector<size_t> shape(dim);
   std::vector<T>      values(dst.BellNumbers_[dim]);
   for(size_t i=0; i<dim; ++i) {
      shape[i]=*indexInIterator;
      ++indexInIterator;
   }
   for(size_t i=0; i<values.size(); ++i) {
      values[i] = *valueInIterator;
      ++valueInIterator;
   }
   dst = PottsGFunction<T, I, L>(shape.begin(), shape.end(), values.begin());
}

} // namespace opengm

#endif // #ifndef OPENGM_POTTS_G_FUNCTION_HXX
