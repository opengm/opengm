#pragma once
#ifndef OPENGM_EXPLICIT_FUNCTION_HXX
#define OPENGM_EXPLICIT_FUNCTION_HXX

#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// Function encoded as a dense multi-dimensional array, marray::Marray
///
/// \ingroup functions
template<class T, class I=size_t, class L=size_t>
class ExplicitFunction
:  public marray::Marray<T>,
   public FunctionBase<ExplicitFunction<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef L LabelType;
   typedef I IndexType;
      
   ExplicitFunction()
   : marray::Marray<T>()
   {}

   /// construct a constant explicit function of order 0
   ExplicitFunction(const T& value)
   : marray::Marray<T>(value)
   {}

   ExplicitFunction(const ExplicitFunction& other)
   : marray::Marray<T>(other)
   {}

   ExplicitFunction& operator=(const ExplicitFunction& other)
   { 
      marray::Marray<T>::operator=(other); 
      return *this;
   }

   /// construct a function encoded by a value table (whose entries are initialized as 0)
   ///
   /// Example: A function depending on two variables with 3 and 4 labels, respectively.
   /// \code
   /// size_t shape[] = {3, 4};
   /// ExplicitFunction f(shape, shape + 2};
   /// \endcode
   ///
   template <class SHAPE_ITERATOR>
   ExplicitFunction(SHAPE_ITERATOR shapeBegin, SHAPE_ITERATOR shapeEnd)
   : marray::Marray<T>(shapeBegin, shapeEnd)
   {}

   /// construct a function encoded by a value table (whose entries are initialized with the same value)
   template <class SHAPE_ITERATOR>
   ExplicitFunction(SHAPE_ITERATOR shapeBegin, SHAPE_ITERATOR shapeEnd, const T & value)
   : marray::Marray<T>(shapeBegin, shapeEnd, value)
   {}
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class T, class I, class L>
struct FunctionRegistration< ExplicitFunction<T, I, L> >{
   enum ID {
      Id=opengm::FUNCTION_TYPE_ID_OFFSET
   };
};

/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization< ExplicitFunction<T, I, L> >{
public:
   typedef typename ExplicitFunction<T, I, L>::value_type ValueType;

   static size_t indexSequenceSize(const ExplicitFunction<T, I, L> &);
   static size_t valueSequenceSize(const ExplicitFunction<T, I, L> &);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
      static void serialize(const ExplicitFunction<T, I, L>  &, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR );
   template<class INDEX_INPUT_ITERATOR , class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, ExplicitFunction<T, I, L>  &);
};
/// \endcond

template<class T, class I, class L>
inline size_t FunctionSerialization<ExplicitFunction<T, I, L> >::indexSequenceSize
(
   const ExplicitFunction<T, I, L> & src
) {
   return src.dimension() +1;
}

template<class T, class I, class L>
inline size_t FunctionSerialization<ExplicitFunction<T, I, L> >::valueSequenceSize
(
   const ExplicitFunction<T, I, L> & src
) {
   return src.size();
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
void FunctionSerialization< ExplicitFunction<T, I, L> >::serialize
(
   const ExplicitFunction<T, I, L> & src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   if(src.dimension()==0) {
      *indexOutIterator=0;
      *valueOutIterator=src(0);
   }
   else{
      *indexOutIterator=src.dimension();
      ++indexOutIterator;
      for(size_t i=0;i<src.dimension();++i) {
         *indexOutIterator=src.shape(i);
         ++indexOutIterator;
      }
      for(size_t i=0;i<src.size();++i) {
         *valueOutIterator=src(i);
         ++valueOutIterator;
      }
   }
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
void FunctionSerialization<ExplicitFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexOutIterator,
   VALUE_INPUT_ITERATOR valueOutIterator,
   ExplicitFunction<T, I, L> & dst
) {
   if(*indexOutIterator==0) {
      dst.assign();
      dst=ExplicitFunction<T, I, L>(*valueOutIterator);
   }
   else{
      const size_t dim=*indexOutIterator;
      std::vector<size_t> shape(dim);
      ++indexOutIterator;
      for(size_t i=0;i<dim;++i) {
         shape[i]=*indexOutIterator;
         ++indexOutIterator;
      }
      dst.assign();
      dst.resize(shape.begin(), shape.end() );
      for(size_t i=0;i<dst.size();++i) {
         dst(i)=*valueOutIterator;
         ++valueOutIterator;
      }
   }
}

} // namespace opengm

#endif // OPENGM_EXPLICIT_FUNCTION_HXX
