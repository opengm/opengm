#pragma once
#ifndef OPENGM_SPARSE_FUNCTION
#define OPENGM_SPARSE_FUNCTION

#include "opengm/datastructures/sparsemarray/sparsemarray.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class T,class I,class L,class C>
struct FunctionRegistration<opengm::SparseFunction<T,I,L,C> > {
   enum ID { Id = opengm::FUNCTION_TYPE_ID_OFFSET + 1 };
};

/// FunctionSerialization
///
/// \ingroup functions
template<class T,class I,class L,class C>
class FunctionSerialization< opengm::SparseFunction<T,I,L,C> > {
public:
   typedef typename opengm::SparseFunction<T,I,L,C>::ValueType ValueType;

   static size_t indexSequenceSize(const opengm::SparseFunction<T,I,L,C>&);
   static size_t valueSequenceSize(const opengm::SparseFunction<T,I,L,C>&);
   template<class INDEX_OUTPUT_ITERATOR,class VALUE_OUTPUT_ITERATOR >
      static void serialize(const opengm::SparseFunction<T,I,L,C>&, INDEX_OUTPUT_ITERATOR,VALUE_OUTPUT_ITERATOR );
   template<class INDEX_INPUT_ITERATOR ,class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR,VALUE_INPUT_ITERATOR, opengm::SparseFunction<T,I,L,C>&);
};
/// \endcond

template<class T,class I,class L,class C>
inline size_t
FunctionSerialization<opengm::SparseFunction<T,I,L,C> >::indexSequenceSize
(
   const opengm::SparseFunction<T,I,L,C>& src
) {
   return 1 + src.dimension()+1+src.container().size();
}

template<class T,class I,class L,class C>
inline size_t
FunctionSerialization<opengm::SparseFunction<T,I,L,C> >::valueSequenceSize
(
   const opengm::SparseFunction<T,I,L,C>& src
) {

   return src.container().size()+1;
}

template<class T,class I,class L,class C>
template<class INDEX_OUTPUT_ITERATOR,class VALUE_OUTPUT_ITERATOR >
void
FunctionSerialization<opengm::SparseFunction<T,I,L,C> >::serialize
(
   const opengm::SparseFunction<T,I,L,C>& src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   *indexOutIterator=src.dimension();
   ++indexOutIterator;
   for(size_t i=0;i<src.dimension();++i) {
      *indexOutIterator=src.shape(i);
      ++indexOutIterator;
   }
   //save the default value
   *valueOutIterator=src.defaultValue();
   ++valueOutIterator;
   //save how many not default values are in the SparseFunction
   *indexOutIterator=src.container().size();
   ++indexOutIterator;
   typedef typename opengm::SparseFunction<T,I,L,C>::ContainerType::const_iterator IterType;
   IterType srcIter=src.container().begin();
   for(size_t i=0;i<src.container().size();++i) {
      *indexOutIterator=srcIter->first;
      *valueOutIterator=srcIter->second;
      ++valueOutIterator;
      ++indexOutIterator;
      ++srcIter;
   }
}

template<class T,class I,class L,class C>
template<class INDEX_INPUT_ITERATOR,class VALUE_INPUT_ITERATOR >
void
FunctionSerialization<opengm::SparseFunction<T,I,L,C> >::deserialize
(
   INDEX_INPUT_ITERATOR indexOutIterator,
   VALUE_INPUT_ITERATOR valueOutIterator,
   opengm::SparseFunction<T,I,L,C>& dst
) {

   const size_t dim=*indexOutIterator;
   std::vector<size_t> shape(dim);
   ++indexOutIterator;
   for(size_t i=0;i<dim;++i) {
      shape[i]=*indexOutIterator;
      ++indexOutIterator;
   }
   dst = opengm::SparseFunction<T,I,L,C>(shape.begin(),shape.end(),*valueOutIterator);
   ++valueOutIterator;
   const size_t nNonDefault=*indexOutIterator;
   ++indexOutIterator;
   typedef typename opengm::SparseFunction<T,I,L,C>::KeyValPairType KeyValPairType;
   for(size_t i=0;i<nNonDefault;++i) {

      dst.container().insert(KeyValPairType(*indexOutIterator,*valueOutIterator));
      ++valueOutIterator;
      ++indexOutIterator;
   }
   
}

} // namepsace opengm

#endif // OPENGM_SPARSE_FUNCTION

