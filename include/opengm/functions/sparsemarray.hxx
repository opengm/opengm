#pragma once
#ifndef OPENGM_SPARSEMARRAY_FUNCTION_HXX
#define OPENGM_SPARSEMARRAY_FUNCTION_HXX

#include "opengm/datastructures/sparsemarray/sparsemarray.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class T>
struct FunctionRegistration<opengm::SparseMarray<T> > {
   enum ID { Id = opengm::FUNCTION_TYPE_ID_OFFSET + 1 };
};

/// FunctionSerialization
template<class T>
class FunctionSerialization< opengm::SparseMarray<T> > {
public:
   typedef typename opengm::SparseMarray<T>::ValueType ValueType;

   static size_t indexSequenceSize(const opengm::SparseMarray<T>&);
   static size_t valueSequenceSize(const opengm::SparseMarray<T>&);
   template<class INDEX_OUTPUT_ITERATOR,class VALUE_OUTPUT_ITERATOR >
      static void serialize(const opengm::SparseMarray<T>&, INDEX_OUTPUT_ITERATOR,VALUE_OUTPUT_ITERATOR );
   template<class INDEX_INPUT_ITERATOR ,class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR,VALUE_INPUT_ITERATOR, opengm::SparseMarray<T>&);
};
/// \endcond

template<class T>
inline size_t
FunctionSerialization<opengm::SparseMarray<T> >::indexSequenceSize
(
   const opengm::SparseMarray<T>& src
) {
   if(src.dimension() == 0) {
      return 1;
   }
   else {
      return 1 + src.dimension()+1+src.getAssociativeContainer().size();
   }
}

template<class T>
inline size_t
FunctionSerialization<opengm::SparseMarray<T> >::valueSequenceSize
(
   const opengm::SparseMarray<T>& src
) {
   if(src.dimension()==0) {
      return 1;
   }
   else{
      return src.getAssociativeContainer().size()+1;
   }
}

template<class T>
template<class INDEX_OUTPUT_ITERATOR,class VALUE_OUTPUT_ITERATOR >
void
FunctionSerialization<opengm::SparseMarray<T> >::serialize
(
   const opengm::SparseMarray<T>& src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   if(src.dimension()==0) {
      *indexOutIterator=0;
      *valueOutIterator=src(0);
   }
   else{
      for(size_t i=0;i<src.dimension();++i) {
         *indexOutIterator=src.shape(i);
         ++indexOutIterator;
      }
      //save the default value
      *valueOutIterator=src.getDefaultValue();
      ++valueOutIterator;
      //save how many not default values are in the sparsemarray
       *indexOutIterator=src.getAssociativeContainer().size();
      //map begin
      typedef typename opengm::SparseMarray<T>::const_assigned_assoziative_iterator IterType;
      IterType srcIter=src.assigned_assoziative_begin();

      for(size_t i=0;i<src.getAssociativeContainer().size();++i) {
         *indexOutIterator=srcIter->first;
         *valueOutIterator=srcIter->second;
         ++valueOutIterator;
         ++indexOutIterator;
         ++srcIter;
      }
   }
}

template<class T>
template<class INDEX_INPUT_ITERATOR,class VALUE_INPUT_ITERATOR >
void
FunctionSerialization<opengm::SparseMarray<T> >::deserialize
(
   INDEX_INPUT_ITERATOR indexOutIterator,
   VALUE_INPUT_ITERATOR valueOutIterator,
   opengm::SparseMarray<T>& dst
) {
   if(*indexOutIterator==0) {
      size_t shape [] ={0};
      //empty shape to get a scalar array
      dst.init(shape,shape,*valueOutIterator);
   }
   else{
      const size_t dim=*indexOutIterator;
      std::vector<size_t> shape(dim);
      ++indexOutIterator;
      for(size_t i=0;i<dim;++i) {
         shape[i]=*indexOutIterator;
         ++indexOutIterator;
      }
      dst.init(shape,shape,*valueOutIterator);
      const size_t nNonDefault=*indexOutIterator;
      for(size_t i=0;i<nNonDefault;++i) {
         dst.reference(*indexOutIterator)=*valueOutIterator;
         ++valueOutIterator;
         ++indexOutIterator;
      }
   }
}

} // namepsace opengm

#endif // OPENGM_SPARSEMARRAY_FUNCTION_HXX

