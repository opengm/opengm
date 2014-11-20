#pragma once
#ifndef OPENGM_LEARNABLE_SUM_OF_EXPERTS_FUNCTION_HXX
#define OPENGM_LEARNABLE_SUM_OF_EXPERTS_FUNCTION_HXX

#include <algorithm>
#include <vector>
#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"
#include "opengm/datastructures/marray/marray.hxx"

namespace opengm {
namespace functions {
namespace learnable {

/// Learnable feature function for two variables
///
/// f(x) = \sum_i w(i) * feat(i)(x)
///  - w    = parameter vector
///  - feat = feature vector
///
///  
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class SumOfExperts
   : public opengm::FunctionBase<opengm::functions::learnable::SumOfExperts<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef L LabelType;
   typedef I IndexType;
 
   SumOfExperts();
   SumOfExperts( 
      const std::vector<L>& shape,
      const Parameters<T,I>& parameters,
      const std::vector<size_t>& parameterIDs,
      const std::vector<marray::Marray<T> >& feat
      ); 
 
   L shape(const size_t) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> T operator()(ITERATOR) const;
 
   // parameters
   void setParameters(const Parameters<T,I>& parameters)
      {parameters_ = &parameters;}
   size_t numberOfParameters()const
     {return parameterIDs_.size();}
   I parameterIndex(const size_t paramNumber) const
     {return parameterIDs_[paramNumber];} //dummy
   template<class ITERATOR> 
   T parameterGradient(size_t,ITERATOR) const;

protected:
   const Parameters<T,I>*                  parameters_;
   std::vector<L>                          shape_;
   std::vector<size_t>                     parameterIDs_;
   std::vector<marray::Marray<T> > feat_;

   friend class opengm::FunctionSerialization<opengm::functions::learnable::SumOfExperts<T, I, L> >;
};


template <class T, class I, class L>
inline
SumOfExperts<T, I, L>::SumOfExperts
( 
   const std::vector<L>&                           shape,
   const Parameters<T,I>&                          parameters,
   const std::vector<size_t>&                      parameterIDs,
   const std::vector<marray::Marray<T> >&  feat
   )
   :   shape_(shape), parameters_(&parameters), parameterIDs_(parameterIDs),feat_(feat)
{
   OPENGM_ASSERT( size() == feat_[0].size() );
   OPENGM_ASSERT( parameterIDs_.size() == feat_.size() );
}

template <class T, class I, class L>
inline
SumOfExperts<T, I, L>::SumOfExperts()
   : shape_(std::vector<L>(0)), parameterIDs_(std::vector<size_t>(0)), feat_(std::vector<marray::Marray<T> >(0))
{
   ;
}


template <class T, class I, class L>
template <class ITERATOR>
inline T
SumOfExperts<T, I, L>::parameterGradient 
(
   size_t parameterNumber,
   ITERATOR begin
) const {
  OPENGM_ASSERT(parameterNumber< numberOfParameters());
  return feat_[parameterNumber](begin);
}

template <class T, class I, class L>
template <class ITERATOR>
inline T
SumOfExperts<T, I, L>::operator()
(
   ITERATOR begin
) const {
   T val = 0;
   for(size_t i=0;i<numberOfParameters();++i){
      val += parameters_->getParameter(i) * parameterGradient(i,begin);
   }
   return val;
}


template <class T, class I, class L>
inline L
SumOfExperts<T, I, L>::shape
(
   const size_t i
) const {
   return shape_[i];
}

template <class T, class I, class L>
inline size_t
SumOfExperts<T, I, L>::dimension() const {
   return shape_.size();
}

template <class T, class I, class L>
inline size_t
SumOfExperts<T, I, L>::size() const {
   size_t s = 1;
   for(size_t i=0; i<dimension(); ++i)
      s *=shape_[i];
   return s;
}

} // namespace learnable
} // namespace functions


/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<opengm::functions::learnable::SumOfExperts<T, I, L> > {
public:
   typedef typename opengm::functions::learnable::SumOfExperts<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const opengm::functions::learnable::SumOfExperts<T, I, L>&);
   static size_t valueSequenceSize(const opengm::functions::learnable::SumOfExperts<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const opengm::functions::learnable::SumOfExperts<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, opengm::functions::learnable::SumOfExperts<T, I, L>&);
};

template<class T, class I, class L>
struct FunctionRegistration<opengm::functions::learnable::SumOfExperts<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 100 + 66
   };
};

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::SumOfExperts<T, I, L> >::indexSequenceSize
(
   const opengm::functions::learnable::SumOfExperts<T, I, L> & src
) {
   return 1+src.shape_.size()+1+src.parameterIDs_.size();
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::SumOfExperts<T, I, L> >::valueSequenceSize
(
   const opengm::functions::learnable::SumOfExperts<T, I, L> & src
) {
   return src.feat_.size()*src.dimension();
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<opengm::functions::learnable::SumOfExperts<T, I, L> >::serialize
(
   const opengm::functions::learnable::SumOfExperts<T, I, L> & src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   // save shape
   *indexOutIterator = src.shape_.size();
   ++indexOutIterator; 
   for(size_t i=0; i<src.shape_.size();++i){
      *indexOutIterator = src.shape_[i];
      ++indexOutIterator; 
   }
   //save parameter ids
   *indexOutIterator = src.parameterIDs_.size();
   ++indexOutIterator; 
   for(size_t i=0; i<src.parameterIDs_.size();++i){
      *indexOutIterator = src.parameterIDs_[i];
      ++indexOutIterator; 
   }

   // save features  
   for(size_t i=0; i<src.parameterIDs_.size();++i){
      for(size_t j=0; j<src.feat_[i].size();++j){
         *valueOutIterator = src.feat_[i](j);
         ++valueOutIterator;
      }
   }
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<opengm::functions::learnable::SumOfExperts<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   opengm::functions::learnable::SumOfExperts<T, I, L> & dst
) { 
   //read shape
   size_t dim  = *indexInIterator;
   size_t size = 1;
   ++indexInIterator;
   std::vector<L> shape(dim);
   for(size_t i=0; i<dim;++i){
      shape[i] = *indexInIterator;
      size    *= *indexInIterator; 
      ++indexInIterator;
   }
   //read parameter ids
   size_t numW =*indexInIterator;
   ++indexInIterator;
   std::vector<size_t> parameterIDs(numW);
   for(size_t i=0; i<numW;++i){ 
      parameterIDs[i] = *indexInIterator;
      ++indexInIterator;
   }
   //read features
   std::vector<marray::Marray<T,L> > feat(numW,marray::Marray<T,L>(shape.begin(),shape.end()));
   for(size_t i=0; i<numW;++i){   
      for(size_t j=0; j<size;++j){
         feat[i](j)=*valueInIterator;
         ++valueInIterator;
      }
   }   
}

} // namespace opengm

#endif // #ifndef OPENGM_LEARNABLE_FUNCTION_HXX
