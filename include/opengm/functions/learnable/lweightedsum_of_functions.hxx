#pragma once
#ifndef OPENGM_LEARNABLE_LWEIGHTEDSUM_OF_FUNCTIONS_FUNCTION_HXX
#define OPENGM_LEARNABLE_LWEIGHTEDSUM_OF_FUNCTIONS_FUNCTION_HXX

#include <algorithm>
#include <vector>
#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"
#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/graphicalmodel/weights.hxx"

namespace opengm {
namespace functions {
namespace learnable {

/// Learnable weighted sum of feature-functions
///
/// f(x) = \sum_i w(i) * feat(i)(x)
///  - w    = parameter vector
///  - feat = feature-function vector
///
///  
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class LWeightedSumOfFunctions
   : public opengm::FunctionBase<opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef L LabelType;
   typedef I IndexType;
 
   LWeightedSumOfFunctions();
   LWeightedSumOfFunctions(const std::vector<L>& shape,
      const opengm::learning::Weights<T>& weights,
      const std::vector<size_t>& weightIDs,
      const std::vector<marray::Marray<T> >& feat
      );
 
   L shape(const size_t) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> T operator()(ITERATOR) const;
 
   // parameters
   void setWeights(const opengm::learning::Weights<T>& weights) const
      {weights_ = &weights;}
   size_t numberOfWeights()const
     {return weightIDs_.size();}
   I weightIndex(const size_t weightNumber) const
     {return weightIDs_[weightNumber];} //dummy
   template<class ITERATOR> 
   T weightGradient(size_t,ITERATOR) const;

protected:
   mutable const opengm::learning::Weights<T>* weights_;
   std::vector<L>                          shape_;
   std::vector<size_t>                     weightIDs_;
   std::vector<marray::Marray<T> >         feat_;

   friend class opengm::FunctionSerialization<opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> >;
};


template <class T, class I, class L>
inline
LWeightedSumOfFunctions<T, I, L>::LWeightedSumOfFunctions
( 
   const std::vector<L>&                           shape,
   const opengm::learning::Weights<T>&             weights,
   const std::vector<size_t>&                      weightIDs,
   const std::vector<marray::Marray<T> >&          feat
   )
   :   shape_(shape), weights_(&weights), weightIDs_(weightIDs),feat_(feat)
{
   OPENGM_ASSERT( weightIDs_.size() == feat_.size() );
   for(size_t i=0; i<weightIDs_.size(); ++i){
      OPENGM_ASSERT( size() == feat_[i].size() );
      for(size_t j=0; j<dimension(); ++j)
          OPENGM_ASSERT( shape_[j] == feat_[i].shape(j))
   }
}

template <class T, class I, class L>
inline
LWeightedSumOfFunctions<T, I, L>::LWeightedSumOfFunctions()
   : shape_(std::vector<L>(0)), weightIDs_(std::vector<size_t>(0)), feat_(std::vector<marray::Marray<T> >(0))
{
   ;
}


template <class T, class I, class L>
template <class ITERATOR>
inline T
LWeightedSumOfFunctions<T, I, L>::weightGradient
(
   size_t weightNumber,
   ITERATOR begin
) const {
  OPENGM_ASSERT(weightNumber< numberOfWeights());
  return feat_[weightNumber](begin);
}

template <class T, class I, class L>
template <class ITERATOR>
inline T
LWeightedSumOfFunctions<T, I, L>::operator()
(
   ITERATOR begin
) const {
   T val = 0;
   for(size_t i=0;i<numberOfWeights();++i){
      val += weights_->getWeight(weightIDs_[i]) * weightGradient(i,begin);
   }
   return val;
}


template <class T, class I, class L>
inline L
LWeightedSumOfFunctions<T, I, L>::shape
(
   const size_t i
) const {
   return shape_[i];
}

template <class T, class I, class L>
inline size_t
LWeightedSumOfFunctions<T, I, L>::dimension() const {
   return shape_.size();
}

template <class T, class I, class L>
inline size_t
LWeightedSumOfFunctions<T, I, L>::size() const {
   size_t s = 1;
   for(size_t i=0; i<dimension(); ++i)
      s *=shape_[i];
   return s;
}

} // namespace learnable
} // namespace functions


/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> > {
public:
   typedef typename opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L>&);
   static size_t valueSequenceSize(const opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L>&);
};

template<class T, class I, class L>
struct FunctionRegistration<opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 100 + 67
   };
};

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> >::indexSequenceSize
(
   const opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> & src
) {
   return 1+src.shape_.size()+1+src.weightIDs_.size();
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> >::valueSequenceSize
(
   const opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> & src
) {
   return src.feat_.size()*src.size();
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> >::serialize
(
   const opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> & src,
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
   *indexOutIterator = src.weightIDs_.size();
   ++indexOutIterator; 
   for(size_t i=0; i<src.weightIDs_.size();++i){
      *indexOutIterator = src.weightIDs_[i];
      ++indexOutIterator; 
   }

   OPENGM_ASSERT_OP(src.weightIDs_.size(), ==, src.feat_.size());

   // save features  
   for(size_t i=0; i<src.weightIDs_.size();++i){
      for(size_t j=0; j<src.feat_[i].size();++j){
         *valueOutIterator = src.feat_[i](j);
         ++valueOutIterator;
      }
   }
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   opengm::functions::learnable::LWeightedSumOfFunctions<T, I, L> & dst
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
   std::vector<marray::Marray<T> > feat(numW,marray::Marray<T>(shape.begin(),shape.end()));
   for(size_t i=0; i<numW;++i){   
      for(size_t j=0; j<size;++j){
         feat[i](j)=*valueInIterator;
         ++valueInIterator;
      }
   }   
}

} // namespace opengm

#endif //OPENGM_LEARNABLE_LWEIGHTEDSUM_OF_FUNCTIONS_FUNCTION_HXX
