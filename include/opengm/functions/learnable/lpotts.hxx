#pragma once
#ifndef OPENGM_LEARNABLE_POTTS_FUNCTION_HXX
#define OPENGM_LEARNABLE_POTTS_FUNCTION_HXX

#include <algorithm>
#include <vector>
#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"
#include "opengm/graphicalmodel/weights.hxx"

namespace opengm {
namespace functions {
namespace learnable {

/// Learnable feature function for two variables
///
/// f(u,v) = (\sum_i w_i * feat_i) I(u!=v)
///  - w    = parameter vector
///  - feat = feature vector
///
/// derive from this class and implement the function
///   paramaterGradient(i,x)= A(x)_{i,*}*feat
///  
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class LPotts
   : public opengm::FunctionBase<opengm::functions::learnable::LPotts<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef L LabelType;
   typedef I IndexType;
 
   LPotts();
   LPotts(const opengm::learning::Weights<T>& weights,
      const L numLabels,
      const std::vector<size_t>& weightIDs,
      const std::vector<T>& feat
      );
   LPotts(const L numLabels,
      const std::vector<size_t>& weightIDs,
      const std::vector<T>& feat
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

   bool isPotts() const {return true;}
   bool isGeneralizedPotts() const {return true;}

protected:
   mutable const opengm::learning::Weights<T> * weights_;
   L numLabels_;
   std::vector<size_t> weightIDs_;
   std::vector<T> feat_;


    friend class opengm::FunctionSerialization<opengm::functions::learnable::LPotts<T, I, L> >;
};


template <class T, class I, class L>
inline
LPotts<T, I, L>::LPotts
( 
   const opengm::learning::Weights<T>& weights,
   const L numLabels,
   const std::vector<size_t>& weightIDs,
   const std::vector<T>& feat
   )
   :  weights_(&weights), numLabels_(numLabels), weightIDs_(weightIDs),feat_(feat)
{
  OPENGM_ASSERT( weightIDs_.size()==feat_.size() );
}

template <class T, class I, class L>
inline
LPotts<T, I, L>::LPotts
( 
   const L numLabels,
   const std::vector<size_t>& weightIDs,
   const std::vector<T>& feat
   )
   : numLabels_(numLabels), weightIDs_(weightIDs),feat_(feat)
{
  OPENGM_ASSERT( weightIDs_.size()==feat_.size() );
}

template <class T, class I, class L>
inline
LPotts<T, I, L>::LPotts
( )
   : numLabels_(0), weightIDs_(std::vector<size_t>(0)), feat_(std::vector<T>(0))
{
  OPENGM_ASSERT( weightIDs_.size()==feat_.size() );
}


template <class T, class I, class L>
template <class ITERATOR>
inline T
LPotts<T, I, L>::weightGradient 
(
   size_t weightNumber,
   ITERATOR begin
) const {
  OPENGM_ASSERT(weightNumber< numberOfWeights());
  if( *(begin) != *(begin+1) )
    return (*this).feat_[weightNumber];
  return 0;
}

template <class T, class I, class L>
template <class ITERATOR>
inline T
LPotts<T, I, L>::operator()
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
LPotts<T, I, L>::shape
(
   const size_t i
) const {
   return numLabels_;
}

template <class T, class I, class L>
inline size_t
LPotts<T, I, L>::dimension() const {
   return 2;
}

template <class T, class I, class L>
inline size_t
LPotts<T, I, L>::size() const {
   return numLabels_*numLabels_;
}

} // namespace learnable
} // namespace functions


/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<opengm::functions::learnable::LPotts<T, I, L> > {
public:
   typedef typename opengm::functions::learnable::LPotts<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const opengm::functions::learnable::LPotts<T, I, L>&);
   static size_t valueSequenceSize(const opengm::functions::learnable::LPotts<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const opengm::functions::learnable::LPotts<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, opengm::functions::learnable::LPotts<T, I, L>&);
};

template<class T, class I, class L>
struct FunctionRegistration<opengm::functions::learnable::LPotts<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 100 + 65
   };
};

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::LPotts<T, I, L> >::indexSequenceSize
(
   const opengm::functions::learnable::LPotts<T, I, L> & src
) {
  return 2+src.weightIDs_.size();
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::LPotts<T, I, L> >::valueSequenceSize
(
   const opengm::functions::learnable::LPotts<T, I, L> & src
) {
  return src.feat_.size();
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<opengm::functions::learnable::LPotts<T, I, L> >::serialize
(
   const opengm::functions::learnable::LPotts<T, I, L> & src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   *indexOutIterator = src.numLabels_;
   ++indexOutIterator; 
   *indexOutIterator = src.feat_.size();
   ++indexOutIterator;
   for(size_t i=0; i<src.weightIDs_.size();++i){
     *indexOutIterator = src.weightIndex(i);
     ++indexOutIterator;
   } 
   for(size_t i=0; i<src.feat_.size();++i){
     *valueOutIterator = src.feat_[i];
     ++valueOutIterator;
   }
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<opengm::functions::learnable::LPotts<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   opengm::functions::learnable::LPotts<T, I, L> & dst
) { 
   dst.numLabels_=*indexInIterator;
   ++ indexInIterator;
   const size_t numW=*indexInIterator;
   ++indexInIterator;
   dst.feat_.resize(numW);
   dst.weightIDs_.resize(numW);
   for(size_t i=0; i<numW;++i){
     dst.feat_[i]=*valueInIterator;
     dst.weightIDs_[i]=*indexInIterator;
     ++indexInIterator;
     ++valueInIterator;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_LEARNABLE_FUNCTION_HXX
