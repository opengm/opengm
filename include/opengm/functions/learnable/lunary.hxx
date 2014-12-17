#pragma once
#ifndef OPENGM_LEARNABLE_UNARY_FUNCTION_HXX
#define OPENGM_LEARNABLE_UNARY_FUNCTION_HXX

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


template<class V, class I>
struct FeaturesAndIndices{
    std::vector<V> features;
    std::vector<I> indices;
};




template<class T, class I , class L>
class LUnary
   : public opengm::FunctionBase<opengm::functions::learnable::LUnary<T, I, L>, T, I, L>
{
public:
    typedef T ValueType;
    typedef T V;
    typedef L LabelType;
    typedef I IndexType;


    LUnary(
        const opengm::learning::Weights<T>&     weights,
        std::vector<FeaturesAndIndices<T, I> >  featuresAndIndicesPerLabel
    );
    L shape(const size_t) const;
    size_t size() const;
    size_t dimension() const;
    template<class ITERATOR> T operator()(ITERATOR) const;

    // parameters
    void setWeights(const opengm::learning::Weights<T>& weights){
        weights_ = &weights;
    }

    size_t numberOfWeights()const{
        return numWeights_;
    }

    I weightIndex(const size_t weightNumber) const{
        return weightIds_[weightNumber];
    } 

    template<class ITERATOR> 
    T weightGradient(size_t,ITERATOR) const;

private:
    bool isMatchingWeight(const LabelType l , const size_t i ){

    }
protected:
    const opengm::learning::Weights<T> *    weights_;
    std::vector<size_t> labelOffset_;
    std::vector<size_t> weightIds_;
    std::vector<V>      features_;
    size_t numWeights_;
    friend class opengm::FunctionSerialization<opengm::functions::learnable::LUnary<T, I, L> >;


};


template <class T, class I, class L>
inline
LUnary<T, I, L>::LUnary
( 
   const opengm::learning::Weights<T> & weights =  opengm::learning::Weights<T>(),
   std::vector<FeaturesAndIndices<V, I> >  featuresAndIndicesPerLabel = std::vector<FeaturesAndIndices<V, I> >()
)
:  
weights_(&weights), 
labelOffset_(featuresAndIndicesPerLabel.size(),0), 
weightIds_(),
features_(),
numWeights_(0){

    // collect how many weights there are at all 
    // for this function
    size_t offset = 0 ;
    for(size_t l=0; l<featuresAndIndicesPerLabel.size(); ++l){
        const size_t nForThisL = featuresAndIndicesPerLabel[l].features.size();
        numWeights_ += nForThisL;
    }

    weightIds_.resize(numWeights_);
    features_.resize(numWeights_);

    for(size_t l=0; l<featuresAndIndicesPerLabel.size(); ++l){
        labelOffset_[l] = offset;
        const size_t nForThisL = featuresAndIndicesPerLabel[l].features.size();
        for(size_t i=0; i<nForThisL; ++i){

            // as many features as labels
            OPENGM_CHECK_OP( featuresAndIndicesPerLabel[l].indices.size(), == ,
                             featuresAndIndicesPerLabel[l].features.size() ,
                             "features and weights must be of same length");

            weightIds_[offset + i] = featuresAndIndicesPerLabel[l].indices[i];
            features_[offset + i] = featuresAndIndicesPerLabel[l].features[i];

        }
        offset+=nForThisL;
    }
}



template <class T, class I, class L>
template <class ITERATOR>
inline T
LUnary<T, I, L>::weightGradient 
(
   size_t weightNumber,
   ITERATOR begin
) const {
    OPENGM_CHECK_OP(weightNumber,<,numberOfWeights(), 
        "weightNumber must be smaller than number of weights");
    const L l = *begin;

    if(l == size()-1){
        size_t start = labelOffset_[l];
        if(weightNumber>=start){
            return features_[weightNumber];
        }
    }
    else{
        size_t start = labelOffset_[l];
        size_t   end = labelOffset_[l+1];
        if(weightNumber>= start && weightNumber<end){
            return features_[weightNumber];
        }
    }
    return V(0);

}

template <class T, class I, class L>
template <class ITERATOR>
inline T
LUnary<T, I, L>::operator()
(
   ITERATOR begin
) const {
   T val = 0;
   size_t end = (*begin == size()-1 ? numberOfWeights() : labelOffset_[*begin+1] );

   //std::cout<<"label "<<*begin<<"\n";
   //std::cout<<"s e "<<labelOffset_[*begin]<<" "<<end<<"\n";
   for(size_t i=labelOffset_[*begin];i<end;++i){
        //std::cout<<"    i="<<i<<" wi="<<weightIds_[i]<<" w="<< weights_->getWeight(weightIds_[i])  <<" f="<<features_[i]<<"\n";
        val += weights_->getWeight(weightIds_[i]) * features_[i];
   }
   return val;
}


template <class T, class I, class L>
inline L
LUnary<T, I, L>::shape
(
   const size_t i
) const {
   return labelOffset_.size();
}

template <class T, class I, class L>
inline size_t
LUnary<T, I, L>::dimension() const {
   return 1;
}

template <class T, class I, class L>
inline size_t
LUnary<T, I, L>::size() const {
   return labelOffset_.size();
}

} // namespace learnable
} // namespace functions


/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<opengm::functions::learnable::LUnary<T, I, L> > {
public:
   typedef typename opengm::functions::learnable::LUnary<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const opengm::functions::learnable::LUnary<T, I, L>&);
   static size_t valueSequenceSize(const opengm::functions::learnable::LUnary<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const opengm::functions::learnable::LUnary<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, opengm::functions::learnable::LUnary<T, I, L>&);
};

template<class T, class I, class L>
struct FunctionRegistration<opengm::functions::learnable::LUnary<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 100 + 66
   };
};

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::LUnary<T, I, L> >::indexSequenceSize
(
   const opengm::functions::learnable::LUnary<T, I, L> & src
) {
  return 2 + src.size() + src.numberOfWeights(); 
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::LUnary<T, I, L> >::valueSequenceSize
(
   const opengm::functions::learnable::LUnary<T, I, L> & src
) {
  return src.numberOfWeights();
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<opengm::functions::learnable::LUnary<T, I, L> >::serialize
(
   const opengm::functions::learnable::LUnary<T, I, L> & src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) {
   *indexOutIterator = src.size();
   ++indexOutIterator; 

   *indexOutIterator = src.numberOfWeights();
   ++indexOutIterator;

    for(size_t l=0; l<src.size(); ++l){
        *indexOutIterator = src.labelOffset_[l];
        ++indexOutIterator; 
    }

    for(size_t i=0; i<src.numberOfWeights(); ++i){
        *indexOutIterator = src.weightIds_[i];
        ++indexOutIterator;

        *valueOutIterator = src.features_[i];
        ++valueOutIterator;
    }

    
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<opengm::functions::learnable::LUnary<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   opengm::functions::learnable::LUnary<T, I, L> & dst
) { 
    const size_t numLabels = *indexInIterator;
    ++indexInIterator;

    dst.numWeights_ = *indexInIterator;
    ++indexInIterator;

    dst.labelOffset_.resize(numLabels);
    dst.weightIds_.resize(dst.numWeights_);
    dst.features_.resize(dst.numWeights_);

    // label offset
    for(size_t l=0; l<numLabels; ++l){
        dst.labelOffset_[l] = *indexInIterator;
        ++indexInIterator;
    }

    for(size_t i=0; i<dst.numWeights_; ++i){
        dst.weightIds_[i] = *indexInIterator;
        ++indexInIterator;

        dst.features_[i] = *valueInIterator;
        ++valueInIterator;
    }


 
}

} // namespace opengm

#endif // #ifndef OPENGM_LEARNABLE_FUNCTION_HXX
