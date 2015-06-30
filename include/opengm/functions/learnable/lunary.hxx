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





template<class V, class I>
struct FeaturesAndIndices{
    std::vector<V> features;
    std::vector<I> weightIds;
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

    LUnary()
    :  
    weights_(NULL),
    numberOfLabels_(0), 
    offsets_(),
    weightIds_(),
    features_()
    {

    }

    LUnary(
        const opengm::learning::Weights<T>&     weights,
        std::vector<FeaturesAndIndices<T, I> >  featuresAndIndicesPerLabel
    );

    LUnary(
        const opengm::learning::Weights<T>& weights,    
        const LabelType                     numberOfLabels,
        marray::Marray< size_t >            weightIds,
        marray::Marray< ValueType>          features,
        const bool                          makeFirstEntryConst
    );


    L shape(const size_t) const;
    size_t size() const;
    size_t dimension() const;
    template<class ITERATOR> T operator()(ITERATOR) const;

    // parameters
    void setWeights(const opengm::learning::Weights<T>& weights) const{
        weights_ = &weights;
    }

    size_t numberOfWeights()const{
        return weightIds_.size();
    }

    I weightIndex(const size_t weightNumber) const{
        return weightIds_[weightNumber];
    } 

    template<class ITERATOR> 
    T weightGradient(size_t,ITERATOR) const;

private:


protected:

    size_t numWeightsForL(const LabelType l )const{
        return offsets_[0*numberOfLabels_ + l];
    }
    size_t weightIdOffset(const LabelType l )const{
        return offsets_[1*numberOfLabels_ + l];
    }
    size_t featureOffset(const LabelType l )const{
        return offsets_[2*numberOfLabels_ + l];
    }

    mutable const opengm::learning::Weights<T> *    weights_;

    opengm::UInt16Type numberOfLabels_;
    std::vector<opengm::UInt16Type> offsets_;
    std::vector<size_t> weightIds_;
    std::vector<ValueType> features_;


    friend class opengm::FunctionSerialization<opengm::functions::learnable::LUnary<T, I, L> >;


};

template <class T, class I, class L>
LUnary<T, I, L>::LUnary(
    const opengm::learning::Weights<T>& weights,    
    const LabelType                     numberOfLabels,
    marray::Marray< size_t >            weightIds,
    marray::Marray< ValueType>          features,
    const bool                          makeFirstEntryConst
)
:  
weights_(&weights),
numberOfLabels_(numberOfLabels), 
offsets_(numberOfLabels*3),
weightIds_(),
features_()
{
    const size_t pFeatDim       = features.dimension();
    const size_t pWeightIdDim   = weightIds.dimension();

    OPENGM_CHECK_OP(weightIds.dimension(), ==, 2 , "wrong dimension");
    OPENGM_CHECK_OP(weightIds.shape(0)+int(makeFirstEntryConst), ==, numberOfLabels , "wrong shape");


    const size_t nWeights = weightIds.size();
    weightIds_.resize(nWeights);

    const size_t nFeat  = features.size();
    features_.resize(nFeat);


    OPENGM_CHECK_OP(features.dimension(), == , 1 , "feature dimension must be 1 ");
    OPENGM_CHECK_OP(features.shape(0), == , weightIds.shape(1) , "feature dimension must be 1");

    // copy features
    for(size_t fi=0; fi<nFeat; ++fi){
        features_[fi] = features(fi);
    }

    size_t nwForL = weightIds.shape(1);
    size_t wOffset = 0;

    if(makeFirstEntryConst){

        OPENGM_CHECK_OP(numberOfLabels_-1, == , weightIds.shape(0),"internal error");

        offsets_[0*numberOfLabels_ + 0] = 0;
        offsets_[1*numberOfLabels_ + 0] = 0;
        offsets_[2*numberOfLabels_ + 0] = 0;

        for(LabelType l=1; l<numberOfLabels_; ++l){
            offsets_[0*numberOfLabels_ + l] = nwForL;
            offsets_[1*numberOfLabels_ + l] = wOffset;
            offsets_[2*numberOfLabels_ + l] = 0;
            // copy weight ids
            for(size_t wi=0; wi<nwForL; ++wi){
                weightIds_[wOffset + wi] = weightIds(l-1,wi);
            }
            wOffset += nwForL;
        }
    }
    else{
        OPENGM_CHECK_OP(numberOfLabels_, == , weightIds.shape(0),"internal error");
        for(LabelType l=0; l<numberOfLabels_; ++l){

            offsets_[0*numberOfLabels_ + l] = nwForL;
            offsets_[1*numberOfLabels_ + l] = wOffset;
            offsets_[2*numberOfLabels_ + l] = 0;
            // copy weight ids
            for(size_t wi=0; wi<nwForL; ++wi){
                weightIds_[wOffset + wi] = weightIds(l,wi);
            }
            wOffset += nwForL;
        }
    }

}

template <class T, class I, class L>
inline
LUnary<T, I, L>::LUnary
( 
   const opengm::learning::Weights<T> & weights, 
   std::vector<FeaturesAndIndices<V, I> >  featuresAndIndicesPerLabel 
)
:  
weights_(&weights),
numberOfLabels_(featuresAndIndicesPerLabel.size()), 
offsets_(featuresAndIndicesPerLabel.size()*3),
weightIds_(),
features_()
{

    size_t fOffset = 0;
    size_t wOffset = 0;


    // fetch the offsets
    for(size_t l=0; l<featuresAndIndicesPerLabel.size(); ++l){
        const size_t nwForL  = featuresAndIndicesPerLabel[l].weightIds.size();
        const size_t nfForL  = featuresAndIndicesPerLabel[l].features.size();
        OPENGM_CHECK_OP(nwForL, == , nfForL, "number of features and weighs"
            "must be the same for a given label within this overload of LUnary<T, I, L>::LUnary");

        offsets_[0*numberOfLabels_ + l] = nwForL;
        offsets_[1*numberOfLabels_ + l] = wOffset;
        offsets_[2*numberOfLabels_ + l] = fOffset;

        wOffset += nwForL;
        fOffset += nfForL;
    }

    weightIds_.resize(wOffset);
    features_.resize(fOffset);

    // write weightIDs and features
    for(size_t l=0; l<featuresAndIndicesPerLabel.size(); ++l){
        const size_t nwForL = numWeightsForL(l);
        for(size_t i=0; i<nwForL; ++i){
            weightIds_[featureOffset(l)+i] = featuresAndIndicesPerLabel[l].weightIds[i];
            features_[featureOffset(l)+i] = featuresAndIndicesPerLabel[l].features[i];
        }
    }

    // check that there are no duplicates
    RandomAccessSet<size_t> idSet;
    idSet.reserve(weightIds_.size());
    idSet.insert(weightIds_.begin(), weightIds_.end());

    OPENGM_CHECK_OP(idSet.size(), == , weightIds_.size(), "weightIds has duplicates");
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
    const LabelType l(*begin);
    const size_t nwForL = numWeightsForL(l);
    if(nwForL>0){
        const size_t wiStart = weightIdOffset(l);
        const size_t wiEnd   = weightIdOffset(l)+nwForL;
        if(weightNumber >= wiStart && weightNumber < wiEnd ){
            const size_t wii = weightNumber - wiStart;
            return features_[featureOffset(l) + wii];
        }
    }
    return static_cast<ValueType>(0);
}

template <class T, class I, class L>
template <class ITERATOR>
inline T
LUnary<T, I, L>::operator()
(
   ITERATOR begin
) const {

    //std::cout<<"LUnary::operator()\n";
    //OPENGM_CHECK_OP( int(weights_==NULL),==,int(false),"foo");
    T val = 0;
    const LabelType l(*begin);
    const size_t nwForL = numWeightsForL(l);
    //std::cout<<"nw for l "<<nwForL<<"\n";
    //std::cout<<"wsize "<<weights_->size()<<"\n";

    for(size_t i=0; i<nwForL; ++i){
        //std::cout<<" i "<<i<<"\n";
        //OPENGM_CHECK_OP(weightIdOffset(l)+i,<,weightIds_.size(),"foo");
        //OPENGM_CHECK_OP(featureOffset(l)+i,<,features_.size(),"foo");
        const size_t wi = weightIds_[weightIdOffset(l)+i];
        //OPENGM_CHECK_OP(wi,<,weights_->size(),"foo");

        val += weights_->getWeight(wi) * features_[featureOffset(l)+i];
    }
    //d::cout<<"LUnary::return operator()\n";
    return val;
}


template <class T, class I, class L>
inline L
LUnary<T, I, L>::shape
(
   const size_t i
) const {
   return numberOfLabels_;
}

template <class T, class I, class L>
inline size_t
LUnary<T, I, L>::dimension() const {
   return 1;
}

template <class T, class I, class L>
inline size_t
LUnary<T, I, L>::size() const {
   return numberOfLabels_;
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

    size_t size = 0;
    size += 1; // numberOfLabels
    size += 1; // numberOfWeights
    size += 1; // numberOfFeatures

    size += 3*src.shape(0);         // offsets serialization 
    size += src.weightIds_.size();  // weight id serialization

    return size;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<opengm::functions::learnable::LUnary<T, I, L> >::valueSequenceSize
(
   const opengm::functions::learnable::LUnary<T, I, L> & src
) {
  return src.features_.size(); // feature serialization
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

    ///////////////////////////////////////
    /// INDEX SERIALIZATION
    ////////////////////////////////////////
    // number of labels
    *indexOutIterator = src.shape(0);
    ++indexOutIterator; 

    // number of weights
    *indexOutIterator = src.weightIds_.size();
    ++indexOutIterator; 
    
    // number of features
    *indexOutIterator = src.features_.size();
    ++indexOutIterator; 

    // offset serialization
    for(size_t i=0; i<src.offsets_.size(); ++i){
        *indexOutIterator = src.offsets_[i];
        ++indexOutIterator;
    }

    // weight id serialization
    for(size_t i=0; i<src.weightIds_.size(); ++i){
        *indexOutIterator = src.weightIds_[i];
        ++indexOutIterator;
    }

    ///////////////////////////////////////
    /// VALUE SERIALIZATION
    ////////////////////////////////////////
    // feature serialization
    for(size_t i=0; i<src.features_.size(); ++i){
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



    ///////////////////////////////////////
    /// INDEX DESERIALIZATION
    ////////////////////////////////////////
    // number of labels
    dst.numberOfLabels_ = *indexInIterator;
    ++indexInIterator;
    // resize offset accordingly
    dst.offsets_.resize(3 * dst.numberOfLabels_);


    // number of weights
    const size_t nW =*indexInIterator;
    ++indexInIterator;
    // resize weightIds accordingly
    dst.weightIds_.resize(nW);

    // number of features
    const size_t nF = *indexInIterator;
    ++indexInIterator;
    // resize weightIds accordingly
    dst.features_.resize(nF);

    // offset deserialization
    for(size_t i=0; i<dst.offsets_.size(); ++i){
        dst.offsets_[i] = *indexInIterator;
        ++indexInIterator;
    }

    // weight id deserialization
    for(size_t i=0; i<dst.weightIds_.size(); ++i){
        dst.weightIds_[i] = *indexInIterator;
        ++indexInIterator;
    }

    ///////////////////////////////////////
    /// VALUE DESERIALIZATION
    ////////////////////////////////////////
    // feature deserialization
    for(size_t i=0; i<dst.features_.size(); ++i){
        dst.features_[i] = *valueInIterator;
        ++valueInIterator;
    } 
}

} // namespace opengm

#endif // #ifndef OPENGM_LEARNABLE_FUNCTION_HXX
