#pragma once
#ifndef OPENGM_FoE_FUNCTION_HXX
#define OPENGM_FoE_FUNCTION_HXX

#include <algorithm>
#include <vector>
#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"
#include "opengm/datastructures/marray/marray.hxx"

namespace opengm {

/// Field of Expert function
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class FoEFunction
: public FunctionBase<FoEFunction<T, I, L>, T, size_t, size_t>
{
public:
   typedef T ValueType;
   typedef L LabelType;
   typedef I IndexType;

   FoEFunction(const std::vector<T>&, const std::vector<T>&, const L);
   template<class IT>
   FoEFunction(IT, IT, const L, const I, const size_t);
   FoEFunction();
   LabelType shape(const size_t) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> ValueType operator()(ITERATOR) const;

   void setDefault();
 
protected:
   std::vector<T> experts_;
   std::vector<T> alphas_;
   L numLabels_;
   I order_;
   mutable std::vector<T> l_;

friend class FunctionSerialization<FoEFunction<T, I, L> > ;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class T, class I, class L>
struct FunctionRegistration<FoEFunction<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 33
   };
};

/// FunctionSerialization
template<class T, class I, class L>
class FunctionSerialization<FoEFunction<T, I, L> > {
public:
   typedef typename FoEFunction<T, I, L>::ValueType ValueType;

   static size_t indexSequenceSize(const FoEFunction<T, I, L>&);
   static size_t valueSequenceSize(const FoEFunction<T, I, L>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const FoEFunction<T, I, L>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, FoEFunction<T, I, L>&);
};
/// \endcond

/// constructor for demo function
   template <class T, class I, class L>
   inline
   FoEFunction<T, I, L>::FoEFunction
   ()    
   {
      experts_.resize(0);
      alphas_.resize(0);
      numLabels_= 0;
      order_ = 0;
      l_.resize(order_);
   }

   template <class T, class I, class L>
   inline void
   FoEFunction<T, I, L>::setDefault
   ()    
   {
      experts_.resize(3*4);
      alphas_.resize(3);

      const double a_alpha[3] = {0.586612685392731, 1.157638405566669, 0.846059486257292};
      const double a_expert[3][4] = {
         {-0.0582774013402734, 0.0339010363051084, -0.0501593018104054, 0.0745568557931712},
         {0.0492112815304123, -0.0307820846538285, -0.123247230948424, 0.104812330861557},
         {0.0562633568728865, 0.0152832583489560, -0.0576215592718086, -0.0139673758425540}
      }; 
      for(size_t e=0; e<3; ++e){
         alphas_[e] =  a_alpha[e]; 
         OPENGM_ASSERT(   alphas_[e] ==  a_alpha[e]);
         for(size_t i=0; i<4;++i){
            experts_[e+i*3] = a_expert[e][i];
         }
      }
      numLabels_= 256;
      order_ = 4;
      l_.resize(order_);
   }


/// constructor
/// \param e:     Expert-matrix (dim = numExp x order)
/// \param a:     Vector weighting the experts
/// \param numL:  NumberOfLabels
   template <class T, class I, class L>
   inline
   FoEFunction<T, I, L>::FoEFunction
   (const std::vector<T>& e, const std::vector<T>& a, const L numL)     
   {
      experts_   = e;
      alphas_    = a;
      numLabels_ = numL;
      order_     = e.size()/a.size();
      l_.resize(order_);
      OPENGM_ASSERT(order_*alphas_.size() == experts_.size());
   }

   /// constructor
   /// \param e:     Expert-iterator (dim = numExp x order)
   /// \param a:     Iterator weighting the experts
   /// \param numL:  NumberOfLabels
   template <class T, class I, class L>
   template<class IT>
   inline
   FoEFunction<T, I, L>::FoEFunction(IT ite, IT ita, const L numLabels, const I numVars, const size_t numExperts){
      numLabels_ = numLabels;
      order_     = numVars;
      experts_.resize(numVars*numExperts);
      alphas_.resize(numExperts);
      l_.resize(order_);
      for(size_t i=0; i<numVars*numExperts; ++i,++ite)
         experts_[i]  = *ite;
      for(size_t i=0; i<numExperts; ++i,++ita)
         alphas_[i]  = *ita;
   }


template <class T, class I, class L>
template <class ITERATOR>
inline T
FoEFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   //copy/cast one time to avoid additional casts
   for (size_t j = 0; j < order_; ++j) {
      l_[j] = static_cast<T>(*begin);
      ++begin;
      OPENGM_ASSERT(l_[j]< numLabels_);
   }
   ValueType val = 0.0;
   size_t i = 0;
   for (size_t e = 0; e < alphas_.size(); ++e) {
      ValueType dot = 0.0;
      for (size_t j = 0; j < order_; ++j,++i) {
         dot += experts_[i] * l_[j];
      }
      val += alphas_[e] * std::log(1 + 0.5 * dot * dot);
   }
   return val;
}
   
template <class T, class I, class L>
inline L
FoEFunction<T, I, L>::shape
(
   const size_t i
) const {
   OPENGM_ASSERT(i < dimension());
   return numLabels_;
}

template <class T, class I, class L>
inline size_t
FoEFunction<T, I, L>::dimension() const {
   return order_;
}

template <class T, class I, class L>
inline size_t
FoEFunction<T, I, L>::size() const {
   size_t s=1;
   for(size_t i=0; i<dimension(); ++i)
      s *= numLabels_;
   return s;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<FoEFunction<T, I, L> >::indexSequenceSize
(
   const FoEFunction<T, I, L> & src
) {
   return 3;
}

template<class T, class I, class L>
inline size_t
FunctionSerialization<FoEFunction<T, I, L> >::valueSequenceSize
(
   const FoEFunction<T, I, L> & src
) {
   return src.experts_.size()+src.alphas_.size();
}

template<class T, class I, class L>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR >
inline void
FunctionSerialization<FoEFunction<T, I, L> >::serialize
(
   const FoEFunction<T, I, L> & src,
   INDEX_OUTPUT_ITERATOR indexOutIterator,
   VALUE_OUTPUT_ITERATOR valueOutIterator
) { 
   for(size_t i=0; i<src.experts_.size(); ++i){
      *valueOutIterator = src.experts_[i];
      ++valueOutIterator;
   }
   std::cout <<std::endl;
   for(size_t i=0; i<src.alphas_.size(); ++i){
      *valueOutIterator = src.alphas_[i];
      ++valueOutIterator;
   }

   *indexOutIterator = src.alphas_.size();
   ++indexOutIterator; 
   *indexOutIterator = src.order_;
   ++indexOutIterator; 
   *indexOutIterator = src.numLabels_;
}

template<class T, class I, class L>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR >
inline void
FunctionSerialization<FoEFunction<T, I, L> >::deserialize
(
   INDEX_INPUT_ITERATOR indexInIterator,
   VALUE_INPUT_ITERATOR valueInIterator,
   FoEFunction<T, I, L> & dst
) {
   size_t numExperts = *indexInIterator;
   ++indexInIterator;
   size_t order = *indexInIterator;
   ++indexInIterator; 
   size_t numLabels = *indexInIterator;

   std::vector<T> alphas(numExperts);
   std::vector<T> experts(numExperts*order);
 
   for(size_t i=0; i<numExperts*order; ++i){
      experts[i] = *valueInIterator;
      ++valueInIterator;
   }
   for(size_t i=0; i<numExperts; ++i){
      alphas[i] = *valueInIterator;
      ++valueInIterator;   
   }

   dst=FoEFunction<T, I, L>(experts,alphas,numLabels);
}

} // namespace opengm

#endif // #ifndef OPENGM_FoE_FUNCTION_HXX
