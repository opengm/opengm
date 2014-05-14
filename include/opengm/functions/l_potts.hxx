#pragma once
#ifndef OPENGM_L_POTTS_FUNCTION_HXX
#define OPENGM_L_POTTS_FUNCTION_HXX

#include <algorithm>
#include <vector>
#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// Potts function for two variables
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class LPottsFunction
: public FunctionBase<LPottsFunction<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef L LabelType;
   typedef I IndexType;

   LPottsFunction(
      const LabelType,
      const LabelType,
      const Parameters<ValueType,IndexType> & parameters,
      const IndexType valueNotEqual
   );
   LabelType shape(const size_t) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> ValueType operator()(ITERATOR) const;
   bool operator==(const LPottsFunction& ) const;
   // specializations
   bool isPotts() const;
   bool isGeneralizedPotts() const;
   ValueType min() const;
   ValueType max() const;
   ValueType sum() const;
   ValueType product() const;
   MinMaxFunctor<ValueType> minMax() const;

   // parameters
   size_t numberOfParameters()const{
      return 1;
   }
   IndexType parameterIndex(const size_t paramNumber)const{
      return piValueNotEqual_;
   }


private:
   LabelType numberOfLabels1_;
   LabelType numberOfLabels2_;

   const Parameters<ValueType,IndexType> * params_;

   IndexType piValueNotEqual_;

friend class FunctionSerialization<LPottsFunction<T, I, L> > ;
};


template<class T, class I, class L>
struct FunctionRegistration<LPottsFunction<T, I, L> > {
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 100 + 6
   };
};





template <class T, class I, class L>
inline
LPottsFunction<T, I, L>::LPottsFunction
(
   const L numberOfLabels1,
   const L numberOfLabels2,
   const Parameters<ValueType,IndexType> & parameters,
   const IndexType valueNotEqual
)
:  numberOfLabels1_(numberOfLabels1),
   numberOfLabels2_(numberOfLabels2),
   params_(&parameters),
   piValueNotEqual_(valueNotEqual)
{}

template <class T, class I, class L>
template <class ITERATOR>
inline T
LPottsFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   return (begin[0]==begin[1] ? 
      static_cast<ValueType>(0.0) : params_->getParameter(piValueNotEqual_) );
}



template <class T, class I, class L>
inline L
LPottsFunction<T, I, L>::shape
(
   const size_t i
) const {
   OPENGM_ASSERT(i < 2);
   return (i==0 ? numberOfLabels1_ : numberOfLabels2_);
}

template <class T, class I, class L>
inline size_t
LPottsFunction<T, I, L>::dimension() const {
   return 2;
}

template <class T, class I, class L>
inline size_t
LPottsFunction<T, I, L>::size() const {
   return numberOfLabels1_*numberOfLabels2_;
}


template<class T, class I, class L>
inline bool
LPottsFunction<T, I, L>::operator==
(
   const LPottsFunction & fb
   )const{
   return  numberOfLabels1_ == fb.numberOfLabels1_ &&
      numberOfLabels2_ == fb.numberOfLabels2_ &&
      piValueNotEqual_   == fb.piValueNotEqual_;
}


template<class T, class I, class L>
inline bool
LPottsFunction<T, I, L>::isPotts() const
{
   return true;
}

template<class T, class I, class L>
inline bool
LPottsFunction<T, I, L>::isGeneralizedPotts() const
{
   return true;
}

template<class T, class I, class L>
inline typename LPottsFunction<T, I, L>::ValueType
LPottsFunction<T, I, L>::min() const
{
   const T val = params_->getParameter(piValueNotEqual_);
   return 0.0<val ? 0.0 :val;
}

template<class T, class I, class L>
inline typename LPottsFunction<T, I, L>::ValueType
LPottsFunction<T, I, L>::max() const
{
  const T val = params_->getParameter(piValueNotEqual_);
  return 0.0>val ? 0.0 :val;
}

template<class T, class I, class L>
inline typename LPottsFunction<T, I, L>::ValueType
LPottsFunction<T, I, L>::sum() const
{
    const T val = params_->getParameter(piValueNotEqual_);
    const LabelType minLabels = std::min(numberOfLabels1_, numberOfLabels2_);
    return val * static_cast<T>(numberOfLabels1_ * numberOfLabels2_ - minLabels);
}

template<class T, class I, class L>
inline typename LPottsFunction<T, I, L>::ValueType
LPottsFunction<T, I, L>::product() const
{
   return static_cast<ValueType>(0);
}

template<class T, class I, class L>
inline MinMaxFunctor<typename LPottsFunction<T, I, L>::ValueType>
LPottsFunction<T, I, L>::minMax() const
{
   if(static_cast<ValueType>(0) < piValueNotEqual_) {
      return MinMaxFunctor<T>(static_cast<ValueType>(0), params_[piValueNotEqual_]);
   }
   else {
      return MinMaxFunctor<T>(params_[piValueNotEqual_], static_cast<ValueType>(0));
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_L_POTTS_FUNCTION_HXX
