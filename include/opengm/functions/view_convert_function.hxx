#pragma once
#ifndef OPENGM_VIEW_CONVERT_FUNCTION_HXX
#define OPENGM_VIEW_CONVERT_FUNCTION_HXX

#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS
namespace detail_convert_function {
   template<class OPERATOR, class ACCUMULATOR, class PROBABILITY>
   struct ValueToProbability;

   template<class PROBABILITY>
   struct ValueToProbability<Multiplier, Maximizer, PROBABILITY>
   {
      typedef PROBABILITY ProbabilityType;
      template<class T>
         static ProbabilityType convert(const T x)
            { return static_cast<ProbabilityType>(x); }
   };

   template<class PROBABILITY>
   struct ValueToProbability<Multiplier, Minimizer, PROBABILITY>
   {
      typedef PROBABILITY ProbabilityType;
      template<class T>
         static ProbabilityType convert(const T x)
            { return static_cast<ProbabilityType>(1) / static_cast<ProbabilityType>(x); }
   };

   template<class PROBABILITY>
   struct ValueToProbability<Adder, Maximizer, PROBABILITY>
   {
      typedef PROBABILITY ProbabilityType;
      template<class T>
         static ProbabilityType convert(const T x)
            { return static_cast<ProbabilityType>(std::exp(x)); }
   };

   template<class PROBABILITY>
   struct ValueToProbability<Adder, Minimizer, PROBABILITY>
   {
      typedef PROBABILITY ProbabilityType;
      template<class T>
         static ProbabilityType convert(const T x)
            { return static_cast<ProbabilityType>(std::exp(-x)); }
   };
}
/// \endcond

/// ViewConvertFunction convert semi-ring in a lazy fashion
///
/// \ingroup functions
template<class GM,class ACC,class VALUE_TYPE>
class ViewConvertFunction
: public FunctionBase<ViewConvertFunction<GM,ACC,VALUE_TYPE>, 
    typename GM::ValueType, typename GM::IndexType, typename GM::LabelType>
{
public:
   typedef VALUE_TYPE ValueType;
   typedef VALUE_TYPE value_type;
   typedef typename GM::FactorType FactorType;
   typedef typename GM::OperatorType OperatorType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;

   ViewConvertFunction();
   ViewConvertFunction(const FactorType &);
   template<class Iterator> ValueType operator()(Iterator begin) const;
   IndexType shape(const IndexType) const;
   IndexType dimension() const;
   IndexType size() const;

private:
   FactorType const* factor_;
};

template<class GM,class ACC,class VALUE_TYPE>
inline
ViewConvertFunction<GM,ACC,VALUE_TYPE>::ViewConvertFunction()
:  factor_(NULL)
{}

template<class GM,class ACC,class VALUE_TYPE>
inline
ViewConvertFunction<GM,ACC,VALUE_TYPE>::ViewConvertFunction
(
   const typename ViewConvertFunction<GM,ACC,VALUE_TYPE>::FactorType & factor
)
:  factor_(&factor)
{}

template<class GM,class ACC,class VALUE_TYPE>
template<class Iterator>
inline typename ViewConvertFunction<GM,ACC,VALUE_TYPE>::ValueType
ViewConvertFunction<GM,ACC,VALUE_TYPE>::operator()
(
   Iterator begin
) const {
   return detail_convert_function::ValueToProbability<OperatorType,ACC,ValueType>::convert(factor_->operator()(begin));
}

template<class GM,class ACC,class VALUE_TYPE>
inline typename ViewConvertFunction<GM,ACC,VALUE_TYPE>::IndexType
ViewConvertFunction<GM,ACC,VALUE_TYPE>::shape
(
   const typename ViewConvertFunction<GM,ACC,VALUE_TYPE>::IndexType index
) const{
   return factor_->numberOfLabels(index);
}

template<class GM,class ACC,class VALUE_TYPE>
inline typename ViewConvertFunction<GM,ACC,VALUE_TYPE>::IndexType
ViewConvertFunction<GM,ACC,VALUE_TYPE>::dimension() const {
   return factor_->numberOfVariables();
}

template<class GM,class ACC,class VALUE_TYPE>
inline typename ViewConvertFunction<GM,ACC,VALUE_TYPE>::IndexType
ViewConvertFunction<GM,ACC,VALUE_TYPE>::size() const {
   return factor_->size( );
}

} // namespace opengm

#endif // #ifndef OPENGM_VIEW_CONVERT_FUNCTION_HXX
