#pragma once
#ifndef OPENGM_VIEW_FUNCTION_HXX
#define OPENGM_VIEW_FUNCTION_HXX

#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// reference to a Factor of a GraphicalModel
///
/// \ingroup functions
template<class GM>
class ViewFunction
: public FunctionBase<ViewFunction<GM>, 
    typename GM::ValueType,typename GM::IndexType, typename GM::LabelType>
{
public:
   typedef typename GM::ValueType ValueType;
   typedef ValueType value_type;
   typedef typename GM::FactorType FactorType;
   typedef typename GM::OperatorType OperatorType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::IndexType LabelType;

   ViewFunction();
   ViewFunction(const FactorType &);
   template<class Iterator>
      ValueType operator()(Iterator begin)const;
   LabelType shape(const IndexType)const;
   IndexType dimension()const;
   IndexType size()const;

private:
   FactorType const* factor_;
};

template<class GM>
inline
ViewFunction<GM>::ViewFunction()
:  factor_(NULL)
{}

template<class GM>
inline
ViewFunction<GM>::ViewFunction
(
   const typename ViewFunction<GM>::FactorType & factor
)
:  factor_(&factor)
{}

template<class GM>
template<class Iterator>
inline typename ViewFunction<GM>::ValueType
ViewFunction<GM>::operator()
(
   Iterator begin
) const {
   return factor_->operator()(begin);
}

template<class GM>
inline typename ViewFunction<GM>::LabelType
ViewFunction<GM>::shape
(
   const typename ViewFunction<GM>::IndexType index
) const{
   return factor_->numberOfLabels(index);
}

template<class GM>
inline typename ViewFunction<GM>::IndexType
ViewFunction<GM>::dimension() const {
   return factor_->numberOfVariables();
}

template<class GM>
inline typename ViewFunction<GM>::IndexType
ViewFunction<GM>::size() const {
   return factor_->size( );
}

} // namespace opengm

#endif // #ifndef OPENGM_VIEW_FUNCTION_HXX
