#pragma once
#ifndef OPENGM_SCALEDVIEWFUNCTION_HXX
#define OPENGM_SCALEDVIEWFUNCTION_HXX

#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// Function that scales a factor of another graphical model
///
/// \ingroup functions
template<class GM> class ScaledViewFunction
: public FunctionBase<ScaledViewFunction<GM>,
   typename GM::ValueType,
   typename GM::IndexType,
   typename GM::LabelType>
{
public:
   typedef typename GM::ValueType ValueType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;
   typedef typename GM::OperatorType OperatorType;

   ScaledViewFunction(const std::vector<IndexType>&);
   ScaledViewFunction(const GM&, const IndexType, const ValueType);
   template<class Iterator> ValueType operator()(Iterator begin) const;
   size_t dimension() const;
   size_t shape(const size_t) const;
   size_t size() const;

private:
   GM const* gm_;
   IndexType factorIndex_;
   ValueType scale_;
   std::vector<IndexType> shape_;
   size_t size_;
};

/// Constructor
/// \param gm graphical model we want to view
/// \param factorIndex index of the factor of gm we want to view
/// \param scale scaling factor of the view function
template<class GM>
inline
ScaledViewFunction<GM>::ScaledViewFunction
(
   const GM& gm,
   const typename ScaledViewFunction<GM>::IndexType factorIndex,
   const ValueType scale
)
:  gm_(&gm),
   factorIndex_(factorIndex),
   scale_(scale)
{
   size_=1;
   shape_.resize(gm[factorIndex].numberOfVariables());
   for(size_t i=0; i<gm[factorIndex].numberOfVariables();++i) {
      shape_[i] = gm[factorIndex].numberOfLabels(i);
      size_*=gm[factorIndex].numberOfLabels(i);
   }
}
/// Constructor
/// \param shape shape of the function
template<class GM>
inline
ScaledViewFunction<GM>::ScaledViewFunction
(
   const std::vector<IndexType>& shape
)
:  gm_(NULL),
   factorIndex_(0),
   scale_(0),
   shape_(shape)
{
   size_=1;
   for(size_t i=0; i<shape_.size();++i) {
      size_*=shape[i];
   }
}

template<class GM>
inline size_t
ScaledViewFunction<GM>::size()const
{
   return size_;
}

template<class GM>
template<class Iterator>
inline typename ScaledViewFunction<GM>::ValueType
ScaledViewFunction<GM>::operator()
(
   Iterator begin
) const
{
   if(gm_==NULL) {
      return OperatorType::template neutral<ValueType>();
   }
   else {
      return scale_*gm_->operator[](factorIndex_)(begin);
   }
}

template<class GM>
inline size_t
ScaledViewFunction<GM>::shape(
   const size_t i
) const {
   return shape_[i];
}

template<class GM>
inline size_t
ScaledViewFunction<GM>::dimension() const {
   return shape_.size();
}

} // namespace opengm

#endif // #ifndef OPENGM_SCALEDVIEWFUNCTION_HXX
