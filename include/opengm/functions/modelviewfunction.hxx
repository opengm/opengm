#pragma once
#ifndef OPENGM_MODELVIEWFUNCTION_HXX
#define OPENGM_MODELVIEWFUNCTION_HXX

#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// Function that refers to a factor of another GraphicalModel
///
/// \tparam GM type of the graphical model which we want to view
/// \tparam MARRAY type of the array that holds the offset
///
/// \ingroup functions
template<class GM, class MARRAY>
class ModelViewFunction
: public FunctionBase<ModelViewFunction<GM,MARRAY>,
            typename GM::ValueType,
            typename GM::IndexType,
            typename GM::LabelType>
{
public:
   typedef GM GraphicalModelType;
   typedef MARRAY OffsetType;
   typedef typename GM::ValueType ValueType;
   typedef typename GM::ValueType value_type;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;

   ModelViewFunction(const GraphicalModelType& gm, const IndexType factorIndex, const ValueType scale, OffsetType const* offset);
   ModelViewFunction(const GraphicalModelType& gm, const IndexType factorIndex, const ValueType scale);
   ModelViewFunction(OffsetType const* offset);

   template<class Iterator> ValueType operator()(Iterator begin) const;
   size_t size() const;
   LabelType shape(const size_t i) const;
   size_t dimension() const;

private:
   /// ViewType
   enum ViewType {
      /// only view
      VIEW,
      /// view with a offset
      VIEWOFFSET,
      /// only offset
      OFFSET
   };

   GraphicalModelType const* gm_;
   IndexType factorIndex_;
   ValueType scale_;
   OffsetType const* offset_;
   ViewType viewType_;
};

/// constructor
/// \param gm graphical model we want to view
/// \param factorIndex index of the factor of gm we want to view
/// \param scale scaling factor of the view function
/// \param offset pointer to the offset marray
template<class GM, class MARRAY>
inline ModelViewFunction<GM, MARRAY>::ModelViewFunction
(
   const GM& gm ,
   const typename  ModelViewFunction<GM, MARRAY>::IndexType factorIndex,
   const typename  ModelViewFunction<GM, MARRAY>::ValueType scale,
   MARRAY const* offset
)
:  gm_(&gm),
   factorIndex_(factorIndex),
   scale_(scale),
   offset_(offset),
   viewType_(VIEWOFFSET)
{
   //viewType_ = VIEWOFFSET;
   OPENGM_ASSERT((*offset_).size() == gm_->operator[](factorIndex_).size());
   OPENGM_ASSERT((*offset_).dimension() == gm_->operator[](factorIndex_).numberOfVariables());
   for(size_t i=0; i<(*offset_).dimension();++i)
      OPENGM_ASSERT((*offset_).shape(i) == gm_->operator[](factorIndex_).numberOfLabels(i));
}

/// Constructor
/// \param gm graphical model we want to view
/// \param factorIndex index of the factor of gm we want to view
/// \param scale scaling factor of the view function
template<class GM, class MARRAY>
inline ModelViewFunction<GM, MARRAY>::ModelViewFunction
(
   const GM& gm,
   const typename  ModelViewFunction<GM, MARRAY>::IndexType factorIndex,
   const ValueType scale
)
:  gm_(&gm),
   factorIndex_(factorIndex),
   scale_(scale),
   viewType_(VIEW)
{
}

/// Constructor
/// \param offset pointer to the offset marray
template<class GM, class MARRAY>
inline ModelViewFunction<GM, MARRAY>::ModelViewFunction
(
   MARRAY const* offset
)
:  gm_(NULL),
   factorIndex_(0),
   scale_(0),
   offset_(offset),
   viewType_(OFFSET)
{
}

template<class GM, class MARRAY>
template<class Iterator>
inline typename opengm::ModelViewFunction<GM, MARRAY>::ValueType
ModelViewFunction<GM, MARRAY>::operator()
(
   Iterator begin
) const
{
   switch(viewType_) {
      case VIEWOFFSET:
         return scale_*gm_->operator[](factorIndex_)(begin) + (*offset_)(begin);
      case VIEW:
         return scale_*gm_->operator[](factorIndex_)(begin);
      case OFFSET:
         return (*offset_)(begin);
      default:
         break;
   }
   return 0;
}

template<class GM, class MARRAY>
inline typename ModelViewFunction<GM, MARRAY>::LabelType
ModelViewFunction<GM, MARRAY>::shape(const size_t i) const
{
   switch(viewType_) {
      case VIEWOFFSET:
         OPENGM_ASSERT(gm_->operator[](factorIndex_).shape(i)==(*offset_).shape(i));
         return (*offset_).shape(i);
      case VIEW:
         return gm_->operator[](factorIndex_).shape(i);
      case OFFSET:
         return (*offset_).shape(i);
      //default:
   }
   // To avoid compiler error "warning : control reached end
   return 0;
}

template<class GM, class MARRAY>
inline size_t ModelViewFunction<GM, MARRAY>::size() const
{
   switch(viewType_) {
      case VIEWOFFSET:
         return (*offset_).size();
      case VIEW:
         return gm_->operator[](factorIndex_).size();
      case OFFSET:
         return (*offset_).size();
      //default:
   }
   return 0;
}

template<class GM, class MARRAY>
inline size_t ModelViewFunction<GM, MARRAY>::dimension() const
{
   switch(viewType_) {
   case VIEWOFFSET:
      OPENGM_ASSERT(gm_->operator[](factorIndex_).numberOfVariables()==(*offset_).dimension());
      return (*offset_).dimension();
   case VIEW:
      return gm_->operator[](factorIndex_).numberOfVariables();
   case OFFSET:
      return (*offset_).dimension();
   default:
      ;
   }
   // To avoid compiler error "warning : control reached end
   return 0;
}

} // namespace opengm

#endif // #ifndef OPENGM_MODELVIEWFUNCTION_HXX
