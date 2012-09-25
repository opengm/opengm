#pragma once
#ifndef OPENGM_VIEW_FIX_VARIABLES_FUNCTION_HXX
#define OPENGM_VIEW_FIX_VARIABLES_FUNCTION_HXX

#include "opengm/functions/function_properties_base.hxx"

namespace opengm {
   
/// \cond HIDDEN_SYMBOLS
template<class I, class L>
struct PositionAndLabel {
   PositionAndLabel(const I = 0, const L = 0);
   I position_;
   L label_;
};
/// \endcond

/// Funcion that refers to a factor of another GraphicalModel in which some variables are fixed
///
/// \ingroup functions
template<class GM>
class ViewFixVariablesFunction
: public FunctionBase<ViewFixVariablesFunction<GM>, typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> {
public:
   typedef typename GM::ValueType ValueType;
   typedef ValueType value_type;
   typedef typename GM::FactorType FactorType;
   typedef typename GM::OperatorType OperatorType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;

   ViewFixVariablesFunction();
   ViewFixVariablesFunction(const FactorType &, const std::vector<PositionAndLabel<IndexType, LabelType> > &);
   template<class POSITION_AND_TYPE_CONTAINER>
   ViewFixVariablesFunction(const FactorType &, const POSITION_AND_TYPE_CONTAINER &);
   template<class Iterator>
      ValueType operator()(Iterator begin)const;
   IndexType shape(const IndexType)const;
   IndexType dimension()const;
   IndexType size()const;

private:
   FactorType const* factor_;
   std::vector<PositionAndLabel<IndexType, LabelType> > positionAndLabels_;
   mutable std::vector<LabelType> iteratorBuffer_;
   mutable bool computedSize_;
   mutable size_t size_;
   std::vector<size_t> lookUp_;
};

template<class I, class L>
PositionAndLabel<I, L>::PositionAndLabel
(
   const I position,
   const L label
)
:  position_(position),
   label_(label)
{}

template<class GM>
inline
ViewFixVariablesFunction<GM>::ViewFixVariablesFunction()
:  factor_(NULL),
   iteratorBuffer_(),
   computedSize_(false),
   size_(1)
{}

template<class GM>
inline
ViewFixVariablesFunction<GM>::ViewFixVariablesFunction
(
   const typename ViewFixVariablesFunction<GM>::FactorType& factor,
   const std::vector<PositionAndLabel<typename GM::IndexType, typename GM::LabelType> >& positionAndLabels
)
:  factor_(&factor),
   positionAndLabels_(positionAndLabels),
   iteratorBuffer_(factor.numberOfVariables()),
   computedSize_(false),
   size_(1),
   lookUp_(factor.numberOfVariables()-positionAndLabels.size())
{
   if(opengm::NO_DEBUG==false) {
      for(size_t i=0; i<positionAndLabels_.size(); ++i) {
         OPENGM_ASSERT(positionAndLabels_[i].label_ < factor_->numberOfLabels(positionAndLabels_[i].position_));
      }
   }
   for(size_t ind=0; ind<lookUp_.size(); ++ind) {
      size_t add=0;
      for(IndexType i=0; i<positionAndLabels_.size(); ++i) {
         if( positionAndLabels_[i].position_ <= ind+add) {
            ++add;
         }
      }
      lookUp_[ind]=ind+add;
   }
}

/// constructor
/// \tparam POSITION_AND_TYPE_CONTAINER container holding positions and labels of the variable to fix
/// \param factor the factor to reference
/// \param positionAndLabels positions and labels of the variable to fix
template<class GM>
template<class POSITION_AND_TYPE_CONTAINER>
inline
ViewFixVariablesFunction<GM>::ViewFixVariablesFunction
(
   const typename ViewFixVariablesFunction<GM>::FactorType& factor,
   const POSITION_AND_TYPE_CONTAINER& positionAndLabels
)
:  factor_(&factor),
   positionAndLabels_(positionAndLabels.begin(), positionAndLabels.end()),
   iteratorBuffer_(factor.numberOfVariables()),
   computedSize_(false),
   size_(1),
   lookUp_(factor.numberOfVariables()-positionAndLabels.size())
{
   if(!(opengm::NO_DEBUG)) {
      for(size_t i=0; i<positionAndLabels_.size(); ++i) {
         OPENGM_ASSERT(positionAndLabels_[i].label_ < factor_->numberOfLabels(positionAndLabels_[i].position_));
      }
   }
   for(size_t ind=0; ind<lookUp_.size(); ++ind) {
      size_t add = 0;
      for(IndexType i=0; i<positionAndLabels_.size(); ++i) {
         if( positionAndLabels_[i].position_ <= ind+add) {
            ++add;
         }
      }
      lookUp_[ind]=ind+add;
   }
}

template<class GM>
template<class Iterator>
inline typename ViewFixVariablesFunction<GM>::ValueType
ViewFixVariablesFunction<GM>::operator()
(
   Iterator begin
) const
{
   OPENGM_ASSERT(factor_ != NULL);
   for(size_t ind=0; ind<lookUp_.size(); ++ind) {
      iteratorBuffer_[lookUp_[ind]]=*begin;
      ++begin;
   }
   for(size_t i=0; i<positionAndLabels_.size(); ++i) {
      iteratorBuffer_[positionAndLabels_[i].position_]
         = positionAndLabels_[i].label_;
   }
   return factor_->operator()(iteratorBuffer_.begin());
}

template<class GM>
inline typename ViewFixVariablesFunction<GM>::IndexType
ViewFixVariablesFunction<GM>::shape
(
   const typename ViewFixVariablesFunction<GM>::IndexType index
) const
{
   OPENGM_ASSERT(factor_ != NULL);
   size_t add = 0;
   for(IndexType i=0; i<positionAndLabels_.size(); ++i) {
      if( positionAndLabels_[i].position_ <= index+add) {
         ++add;
      }
   }
   OPENGM_ASSERT(index + add < factor_->numberOfVariables());
   return factor_->numberOfLabels(index + add);
}

template<class GM>
inline typename ViewFixVariablesFunction<GM>::IndexType
ViewFixVariablesFunction<GM>::dimension() const
{
   OPENGM_ASSERT(factor_!=NULL);
   return factor_->numberOfVariables() - positionAndLabels_.size();
}

template<class GM>
inline typename ViewFixVariablesFunction<GM>::IndexType
ViewFixVariablesFunction<GM>::size() const
{
   OPENGM_ASSERT(factor_!=NULL);
   if(computedSize_== false) {
      size_ = factor_->size();
      for(IndexType j=0; j<positionAndLabels_.size(); ++j) {
         size_ /= (factor_->numberOfLabels(positionAndLabels_[j].position_));
      }
      computedSize_ = true;
   }
   return size_;
}

} // namespace opengm

#endif // #ifndef OPENGM_VIEW_FIX_VARIABLES_FUNCTION_HXX
