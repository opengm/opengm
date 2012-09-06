#pragma once
#ifndef OPENGM_VECTOR_VIEW_SPACE_HXX
#define OPENGM_VECTOR_VIEW_SPACE_HXX

/// \cond HIDDEN_SYMBOLS

#include <vector>
#include <limits>

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/space/space_base.hxx"

namespace opengm {

/// Label space in which variables can have different numbers of labels
///
/// \ingroup spaces
template<class I = std::size_t, class L = std::size_t>
class VectorViewSpace
: public SpaceBase<VectorViewSpace<I,L>,I,L>
{
public:
   typedef I IndexType;
   typedef L LabelType;

   VectorViewSpace();
   VectorViewSpace(const std::vector<LabelType>&);
   IndexType addVariable(const LabelType);
   IndexType numberOfVariables() const;
   LabelType numberOfLabels(const IndexType) const;

private:
   std::vector<LabelType> const* numbersOfLabels_;
};

template<class I, class L>
inline
VectorViewSpace<I, L>::VectorViewSpace()
:  numbersOfLabels_(NULL)
{}

template<class I, class L>
inline
VectorViewSpace<I, L>::VectorViewSpace
(
   const std::vector<LabelType>& spaceVector
)
:  numbersOfLabels_(&spaceVector)
{
   OPENGM_ASSERT(numbersOfLabels_->size()>0);
}

template<class I, class L>
inline typename VectorViewSpace<I, L>::IndexType
VectorViewSpace<I, L>::addVariable(
   const LabelType numberOfLabels
) {
   throw opengm::RuntimeError("attempt to add a variable to VectorViewSpace");
}

template<class I, class L>
inline typename VectorViewSpace<I, L>::IndexType
VectorViewSpace<I, L>::numberOfVariables() const
{
   return static_cast<IndexType>(numbersOfLabels_->size());
}

template<class I, class L>
inline typename VectorViewSpace<I, L>::LabelType
VectorViewSpace<I, L>::numberOfLabels(
   const IndexType dimension
) const
{
   OPENGM_ASSERT(dimension<numbersOfLabels_->size());
   return numbersOfLabels_->operator[](dimension);
}

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_VECTOR_VIEW_SPACE_HXX
