#pragma once
#ifndef OPENGM_VIEW_SPACE_HXX
#define OPENGM_VIEW_SPACE_HXX

#include <vector>
#include <limits>

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/space/space_base.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm {

/// View Space 
///
/// \ingroup spaces
template<class GM>
class ViewSpace
: public SpaceBase<ViewSpace<GM>, typename GM::IndexType, typename GM::LabelType> {
public:
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;
   typedef typename GM::SpaceType SrcSpaceType;

   ViewSpace();
   ViewSpace(const GM &);
   void assign(const GM &);
   IndexType addVariable(const LabelType);
   IndexType numberOfVariables() const;
   LabelType numberOfLabels(const IndexType) const;

private:
   SrcSpaceType const* space_;
};

template<class GM>
inline
ViewSpace<GM>::ViewSpace()
:  space_(NULL)
{}

template<class GM>
inline
ViewSpace<GM>::ViewSpace(
   const GM & gm
)
:  space_(&gm.space())
{}

template<class GM>
inline void
ViewSpace<GM>::assign(
   const GM & gm
)
{
   space_=& gm.space();
}

template<class GM>
inline typename  ViewSpace<GM>::IndexType
ViewSpace<GM>::addVariable(
   const LabelType numberOfLabels
)
{
   opengm::RuntimeError("cannot add Variable with a ViewSpace as a space object");
}

template<class GM>
inline typename ViewSpace<GM>::IndexType
ViewSpace<GM>::numberOfVariables() const
{
   OPENGM_ASSERT(space_!=NULL);
   return space_->numberOfVariables();
}

template<class GM>
inline typename ViewSpace<GM>::IndexType
ViewSpace<GM>::numberOfLabels
(
   const IndexType dimension
) const
{
   OPENGM_ASSERT(space_!=NULL);
   return space_->numberOfLabels(dimension);
}

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_VIEW_SPACE_HXX
