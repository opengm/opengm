#pragma once
#ifndef OPENGM_GRID_DISCRETE_SPACE_HXX
#define OPENGM_GRID_DISCRETE_SPACE_HXX

/// \cond HIDDEN_SYMBOLS

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/space/space_base.hxx"

namespace opengm {

/// \ingroup spaces
///
/// \ingroup spaces
template<class I = std::size_t , class L = std::size_t >
class GridSpace :public SpaceBase<GridSpace<I,L>,I,L>{
public:
   typedef I IndexType;
   typedef L LabelType;

   GridSpace();
   GridSpace(const IndexType,const IndexType, const LabelType);
   void assign(const IndexType ,const IndexType, const LabelType);
   IndexType numberOfVariables() const;
   IndexType dimX() const;
   IndexType dimY() const;
   LabelType numberOfLabels()const;
   LabelType numberOfLabels(const IndexType) const;
   LabelType numberOfLabels(const IndexType,const IndexType) const;
   bool isSimpleSpace() const;
private:
   IndexType dimX_;
   IndexType dimY_;
   LabelType numberOfStates_;
};

template<class I, class L>
inline
GridSpace<I, L>::GridSpace()
:  dimX_(),
   dimY_(),
   numberOfStates_()
{}

template<class I, class L>
inline
GridSpace<I, L>::GridSpace
(
   const typename GridSpace<I, L>::IndexType dimX,
   const typename GridSpace<I, L>::IndexType dimY,
   const typename GridSpace<I, L>::LabelType numberOfLabels
)
:  dimX_(dimX),
   dimY_(dimY),
   numberOfStates_(numberOfLabels)
{}

template<class I, class L>
inline void
GridSpace<I, L>::assign
(
   const typename GridSpace<I, L>::IndexType dimX,
   const typename GridSpace<I, L>::IndexType dimY,
   const typename GridSpace<I, L>::LabelType numberOfLabels
) {
   numberOfStates_ = numberOfLabels;
   dimX_ = dimX;
   dimY_ = dimY;
}

template<class I, class L>
inline typename GridSpace<I, L>::IndexType
GridSpace<I, L>::numberOfVariables() const{
   return dimX_*dimY_;
}

template<class I, class L>
inline typename GridSpace<I, L>::IndexType
GridSpace<I, L>::dimX() const{
   return dimX_;
}

template<class I, class L>
inline typename GridSpace<I, L>::IndexType
GridSpace<I, L>::dimY() const{
   return dimY_;
}

template<class I, class L>
inline typename GridSpace<I, L>::LabelType
GridSpace<I, L>::numberOfLabels() const{
   return numberOfStates_;
}

template<class I, class L>
inline typename GridSpace<I, L>::LabelType
GridSpace<I, L>::numberOfLabels
(
   const typename GridSpace<I, L>::IndexType dimension
) const{
   return numberOfStates_;
}

template<class I, class L>
inline bool
GridSpace<I, L>::isSimpleSpace() const{
   return true;
}

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_GRID_DISCRETE_SPACE_HXX
