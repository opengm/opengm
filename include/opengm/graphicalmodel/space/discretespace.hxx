#pragma once
#ifndef OPENGM_DISCRETE_SPACE_HXX
#define OPENGM_DISCRETE_SPACE_HXX

#include <vector>
#include <limits>

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/space/space_base.hxx"

namespace opengm {

/// Discrete space in which variables can have differently many labels
///
/// \ingroup spaces
template<class I = std::size_t, class L = std::size_t>
class DiscreteSpace 
:  public SpaceBase<DiscreteSpace<I, L>, I, L> {
public:
   typedef I IndexType;
   typedef L LabelType;

   DiscreteSpace();
   template<class Iterator> DiscreteSpace(Iterator, Iterator);
   template<class Iterator> void assign(Iterator, Iterator);
   template<class Iterator> void assignDense(Iterator, Iterator);
   IndexType addVariable(const LabelType);
   IndexType numberOfVariables() const;
   LabelType numberOfLabels(const IndexType) const;
   void reserve(const IndexType);

private:
   std::vector<LabelType> numbersOfLabels_;
};

/// construct an empty label space (with zero variables)
template<class I, class L>
inline
DiscreteSpace<I, L>::DiscreteSpace()
:  numbersOfLabels_() {
}


/// construct a label space in which each variable can attain a different number of labels
/// \param begin iterator to the beginning of a sequence of numbers of labels
/// \param end iterator to the end of a sequence of numbers of labels
///
/// Example:
/// \code
/// size_t numbersOfLabels[] = {4, 2, 3};
/// opengm::DiscreteSpace<> space(numbersOfLabels, numbersOfLabels + 3);
/// \endcode
template<class I, class L>
template<class Iterator>
inline
DiscreteSpace<I, L>::DiscreteSpace
(
   Iterator begin, 
   Iterator end 
)
:  numbersOfLabels_(begin, end) {
   OPENGM_ASSERT(numbersOfLabels_.size()>=0);
   OPENGM_ASSERT(std::numeric_limits<IndexType>::max()>numbersOfLabels_.size());
}

/// assign a new sequence of numbers of labels to an existing space
/// \param begin iterator to the beginning of a sequence of numbers of labels
/// \param end iterator to the end of a sequence of numbers of labels
template<class I, class L>
template<class Iterator>
inline void
DiscreteSpace<I, L>::assign
(
   Iterator begin,
   Iterator end
) {
   numbersOfLabels_.assign(begin, end);
   OPENGM_ASSERT(std::numeric_limits<IndexType>::max()>numbersOfLabels_.size());
}

/// allocate memory for a fixed number of variables
template<class I, class L>
inline void
DiscreteSpace<I, L>::reserve
(
   const I numberOfVariables
) {
   this->numbersOfLabels_.reserve(numberOfVariables);
}

template<class I, class L>
template<class Iterator>
inline void
DiscreteSpace<I, L>::assignDense
(
   Iterator begin,
   Iterator end
) {
   this->assign(begin, end);
}

/// add one more variable
template<class I, class L>
inline typename DiscreteSpace<I, L>::IndexType
DiscreteSpace<I, L>::addVariable
(
   const LabelType numberOfLabels
) {
   numbersOfLabels_.push_back(numberOfLabels);
   OPENGM_ASSERT(std::numeric_limits<IndexType>::max()>numbersOfLabels_.size());
   return numbersOfLabels_.size() - 1;
}

template<class I, class L>
inline typename DiscreteSpace<I, L>::IndexType
DiscreteSpace<I, L>::numberOfVariables() const
{
   return static_cast<IndexType>(numbersOfLabels_.size());
}

template<class I, class L>
inline typename DiscreteSpace<I, L>::LabelType
DiscreteSpace<I, L>::numberOfLabels
(
   const IndexType dimension
) const
{
   return numbersOfLabels_[dimension];
}

} // namespace opengm

#endif // #ifndef OPENGM_DISCRETE_SPACE_HXX
