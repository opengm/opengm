#pragma once
#ifndef OPENGM_STATIC_SIMPLE_DISCRETE_SPACE_HXX
#define OPENGM_STATIC_SIMPLE_DISCRETE_SPACE_HXX

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/space/space_base.hxx"

namespace opengm {

/// Discrete space in which all variables have the same number of labels
///
/// \ingroup spaces
template<size_t LABELS, class I = std::size_t , class L = std::size_t>
class StaticSimpleDiscreteSpace
:  public SpaceBase<StaticSimpleDiscreteSpace<LABELS, I, L>, I, L>
{
public:
   typedef I IndexType;
   typedef L LabelType;

   StaticSimpleDiscreteSpace();
   StaticSimpleDiscreteSpace(const IndexType);
   void assign(const IndexType);
   template<class Iterator> void assignDense(Iterator, Iterator);
   IndexType addVariable(const LabelType );
   IndexType numberOfVariables() const;
   LabelType numberOfLabels(const IndexType) const;
   bool isSimpleSpace()const;

private:
   IndexType numberOfVariables_;
};

template<size_t LABELS,class I, class L>
inline
StaticSimpleDiscreteSpace<LABELS,I, L>::StaticSimpleDiscreteSpace()
:  numberOfVariables_()
{}

template<size_t LABELS,class I, class L>
inline
StaticSimpleDiscreteSpace<LABELS,I, L>::StaticSimpleDiscreteSpace(
   const IndexType numberOfVariables
)
:  numberOfVariables_(numberOfVariables)
{}

template<size_t LABELS,class I, class L>
template<class Iterator>
inline void
StaticSimpleDiscreteSpace<LABELS,I, L>::assignDense(
   Iterator begin,
   Iterator end
) {
   numberOfVariables_=std::distance(begin, end);
   numberOfVariables_=static_cast<L>(*begin);
   while(begin!=end) {
      if(LABELS!=static_cast<size_t>(*begin)) {
         throw opengm::RuntimeError("*begin==LABELS_ is violated \
         in StaticSimpleDiscreteSpace::assignDense ");
      }
      ++begin;
   }
}

template<size_t LABELS,class I, class L>
inline void
StaticSimpleDiscreteSpace<LABELS,I, L>::assign(
   const IndexType numberOfVariables
) {
   numberOfVariables_ = numberOfVariables;
}

template<size_t LABELS,class I, class L>
inline I
StaticSimpleDiscreteSpace<LABELS,I, L>::addVariable(
   const L numberOfLabels
) {
   if(numberOfLabels!=static_cast<L> (LABELS)) {
      throw opengm::RuntimeError("numberOfLabels==LABELS is violated \
      in StaticSimpleDiscreteSpace::addVariable ");
   }
}

template<size_t LABELS,class I, class L>
inline typename StaticSimpleDiscreteSpace<LABELS,I, L>::IndexType
StaticSimpleDiscreteSpace<LABELS,I, L>::numberOfVariables() const {
   return numberOfVariables_;
}

template<size_t LABELS,class I, class L>
inline typename StaticSimpleDiscreteSpace<LABELS,I, L>::LabelType
StaticSimpleDiscreteSpace<LABELS,I, L>::numberOfLabels(
   const IndexType dimension
) const {
   return static_cast<L> LABELS;
}

template<size_t LABELS,class I, class L>
inline bool
StaticSimpleDiscreteSpace<LABELS,I, L>::isSimpleSpace() const
{
   return true;
}

} // namespace opengm

#endif // #ifndef OPENGM_STATIC_SIMPLE_DISCRETE_SPACE_HXX
