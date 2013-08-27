#pragma once
#ifndef OPENGM_SPACE_BASE_HXX
#define OPENGM_SPACE_BASE_HXX

#include <vector>
#include <limits>
#include <typeinfo>

#include "opengm/opengm.hxx"

namespace opengm {

/// Interface of label spaces
template<class SPACE, class I = std::size_t, class L = std::size_t>
class SpaceBase {
public:   
   typedef I IndexType;
   typedef L LabelType;

   IndexType numberOfVariables() const; // must be implemented in Space
   LabelType numberOfLabels(const IndexType) const; // must be implemented in Space
   template<class Iterator> void assignDense(Iterator, Iterator);
   IndexType addVariable(const LabelType);
   bool isSimpleSpace() const;
};

template<class SPACE, class I, class L>
inline bool
SpaceBase<SPACE, I, L>::isSimpleSpace() const {
   const IndexType numVar = static_cast<SPACE const*>(this)->numberOfVariables();
   const IndexType l = static_cast<SPACE const*>(this)->numberOfLabels(0);
   for(size_t i=1;i<numVar;++i) {
      if(l!=static_cast<SPACE const *>(this)->numberOfLabels(i)) {
         return false;
      }
   }
   return true;
}

template<class SPACE, class I, class L>
template<class Iterator>
inline void
SpaceBase<SPACE, I, L>::assignDense
(
   Iterator begin,
   Iterator end
) {
   throw RuntimeError(std::string("assignDense(begin, end) is not implemented in ")+typeid(SPACE).name());
}

template<class SPACE, class I, class L>
inline typename SpaceBase<SPACE, I, L>::IndexType
SpaceBase<SPACE, I, L>::addVariable
(
   const LabelType numberOfLabels
) {
   throw RuntimeError(std::string("addVariable(numberOfLabels) is not implemented in ")+typeid(SPACE).name());
   return IndexType();
}

} // namespace opengm

#endif // #ifndef OPENGM_SPACE_BASE_HXX
