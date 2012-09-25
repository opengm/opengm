#pragma once
#ifndef OPENGM_SORTING_HXX
#define OPENGM_SORTING_HXX

/// \cond HIDDEN_SYMBOLS

#include "opengm/datastructures/fast_sequence.hxx"

namespace opengm {
   
template<class ITERATOR,class TYPE_TO_FIND, class INDEXTYPE>
inline bool 
findInSortedSequence
(
   ITERATOR sequenceBegin,
   const INDEXTYPE sequenceSize,
   const TYPE_TO_FIND & toFind,
   INDEXTYPE & position
) {
   if(toFind>static_cast<TYPE_TO_FIND>(sequenceBegin[sequenceSize-1]))
      return false;
   for(INDEXTYPE p=0;p<sequenceSize;++p) {
      if(toFind<static_cast<TYPE_TO_FIND>(sequenceBegin[p]))
         return false;
      else if(toFind==static_cast<TYPE_TO_FIND>(sequenceBegin[p])) {
         position=p;
         return true;
      }
   }
   return false;
}
   
template<class Iterator>
inline bool
isSorted(Iterator begin, Iterator end) {
   typedef typename std::iterator_traits<Iterator>::value_type ValueType;
   if(std::distance(begin, end) > 1) {
      ValueType tmp = static_cast<ValueType>(*begin);
      while(begin != end) {
         if(*begin < tmp) {
            return false;
         }
         tmp = static_cast<ValueType>(*begin);
         ++begin;
      }
   }
   return true;
}

template<class Iterator, class IteratorTag>
struct IteratorAt
{
   inline typename std::iterator_traits<Iterator>::value_type operator()(Iterator iter, const size_t i) const {
      iter += i;
      return *iter;
   }
};

template<class Iterator>
struct IteratorAt<std::random_access_iterator_tag, Iterator>
{
   inline typename std::iterator_traits<Iterator>::value_type operator()(Iterator iter, const size_t i) const {
      return iter[i];
   }
};

template<class Iterator>
typename std::iterator_traits<Iterator>::value_type
   iteratorAt(Iterator iterator, const size_t i) {
      IteratorAt<Iterator, typename std::iterator_traits<Iterator>::iterator_category> iat;
      return iat(iterator, i);
}

template<class T>
struct sortPairByFirst {
   bool operator()(const T & a, const T & b)
      { return a.first < b.first; }
};

template<class Iterator_A, class Iterator_B>
inline void
doubleSort(Iterator_A beginA, Iterator_A endA, Iterator_B beginB) {
   typedef typename std::iterator_traits<Iterator_A>::value_type ValueType_a;
   typedef typename std::iterator_traits<Iterator_B>::value_type ValueType_b;
   typedef std::pair< ValueType_a, ValueType_b> PairType;
   opengm::FastSequence< PairType > pairvector(std::distance(beginA, endA));
   Iterator_A beginA_ = beginA;
   Iterator_B beginB_ = beginB;
   size_t counter = 0;
   while(beginA_ != endA) {
      pairvector[counter].first = *beginA_;
      pairvector[counter].second = *beginB_;
      ++beginA_;
      ++beginB_;
      ++counter;
   }
   sortPairByFirst<PairType > sortfunctor;
   std::sort(pairvector.begin(), pairvector.end(), sortfunctor);
   counter = 0;
   while(beginA != endA) {
      *beginA = pairvector[counter].first;
      *beginB = pairvector[counter].second;
      ++beginA;
      ++beginB;
      ++counter;
   }
}

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_SORTING_HXX

