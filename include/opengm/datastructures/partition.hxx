#pragma once
#ifndef OPENGM_PARTITION_HXX
#define OPENGM_PARTITION_HXX

#include <vector>
#include <map>

#ifdef WITH_BOOST
#include <boost/unordered_map.hpp>
#endif


namespace opengm {

/// Disjoint set data structure with path compression.
/// \ingroup datastructures
template<class T = size_t>
class Partition {
public:
   typedef T value_type;

   Partition();
   Partition(const value_type&);

   // query
   value_type find(const value_type&) const; // without path compression
   value_type find(value_type); // with path compression
   value_type numberOfElements() const;
   value_type numberOfSets() const;
   template<class Iterator> void elementLabeling(Iterator) const;
   template<class Iterator> void representatives(Iterator) const;
   void representativeLabeling(std::map<value_type, value_type>&) const;


   #ifdef WITH_BOOST
   void representativeLabeling(boost::unordered_map<value_type, value_type>&) const;
   #endif



   // manipulation
   void reset(const value_type&);
   void merge(value_type, value_type);
   void insert(const value_type&);

private:
   std::vector<value_type> parents_;
   std::vector<value_type> ranks_;
   value_type numberOfElements_;
   value_type numberOfSets_;
};

/// Construct a partition.
template<class T>
Partition<T>::Partition()
: parents_(),
  ranks_(),
  numberOfElements_(0),
  numberOfSets_(0)
{}

/// Construct a partition.
///
/// \param size Number of distinct sets.
///
template<class T>
inline
Partition<T>::Partition
(
   const value_type& size
)
: parents_(static_cast<size_t>(size)),
  ranks_(static_cast<size_t>(size)),
  numberOfElements_(size),
  numberOfSets_(size)
{
   for(T j=0; j<size; ++j) {
      parents_[static_cast<size_t>(j)] = j;
   }
}

/// Reset a partition such that each set contains precisely one element
///
/// \param size Number of distinct sets.
///
template<class T>
inline void
Partition<T>::reset
(
   const value_type& size
)
{
   numberOfElements_ = size;
   numberOfSets_ = size;
   ranks_.resize(static_cast<size_t>(size));
   parents_.resize(static_cast<size_t>(size));
   for(T j=0; j<size; ++j) {
      ranks_[static_cast<size_t>(j)] = 0;
      parents_[static_cast<size_t>(j)] = j;
   }
}

/// Find the representative element of the set that contains the given element.
///
/// This constant function does not compress the search path.
///
/// \param element Element.
///
template<class T>
inline typename Partition<T>::value_type
Partition<T>::find
(
   const value_type& element
) const
{
   // find the root
   value_type root = element;
   while(parents_[static_cast<size_t>(root)] != root) {
      root = parents_[static_cast<size_t>(root)];
   }
   return root;
}

/// Find the representative element of the set that contains the given element.
///
/// This mutable function compresses the search path.
///
/// \param element Element.
///
template<class T>
inline typename Partition<T>::value_type
Partition<T>::find
(
   value_type element // copy to work with
)
{
   // find the root
   value_type root = element;
   while(parents_[static_cast<size_t>(root)] != root) {
      root = parents_[static_cast<size_t>(root)];
   }
   // path compression
   while(element != root) {
      value_type tmp = parents_[static_cast<size_t>(element)];
      parents_[static_cast<size_t>(element)] = root;
      element = tmp;
   }
   return root;
}

/// Merge two sets.
///
/// \param element1 Element in the first set.
/// \param element2 Element in the second set.
///
template<class T>
inline void
Partition<T>::merge
(
   value_type element1,
   value_type element2
)
{
   // merge by rank
   element1 = find(element1);
   element2 = find(element2);
   if(ranks_[static_cast<size_t>(element1)] < ranks_[static_cast<size_t>(element2)]) {
      parents_[static_cast<size_t>(element1)] = element2;
      --numberOfSets_;
   }
   else if(ranks_[static_cast<size_t>(element1)] > ranks_[static_cast<size_t>(element2)]) {
      parents_[static_cast<size_t>(element2)] = element1;
      --numberOfSets_;
   }
   else if(element1 != element2) {
      parents_[static_cast<size_t>(element2)] = element1;
      ++ranks_[static_cast<size_t>(element1)];
      --numberOfSets_;
   }
}

/// Insert new sets.
///
/// \param number Number of sets to insert.
///
template<class T>
inline void
Partition<T>::insert
(
   const value_type& number
)
{
   ranks_.insert(ranks_.end(), static_cast<size_t>(number), T(0));
   parents_.insert(parents_.end(), static_cast<size_t>(number), T(0));
   for(value_type j=numberOfElements_; j<numberOfElements_+number; ++j) {
      parents_[static_cast<size_t>(j)] = j;
   }
   numberOfElements_ += number;
   numberOfSets_ += number;
}

/// Output all elements which are set representatives.
///
/// \param it (Output) Iterator into a container.
///
template<class T>
template<class Iterator>
inline void
Partition<T>::representatives
(
   Iterator it
) const
{
   for(value_type j=0; j<numberOfElements(); ++j) {
      if(parents_[static_cast<size_t>(j)] == j) {
         *it = j;
         ++it;
      }
   }
}

/// Output a continuous labeling of the representative elements.
///
/// \param out (Output) A map that assigns each representative element to its label.
///
template<class T>
inline void
Partition<T>::representativeLabeling
(
   std::map<value_type, value_type>& out
) const
{
   out.clear();
   std::vector<value_type> r(static_cast<size_t>(numberOfSets()));
   representatives(r.begin());
   for(T j=0; j<numberOfSets(); ++j) {
      out[ r[static_cast<size_t>(j)] ] = j;
   }
}


#ifdef WITH_BOOST
/// Output a continuous labeling of the representative elements.
///
/// \param out (Output) A map that assigns each representative element to its label.
///
template<class T>
inline void
Partition<T>::representativeLabeling
(
   boost::unordered_map<value_type, value_type>& out
) const
{
   out.clear();
   std::vector<value_type> r(static_cast<size_t>(numberOfSets()));
   representatives(r.begin());
   for(T j=0; j<numberOfSets(); ++j) {
      out[ r[static_cast<size_t>(j)] ] = j;
   }
}

#endif 

/// Output a continuous labeling of all elements.
///
/// \param out (Output) Iterator into a container in which the j-th entry is the label of the element j.
///
template<class T>
template<class Iterator>
inline void
Partition<T>::elementLabeling
(
   Iterator out
) const
{
   std::map<value_type, value_type> rl;
   representativeLabeling(rl);
   for(value_type j=0; j<numberOfElements(); ++j) {
      *out = rl[find(j)];
      ++out;
   }
}

template<class T>
inline typename Partition<T>::value_type
Partition<T>::numberOfElements() const
{
   return numberOfElements_;
}

template<class T>
inline typename Partition<T>::value_type
Partition<T>::numberOfSets() const
{
   return numberOfSets_;
}

} // namespace opengm

#endif // #ifndef OPENGM_PARTITION_HXX
