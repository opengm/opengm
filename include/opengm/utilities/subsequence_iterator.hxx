#ifndef OPENGM_SUBSEQUENCE_ITERATOR_HXX_
#define OPENGM_SUBSEQUENCE_ITERATOR_HXX_

#include <iterator>

namespace opengm {

/*********************
 * class definitions *
 *********************/
template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
class SubsequenceIterator : public std::iterator<std::random_access_iterator_tag, typename SEQUENCE_ITERATOR_TYPE::value_type > {
public:
   // typedefs
   typedef SEQUENCE_ITERATOR_TYPE                                                               SequenceIteratorType;
   typedef SUBSEQUENCE_INDICES_ITERATOR_TYPE                                                    SubsequenceIndicesIteratorType;
   typedef SubsequenceIterator<SequenceIteratorType, SubsequenceIndicesIteratorType>            SubsequenceIteratorType;
   typedef typename SequenceIteratorType::value_type                                            value_type;
   typedef std::random_access_iterator_tag                                                      iterator_category;
   typedef typename std::iterator<std::random_access_iterator_tag, value_type>::difference_type difference_type;
   typedef value_type&                                                                          reference;
   typedef value_type*                                                                          pointer;

   // constructor
   SubsequenceIterator();
   SubsequenceIterator(const SubsequenceIteratorType& copy);
   SubsequenceIterator(const SequenceIteratorType sequenceBegin, const SubsequenceIndicesIteratorType subsequenceIndicesBegin, const size_t subsequenceIndicesPosition = 0);

   // assignment
   SubsequenceIteratorType& operator=(const SubsequenceIteratorType& copy);

   // increment
   SubsequenceIteratorType& operator++();  // PREFIX
   SubsequenceIteratorType  operator++(int);  // POSTFIX
   SubsequenceIteratorType  operator+(const difference_type& n) const;
   SubsequenceIteratorType& operator+=(const difference_type& n);

   // decrement
   SubsequenceIteratorType& operator--();  // PREFIX
   SubsequenceIteratorType  operator--(int);  // POSTFIX
   SubsequenceIteratorType  operator-(const difference_type& n) const;
   SubsequenceIteratorType& operator-=(const difference_type& n);

   // access
   const value_type& operator*() const;
   const value_type* operator->() const;
   const value_type& operator[](const difference_type& n) const;
protected:
   // storage
   SequenceIteratorType           sequenceBegin_;
   SubsequenceIndicesIteratorType subsequenceIndicesBegin_;
   size_t                         subsequenceIndicesPosition_;

   // friends
   inline friend bool            operator==(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2) {
      return (iter1.sequenceBegin_ == iter2.sequenceBegin_) && (iter1.subsequenceIndicesBegin_ == iter2.subsequenceIndicesBegin_) && (iter1.subsequenceIndicesPosition_ == iter2.subsequenceIndicesPosition_);
   }
   inline friend bool            operator!=(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2) {
      return (iter1.sequenceBegin_ != iter2.sequenceBegin_) || (iter1.subsequenceIndicesBegin_ != iter2.subsequenceIndicesBegin_) || (iter1.subsequenceIndicesPosition_ != iter2.subsequenceIndicesPosition_);
   }
   inline friend bool            operator<(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2) {
      return (iter1.sequenceBegin_ == iter2.sequenceBegin_) && (iter1.subsequenceIndicesBegin_ == iter2.subsequenceIndicesBegin_) && (iter1.subsequenceIndicesPosition_ < iter2.subsequenceIndicesPosition_);
   }
   inline friend bool            operator>(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2) {
      return (iter1.sequenceBegin_ == iter2.sequenceBegin_) && (iter1.subsequenceIndicesBegin_ == iter2.subsequenceIndicesBegin_) && (iter1.subsequenceIndicesPosition_ > iter2.subsequenceIndicesPosition_);
   }
   inline friend bool            operator<=(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2) {
      return (iter1.sequenceBegin_ == iter2.sequenceBegin_) && (iter1.subsequenceIndicesBegin_ == iter2.subsequenceIndicesBegin_) && (iter1.subsequenceIndicesPosition_ <= iter2.subsequenceIndicesPosition_);
   }
   inline friend bool            operator>=(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2) {
      return (iter1.sequenceBegin_ == iter2.sequenceBegin_) && (iter1.subsequenceIndicesBegin_ == iter2.subsequenceIndicesBegin_) && (iter1.subsequenceIndicesPosition_ >= iter2.subsequenceIndicesPosition_);
   }
   inline friend difference_type operator-(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2) {
      return iter1.subsequenceIndicesPosition_ - iter2.subsequenceIndicesPosition_;
   }
};

/***********************
 * class documentation *
 ***********************/
/*! \file subsequence_iterator.hxx
 *  \brief Provides implementation for class SubsequenceIterator.
 */

/*! \class SubsequenceIterator
 *  \brief Defines the const iterator type to iterate over the subset of a
 *         sequence.
 *
 *  The SubsequenceIterator class implements an STL compliant constant random
 *  access iterator to iterate over the subset of a sequence.
 *
 *  \tparam SEQUENCE_ITERATOR_TYPE The iterator type used to iterate over the
 *                                 sequence. Has to be a random access iterator.
 *  \tparam SUBSEQUENCE_INDICES_ITERATOR_TYPE The iterator type used to iterate
 *                                            over the subsequence indices. Has
 *                                            to be a random access iterator.
 */

/*! \typedef SubsequenceIterator::SequenceIteratorType
 *  \brief Typedef of the SEQUENCE_ITERATOR_TYPE template parameter from the
 *         class SubsequenceIterator.
 */

/*! \typedef SubsequenceIterator::SubsequenceIndicesIteratorType
 *  \brief Typedef of the SUBSEQUENCE_INDICES_ITERATOR_TYPE template parameter
 *         from the class SubsequenceIterator.
 */

/*! \typedef SubsequenceIterator::SubsequenceIteratorType
 *  \brief Typedef of the class SubsequenceIterator with appropriate template
 *         parameter.
 */

/*! \typedef SubsequenceIterator::value_type
 *  \brief Value type of the sequence adapted from
 *         SubsequenceIterator::SequenceIteratorType.
 */

/*! \typedef SubsequenceIterator::iterator_category
 *  \brief STL compliant typedef of the iterator category.
 */

/*! \typedef SubsequenceIterator::difference_type
 *  \brief STL compliant typedef of the iterator difference type.
 */

/*! \typedef SubsequenceIterator::reference
 *  \brief STL compliant typedef of the iterator reference type.
 */

/*! \typedef SubsequenceIterator::pointer
 *  \brief STL compliant typedef of the iterator pointer type.
 */

/*! \fn SubsequenceIterator::SubsequenceIterator()
 *  \brief SubsequenceIterator constructor.
 *
 *  This constructor will create an empty SubsequenceIterator.
 */

/*! \fn SubsequenceIterator::SubsequenceIterator(const SubsequenceIteratorType& copy)
 *  \brief SubsequenceIterator constructor.
 *
 *  This constructor will create a copy of an existing SubsequenceIterator.
 *
 *  \param[in] copy The iterator which will be copied.
 */

/*! \fn SubsequenceIterator::SubsequenceIterator(const SequenceIteratorType sequenceBegin, const SubsequenceIndicesIteratorType subsequenceIndicesBegin, const size_t subsequenceIndicesPosition = 0)
 *  \brief SubsequenceIterator constructor.
 *
 *  This constructor will create a new SubsequenceIterator.
 *
 *  \param[in] sequenceBegin Iterator pointing to the begin of the sequence over
 *                           which the subsequence iterator will iterate.
 *  \param[in] subsequenceIndicesBegin Iterator pointing to the begin of the
 *                                     subsequence indices. They are used as
 *                                     indices for the complete sequence.
 *  \param[in] subsequenceIndicesPosition Current position of the
 *                                        subsequence indices.
 */

/*! \fn SubsequenceIteratorType& SubsequenceIterator::operator=(const SubsequenceIteratorType& copy)
 *  \brief SubsequenceIterator assignment operator.
 *
 *  \param[in] copy The iterator which will be copied.
 *
 *  \return Reference to iterator which got updated by the new assignment.
 */

/*! \fn SubsequenceIteratorType& SubsequenceIterator::operator++()
 *   \brief Prefix increment operator.
 *
 *  \return Reference to iterator which got incremented by one.
 */

/*! \fn SubsequenceIteratorType SubsequenceIterator::operator++(int)
 *  \brief Postfix increment operator.
 *
 *  \return A new iterator which points to the same object as the iterator
 *          before the call to the postfix increment operator.
 */

/*! \fn SubsequenceIteratorType SubsequenceIterator::operator+(const difference_type& n) const
 *  \brief Arithmetic operator to increment iterator by an integer value.
 *
 *  \param[in] n The integer value by which the iterator will be incremented.
 *
 *  \return A new iterator pointing to the object which is n objects behind the
 *          object pointed at by the current iterator.
 */

/*! \fn SubsequenceIteratorType& SubsequenceIterator::operator+=(const difference_type& n)
 *  \brief Compound assignment operator to increment iterator by an integer value.
 *
 *  \param[in] n The integer value by which the iterator will be incremented.
 *
 *  \return Reference to iterator which got incremented by n.
 */

/*! \fn SubsequenceIteratorType& SubsequenceIterator::operator--()
 *   \brief Prefix decrement operator.
 *
 *  \return Reference to iterator which got decremented by one.
 */

/*! \fn SubsequenceIteratorType SubsequenceIterator::operator--(int)
 *  \brief Postfix decrement operator.
 *
 *  \return A new iterator which points to the same object as the iterator
 *          before the call to the postfix decrement operator.
 */

/*! \fn SubsequenceIteratorType SubsequenceIterator::operator-(const difference_type& n) const
 *  \brief Arithmetic operator to decrement iterator by an integer value.
 *
 *  \return A new iterator pointing to the object which is n objects ahead
 *          of the object pointed at by the current iterator.
 */

/*! \fn SubsequenceIteratorType& SubsequenceIterator::operator-=(const difference_type& n)
 *  \brief Compound assignment operator to decrement iterator by an integer
 *         value.
 *
 *  \param[in] n The integer value by which the iterator will be decremented.
 *
 *  \return Reference to iterator which got decremented by n.
 */

/*! \fn const reference SubsequenceIterator::operator*() const
 *  \brief Dereference operator.
 *
 *  \return Const reference to the object which is pointed at by the subsequence
 *          iterator.
 */

/*! \fn const pointer SubsequenceIterator::operator->() const
 *   \brief Pointer operator.
 *
 *  \return Const Pointer operator for the object which is pointed at by the
 *          subsequence iterator.
 */

/*! \fn const reference SubsequenceIterator::operator[](const difference_type& n) const
 *  \brief Offset dereference operator.
 *
 *  \return Const reference to the object which is n objects behind the object
 *          pointed at by the subsequence iterator.
 */

/*! \var SubsequenceIterator::sequenceBegin_
 *  \brief Iterator pointing to the begin of a sequence of values. The
 *         subsequence iterator will iterate over the elements of this sequence
 *         which are given by
 *         SubsequenceIterator::subsequenceIndicesBegin_.
 */


/*! \var SubsequenceIterator::subsequenceIndicesBegin_
 *  \brief Iterator pointing to the begin of a sequence where the indices for
 *         the elements of the subsequence are stored. Hence
 *         sequenceBegin_[subsequenceIndicesBegin_[n]] will give the n-th
 *         element of the subsequence.
 */

/*! \var SubsequenceIterator::subsequenceIndicesPosition_
 *  \brief Current Position of the iterator. Hence subsequenceIndicesPosition_
 *         contains the number of the element of the subsequence at which the
 *         iterator is pointing at the moment.
 */

/*! \fn bool SubsequenceIterator::operator==(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2)
 *  \brief Equality comparison operator for two SubsequenceIterator iterators.
 *
 *  \param[in] iter1 First iterator.
 *  \param[in] iter2 Second iterator.
 *
 *  \return True if the two iterators point to the same object. False otherwise.
 */

/*! \fn bool SubsequenceIterator::operator!=(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2)
 *  \brief Inequality comparison operator for two SubsequenceIterator iterators.
 *
 *  \param[in] iter1 First iterator.
 *  \param[in] iter2 Second iterator.
 *
 *  \return True if the two iterators point to different objects. False
 *          otherwise.
 */

/*! \fn bool SubsequenceIterator::operator<(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2)
 *  \brief Less comparison operator for two SubsequenceIterator iterators.
 *
 *  \param[in] iter1 First iterator.
 *  \param[in] iter2 Second iterator.
 *
 *  \return True if the first iterators point to an objects which comes in a
 *          sequence before the object pointed to by the second iterator. False
 *          otherwise.
 */

/*! \fn bool SubsequenceIterator::operator>(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2)
 *  \brief Greater comparison operator for two SubsequenceIterator iterators.
 *
 *  \param[in] iter1 First iterator.
 *  \param[in] iter2 Second iterator.
 *
 *  \return True if the first iterators point to an objects which comes in a
 *          sequence after the object pointed to by the second iterator. False
 *          otherwise.
 */

/*! \fn bool SubsequenceIterator::operator<=(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2)
 *  \brief Less or equal comparison operator for two SubsequenceIterator
 *         iterators.
 *
 *  \param[in] iter1 First iterator.
 *  \param[in] iter2 Second iterator.
 *
 *  \return True if the first iterators point to an objects which comes in a
 *          sequence before the object pointed to by the second iterator or if
 *          both objects are the same. False otherwise.
 */

/*! \fn bool SubsequenceIterator::operator>=(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2)
 *  \brief Greater or equal comparison operator for two SubsequenceIterator
 *         iterators.
 *
 *  \param[in] iter1 First iterator.
 *  \param[in] iter2 Second iterator.
 *
 *  \return True if the first iterators point to an objects which comes in a
 *          sequence after the object pointed to by the second iterator or if
 *          both objects are the same. False otherwise.
 */

/*! \fn SubsequenceIterator::difference_type SubsequenceIterator::operator-(const SubsequenceIteratorType& iter1, const SubsequenceIteratorType& iter2)
 *  \brief Difference operator for two subsequence iterators.
 *
 *  \param[in] iter1 First iterator.
 *  \param[in] iter2 Second iterator.
 *
 *  \return The distance between the two subsequence iterators.
 */

/******************
 * implementation *
 ******************/
template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::SubsequenceIterator()
   : sequenceBegin_(), subsequenceIndicesBegin_(), subsequenceIndicesPosition_() {

}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::SubsequenceIterator(const SubsequenceIteratorType& copy)
   : sequenceBegin_(copy.sequenceBegin_),
     subsequenceIndicesBegin_(copy.subsequenceIndicesBegin_),
     subsequenceIndicesPosition_(copy.subsequenceIndicesPosition_) {

}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::SubsequenceIterator(const SequenceIteratorType sequenceBegin, const SubsequenceIndicesIteratorType subsequenceIndicesBegin, const size_t subsequenceIndicesPosition)
   : sequenceBegin_(sequenceBegin), subsequenceIndicesBegin_(subsequenceIndicesBegin), subsequenceIndicesPosition_(subsequenceIndicesPosition) {

}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>& SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator=(const SubsequenceIteratorType& copy) {
   sequenceBegin_ = copy.sequenceBegin_;
   subsequenceIndicesBegin_ = copy.subsequenceIndicesBegin_;
   subsequenceIndicesPosition_ = copy.subsequenceIndicesPosition_;
   return *this;
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>& SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator++() {
   ++subsequenceIndicesPosition_;
   return *this;
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE> SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator++(int) {
   return SubsequenceIteratorType(sequenceBegin_, subsequenceIndicesBegin_, subsequenceIndicesPosition_++);
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE> SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator+(const difference_type& n) const {
   return SubsequenceIteratorType(sequenceBegin_, subsequenceIndicesBegin_, subsequenceIndicesPosition_ + n);
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>& SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator+=(const difference_type& n) {
   subsequenceIndicesPosition_ += n;
   return *this;
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>& SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator--() {
   --subsequenceIndicesPosition_;
   return *this;
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE> SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator--(int) {
   return SubsequenceIteratorType(sequenceBegin_, subsequenceIndicesBegin_, subsequenceIndicesPosition_--);
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE> SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator-(const difference_type& n) const {
   return SubsequenceIteratorType(sequenceBegin_, subsequenceIndicesBegin_, subsequenceIndicesPosition_ - n);
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>& SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator-=(const difference_type& n) {
   subsequenceIndicesPosition_ -= n;
   return *this;
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline const typename SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::value_type& SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator*() const {
   return sequenceBegin_[subsequenceIndicesBegin_[subsequenceIndicesPosition_]];
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline const typename SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::value_type* SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator->() const {
   return &(sequenceBegin_[subsequenceIndicesBegin_[subsequenceIndicesPosition_]]);
}

template<class SEQUENCE_ITERATOR_TYPE, class SUBSEQUENCE_INDICES_ITERATOR_TYPE>
inline const typename SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::value_type& SubsequenceIterator<SEQUENCE_ITERATOR_TYPE, SUBSEQUENCE_INDICES_ITERATOR_TYPE>::operator[](const difference_type& n) const {
   return sequenceBegin_[subsequenceIndicesBegin_[subsequenceIndicesPosition_ + n]];
}

} // namespace opengm

#endif /* OPENGM_SUBSEQUENCE_ITERATOR_HXX_ */
