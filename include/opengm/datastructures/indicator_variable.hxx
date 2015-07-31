#ifndef OPENGM_INDICATOR_VARIABLE_HXX_
#define OPENGM_INDICATOR_VARIABLE_HXX_

#include <utility> // for std::pair
#include <vector>
#include <algorithm>

namespace opengm {

/*********************
 * class definition *
 *********************/
template <class INDEX_TYPE = size_t, class LABEL_TYPE = size_t>
class IndicatorVariable {
public:
   // typedefs
   typedef INDEX_TYPE IndexType;
   typedef LABEL_TYPE LabelType;

   enum LogicalOperatorType {And, Or, Not};

   typedef std::pair<IndexType, LabelType> VariableLabelPair;
   typedef std::vector<VariableLabelPair>  VariableLabelPairContainerType;

   typedef typename VariableLabelPairContainerType::const_iterator IteratorType;

   // constructors
   IndicatorVariable();
   IndicatorVariable(const IndexType variable, const LabelType label, const LogicalOperatorType logicalOperatorType = And);
   IndicatorVariable(const VariableLabelPair& variableLabelPair, const LogicalOperatorType logicalOperatorType = And);
   IndicatorVariable(const VariableLabelPairContainerType& variableLabelPairs, const LogicalOperatorType logicalOperatorType = And);
   template<class ITERATOR_TYPE>
   IndicatorVariable(const ITERATOR_TYPE variableLabelPairsBegin, const ITERATOR_TYPE variableLabelPairsEnd, const LogicalOperatorType logicalOperatorType = And);

   // modify
   void reserve(const size_t numPairs);
   void add(const IndexType variable, const LabelType label);
   void add(const VariableLabelPair& variableLabelPair);
   void add(const VariableLabelPairContainerType& variableLabelPairs);
   template<class ITERATOR_TYPE>
   void add(const ITERATOR_TYPE variableLabelPairsBegin, const ITERATOR_TYPE variableLabelPairsEnd);
   void setLogicalOperatorType(const LogicalOperatorType logicalOperatorType);

   // evaluate
   template<class ITERATOR_TYPE>
   bool operator()(const ITERATOR_TYPE statesBegin) const;

   // const access
   IteratorType begin() const;
   IteratorType end() const;
   LogicalOperatorType getLogicalOperatorType() const;
protected:
   // storage
   VariableLabelPairContainerType variableLabelPairs_;
   LogicalOperatorType            logicalOperatorType_;

   // friends
   template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
   friend bool operator==(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
   template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
   friend bool operator!=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
   template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
   friend bool operator<(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
   template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
   friend bool operator>(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
   template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
   friend bool operator<=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
   template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
   friend bool operator>=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
};

// operators for IndicatorVariable
template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator==(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator!=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator<(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator>(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator<=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator>=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);

/***********************
 * class documentation *
 ***********************/
/*! \file indicator_variable.hxx
 *  \brief Provides implementation for class IndicatorVariable.
 */

/*! \class IndicatorVariable
 *  \brief Combine a group of variables to a new variable.
 *
 *  An indicator variable defines a new boolean variable from a set of variable
 *  label pairs. Depending on the selected logical operator type an indicator
 *  variable will be interpreted as 1 if all (logical and), at least one
 *  (logical or) or none (logical not) of the variables take the labels stated
 *  by the variable label pairs.
 *
 *  \tparam INDEX_TYPE Index type.
 *  \tparam LABEL_TYPE Label type.
 *
 *  \note The set of variable label pairs will be stored in an ordered sequence.
 *        This means a variable label pair will come in the sequence before all
 *        other variable label pairs whose variable index is greater. And if two
 *        variable label pairs have the same variable index the one with the
 *        smaller label will come first.
 */

/*! \typedef IndicatorVariable::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         IndicatorVariable.
 */

/*! \typedef IndicatorVariable::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         IndicatorVariable.
 */

/*! \enum IndicatorVariable::LogicalOperatorType
 *  \brief This enum defines the logical operator types which are supported by
 *         the indicator variables.
 */

/*! \var IndicatorVariable::LogicalOperatorType IndicatorVariable::And
 *  \brief The indicator variable will be interpreted as 1 if all variables
 *         associated by the indicator variable take the labels stated by
 *         the variable label pairs of the indicator variable. Otherwise it will
 *         be interpreted as 0.
 */

/*! \var IndicatorVariable::LogicalOperatorType IndicatorVariable::Or
 *  \brief The indicator variable will be interpreted as 1 if at least one of
 *          the variables associated by the indicator variable takes the label
 *          stated by the variable label pairs of the indicator variable.
 *          Otherwise it will be interpreted as 0.
 */

/*! \var IndicatorVariable::LogicalOperatorType IndicatorVariable::Not
 *  \brief The indicator variable will be interpreted as 1 if none of the
 *         variables associated by the indicator variable takes the label
 *         stated by the variable label pairs of the indicator variable.
 *         Otherwise it will be interpreted as 0.
 */

/*! \typedef IndicatorVariable::VariableLabelPair
 *  \brief A pair representing a single state of a variable.
 */

/*! \typedef IndicatorVariable::VariableLabelPairContainerType
 *  \brief A vector containing VariableLabelPair elements.
 */

/*! \typedef IndicatorVariable::IteratorType
 *  \brief A const iterator to iterate over the VariableLabelPair elements.
 */

/*! \fn IndicatorVariable::IndicatorVariable()
 *  \brief IndicatorVariable constructor.
 *
 *  This constructor will create an empty IndicatorVariable.
 */

/*! \fn IndicatorVariable::IndicatorVariable(const IndexType variable, const LabelType label, const LogicalOperatorType logicalOperatorType)
 *  \brief IndicatorVariable constructor.
 *
 *  This constructor will create an IndicatorVariable containing just one
 *  variable label pair.
 *
 *  \param[in] variable Index of the variable.
 *  \param[in] label Label of the variable.
 *  \param[in] logicalOperatorType The logical operator type by which the
 *                                 indicator variable is evaluated.
 */

/*! \fn IndicatorVariable::IndicatorVariable(const VariableLabelPair& variableLabelPair, const LogicalOperatorType logicalOperatorType)
 *  \brief IndicatorVariable constructor.
 *
 *  This constructor will create an IndicatorVariable containing just one
 *  variable label pair.
 *
 *  \param[in] variableLabelPair A pair containing index and label of a
 *                               variable.
 *  \param[in] logicalOperatorType The logical operator type by which the
 *                                 indicator variable is evaluated.
 */

/*! \fn IndicatorVariable::IndicatorVariable(const IndicatorVariableLabelPairContainerTypeorVariable, const LogicalOperatorType logicalOperatorType)
 *  \brief IndicatorVariable constructor.
 *
 *  This constructor will create an IndicatorVariable by copying the indicator
 *  variable vector.
 *
 *  \param[in] indicatorVariable A vector of VariableLabelPair elements which
 *                               define a IndicatorVariable.
 *  \param[in] logicalOperatorType The logical operator type by which the
 *                                 indicator variable is evaluated.
 */

/*! \fn IndicatorVariable::IndicatorVariable(const ITERATOR_TYPE variableLabelPairsBegin, const ITERATOR_TYPE variableLabelPairsEnd, const LogicalOperatorType logicalOperatorType)
 *  \brief IndicatorVariable constructor.
 *
 *  This constructor will create an IndicatorVariable by copying variable label
 *  pairs.
 *
 *  \tparam ITERATOR_TYPE Iterator type.
 *
 *  \param[in] variableLabelPairsBegin Iterator pointing to the begin of a
 *                                     sequence of variable label pairs.
 *  \param[in] variableLabelPairsEnd Iterator pointing to the end of a sequence
 *                                   of variable label pairs.
 *  \param[in] logicalOperatorType The logical operator type by which the
 *                                 indicator variable is evaluated.
 */

/*! \fn void IndicatorVariable::reserve(const size_t numPairs)
 *  \brief Preallocate memory.
 *
 *  The reserve function fill preallocate enough memory to store at least the
 *  stated number of variable label pairs.
 *
 *  \param[in] numPairs The number of variable label pairs for which memory will
 *                      be allocated.
 */

/*! \fn void IndicatorVariable::add(const IndexType variable, const LabelType label)
 *  \brief Add a variable label pair to the indicator variable.
 *
 *  \param[in] variable Index of the variable.
 *  \param[in] label Label of the variable.
 */

/*! \fn void IndicatorVariable::add(const VariableLabelPair& variableLabelPair)
 *  \brief Add a variable label pair to the indicator variable.
 *
 *  \param[in] variableLabelPair A pair containing index and label of a
 *                               variable.
 *
 */

/*! \fn void IndicatorVariable::add(const VariableLabelPairContainerType& indicatorVariable)
 *  \brief Add a sequence of variable label pairs to the indicator variable.
 *
 *  \param[in] indicatorVariable A vector of VariableLabelPair elements which
 *                               define an IndicatorVariable.
 */

/*! \fn void IndicatorVariable::add(const ITERATOR_TYPE variableLabelPairsBegin, const ITERATOR_TYPE variableLabelPairsEnd)
 *  \brief Add a sequence of variable label pairs to the indicator variable.
 *
 *  \tparam ITERATOR_TYPE Iterator type.
 *
 *  \param[in] variableLabelPairsBegin Iterator pointing to the begin of a
 *                                     sequence of variable label pairs.
 *  \param[in] variableLabelPairsEnd Iterator pointing to the end of a sequence
 *                                   of variable label pairs.
 */

/*! \fn void IndicatorVariable::setLogicalOperatorType(const LogicalOperatorType logicalOperatorType)
 *  \brief Set the logical operator type of the indicator variable.
 *
 *  \param[in] logicalOperatorType The new logical operator type.
 */

/*! \fn bool IndicatorVariable::operator()(const ITERATOR_TYPE statesBegin) const
 *  \brief Evaluation operator to check if the indicator variable is active for
 *         the given labeling.
 *
 *  \tparam ITERATOR_TYPE Iterator type.
 *
 *  \param[in] statesBegin Iterator pointing to the begin of the labeling.
 *
 *  \return True if the indicator variable is active for the given labeling,
 *          false otherwise.
 *
 *  \warning No boundary check is performed.
 */

/*! \fn IndicatorVariable::IteratorType IndicatorVariable::begin() const
 *  \brief Get the iterator over the sequence of variable label pairs from the
 *         indicator variable.
 *
 *  \return Returns a const iterator to the begin of the sequence of variable
 *          label pairs from the indicator variable.
 */

/*! \fn IndicatorVariable::IteratorType IndicatorVariable::end() const
 *  \brief Get the end iterator of the sequence of variable label pairs from the
 *         indicator variable.
 *
 *  \return Returns a const iterator to the end of the sequence of variable
 *          label pairs from the indicator variable.
 */

/*! \fn IndicatorVariable::LogicalOperatorType IndicatorVariable::getLogicalOperatorType()
 *  \brief Get the logical operator type of the indicator variable.
 *
 *  \return The current logical operator type of the indicator variable.
 */

/*! \var IndicatorVariable::variableLabelPairs_
 *  \brief Storage for the variable label pairs.
 */

/*! \var IndicatorVariable::logicalOperatorType_
 *  \brief Storage for the logical operator type of the indicator variable.
 */

/*! \fn bool IndicatorVariable::operator==(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
 *  \brief Equality operator for IndicatorVariable.
 *
 *  \tparam INDEX1_TYPE Index type of the first indicator variable.
 *  \tparam LABEL1_TYPE Label type of the first indicator variable.
 *  \tparam INDEX2_TYPE Index type of the second indicator variable.
 *  \tparam LABEL2_TYPE Label type of the second indicator variable.
 *
 *  \param[in] indicatorVar1 First indicator variable.
 *  \param[in] indicatorVar2 Second indicator variable.
 *
 *  \return Returns true if two IndicatorVariables are equal. False otherwise.
 */

/*! \fn bool IndicatorVariable::operator!=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
 *  \brief Inequality operator for IndicatorVariable.
 *
 *  \tparam INDEX1_TYPE Index type of the first indicator variable.
 *  \tparam LABEL1_TYPE Label type of the first indicator variable.
 *  \tparam INDEX2_TYPE Index type of the second indicator variable.
 *  \tparam LABEL2_TYPE Label type of the second indicator variable.
 *
 *  \param[in] indicatorVar1 First indicator variable.
 *  \param[in] indicatorVar2 Second indicator variable.
 *
 *  \return Returns true if two IndicatorVariables are different. False
 *          otherwise.
 */

/*! \fn bool IndicatorVariable::operator<(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
 *  \brief Less operator for IndicatorVariable.
 *
 *  \tparam INDEX1_TYPE Index type of the first indicator variable.
 *  \tparam LABEL1_TYPE Label type of the first indicator variable.
 *  \tparam INDEX2_TYPE Index type of the second indicator variable.
 *  \tparam LABEL2_TYPE Label type of the second indicator variable.
 *
 *  \param[in] indicatorVar1 First indicator variable.
 *  \param[in] indicatorVar2 Second indicator variable.
 *
 *  \return Returns true if an IndicatorVariables is smaller than an other one.
 *          False otherwise.
 *
 *  \note Indicator variables will be ordered first by the logical operator type
 *        (And < Or < Not), then by the number of variable label pairs. If both
 *        the logical operator type and the number of variable label pairs are
 *        equal the first variable label pairs which are different in the first
 *        and the second indicator variable will be used for comparison.
 */

/*! \fn bool IndicatorVariable::operator>(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
 *  \brief Greater operator for IndicatorVariable.
 *
 *  \tparam INDEX1_TYPE Index type of the first indicator variable.
 *  \tparam LABEL1_TYPE Label type of the first indicator variable.
 *  \tparam INDEX2_TYPE Index type of the second indicator variable.
 *  \tparam LABEL2_TYPE Label type of the second indicator variable.
 *
 *  \param[in] indicatorVar1 First indicator variable.
 *  \param[in] indicatorVar2 Second indicator variable.
 *
 *  \return Returns true if an IndicatorVariables is greater than an other one.
 *          False otherwise.
 *
 *  \note Indicator variables will be ordered first by the logical operator type
 *        (And < Or < Not), then by the number of variable label pairs. If both
 *        the logical operator type and the number of variable label pairs are
 *        equal the first variable label pairs which are different in the first
 *        and the second indicator variable will be used for comparison.
 */

/*! \fn bool IndicatorVariable::operator<=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
 *  \brief Less equal operator for IndicatorVariable.
 *
 *  \tparam INDEX1_TYPE Index type of the first indicator variable.
 *  \tparam LABEL1_TYPE Label type of the first indicator variable.
 *  \tparam INDEX2_TYPE Index type of the second indicator variable.
 *  \tparam LABEL2_TYPE Label type of the second indicator variable.
 *
 *  \param[in] indicatorVar1 First indicator variable.
 *  \param[in] indicatorVar2 Second indicator variable.
 *
 *  \return Returns true if an IndicatorVariables is smaller than or equal to an
 *          other one. False otherwise.
 *
 *  \note Indicator variables will be ordered first by the logical operator type
 *        (And < Or < Not), then by the number of variable label pairs. If both
 *        the logical operator type and the number of variable label pairs are
 *        equal the first variable label pairs which are different in the first
 *        and the second indicator variable will be used for comparison.
 */

/*! \fn bool IndicatorVariable::operator>=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2);
 *  \brief Greater equal operator for IndicatorVariable.
 *
 *  \tparam INDEX1_TYPE Index type of the first indicator variable.
 *  \tparam LABEL1_TYPE Label type of the first indicator variable.
 *  \tparam INDEX2_TYPE Index type of the second indicator variable.
 *  \tparam LABEL2_TYPE Label type of the second indicator variable.
 *
 *  \param[in] indicatorVar1 First indicator variable.
 *  \param[in] indicatorVar2 Second indicator variable.
 *
 *  \return Returns true if an IndicatorVariables is greater than or equal to an
 *          other one. False otherwise.
 *
 *  \note Indicator variables will be ordered first by the logical operator type
 *        (And < Or < Not), then by the number of variable label pairs. If both
 *        the logical operator type and the number of variable label pairs are
 *        equal the first variable label pairs which are different in the first
 *        and the second indicator variable will be used for comparison.
 */

/******************
 * implementation *
 ******************/

// constructors
template <class INDEX_TYPE, class LABEL_TYPE>
inline IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::IndicatorVariable()
: variableLabelPairs_(), logicalOperatorType_(And) {

}

template <class INDEX_TYPE, class LABEL_TYPE>
inline IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::IndicatorVariable(const IndexType variable, const LabelType label, const LogicalOperatorType logicalOperatorType)
: variableLabelPairs_(1, VariableLabelPair(variable, label)),
  logicalOperatorType_(logicalOperatorType) {

}

template <class INDEX_TYPE, class LABEL_TYPE>
inline IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::IndicatorVariable(const VariableLabelPair& variableLabelPair, const LogicalOperatorType logicalOperatorType)
: variableLabelPairs_(1, variableLabelPair),
  logicalOperatorType_(logicalOperatorType) {

}

template <class INDEX_TYPE, class LABEL_TYPE>
inline IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::IndicatorVariable(const VariableLabelPairContainerType& variableLabelPairs, const LogicalOperatorType logicalOperatorType)
: variableLabelPairs_(variableLabelPairs),
  logicalOperatorType_(logicalOperatorType) {
   std::sort(variableLabelPairs_.begin(), variableLabelPairs_.end());
}

template <class INDEX_TYPE, class LABEL_TYPE>
template<class ITERATOR_TYPE>
inline IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::IndicatorVariable(const ITERATOR_TYPE variableLabelPairsBegin, const ITERATOR_TYPE variableLabelPairsEnd, const LogicalOperatorType logicalOperatorType)
: variableLabelPairs_(variableLabelPairsBegin, variableLabelPairsEnd),
  logicalOperatorType_(logicalOperatorType) {
   std::sort(variableLabelPairs_.begin(), variableLabelPairs_.end());
}

// modify
template <class INDEX_TYPE, class LABEL_TYPE>
inline void IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::reserve(const size_t numPairs) {
   variableLabelPairs_.reserve(numPairs);
}

template <class INDEX_TYPE, class LABEL_TYPE>
inline void IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::add(const IndexType variable, const LabelType label) {
   variableLabelPairs_.push_back(VariableLabelPair(variable, label));
   std::sort(variableLabelPairs_.begin(), variableLabelPairs_.end());
}

template <class INDEX_TYPE, class LABEL_TYPE>
inline void IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::add(const VariableLabelPair& variableLabelPair) {
   variableLabelPairs_.push_back(variableLabelPair);
   std::sort(variableLabelPairs_.begin(), variableLabelPairs_.end());
}

template <class INDEX_TYPE, class LABEL_TYPE>
inline void IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::add(const VariableLabelPairContainerType& variableLabelPairs) {
   variableLabelPairs_.insert(variableLabelPairs_.end(), variableLabelPairs.begin(), variableLabelPairs.end());
   std::sort(variableLabelPairs_.begin(), variableLabelPairs_.end());
}

template <class INDEX_TYPE, class LABEL_TYPE>
template<class ITERATOR_TYPE>
inline void IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::add(const ITERATOR_TYPE variableLabelPairsBegin, const ITERATOR_TYPE variableLabelPairsEnd) {
   variableLabelPairs_.insert(variableLabelPairs_.end(), variableLabelPairsBegin, variableLabelPairsEnd);
   std::sort(variableLabelPairs_.begin(), variableLabelPairs_.end());
}

template <class INDEX_TYPE, class LABEL_TYPE>
inline void IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::setLogicalOperatorType(const LogicalOperatorType logicalOperatorType) {
   logicalOperatorType_ = logicalOperatorType;
}

// evaluate
template <class INDEX_TYPE, class LABEL_TYPE>
template<class ITERATOR_TYPE>
inline bool IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::operator()(const ITERATOR_TYPE statesBegin) const {
   bool result = true;
   if(logicalOperatorType_ == And) {
      for(IteratorType indicatorVariableIter = begin(); indicatorVariableIter != end(); ++indicatorVariableIter) {
         if(indicatorVariableIter->second != statesBegin[indicatorVariableIter->first]) {
            result = false;
            break;
         }
      }
   } else if(logicalOperatorType_ == Or) {
      result = false;
      for(IteratorType indicatorVariableIter = begin(); indicatorVariableIter != end(); ++indicatorVariableIter) {
         if(indicatorVariableIter->second == statesBegin[indicatorVariableIter->first]) {
            result = true;
            break;
         }
      }
   } else {
      // logicalOperatorType_ == Not
      for(IteratorType indicatorVariableIter = begin(); indicatorVariableIter != end(); ++indicatorVariableIter) {
         if(indicatorVariableIter->second == statesBegin[indicatorVariableIter->first]) {
            result = false;
            break;
         }
      }
   }
   return result;
}

// const access
template <class INDEX_TYPE, class LABEL_TYPE>
inline typename IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::IteratorType IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::begin() const {
   return variableLabelPairs_.begin();
}

template <class INDEX_TYPE, class LABEL_TYPE>
inline typename IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::IteratorType IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::end() const {
   return variableLabelPairs_.end();
}

template <class INDEX_TYPE, class LABEL_TYPE>
inline typename IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::LogicalOperatorType IndicatorVariable<INDEX_TYPE, LABEL_TYPE>::getLogicalOperatorType() const {
   return logicalOperatorType_;
}

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator==(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2) {
   return (indicatorVar1.logicalOperatorType_ == indicatorVar2.logicalOperatorType_) && (indicatorVar1.variableLabelPairs_ == indicatorVar2.variableLabelPairs_);
}

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator!=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2) {
   return (indicatorVar1.logicalOperatorType_ != indicatorVar2.logicalOperatorType_) || (indicatorVar1.variableLabelPairs_ != indicatorVar2.variableLabelPairs_);
}

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator<(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2) {
   if(indicatorVar1.logicalOperatorType_ < indicatorVar2.logicalOperatorType_) {
      return true;
   } else if(indicatorVar1.logicalOperatorType_ > indicatorVar2.logicalOperatorType_) {
      return false;
   } else {
      if(indicatorVar1.variableLabelPairs_.size() > indicatorVar2.variableLabelPairs_.size()) {
         return false;
      }
      if(indicatorVar1.variableLabelPairs_.size() == indicatorVar2.variableLabelPairs_.size()) {
         if(indicatorVar1.variableLabelPairs_.size() == 0) {
            return false;
         }
         for(size_t i = 0; i < indicatorVar1.variableLabelPairs_.size(); ++i) {
            if(indicatorVar1.variableLabelPairs_[i].first > indicatorVar2.variableLabelPairs_[i].first) {
               return false;
            } else if(indicatorVar1.variableLabelPairs_[i].first == indicatorVar2.variableLabelPairs_[i].first) {
               if(indicatorVar1.variableLabelPairs_[i].second >= indicatorVar2.variableLabelPairs_[i].second) {
                  return false;
               }
            }
         }
      }
      return true;
   }
}

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator>(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2) {
   if(indicatorVar1.logicalOperatorType_ < indicatorVar2.logicalOperatorType_) {
      return false;
   } else if(indicatorVar1.logicalOperatorType_ > indicatorVar2.logicalOperatorType_) {
      return true;
   } else {
      if(indicatorVar1.variableLabelPairs_.size() < indicatorVar2.variableLabelPairs_.size()) {
         return false;
      }
      if(indicatorVar1.variableLabelPairs_.size() == indicatorVar2.variableLabelPairs_.size()) {
         if(indicatorVar1.variableLabelPairs_.size() == 0) {
            return false;
         }
         for(size_t i = 0; i < indicatorVar1.variableLabelPairs_.size(); ++i) {
            if(indicatorVar1.variableLabelPairs_[i].first < indicatorVar2.variableLabelPairs_[i].first) {
               return false;
            } else if(indicatorVar1.variableLabelPairs_[i].first == indicatorVar2.variableLabelPairs_[i].first) {
               if(indicatorVar1.variableLabelPairs_[i].second <= indicatorVar2.variableLabelPairs_[i].second) {
                  return false;
               }
            }
         }
      }
      return true;
   }
}

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator<=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2) {
   if(indicatorVar1.logicalOperatorType_ < indicatorVar2.logicalOperatorType_) {
      return true;
   } else if(indicatorVar1.logicalOperatorType_ > indicatorVar2.logicalOperatorType_) {
      return false;
   } else {
      if(indicatorVar1.variableLabelPairs_.size() > indicatorVar2.variableLabelPairs_.size()) {
         return false;
      }
      if(indicatorVar1.variableLabelPairs_.size() == indicatorVar2.variableLabelPairs_.size()) {
         for(size_t i = 0; i < indicatorVar1.variableLabelPairs_.size(); ++i) {
            if(indicatorVar1.variableLabelPairs_[i].first > indicatorVar2.variableLabelPairs_[i].first) {
               return false;
            } else if(indicatorVar1.variableLabelPairs_[i].first == indicatorVar2.variableLabelPairs_[i].first) {
               if(indicatorVar1.variableLabelPairs_[i].second > indicatorVar2.variableLabelPairs_[i].second) {
                  return false;
               }
            }
         }
      }
      return true;
   }
}

template<class INDEX1_TYPE, class LABEL1_TYPE, class INDEX2_TYPE, class LABEL2_TYPE>
bool operator>=(const IndicatorVariable<INDEX1_TYPE, LABEL1_TYPE>& indicatorVar1, const IndicatorVariable<INDEX2_TYPE, LABEL2_TYPE>& indicatorVar2) {
   if(indicatorVar1.logicalOperatorType_ < indicatorVar2.logicalOperatorType_) {
      return false;
   } else if(indicatorVar1.logicalOperatorType_ > indicatorVar2.logicalOperatorType_) {
      return true;
   } else {
      if(indicatorVar1.variableLabelPairs_.size() < indicatorVar2.variableLabelPairs_.size()) {
         return false;
      }
      if(indicatorVar1.variableLabelPairs_.size() == indicatorVar2.variableLabelPairs_.size()) {
         for(size_t i = 0; i < indicatorVar1.variableLabelPairs_.size(); ++i) {
            if(indicatorVar1.variableLabelPairs_[i].first < indicatorVar2.variableLabelPairs_[i].first) {
               return false;
            } else if(indicatorVar1.variableLabelPairs_[i].first == indicatorVar2.variableLabelPairs_[i].first) {
               if(indicatorVar1.variableLabelPairs_[i].second < indicatorVar2.variableLabelPairs_[i].second) {
                  return false;
               }
            }
         }
      }
      return true;
   }
}

} // namespace opengm

#endif /* OPENGM_INDICATOR_VARIABLE_HXX_ */
