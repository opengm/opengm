#ifndef OPENGM_LINEAR_CONSTRAINT_HXX_
#define OPENGM_LINEAR_CONSTRAINT_HXX_

#include <iterator>
#include <vector>

#include <opengm/datastructures/indicator_variable.hxx>

namespace opengm {

/*********************
 * class definition *
 *********************/
class LinearConstraintTraits {
public:
   // typedefs
   struct LinearConstraintOperator {enum ValueType {LessEqual, Equal, GreaterEqual};};
};

template<class VALUE_TYPE, class INDEX_TYPE = size_t, class LABEL_TYPE = size_t>
class LinearConstraint {
public:
   // typedefs
   typedef VALUE_TYPE ValueType;
   typedef INDEX_TYPE IndexType;
   typedef LABEL_TYPE LabelType;

   typedef IndicatorVariable<IndexType, LabelType>          IndicatorVariableType;
   typedef std::vector<IndicatorVariableType>               IndicatorVariablesContainerType;
   typedef std::vector<ValueType>                           CoefficientsContainerType;
   typedef ValueType                                        BoundType;
   typedef LinearConstraintTraits::LinearConstraintOperator LinearConstraintOperatorType;
   typedef LinearConstraintOperatorType::ValueType          LinearConstraintOperatorValueType;

   typedef typename IndicatorVariableType::IteratorType             VariableLabelPairsIteratorType;
   typedef typename IndicatorVariablesContainerType::const_iterator IndicatorVariablesIteratorType;
   typedef typename CoefficientsContainerType::const_iterator       CoefficientsIteratorType;

   // constructors
   LinearConstraint();
   LinearConstraint(const IndicatorVariablesContainerType& indicatorVariables, const CoefficientsContainerType& coefficients, const BoundType bound = 0.0, const LinearConstraintOperatorValueType constraintOperator = LinearConstraintOperatorType::LessEqual);
   template<class INDICATOR_VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   LinearConstraint(const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesBegin, const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesEnd, const COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const BoundType bound = 0.0, const LinearConstraintOperatorValueType constraintOperator = LinearConstraintOperatorType::LessEqual);
   LinearConstraint(const LinearConstraint<ValueType, IndexType, LabelType>& linearConstraint);

   // modify
   void reserve(const size_t numIndicatorVariables);
   void add(const IndicatorVariableType& indicatorVariable, const ValueType coefficient);
   void add(const IndicatorVariablesContainerType& indicatorVariables, const CoefficientsContainerType& coefficients);
   template<class INDICATOR_VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void add(const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesBegin, const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesEnd, const COEFFICIENTS_ITERATOR_TYPE coefficientsBegin);
   void setBound(const BoundType bound);
   void setConstraintOperator(const LinearConstraintOperatorValueType constraintOperator);

   // evaluate
   template<class ITERATOR_TYPE>
   ValueType operator()(const ITERATOR_TYPE statesBegin) const;

   // const access
   IndicatorVariablesIteratorType    indicatorVariablesBegin() const;
   IndicatorVariablesIteratorType    indicatorVariablesEnd() const;
   CoefficientsIteratorType          coefficientsBegin() const;
   CoefficientsIteratorType          coefficientsEnd() const;
   BoundType                         getBound() const;
   LinearConstraintOperatorValueType getConstraintOperator() const;
protected:
   // storage
   IndicatorVariablesContainerType   indicatorVariables_;
   CoefficientsContainerType         coefficients_;
   BoundType                         bound_;
   LinearConstraintOperatorValueType constraintOperator_;
};

/***********************
 * class documentation *
 ***********************/
/*! \file linear_constraint.hxx
 *  \brief Provides implementation for class LinearConstraint.
 */

/*! \class LinearConstraintTraits
 *  \brief Traits class for LinearConstraint to provide template independent
 *         enum for ConstraintOperatorType.
 *
 *  This is a traits class for the class LinearConstraint to provide an enum for
 *  ConstraintOperatorType. This has to be done outside the LinearConstraint
 *  class. Otherwise there would be several different types defining the
 *  same enum. One for each instantiation of LinearConstraint, as
 *  LinearConstraint is a template class.
 */

/*! \struct LinearConstraintTraits::LinearConstraintOperator
 *  \brief This struct is used to create an own scope for the
 *         LinearConstraintTraits::LinearConstraintOperator::ValueType enum as
 *         this is not done in C++ by default.
 */

/*! \enum LinearConstraintTraits::LinearConstraintOperator::ValueType
 *  \brief This enum defines the operator type for the linear constraint.
 */

/*! \var LinearConstraintTraits::LinearConstraintOperator::ValueType LinearConstraintTraits::LinearConstraintOperator::LessEqual
 *  \brief Defines the linear constraint operator type to be \f$\leq\f$. Hence
 *         the left hand side of the constraint will be compared against the
 *         bound using the operator <=.
 */

/*! \var LinearConstraintTraits::LinearConstraintOperator::ValueType LinearConstraintTraits::LinearConstraintOperator::Equal
 *  \brief Defines the linear constraint operator type to be \f$=\f$. Hence the
 *         left hand side of the constraint will be compared against the bound
 *         using the operator ==.
 */

/*! \var LinearConstraintTraits::LinearConstraintOperator::ValueType LinearConstraintTraits::LinearConstraintOperator::GreaterEqual
 *  \brief Defines the linear constraint operator type to be \f$\geq\f$. Hence
 *         the left hand side of the constraint will be compared against the
 *         bound using the operator >=.
 */

/*! \class LinearConstraint
 *  \brief Define a linear constraint for a set of indicatorVariables.
 *
 *  This class defines a linear constraint for a set of indicator variables.
 *  Each variable has to be an indicator variable of the class
 *  IndicatorVariable. Each constraint consists of four parts:
 *
 *     1. A set of indicator variables.
 *     2. A set of coefficients. One coefficient for each indicator variable.
 *     3. A bound.
 *     4. An operator type which defines how the left hand side of the
 *        constraint is compared against the bound.
 *
 *  The constraint will be evaluated by the following formula:
 *  \f[
 *     \sum_i c_i \cdot v_i \quad \bigcirc \quad b.
 *  \f]
 *  Here \f$v_i\f$ represents the \f$i\f$-th indicator variable, \f$c_i\f$
 *  represents the coefficient belonging to the \f$i\f$-th indictor variable,
 *  \f$b\f$ represents the bound and \f$\bigcirc\f$ is the selected operator
 *  type (\f$\leq\f$, \f$=\f$ or \f$\geq\f$).
 *
 *  \tparam VALUE_TYPE Value type.
 *  \tparam INDEX_TYPE Index type.
 *  \tparam LABEL_TYPE Label type.
 */

/*! \typedef LinearConstraint::ValueType
 *  \brief Typedef of the VALUE_TYPE template parameter type from the class
 *         LinearConstraint.
 */

/*! \typedef LinearConstraint::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         LinearConstraint.
 */

/*! \typedef LinearConstraint::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         LinearConstraint.
 */

/*! \typedef LinearConstraint::IndicatorVariableType
 *  \brief Typedef of the IndicatorVariable class with appropriate template
 *         parameter.
 */

/*! \typedef LinearConstraint::IndicatorVariablesContainerType
 *  \brief Defines the storage type for the set of indicator variables.
 */

/*! \typedef LinearConstraint::CoefficientsContainerType
 *  \brief Defines the storage type for the set of coefficients for the
 *         indicator variables.
 */

/*! \typedef LinearConstraint::BoundType
 *  \brief Defines the data type for the bound.
 */

/*! \typedef LinearConstraint::LinearConstraintOperatorType
 *  \brief Defines the linear constraint operator.
 */

/*! \typedef LinearConstraint::LinearConstraintOperatorValueType
 *  \brief Defines the linear constraint operator type.
 */

/*! \typedef LinearConstraint::VariableLabelPairsIteratorType
 *  \brief Defines the const iterator type to iterate over the variable label
 *         pairs of an indicator variable.
 */

/*! \typedef LinearConstraint::IndicatorVariablesIteratorType
 *  \brief Defines the const iterator type to iterate over the set of indicator
 *         variables.
 */

/*! \typedef LinearConstraint::CoefficientsIteratorType
 *  \brief Defines the const iterator type to iterate over the set of
 *         coefficients for the indicator variables.
 */

/*! \fn LinearConstraint::LinearConstraint()
 *  \brief LinearConstraint constructor.
 *
 *  This constructor will create an empty LinearConstraint.
 */

/*! \fn LinearConstraint::LinearConstraint(const IndicatorVariablesContainerType& indicatorVariables, const CoefficientsContainerType& coefficients, const BoundType bound = 0.0, const LinearConstraintOperatorValueType constraintOperator = LinearConstraintOperatorType::LessEqual)
 *  \brief LinearConstraint constructor.
 *
 *  This constructor will create a LinearConstraint by copying the set of
 *  indicator variables and coefficients.
 *
 *  \param[in] indicatorVariables The set of indicator variables.
 *  \param[in] coefficients The set of coefficients for the indicator variables.
 *  \param[in] bound The right hand side of the constraint.
 *  \param[in] constraintOperator The comparison operator by which the
 *                                constraint will be evaluated.
 */

/*! \fn LinearConstraint::LinearConstraint(const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesBegin, const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesEnd, const COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const BoundType bound = 0.0, const LinearConstraintOperatorValueType constraintOperator = LinearConstraintOperatorType::LessEqual)
 *  \brief LinearConstraint constructor.
 *
 *  This constructor will create a LinearConstraint by copying the set of
 *  indicator variables and coefficients.
 *
 *  \tparam INDICATOR_VARIABLES_ITERATOR_TYPE Iterator type to iterate over the
 *                                            set of indicator variables.
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type to iterate over the set of
 *                                     coefficients for the indicator variables.
 *
 *  \param[in] indicatorVariablesBegin Iterator pointing to the begin of the set
 *                                     of indicator variables.
 *  \param[in] indicatorVariablesEnd Iterator pointing to the end of the set of
 *                                   indicator variables.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of the set of
 *                               coefficients for the indicator variables.
 *  \param[in] bound The right hand side of the constraint.
 *  \param[in] constraintOperator The comparison operator by which the
 *                                constraint will be evaluated.
 */

/*! \fn LinearConstraint::LinearConstraint(const LinearConstraint<ValueType, IndexType, LabelType>& linearConstraint)
 *  \brief LinearConstraint constructor.
 *
 *  This constructor will create a LinearConstraint by copying an existing
 *  linear constraint.
 *
 *  \param[in] linearConstraint Existing linear constraint which will be copied.
 */

/*! \fn void LinearConstraint::reserve(const size_t numIndicatorVariables)
 *  \brief Preallocate memory
 *
 *  The reserve function fill preallocate enough memory to store at least the
 *  stated number of indicator variables and the corresponding coefficients.
 *
 *  \param[in] numIndicatorVariables The number of indicator variables for which
 *                                   memory will be allocated.
 */

/*! \fn void LinearConstraint::add(const IndicatorVariableType& indicatorVariable, const ValueType coefficient)
 *  \brief Add a single indicator variable and the corresponding coefficient to
 *         the linear constraint.
 *
 *  \param[in] indicatorVariable Indicator variable which will be added to the
 *                               linear constraint.
 *  \param[in] coefficient Coefficient of the indicator variable.
 */

/*! \fn void LinearConstraint::add(const IndicatorVariablesContainerType& indicatorVariables, const CoefficientsContainerType& coefficients)
 *  \brief Add a set of indicator variables and the corresponding coefficients
 *         to the linear constraint.
 *
 *  \param[in] indicatorVariables Set of indicator variables which will be added
 *                                to the linear constraint.
 *  \param[in] coefficients Set of coefficients for the indicator variables.
 */

/*! \fn void LinearConstraint::add(const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesBegin, const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesEnd, const COEFFICIENTS_ITERATOR_TYPE coefficientsBegin)
 *  \brief Add a set of indicator variables and the corresponding coefficients
 *         to the linear constraint.
 *
 *  \tparam INDICATOR_VARIABLES_ITERATOR_TYPE Iterator type to iterate over the
 *                                            set of indicator variables.
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type to iterate over the set of
 *                                     coefficients for the indicator variables.
 *
 *  \param[in] indicatorVariablesBegin Iterator pointing to the begin of the set
 *                                     of indicator variables which will be
 *                                     added to the linear constraint.
 *  \param[in] indicatorVariablesEnd Iterator pointing to the end of the set of
 *                                   indicator variables which will be added to
 *                                   the linear constraint.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of the set of
 *                               coefficients for the indicator variables.
 */

/*! \fn void LinearConstraint::setBound(const BoundType bound)
 *   \brief Set the bound of the linear constraint.
 *
 *  \param[in] bound The new bound for the linear constraint.
 */


/*! \fn void LinearConstraint::setConstraintOperator(const LinearConstraintOperatorValueType constraintOperator)
 *  \brief Set the constraint operator for the linear constraint.
 *
 *  \param[in] constraintOperator The new constraint operator for the linear
 *                                constraint.
 */

/*! \fn LinearConstraint::ValueType LinearConstraint::operator()(const ITERATOR_TYPE statesBegin) const
 *  \brief Evaluation operator to check if the linear constraint is violated by
 *         the given labeling.
 *
 *  \tparam ITERATOR_TYPE Iterator type.
 *
 *  \param[in] statesBegin Iterator pointing to the begin of the labeling.
 *
 *  \return The absolute value by which the constraint is violated (0.0 if the
 *          linear constraint is not violated).
 *
 *  \warning No boundary check is performed.
 */

/*! \fn LinearConstraint::IndicatorVariablesIteratorType LinearConstraint::indicatorVariablesBegin() const
 *  \brief Get the begin iterator to the set of indicator variables.
 *
 *  \return The const iterator pointing to the begin of the set of indicator
 *          variables.
 */

/*! \fn LinearConstraint::IndicatorVariablesIteratorType LinearConstraint::indicatorVariablesEnd() const
 *  \brief Get the end iterator to the set of indicator variables.
 *
 *  \return The const iterator pointing to the end of the set of indicator
 *          variables.
 */

/*! \fn LinearConstraint::CoefficientsIteratorType LinearConstraint::coefficientsBegin() const
 *  \brief Get the begin iterator to the set of coefficients for the indicator
 *         variables.
 *
 *  \return The const iterator pointing to the begin of the set of coefficients
 *          for the indicator variables.
 */

/*! \fn LinearConstraint::CoefficientsIteratorType LinearConstraint::coefficientsEnd() const
 *  \brief Get the end iterator to the set of coefficients for the indicator
 *         variables.
 *
 *  \return The const iterator pointing to the end of the set of coefficients
 *          for the indicator variables.
 */

/*! \fn LinearConstraint::BoundType LinearConstraint::getBound() const
 *  \brief Get the bound of the linear constraint.
 *
 *  \return The bound of the linear constraint.
 */

/*! \fn LinearConstraint::ConstraintOperatorType LinearConstraint::getConstraintOperator() const
 *  \brief Get the constraint operator of the linear constraint.
 *
 *  \return The constraint operator of the linear constraint.
 */

/*! \var LinearConstraint::indicatorVariables_
 *  \brief Storage for the set of indicator variables.
 */

/*! \var LinearConstraint::coefficients_
 *  \brief Storage for the set of coefficients for the indicator variables.
 */

/*! \var LinearConstraint::bound_
 *  \brief Storage for the bound of the linear constraint.
 */

/*! \var LinearConstraint::constraintOperator_
 *  \brief Storage for the constraint operator of the linear constraint.
 */

/******************
 * implementation *
 ******************/

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraint() : indicatorVariables_(),
   coefficients_(), bound_(0.0),
   constraintOperator_(LinearConstraintOperatorType::LessEqual) {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraint(const IndicatorVariablesContainerType& indicatorVariables, const CoefficientsContainerType& coefficients, const BoundType bound, const LinearConstraintOperatorValueType constraintOperator)
   : indicatorVariables_(indicatorVariables), coefficients_(coefficients),
     bound_(bound), constraintOperator_(constraintOperator) {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDICATOR_VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraint(const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesBegin, const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesEnd, const COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const BoundType bound, const LinearConstraintOperatorValueType constraintOperator)
   : indicatorVariables_(indicatorVariablesBegin, indicatorVariablesEnd),
     coefficients_(coefficientsBegin, coefficientsBegin + std::distance(indicatorVariablesBegin, indicatorVariablesEnd)),
     bound_(bound), constraintOperator_(constraintOperator) {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraint(const LinearConstraint<ValueType, IndexType, LabelType>& linearConstraint)
   : indicatorVariables_(linearConstraint.indicatorVariables_),
     coefficients_(linearConstraint.coefficients_),
     bound_(linearConstraint.bound_),
     constraintOperator_(linearConstraint.constraintOperator_) {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::reserve(const size_t numIndicatorVariables) {
   indicatorVariables_.reserve(numIndicatorVariables);
   coefficients_.reserve(numIndicatorVariables);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::add(const IndicatorVariableType& indicatorVariable, const ValueType coefficient) {
   indicatorVariables_.push_back(indicatorVariable);
   coefficients_.push_back(coefficient);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::add(const IndicatorVariablesContainerType& indicatorVariables, const CoefficientsContainerType& coefficients) {
   indicatorVariables_.insert(indicatorVariables_.end(), indicatorVariables.begin(), indicatorVariables.end());
   coefficients_.insert(coefficients_.end(), coefficients.begin(), coefficients.end());
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDICATOR_VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::add(const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesBegin, const INDICATOR_VARIABLES_ITERATOR_TYPE indicatorVariablesEnd, const COEFFICIENTS_ITERATOR_TYPE coefficientsBegin) {
   indicatorVariables_.insert(indicatorVariables_.end(), indicatorVariablesBegin, indicatorVariablesEnd);
   coefficients_.insert(coefficients_.end(), coefficientsBegin, coefficientsBegin + std::distance(indicatorVariablesBegin, indicatorVariablesEnd));
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::setBound(const BoundType bound) {
   bound_ = bound;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::setConstraintOperator(const LinearConstraintOperatorValueType constraintOperator) {
   constraintOperator_ = constraintOperator;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class ITERATOR_TYPE>
inline typename LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::operator()(const ITERATOR_TYPE statesBegin) const {
   ValueType leftHandSide = 0.0;
   for(IndexType i = 0; i < indicatorVariables_.size(); ++i) {
      if(indicatorVariables_[i](statesBegin)) {
         leftHandSide += coefficients_[i];
      }
   }

   // compare left hand side against bound
   const ValueType weight = leftHandSide - bound_;
   switch(constraintOperator_) {
      case LinearConstraintOperatorType::LessEqual : {
         if(weight > 0.0) {
            return weight;
         } else {
            return 0.0;
         }
         break;
      }
      case LinearConstraintOperatorType::Equal : {
         if(weight > 0.0) {
            return weight;
         } else if(weight < 0.0) {
            return -weight;
         } else {
            return 0.0;
         }
         break;
      }
      /*case LinearConstraintOperatorType::GreaterEqual : {
         if(weight < 0.0) {
            return -weight;
         } else {
            return 0.0;
         }
         break;
      } */
      default : { // default corresponds to GreaterEqual case
         if(weight < 0.0) {
            return -weight;
         } else {
            return 0.0;
         }
      }
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::indicatorVariablesBegin() const {
   return indicatorVariables_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::indicatorVariablesEnd() const {
   return indicatorVariables_.end();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::CoefficientsIteratorType LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::coefficientsBegin() const {
   return coefficients_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::CoefficientsIteratorType LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::coefficientsEnd() const {
   return coefficients_.end();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::BoundType LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getBound() const {
   return bound_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintOperatorValueType LinearConstraint<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getConstraintOperator() const {
   return constraintOperator_;
}

} // namespace opengm

#endif /* OPENGM_LINEAR_CONSTRAINT_HXX_ */
