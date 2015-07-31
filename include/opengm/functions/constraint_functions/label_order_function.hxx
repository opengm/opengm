#ifndef OPENGM_LABEL_ORDER_FUNCTION_HXX_
#define OPENGM_LABEL_ORDER_FUNCTION_HXX_

#include <algorithm>
#include <vector>
#include <cmath>

#include <opengm/opengm.hxx>
#include <opengm/functions/function_registration.hxx>

#include <opengm/utilities/subsequence_iterator.hxx>
#include <opengm/functions/constraint_functions/linear_constraint_function_base.hxx>
#include <opengm/datastructures/linear_constraint.hxx>

namespace opengm {

/*********************
 * class definition *
 *********************/
template<class VALUE_TYPE, class INDEX_TYPE = size_t, class LABEL_TYPE = size_t>
class LabelOrderFunction : public LinearConstraintFunctionBase<LabelOrderFunction<VALUE_TYPE,INDEX_TYPE, LABEL_TYPE> > {
public:
   // public function properties
   static const bool useSingleConstraint_    = true;
   static const bool useMultipleConstraints_ = false;

   // typedefs
   typedef LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>                                     LinearConstraintFunctionType;
   typedef LinearConstraintFunctionBase<LinearConstraintFunctionType>                                 LinearConstraintFunctionBaseType;
   typedef LinearConstraintFunctionTraits<LinearConstraintFunctionType>                               LinearConstraintFunctionTraitsType;
   typedef typename LinearConstraintFunctionTraitsType::ValueType                                     ValueType;
   typedef typename LinearConstraintFunctionTraitsType::IndexType                                     IndexType;
   typedef typename LinearConstraintFunctionTraitsType::LabelType                                     LabelType;
   typedef std::vector<ValueType>                                                                     LabelOrderType;
   typedef typename LinearConstraintFunctionTraitsType::LinearConstraintType                          LinearConstraintType;
   typedef typename LinearConstraintFunctionTraitsType::LinearConstraintsContainerType                LinearConstraintsContainerType;
   typedef typename LinearConstraintFunctionTraitsType::LinearConstraintsIteratorType                 LinearConstraintsIteratorType;
   typedef typename LinearConstraintFunctionTraitsType::IndicatorVariablesContainerType               IndicatorVariablesContainerType;
   typedef typename LinearConstraintFunctionTraitsType::IndicatorVariablesIteratorType                IndicatorVariablesIteratorType;
   typedef typename LinearConstraintFunctionTraitsType::VariableLabelPairsIteratorType                VariableLabelPairsIteratorType;
   typedef typename LinearConstraintFunctionTraitsType::ViolatedLinearConstraintsIteratorType         ViolatedLinearConstraintsIteratorType;
   typedef typename LinearConstraintFunctionTraitsType::ViolatedLinearConstraintsWeightsContainerType ViolatedLinearConstraintsWeightsContainerType;
   typedef typename LinearConstraintFunctionTraitsType::ViolatedLinearConstraintsWeightsIteratorType  ViolatedLinearConstraintsWeightsIteratorType;

   // constructors
   LabelOrderFunction();
   LabelOrderFunction(const LabelType numLabelsVar1, const LabelType numLabelsVar2, const LabelOrderType& labelOrder, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0);
   template <class ITERATOR_TYPE>
   LabelOrderFunction(const LabelType numLabelsVar1, const LabelType numLabelsVar2, ITERATOR_TYPE labelOrderBegin, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0);
   ~LabelOrderFunction();

   // function access
   template<class Iterator>
   ValueType   operator()(Iterator statesBegin) const;   // function evaluation
   size_t      shape(const size_t i) const;              // number of labels of the indicated input variable
   size_t      dimension() const;                        // number of input variables
   size_t      size() const;                             // number of parameters

   // specializations
   ValueType                  min() const;
   ValueType                  max() const;
   MinMaxFunctor<ValueType>   minMax() const;

protected:
   // storage
   static const size_t                                   dimension_ = 2;
   LabelType                                             numLabelsVar1_;
   LabelType                                             numLabelsVar2_;
   size_t                                                size_;
   LabelOrderType                                        labelOrder_;
   ValueType                                             returnValid_;
   ValueType                                             returnInvalid_;
   LinearConstraintsContainerType                        constraints_;
   mutable std::vector<size_t>                           violatedConstraintsIds_;
   mutable ViolatedLinearConstraintsWeightsContainerType violatedConstraintsWeights_;
   IndicatorVariablesContainerType                       indicatorVariableList_;

   // implementations for LinearConstraintFunctionBase
   LinearConstraintsIteratorType  linearConstraintsBegin_impl() const;
   LinearConstraintsIteratorType  linearConstraintsEnd_impl() const;
   IndicatorVariablesIteratorType indicatorVariablesOrderBegin_impl() const;
   IndicatorVariablesIteratorType indicatorVariablesOrderEnd_impl() const;
   template <class LABEL_ITERATOR>
   void challenge_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const;
   template <class LABEL_ITERATOR>
   void challengeRelaxed_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const;

   // sanity check
   bool checkLabelOrder() const;

   // helper functions
   void fillIndicatorVariableList();
   void createConstraints();

   // friends
   friend class FunctionSerialization<LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >;
   friend class opengm::LinearConstraintFunctionBase<LabelOrderFunction<VALUE_TYPE,INDEX_TYPE, LABEL_TYPE> >;
};

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
struct LinearConstraintFunctionTraits<LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
   // typedefs
   typedef VALUE_TYPE                                                                                       ValueType;
   typedef INDEX_TYPE                                                                                       IndexType;
   typedef LABEL_TYPE                                                                                       LabelType;
   typedef LinearConstraint<ValueType, IndexType, LabelType>                                                LinearConstraintType;
   typedef std::vector<LinearConstraintType>                                                                LinearConstraintsContainerType;
   typedef typename LinearConstraintsContainerType::const_iterator                                          LinearConstraintsIteratorType;
   typedef typename LinearConstraintType::IndicatorVariablesContainerType                                   IndicatorVariablesContainerType;
   typedef typename LinearConstraintType::IndicatorVariablesIteratorType                                    IndicatorVariablesIteratorType;
   typedef typename LinearConstraintType::VariableLabelPairsIteratorType                                    VariableLabelPairsIteratorType;
   typedef SubsequenceIterator<LinearConstraintsIteratorType, typename std::vector<size_t>::const_iterator> ViolatedLinearConstraintsIteratorType;
   typedef std::vector<double>                                                                              ViolatedLinearConstraintsWeightsContainerType;
   typedef typename ViolatedLinearConstraintsWeightsContainerType::const_iterator                           ViolatedLinearConstraintsWeightsIteratorType;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
struct FunctionRegistration<LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
   enum ID {
      // TODO set final Id
      Id = opengm::FUNCTION_TYPE_ID_OFFSET - 2
   };
};

/// FunctionSerialization
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class FunctionSerialization<LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
public:
   typedef typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType ValueType;

   static size_t indexSequenceSize(const LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   static size_t valueSequenceSize(const LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
};
/// \endcond

/***********************
 * class documentation *
 ***********************/
/*! \file label_order_function.hxx
 *  \brief Provides implementation of a label order function.
 */

/*! \class LabelOrderFunction
 *  \brief A linear constraint function class ensuring the correct label order
 *         for two variables.
 *
 *  This class implements a linear constraint function which ensures the correct
 *  label order for two variables. Each label is associated with a weight and
 *  the function checks the condition \f$w(l_1) \leq w(l_2)\f$ where
 *  \f$w(l_1)\f$ is the weight of the label of the first variable and
 *  \f$w(l_2)\f$ is the weight of the label of the second variable.
 *
 *  \tparam VALUE_TYPE Value type.
 *  \tparam INDEX_TYPE Index type.
 *  \tparam LABEL_TYPE Label type.
 *
 *  \ingroup functions
 */

/*! \var LabelOrderFunction::useSingleConstraint_
 *  \brief Describe the label order constraint in one single linear constraint.
 *
 *  The label order constraint is described by the following linear constraint:
 *  \f[
 *     \sum_i c_i \cdot v^0_i - \sum_j c_j \cdot v^1_j \quad \leq \quad 0.
 *  \f]
 *  Where \f$c_i\f$ is the weight for label i, \f$v^0_i\f$ is the indicator
 *  variable of variable 0 which is 1 if variable 0 is set to label i and 0
 *  otherwise and \f$v^1_j\f$ is the indicator variable of variable 1.
 *
 *  \note LabelOrderFunction::useSingleConstraint_ can be used in combination
 *        with LabelOrderFunction::useMultipleConstraints_ in this case both
 *        descriptions of the label order constraint are considered. At least
 *        one of both variables has to be set to true.
 */

/*! \var LabelOrderFunction::useMultipleConstraints_
 *  \brief Describe the label order constraint in multiple linear constraints.
 *
 *  The label order constraint is described by the following set of linear
 *  constraints:
 *  \f[
 *     c_i \cdot v^0_i - \sum_j c_j \cdot v^1_j \quad \leq \quad 0 \qquad \forall i \in \{0, ..., n - 1\}.
 *  \f]
 *  Where \f$c_i\f$ is the weight for label i, \f$v^0_i\f$ is the indicator
 *  variable of variable 0 which is 1 if variable 0 is set to label i and 0
 *  otherwise, \f$v^1_j\f$ is the indicator variable of variable 1 and n is the
 *  number of labels of variable 0.
 *
 *  \note LabelOrderFunction::useMultipleConstraints_ can be used in combination
 *        with LabelOrderFunction::useSingleConstraint_ in this case both
 *        descriptions of the label order constraint are considered. At least
 *        one of both variables has to be set to true.
 */

/*! \typedef LabelOrderFunction::LinearConstraintFunctionType
 *  \brief Typedef of the LabelOrderFunction class with appropriate
 *         template parameter.
 */

/*! \typedef LabelOrderFunction::LinearConstraintFunctionBaseType
 *  \brief Typedef of the LinearConstraintFunctionBase class with appropriate
 *         template parameter.
 */

/*! \typedef LabelOrderFunction::LinearConstraintFunctionTraitsType
 *  \brief Typedef of the LinearConstraintFunctionTraits class with appropriate
 *         template parameter.
 */

/*! \typedef LabelOrderFunction::ValueType
 *  \brief Typedef of the VALUE_TYPE template parameter type from the class
 *         LabelOrderFunction.
 */

/*! \typedef LabelOrderFunction::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         LabelOrderFunction.
 */

/*! \typedef LabelOrderFunction::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         LabelOrderFunction.
 */

/*! \typedef LabelOrderFunction::LabelOrderType
 *  \brief Type to store the weights of the label order.
 */

/*! \typedef LabelOrderFunction::LinearConstraintType
 *  \brief Typedef of the LinearConstraint class which is used to represent
 *         linear constraints.
 */

/*! \typedef LabelOrderFunction::LinearConstraintsContainerType
 *  \brief Defines the linear constraints container type which is used to store
 *         multiple linear constraints.
 */

/*! \typedef LabelOrderFunction::LinearConstraintsIteratorType
 *  \brief Defines the linear constraints container iterator type which is used
 *         to iterate over the set of linear constraints.
 */

/*! \typedef LabelOrderFunction::IndicatorVariablesContainerType
 *  \brief Defines the indicator variables container type which is used to store
 *         the indicator variables used by the linear constraint function.
 */

/*! \typedef LabelOrderFunction::IndicatorVariablesIteratorType
 *  \brief Defines the indicator variables container iterator type which is used
 *         to iterate over the indicator variables used by the linear constraint
 *         function.
 */

/*! \typedef LabelOrderFunction::VariableLabelPairsIteratorType
 *  \brief Defines the variable label pairs iterator type which is used
 *         to iterate over the variable label pairs of an indicator variable.
 */

/*! \typedef LabelOrderFunction::ViolatedLinearConstraintsIteratorType
 *  \brief Defines the violated linear constraints iterator type which is used
 *         to iterate over the set of violated linear constraints.
 */

/*! \typedef LabelOrderFunction::ViolatedLinearConstraintsWeightsContainerType
 *  \brief Defines the violated linear constraints weights container type which
 *         is used to store the weights of the violated linear constraints.
 */

/*! \typedef LabelOrderFunction::ViolatedLinearConstraintsWeightsIteratorType
 *  \brief Defines the violated linear constraints weights iterator type which
 *         is used to iterate over the weights of the violated linear
 *         constraints.
 */

/*! \fn LabelOrderFunction::LabelOrderFunction()
 *  \brief LabelOrderFunction constructor.
 *
 *  This constructor will create an empty LabelOrderFunction.
 */

/*! \fn LabelOrderFunction::LabelOrderFunction(const LabelType numLabelsVar1, const LabelType numLabelsVar2, const LabelOrderType& labelOrder, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0)
 *  \brief LabelOrderFunction constructor.
 *
 *  This constructor will create a LabelOrderFunction.
 *
 *  \param[in] numLabelsVar1 Number of labels for the first variable.
 *  \param[in] numLabelsVar2 Number of labels for the second variable.
 *  \param[in] labelOrder Weights for the label order.
 *  \param[in] returnValid The value which will be returned by the function
 *                         evaluation if no constraint is violated.
 *  \param[in] returnInvalid The value which will be returned by the function
 *                           evaluation if at least one constraint is violated.
 */

/*! \fn LabelOrderFunction::LabelOrderFunction(const LabelType numLabelsVar1, const LabelType numLabelsVar2, ITERATOR_TYPE labelOrderBegin, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0)
 *  \brief LabelOrderFunction constructor.
 *
 *  This constructor will create a LabelOrderFunction.
 *
 *  \tparam ITERATOR_TYPE Iterator to iterate over the weights of the label
 *                        order.
 *
 *  \param[in] numLabelsVar1 Number of labels for the first variable.
 *  \param[in] numLabelsVar2 Number of labels for the second variable.
 *  \param[in] labelOrderBegin Iterator pointing to the begin of the weights for
 *                             the label order.
 *  \param[in] labelOrderBegin Iterator pointing to the end of the weights for
 *                             the label order.
 *  \param[in] returnValid The value which will be returned by the function
 *                         evaluation if no constraint is violated.
 *  \param[in] returnInvalid The value which will be returned by the function
 *                           evaluation if at least one constraint is violated.
 */

/*! \fn LabelOrderFunction::~LabelOrderFunction()
 *  \brief LabelOrderFunction destructor.
 */

/*! \fn LabelOrderFunction::ValueType LabelOrderFunction::operator()(Iterator statesBegin) const
 *  \brief Function evaluation.
 *
 *  \param[in] statesBegin Iterator pointing to the begin of a sequence of
 *                         labels for the variables of the function.
 *
 *  \return LabelOrderFunction::returnValid_ if no constraint is violated
 *          by the labeling. LabelOrderFunction::returnInvalid_ if at
 *          least one constraint is violated by the labeling.
 */

/*! \fn size_t LabelOrderFunction::shape(const size_t i) const
 *  \brief Number of labels of the indicated input variable.
 *
 *  \param[in] i Index of the variable.
 *
 *  \return Returns the number of labels of the i-th variable.
 */

/*! \fn size_t LabelOrderFunction::dimension() const
 *  \brief Number of input variables.
 *
 *  \return Returns the number of variables.
 */

/*! \fn size_t LabelOrderFunction::size() const
 *  \brief Number of parameters.
 *
 *  \return Returns the number of parameters.
 */

/*! \fn LabelOrderFunction::ValueType LabelOrderFunction::min() const
 *  \brief Minimum value of the label order function.
 *
 *  \return Returns the minimum value of LabelOrderFunction::returnValid_
 *          and LabelOrderFunction::returnInvalid_.
 */

/*! \fn LabelOrderFunction::ValueType LabelOrderFunction::max() const
 *  \brief Maximum value of the label order function.
 *
 *  \return Returns the maximum value of LabelOrderFunction::returnValid_
 *          and LabelOrderFunction::returnInvalid_.
 */

/*! \fn MinMaxFunctor LabelOrderFunction::minMax() const
 *  \brief Get minimum and maximum at the same time.
 *
 *  \return Returns a functor containing the minimum and the maximum value of
 *          the label order function.
 */

/*! \var LabelOrderFunction::dimension_
 *  \brief The dimension of the label order function.
 */

/*! \var LabelOrderFunction::numLabelsVar1_
 *  \brief The number of labels of the first variable.
 */

/*! \var LabelOrderFunction::numLabelsVar2_
 *  \brief The number of labels of the second variable.
 */

/*! \var LabelOrderFunction::size_
 *  \brief Stores the size of the label order function.
 */

/*! \var LabelOrderFunction::labelOrder_
 *  \brief The weights defining the label order.
 */

/*! \var LabelOrderFunction::returnValid_
 *  \brief Stores the return value of LabelOrderFunction::operator() if no
 *         constraint is violated.
 */

/*! \var LabelOrderFunction::returnInvalid_
 *  \brief Stores the return value of LabelOrderFunction::operator() if at
 *         least one constraint is violated.
 */

/*! \var LabelOrderFunction::constraints_
 *  \brief Stores the linear constraints of the label order function.
 */

/*! \var LabelOrderFunction::violatedConstraintsIds_
 *  \brief Stores the indices of the violated constraints which are detected by
 *         LabelOrderFunction::challenge and
 *         LabelOrderFunction::challengeRelaxed.
 */

/*! \var LabelOrderFunction::violatedConstraintsWeights_
 *  \brief Stores the weights of the violated constraints which are detected by
 *         LabelOrderFunction::challenge and
 *         LabelOrderFunction::challengeRelaxed.
 */

/*! \var LabelOrderFunction::indicatorVariableList_
 *  \brief A list of all indicator variables present in the label order
 *         function.
 */

/*! \fn LabelOrderFunction::LinearConstraintsIteratorType LabelOrderFunction::linearConstraintsBegin_impl() const
 *  \brief Implementation of
 *         LinearConstraintFunctionBase::linearConstraintsBegin.
 */

/*! \fn LabelOrderFunction::LinearConstraintsIteratorType LabelOrderFunction::linearConstraintsEnd_impl() const
 *  \brief Implementation of LinearConstraintFunctionBase::linearConstraintsEnd.
 */

/*! \fn LabelOrderFunction::IndicatorVariablesIteratorType LabelOrderFunction::indicatorVariablesOrderBegin_impl() const
 *  \brief Implementation of
 *         LinearConstraintFunctionBase::indicatorVariablesOrderBegin.
 */

/*! \fn LabelOrderFunction::IndicatorVariablesIteratorType LabelOrderFunction::indicatorVariablesOrderEnd_impl() const
 *  \brief Implementation of
 *         LinearConstraintFunctionBase::indicatorVariablesOrderEnd.
 */

/*! \fn void LabelOrderFunction::challenge_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const
 *  \brief Implementation of LinearConstraintFunctionBase::challenge.
 */

/*! \fn void LabelOrderFunction::challengeRelaxed_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const
 *  \brief Implementation of LinearConstraintFunctionBase::challengeRelaxed.
 */

/*! \fn bool LabelOrderFunction::checkLabelOrder() const
 *  \brief Check label order weights. Only used for assertion in debug mode.
 *
 *  \return Returns true if the number of weights is large enough to define a
 *          weight for each label of both variables. False otherwise.
 */

/*! \fn bool LabelOrderFunction::fillIndicatorVariableList()
 *  \brief Helper function to fill LabelOrderFunction::indicatorVariableList_
 *         with all indicator variables used by the label order function.
 */

/*! \fn bool LabelOrderFunction::createConstraints()
 *  \brief Helper function to create all linear constraints which are implied by
 *         the label order function.
 */

/******************
 * implementation *
 ******************/
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelOrderFunction() : numLabelsVar1_(),
   numLabelsVar2_(), size_(), labelOrder_(), returnValid_(), returnInvalid_(),
   constraints_(), violatedConstraintsIds_(), violatedConstraintsWeights_(),
   indicatorVariableList_() {
   if(!(useSingleConstraint_ || useMultipleConstraints_)) {
      throw opengm::RuntimeError("Unsupported configuration for label order function. At least one of LabelOrderFunction::useSingleConstraint_ and LabelOrderFunction::useMultipleConstraints_ has to be set to true.");
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelOrderFunction(const LabelType numLabelsVar1, const LabelType numLabelsVar2, const LabelOrderType& labelOrder, const ValueType returnValid, const ValueType returnInvalid)
   : numLabelsVar1_(numLabelsVar1), numLabelsVar2_(numLabelsVar2),
     size_(numLabelsVar1_ * numLabelsVar2_), labelOrder_(labelOrder),
     returnValid_(returnValid), returnInvalid_(returnInvalid),
     constraints_((useSingleConstraint_ ? 1 : 0) + (useMultipleConstraints_ ? numLabelsVar1_ : 0)),
     violatedConstraintsIds_((useSingleConstraint_ ? 1 : 0) + (useMultipleConstraints_ ? numLabelsVar1_ : 0)),
     violatedConstraintsWeights_((useSingleConstraint_ ? 1 : 0) + (useMultipleConstraints_ ? numLabelsVar1_ : 0)),
     indicatorVariableList_(numLabelsVar1_ + numLabelsVar2_) {
   OPENGM_ASSERT(checkLabelOrder());

   if(!(useSingleConstraint_ || useMultipleConstraints_)) {
      throw opengm::RuntimeError("Unsupported configuration for label order function. At least one of LabelOrderFunction::useSingleConstraint_ and LabelOrderFunction::useMultipleConstraints_ has to be set to true.");
   }

   // fill indicator variable list
   fillIndicatorVariableList();

   // create linear constraints
   createConstraints();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class ITERATOR_TYPE>
inline LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelOrderFunction(const LabelType numLabelsVar1, const LabelType numLabelsVar2, ITERATOR_TYPE labelOrderBegin, const ValueType returnValid, const ValueType returnInvalid)
   : numLabelsVar1_(numLabelsVar1), numLabelsVar2_(numLabelsVar2),
     size_(numLabelsVar1_ * numLabelsVar2_),
     labelOrder_(labelOrderBegin, labelOrderBegin + std::max(numLabelsVar1_, numLabelsVar2_)),
     returnValid_(returnValid), returnInvalid_(returnInvalid),
     constraints_((useSingleConstraint_ ? 1 : 0) + (useMultipleConstraints_ ? numLabelsVar1_ : 0)),
     violatedConstraintsIds_((useSingleConstraint_ ? 1 : 0) + (useMultipleConstraints_ ? numLabelsVar1_ : 0)),
     violatedConstraintsWeights_((useSingleConstraint_ ? 1 : 0) + (useMultipleConstraints_ ? numLabelsVar1_ : 0)),
     indicatorVariableList_(numLabelsVar1_ + numLabelsVar2_) {
   OPENGM_ASSERT(checkLabelOrder());

   if(!(useSingleConstraint_ || useMultipleConstraints_)) {
      throw opengm::RuntimeError("Unsupported configuration for label order function. At least one of LabelOrderFunction::useSingleConstraint_ and LabelOrderFunction::useMultipleConstraints_ has to be set to true.");
   }

   // fill indicator variable list
   fillIndicatorVariableList();

   // create linear constraints
   createConstraints();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::~LabelOrderFunction() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class Iterator>
inline typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::operator()(Iterator statesBegin) const {
   if(labelOrder_[statesBegin[0]] <= labelOrder_[statesBegin[1]]) {
      return returnValid_;
   } else {
      return returnInvalid_;
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::shape(const size_t i) const {
   OPENGM_ASSERT(i < dimension_);
   return (i==0 ? numLabelsVar1_ : numLabelsVar2_);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::dimension() const {
   return dimension_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::size() const {
   return size_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::min() const {
   return returnValid_ < returnInvalid_ ? returnValid_ : returnInvalid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::max() const {
   return returnValid_ > returnInvalid_ ? returnValid_ : returnInvalid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline MinMaxFunctor<typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType> LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::minMax() const {
   if(returnValid_ < returnInvalid_) {
      return MinMaxFunctor<VALUE_TYPE>(returnValid_, returnInvalid_);
   }
   else {
      return MinMaxFunctor<VALUE_TYPE>(returnInvalid_, returnValid_);
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::linearConstraintsBegin_impl() const {
   return constraints_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::linearConstraintsEnd_impl() const {
   return constraints_.end();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::indicatorVariablesOrderBegin_impl() const {
   return indicatorVariableList_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::indicatorVariablesOrderEnd_impl() const {
   return indicatorVariableList_.end();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class LABEL_ITERATOR>
inline void LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::challenge_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance) const {
   const ValueType weight = labelOrder_[labelingBegin[0]] - labelOrder_[labelingBegin[1]];
   if(weight <= tolerance) {
      violatedConstraintsBegin = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
      violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
      violatedConstraintsWeightsBegin = violatedConstraintsWeights_.begin();
      return;
   } else {
      if(useSingleConstraint_) {
         violatedConstraintsIds_[0] = 0;
         violatedConstraintsWeights_[0] = weight;
      }
      if(useMultipleConstraints_) {
         violatedConstraintsIds_[(useSingleConstraint_ ? 1 : 0)] = labelingBegin[0] + (useSingleConstraint_ ? 1 : 0);
         violatedConstraintsWeights_[(useSingleConstraint_ ? 1 : 0)] = weight;

      }

      violatedConstraintsBegin = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
      violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), (useSingleConstraint_ ? 1 : 0) + (useMultipleConstraints_ ? 1 : 0));
      violatedConstraintsWeightsBegin = violatedConstraintsWeights_.begin();
      return;
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class LABEL_ITERATOR>
inline void LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::challengeRelaxed_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance) const {
   size_t numViolatedConstraints = 0;

   double weightVar2 = 0.0;
   for(IndexType i = 0; i < numLabelsVar2_; ++i) {
      weightVar2 -= labelOrder_[i] * labelingBegin[numLabelsVar1_ + i];
   }

   double totalWeight = weightVar2;
   for(IndexType i = 0; i < numLabelsVar1_; ++i) {
      double currentWeight = (labelOrder_[i] * labelingBegin[i]); // - ();

      if(useSingleConstraint_) {
         totalWeight += currentWeight;
      }
      if(useMultipleConstraints_) {
         currentWeight += weightVar2;
         if(currentWeight > tolerance) {
            violatedConstraintsIds_[numViolatedConstraints] = i + (useSingleConstraint_ ? 1 : 0);
            violatedConstraintsWeights_[numViolatedConstraints] = currentWeight;
            ++numViolatedConstraints;
         }
      }
   }

   if(useSingleConstraint_) {
      if(totalWeight > tolerance) {
         violatedConstraintsIds_[numViolatedConstraints] = 0;
         violatedConstraintsWeights_[numViolatedConstraints] = totalWeight;
         ++numViolatedConstraints;
      }
   }

   violatedConstraintsBegin = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
   violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), numViolatedConstraints);
   violatedConstraintsWeightsBegin = violatedConstraintsWeights_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline bool LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::checkLabelOrder() const {
   if(labelOrder_.size() < std::max(numLabelsVar1_, numLabelsVar2_)) {
      return false;
   } else {
      return true;
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::fillIndicatorVariableList() {
   for(size_t i = 0; i < numLabelsVar1_; ++i) {
      indicatorVariableList_[i] = typename LinearConstraintType::IndicatorVariableType(IndexType(0), LabelType(i));
   }
   for(size_t i = 0; i < numLabelsVar2_; ++i) {
      indicatorVariableList_[numLabelsVar1_ + i] = typename LinearConstraintType::IndicatorVariableType(IndexType(1), LabelType(i));
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::createConstraints() {
   if(useSingleConstraint_) {
      constraints_[0].setBound(0.0);
      constraints_[0].setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::LessEqual);
      for(size_t i = 0; i < numLabelsVar1_; ++i) {
         constraints_[0].add(typename LinearConstraintType::IndicatorVariableType(IndexType(0), LabelType(i)), labelOrder_[i]);
      }
      for(size_t i = 0; i < numLabelsVar2_; ++i) {
         constraints_[0].add(typename LinearConstraintType::IndicatorVariableType(IndexType(1), LabelType(i)), -labelOrder_[i]);
      }
   }
   if(useMultipleConstraints_) {
      for(size_t i = 0; i < numLabelsVar1_; ++i) {
         constraints_[i + (useSingleConstraint_ ? 1 : 0)].add(typename LinearConstraintType::IndicatorVariableType(IndexType(0), LabelType(i)), labelOrder_[i]);
         constraints_[i + (useSingleConstraint_ ? 1 : 0)].setBound(0.0);
         constraints_[i + (useSingleConstraint_ ? 1 : 0)].setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::LessEqual);
         for(size_t j = 0; j < numLabelsVar2_; ++j) {
               constraints_[i + (useSingleConstraint_ ? 1 : 0)].add(typename LinearConstraintType::IndicatorVariableType(IndexType(1), LabelType(j)), -labelOrder_[j]);
         }
      }
   }
}

/// \cond HIDDEN_SYMBOLS
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::indexSequenceSize(const LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t shapeSize        = 2; // numLabelsVar1 and numLabelsVar2
   const size_t labelOrderSize   = 1;
   const size_t totalIndexSize   = shapeSize + labelOrderSize;
   return totalIndexSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::valueSequenceSize(const LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t labelOrderSize   = src.labelOrder_.size();
   const size_t returnSize      = 2; //returnValid and returnInvalid
   const size_t totalValueSize   = labelOrderSize + returnSize;
   return totalValueSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
inline void FunctionSerialization<LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::serialize(const LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src, INDEX_OUTPUT_ITERATOR indexOutIterator, VALUE_OUTPUT_ITERATOR valueOutIterator) {
   // index output
   // shape
   *indexOutIterator = src.numLabelsVar1_;
   ++indexOutIterator;
   *indexOutIterator = src.numLabelsVar2_;
   ++indexOutIterator;

   // label order size
   *indexOutIterator = src.labelOrder_.size();

   // value output
   // label order
   for(size_t i = 0; i < src.labelOrder_.size(); ++i) {
      *valueOutIterator = src.labelOrder_[i];
      ++valueOutIterator;
   }

   // return values
   *valueOutIterator = src.returnValid_;
   ++valueOutIterator;
   *valueOutIterator = src.returnInvalid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
inline void FunctionSerialization<LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::deserialize( INDEX_INPUT_ITERATOR indexInIterator, VALUE_INPUT_ITERATOR valueInIterator, LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& dst) {
   // index input
   // shape
   const size_t numLabelsVar1 = *indexInIterator;
   ++indexInIterator;
   const size_t numLabelsVar2 = *indexInIterator;
   ++indexInIterator;

   // label order size
   const size_t labelOrderSize = *indexInIterator;

   // value input
   typename LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LabelOrderType labelOrder(valueInIterator, valueInIterator + labelOrderSize);
   valueInIterator += labelOrderSize;

   // valid value
   VALUE_TYPE returnValid = *valueInIterator;
   ++valueInIterator;

   // invalid value
   VALUE_TYPE returnInvalid = *valueInIterator;

   dst = LabelOrderFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(numLabelsVar1, numLabelsVar2, labelOrder, returnValid, returnInvalid);
}
/// \endcond

} // namespace opengm

#endif /* OPENGM_LABEL_ORDER_FUNCTION_HXX_ */
