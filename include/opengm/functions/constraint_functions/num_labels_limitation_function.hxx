#ifndef OPENGM_NUM_LABELS_LIMITATION_FUNCTION_HXX_
#define OPENGM_NUM_LABELS_LIMITATION_FUNCTION_HXX_

#include <opengm/opengm.hxx>
#include <opengm/functions/function_registration.hxx>

#include <opengm/utilities/subsequence_iterator.hxx>
#include <opengm/functions/constraint_functions/linear_constraint_function_base.hxx>
#include <opengm/datastructures/linear_constraint.hxx>
#include <opengm/utilities/unsigned_integer_pow.hxx>

namespace opengm {

/*********************
 * class definition *
 *********************/
template<class VALUE_TYPE, class INDEX_TYPE = size_t, class LABEL_TYPE = size_t>
class NumLabelsLimitationFunction : public LinearConstraintFunctionBase<NumLabelsLimitationFunction<VALUE_TYPE,INDEX_TYPE, LABEL_TYPE> > {
public:
   // typedefs
   typedef NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>                            LinearConstraintFunctionType;
   typedef LinearConstraintFunctionBase<LinearConstraintFunctionType>                                 LinearConstraintFunctionBaseType;
   typedef LinearConstraintFunctionTraits<LinearConstraintFunctionType>                               LinearConstraintFunctionTraitsType;
   typedef typename LinearConstraintFunctionTraitsType::ValueType                                     ValueType;
   typedef typename LinearConstraintFunctionTraitsType::IndexType                                     IndexType;
   typedef typename LinearConstraintFunctionTraitsType::LabelType                                     LabelType;
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
   NumLabelsLimitationFunction();
   template <class SHAPE_ITERATOR_TYPE>
   NumLabelsLimitationFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LabelType maxNumUsedLabels, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0);
   NumLabelsLimitationFunction(const IndexType numVariables, const LabelType numLabels, const LabelType maxNumUsedLabels, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0);
   ~NumLabelsLimitationFunction();

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
   std::vector<LabelType>                                shape_;
   size_t                                                numVariables_;
   bool                                                  useSameNumLabels_;
   LabelType                                             maxNumLabels_;
   LabelType                                             maxNumUsedLabels_;
   size_t                                                size_;
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

   // helper functions
   void fillIndicatorVariableList();
   void createConstraints();

   // friends
   friend class FunctionSerialization<NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >;
   friend class opengm::LinearConstraintFunctionBase<NumLabelsLimitationFunction<VALUE_TYPE,INDEX_TYPE, LABEL_TYPE> >;
};

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
struct LinearConstraintFunctionTraits<NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
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
struct FunctionRegistration<NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
   enum ID {
      // TODO set final Id
      Id = opengm::FUNCTION_TYPE_ID_OFFSET - 3
   };
};

/// FunctionSerialization
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class FunctionSerialization<NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
public:
   typedef typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType ValueType;

   static size_t indexSequenceSize(const NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   static size_t valueSequenceSize(const NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
   static void serialize(const NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
   static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
};
/// \endcond

/***********************
 * class documentation *
 ***********************/
/*! \file num_labels_limitation_function.hxx
 *  \brief Provides implementation of a number of active labels limitation
 *         function.
 */

/*! \class NumLabelsLimitationFunction
 *  \brief A linear constraint function class for limiting the number of used
 *         labels.
 *
 *  This class implements a linear constraint function which limits the number
 *  of used labels. The number of different labels in the assignment of the
 *  function is counted and if this number exceeds the value of maximum allowed
 *  different labels the function will return an invalid value.
 *
 *  \tparam VALUE_TYPE Value type.
 *  \tparam INDEX_TYPE Index type.
 *  \tparam LABEL_TYPE Label type.
 *
 *  \ingroup functions
 */

/*! \typedef NumLabelsLimitationFunction::LinearConstraintFunctionType
 *  \brief Typedef of the NumLabelsLimitationFunction class with appropriate
 *         template parameter.
 */

/*! \typedef NumLabelsLimitationFunction::LinearConstraintFunctionBaseType
 *  \brief Typedef of the LinearConstraintFunctionBase class with appropriate
 *         template parameter.
 */

/*! \typedef NumLabelsLimitationFunction::LinearConstraintFunctionTraitsType
 *  \brief Typedef of the LinearConstraintFunctionTraits class with appropriate
 *         template parameter.
 */

/*! \typedef NumLabelsLimitationFunction::ValueType
 *  \brief Typedef of the VALUE_TYPE template parameter type from the class
 *         LabelOrderFunction.
 */

/*! \typedef NumLabelsLimitationFunction::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         LabelOrderFunction.
 */

/*! \typedef NumLabelsLimitationFunction::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         LabelOrderFunction.
 */

/*! \typedef NumLabelsLimitationFunction::LinearConstraintType
 *  \brief Typedef of the LinearConstraint class which is used to represent
 *         linear constraints.
 */

/*! \typedef NumLabelsLimitationFunction::LinearConstraintsContainerType
 *  \brief Defines the linear constraints container type which is used to store
 *         multiple linear constraints.
 */

/*! \typedef NumLabelsLimitationFunction::LinearConstraintsIteratorType
 *  \brief Defines the linear constraints container iterator type which is used
 *         to iterate over the set of linear constraints.
 */

/*! \typedef NumLabelsLimitationFunction::IndicatorVariablesContainerType
 *  \brief Defines the indicator variables container type which is used to store
 *         the indicator variables used by the linear constraint function.
 */

/*! \typedef NumLabelsLimitationFunction::IndicatorVariablesIteratorType
 *  \brief Defines the indicator variables container iterator type which is used
 *         to iterate over the indicator variables used by the linear constraint
 *         function.
 */

/*! \typedef NumLabelsLimitationFunction::VariableLabelPairsIteratorType
 *  \brief Defines the variable label pairs iterator type which is used
 *         to iterate over the variable label pairs of an indicator variable.
 */

/*! \typedef NumLabelsLimitationFunction::ViolatedLinearConstraintsIteratorType
 *  \brief Defines the violated linear constraints iterator type which is used
 *         to iterate over the set of violated linear constraints.
 */

/*! \typedef NumLabelsLimitationFunction::ViolatedLinearConstraintsWeightsContainerType
 *  \brief Defines the violated linear constraints weights container type which
 *         is used to store the weights of the violated linear constraints.
 */

/*! \typedef NumLabelsLimitationFunction::ViolatedLinearConstraintsWeightsIteratorType
 *  \brief Defines the violated linear constraints weights iterator type which
 *         is used to iterate over the weights of the violated linear
 *         constraints.
 */

/*! \fn NumLabelsLimitationFunction::NumLabelsLimitationFunction()
 *  \brief NumLabelsLimitationFunction constructor.
 *
 *  This constructor will create an empty NumLabelsLimitationFunction.
 */

/*! \fn NumLabelsLimitationFunction::NumLabelsLimitationFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LabelType maxNumUsedLabels, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0)
 *  \brief NumLabelsLimitationFunction constructor.
 *
 *  This constructor will create a NumLabelsLimitationFunction where each
 *   variable can have a different number of labels.
 *
 *  \tparam SHAPE_ITERATOR_TYPE Iterator type to iterate over the shape of the
 *                              function.
 *
 *  \param[in] shapeBegin Iterator pointing to the begin of a sequence which
 *                        defines the shape of the function.
 *  \param[in] shapeEnd Iterator pointing to the end of a sequence which defines
 *                      the shape of the function.
 *  \param[in] maxNumUsedLabels The maximum number of different labels which are
 *                              allowed in the assignment of the function.
 *  \param[in] returnValid The value which will be returned by the function
 *                         evaluation if the number of different labels which
 *                         are used in the assignment of the function do not
 *                         exceed NumLabelsLimitationFunction::maxNumUsedLabels.
 *  \param[in] returnInvalid The value which will be returned by the function
 *                           evaluation if the number of different labels which
 *                           are used in the assignment of the function exceed
 *                           NumLabelsLimitationFunction::maxNumUsedLabels.
 */

/*! \fn NumLabelsLimitationFunction::NumLabelsLimitationFunction(const IndexType numVariables, const LabelType numLabels, const LabelType maxNumUsedLabels, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0)
 *  \brief NumLabelsLimitationFunction constructor.
 *
 *  This constructor will create a NumLabelsLimitationFunction where each
 *  variable has the same number of labels.
 *
 *  \param[in] numVariables The number of variables of the function
 *  \param[in] numLabels The number of labels of each variable.
 *  \param[in] maxNumUsedLabels The maximum number of different labels which are
 *                              allowed in the assignment of the function.
 *  \param[in] returnValid The value which will be returned by the function
 *                         evaluation if the number of different labels which
 *                         are used in the assignment of the function do not
 *                         exceed NumLabelsLimitationFunction::maxNumUsedLabels.
 *  \param[in] returnInvalid The value which will be returned by the function
 *                           evaluation if the number of different labels which
 *                           are used in the assignment of the function exceed
 *                           NumLabelsLimitationFunction::maxNumUsedLabels.
 */

/*! \fn NumLabelsLimitationFunction::~NumLabelsLimitationFunction()
 *  \brief NumLabelsLimitationFunction destructor.
 */

/*! \fn NumLabelsLimitationFunction::ValueType NumLabelsLimitationFunction::operator()(Iterator statesBegin) const
 *   \brief Function evaluation.
 *
 *  \param[in] statesBegin Iterator pointing to the begin of a sequence of
 *                         labels for the variables of the function.
 *
 *  \return NumLabelsLimitationFunction::returnValid_ if no constraint is
 *          violated by the labeling.
 *          NumLabelsLimitationFunction::returnInvalid_ if at least one
 *          constraint is violated by the labeling.
 */

/*! \fn size_t NumLabelsLimitationFunction::shape(const size_t i) const
 *  \brief Number of labels of the indicated input variable.
 *
 *  \param[in] i Index of the variable.
 *
 *  \return Returns the number of labels of the i-th variable.
 */

/*! \fn size_t NumLabelsLimitationFunction::dimension() const
 *  \brief Number of input variables.
 *
 *  \return Returns the number of variables.
 */

/*! \fn size_t NumLabelsLimitationFunction::size() const
 *  \brief Number of parameters.
 *
 *  \return Returns the number of parameters.
 */

/*! \fn NumLabelsLimitationFunction::ValueType NumLabelsLimitationFunction::min() const
 *  \brief Minimum value of the function.
 *
 *  \return Returns the minimum value of
 *          NumLabelsLimitationFunction::returnValid_ and
 *          NumLabelsLimitationFunction::returnInvalid_.
 */

/*! \fn NumLabelsLimitationFunction::ValueType NumLabelsLimitationFunction::max() const
 *  \brief Maximum value of the function.
 *
 *  \return Returns the maximum value of
 *          NumLabelsLimitationFunction::returnValid_ and
 *          NumLabelsLimitationFunction::returnInvalid_.
 */

/*! \fn MinMaxFunctor NumLabelsLimitationFunction::minMax() const
 *  \brief Get minimum and maximum at the same time.
 *
 *  \return Returns a functor containing the minimum and the maximum value of
 *          the function.
 */

/*! \var NumLabelsLimitationFunction::shape_
 *  \brief The shape of the function (only used if variables have different
 *         number of labels).
 */

/*! \var NumLabelsLimitationFunction::numVariables_
 *  \brief The number of variables of the function.
 */

/*! \var NumLabelsLimitationFunction::useSameNumLabels_
 *  \brief Indicator to tell that all variables have the same number of
 *         variables.
 */

/*! \var NumLabelsLimitationFunction::maxNumLabels_
 *  \brief The maximum number of labels of the variables.
 */

/*! \var NumLabelsLimitationFunction::maxNumUsedLabels_
 *  \brief The maximum number of different labels which are allowed in the
 *         assignment of the function
 */

/*! \var NumLabelsLimitationFunction::size_
 *  \brief Stores the size of the function.
 */

/*! \var NumLabelsLimitationFunction::returnValid_
 *  \brief Stores the return value of NumLabelsLimitationFunction::operator() if
 *         the number of different labels which are used in the assignment of
 *         the function is smaller or equal to
 *         NumLabelsLimitationFunction::maxNumUsedLabels_.
 */

/*! \var NumLabelsLimitationFunction::returnInvalid_
 *  \brief Stores the return value of NumLabelsLimitationFunction::operator() if
 *         the number of different labels which are used in the assignment of
 *         the function exceeds NumLabelsLimitationFunction::maxNumUsedLabels_.
 */

/*! \var NumLabelsLimitationFunction::constraints_
 *  \brief Stores the linear constraints of the function.
 */

/*! \var NumLabelsLimitationFunction::violatedConstraintsIds_
 *  \brief Stores the indices of the violated constraints which are detected by
 *         NumLabelsLimitationFunction::challenge and
 *         NumLabelsLimitationFunction::challengeRelaxed.
 */

/*! \var NumLabelsLimitationFunction::violatedConstraintsWeights_
 *  \brief Stores the weights of the violated constraints which are detected by
 *         NumLabelsLimitationFunction::challenge and
 *         NumLabelsLimitationFunction::challengeRelaxed.
 */

/*! \var NumLabelsLimitationFunction::indicatorVariableList_
 *  \brief A list of all indicator variables present in the function.
 */

/*! \fn NumLabelsLimitationFunction::LinearConstraintsIteratorType NumLabelsLimitationFunction::linearConstraintsBegin_impl() const
 *  \brief Implementation of LinearConstraintFunctionBase::linearConstraintsBegin.
 */

/*! \fn NumLabelsLimitationFunction::LinearConstraintsIteratorType NumLabelsLimitationFunction::linearConstraintsEnd_impl() const
 *  \brief Implementation of LinearConstraintFunctionBase::linearConstraintsEnd.
 */

/*! \fn NumLabelsLimitationFunction::IndicatorVariablesIteratorType NumLabelsLimitationFunction::indicatorVariablesOrderBegin_impl() const
 *  \brief Implementation of
 *         LinearConstraintFunctionBase::indicatorVariablesOrderBegin.
 */

/*! \fn NumLabelsLimitationFunction::IndicatorVariablesIteratorType NumLabelsLimitationFunction::indicatorVariablesOrderEnd_impl() const
 *  \brief Implementation of
 *         LinearConstraintFunctionBase::indicatorVariablesOrderEnd.
 */

/*! \fn void NumLabelsLimitationFunction::challenge_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const
 *  \brief Implementation of LinearConstraintFunctionBase::challenge.
 */

/*! \fn void NumLabelsLimitationFunction::challengeRelaxed_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const
 *  \brief Implementation of LinearConstraintFunctionBase::challengeRelaxed.
 */

/*! \fn bool NumLabelsLimitationFunction::fillIndicatorVariableList()
 *  \brief Helper function to fill
 *         NumLabelsLimitationFunction::indicatorVariableList_ with all
 *         indicator variables used by the function.
 */

/*! \fn bool NumLabelsLimitationFunction::createConstraints()
 *  \brief Helper function to create all linear constraints which are implied by
 *         the function.
 */

/******************
 * implementation *
 ******************/
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::NumLabelsLimitationFunction()
: shape_(), numVariables_(), useSameNumLabels_(), maxNumLabels_(),
  maxNumUsedLabels_(), size_(), returnValid_(), returnInvalid_(),
  constraints_(), violatedConstraintsIds_(), violatedConstraintsWeights_(),
  indicatorVariableList_() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class SHAPE_ITERATOR_TYPE>
inline NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::NumLabelsLimitationFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LabelType maxNumUsedLabels, const ValueType returnValid, const ValueType returnInvalid)
: shape_(shapeBegin, shapeEnd), numVariables_(shape_.size()),
  useSameNumLabels_(numVariables_ > 0 ? std::equal(shape_.begin() + 1, shape_.end(), shape_.begin()) : true),
  maxNumLabels_(numVariables_ > 0 ? *std::max_element(shape_.begin(), shape_.end()) : 0),
  maxNumUsedLabels_(maxNumUsedLabels),
  size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<SHAPE_ITERATOR_TYPE>::value_type>())),
  returnValid_(returnValid), returnInvalid_(returnInvalid),
  constraints_(1), violatedConstraintsIds_(1), violatedConstraintsWeights_(1),
  indicatorVariableList_(maxNumLabels_) {
   // fill indicator variable list
   fillIndicatorVariableList();

   // create linear constraints
   createConstraints();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::NumLabelsLimitationFunction(const IndexType numVariables, const LabelType numLabels, const LabelType maxNumUsedLabels, const ValueType returnValid, const ValueType returnInvalid)
: shape_(), numVariables_(numVariables), useSameNumLabels_(true),
  maxNumLabels_(numLabels), maxNumUsedLabels_(maxNumUsedLabels),
  size_(unsignedIntegerPow(maxNumLabels_, numVariables_)),
  returnValid_(returnValid), returnInvalid_(returnInvalid),
  constraints_(1), violatedConstraintsIds_(1), violatedConstraintsWeights_(1),
  indicatorVariableList_(maxNumLabels_) {
   // fill indicator variable list
   fillIndicatorVariableList();

   // create linear constraints
   createConstraints();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::~NumLabelsLimitationFunction() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class Iterator>
inline typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::operator()(Iterator statesBegin) const {
   std::vector<bool> labelIsUsed(maxNumLabels_, false);
   LabelType maxNumLabelsFound = 0;
   const Iterator statesEnd = statesBegin + numVariables_;
   while(statesBegin != statesEnd) {
      const LabelType currentLabel = *statesBegin;
      OPENGM_ASSERT(currentLabel < maxNumLabels_);
      if(!labelIsUsed[currentLabel]) {
         labelIsUsed[currentLabel] = true;
         ++maxNumLabelsFound;
         if(maxNumLabelsFound > maxNumUsedLabels_) {
            return returnInvalid_;
         }
      }
      ++statesBegin;
   }
   return returnValid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::shape(const size_t i) const {
   OPENGM_ASSERT(i < numVariables_);
   return (useSameNumLabels_ ? maxNumLabels_ : shape_[i]);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::dimension() const {
   return numVariables_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::size() const {
   return size_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::min() const {
   return returnValid_ < returnInvalid_ ? returnValid_ : returnInvalid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::max() const {
   return returnValid_ > returnInvalid_ ? returnValid_ : returnInvalid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline MinMaxFunctor<typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType> NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::minMax() const {
   if(returnValid_ < returnInvalid_) {
      return MinMaxFunctor<VALUE_TYPE>(returnValid_, returnInvalid_);
   }
   else {
      return MinMaxFunctor<VALUE_TYPE>(returnInvalid_, returnValid_);
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::linearConstraintsBegin_impl() const {
   return constraints_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::linearConstraintsEnd_impl() const {
   return constraints_.end();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::indicatorVariablesOrderBegin_impl() const {
   return indicatorVariableList_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::indicatorVariablesOrderEnd_impl() const {
   return indicatorVariableList_.end();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class LABEL_ITERATOR>
inline void NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::challenge_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance) const {
   std::vector<bool> labelIsUsed(maxNumLabels_, false);
   ValueType maxNumLabelsFound = 0;
   const LABEL_ITERATOR labelingEnd = labelingBegin + numVariables_;
   while(labelingBegin != labelingEnd) {
      const LabelType currentLabel = *labelingBegin;
      OPENGM_ASSERT(currentLabel < maxNumLabels_);
      if(!labelIsUsed[currentLabel]) {
         labelIsUsed[currentLabel] = true;
         ++maxNumLabelsFound;
         if(maxNumLabelsFound == static_cast<ValueType>(maxNumLabels_)) {
            // all labels are used
            break;
         }
      }
      ++labelingBegin;
   }

   const ValueType weight = maxNumLabelsFound - static_cast<ValueType>(maxNumUsedLabels_);
   if(weight <= tolerance) {
      violatedConstraintsBegin = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
      violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
      violatedConstraintsWeightsBegin = violatedConstraintsWeights_.begin();
   } else {
      violatedConstraintsIds_[0] = 0;
      violatedConstraintsWeights_[0] = weight;
      violatedConstraintsBegin = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
      violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 1);
      violatedConstraintsWeightsBegin = violatedConstraintsWeights_.begin();
      return;
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class LABEL_ITERATOR>
inline void NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::challengeRelaxed_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance) const {
   double weight = -static_cast<double>(maxNumUsedLabels_);
   for(LabelType i = 0; i < maxNumLabels_; ++i) {
      weight += labelingBegin[i];
   }

   violatedConstraintsBegin = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);

   if(weight > tolerance) {
      violatedConstraintsIds_[0] = 0;
      violatedConstraintsWeights_[0] = weight;
      violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 1);
      violatedConstraintsWeightsBegin = violatedConstraintsWeights_.begin();
   } else {
      violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::fillIndicatorVariableList() {
   for(LabelType i = 0; i < maxNumLabels_; ++i) {
      typename LinearConstraintType::IndicatorVariableType indicatorVariable;
      indicatorVariable.setLogicalOperatorType(LinearConstraintType::IndicatorVariableType::Or);
      for(IndexType j = 0; j < numVariables_; ++j) {
         if(useSameNumLabels_) {
            indicatorVariable.add(j, i);
         } else if(shape_[j] > i) {
            indicatorVariable.add(j, i);
         }
      }
      indicatorVariableList_[i] = indicatorVariable;
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::createConstraints() {
   constraints_[0].setBound(maxNumUsedLabels_);
   constraints_[0].setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::LessEqual);
   for(LabelType i = 0; i < maxNumLabels_; ++i) {
      constraints_[0].add(indicatorVariableList_[i], 1.0);
   }
}

/// \cond HIDDEN_SYMBOLS
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::indexSequenceSize(const NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t sameNumLabelsSize    = 1;
   const size_t numVariablesSize     = 1;
   const size_t shapeSize            = src.useSameNumLabels_ ? 1 : src.shape_.size();
   const size_t maxNumUsedLabelsSize = 1;

   const size_t totalIndexSize = sameNumLabelsSize + numVariablesSize + shapeSize + maxNumUsedLabelsSize;
   return totalIndexSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::valueSequenceSize(const NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t returnSize = 2; //returnValid and returnInvalid

   const size_t totalValueSize = returnSize;
   return totalValueSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
inline void FunctionSerialization<NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::serialize(const NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src, INDEX_OUTPUT_ITERATOR indexOutIterator, VALUE_OUTPUT_ITERATOR valueOutIterator) {
   // index output
   // shape
   *indexOutIterator = static_cast<typename INDEX_OUTPUT_ITERATOR::value_type>(src.useSameNumLabels_);
   ++indexOutIterator;
   *indexOutIterator = src.numVariables_;
   ++indexOutIterator;
   if(src.useSameNumLabels_) {
      *indexOutIterator = src.maxNumLabels_;
      ++indexOutIterator;
   } else {
      for(size_t i = 0; i < src.shape_.size(); ++i) {
         *indexOutIterator = src.shape_[i];
         ++indexOutIterator;
      }
   }

   // max num used labels
   *indexOutIterator = src.maxNumUsedLabels_;

   // value output
   // return values
   *valueOutIterator = src.returnValid_;
   ++valueOutIterator;
   *valueOutIterator = src.returnInvalid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
inline void FunctionSerialization<NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::deserialize(INDEX_INPUT_ITERATOR indexInIterator, VALUE_INPUT_ITERATOR valueInIterator, NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& dst) {
   typedef VALUE_TYPE ValueType;
   typedef INDEX_TYPE IndexType;
   typedef LABEL_TYPE LabelType;

   // index input
   // shape
   const bool useSameNumLabels = *indexInIterator;
   ++indexInIterator;
   const IndexType numVariables = *indexInIterator;
   ++indexInIterator;

   std::vector<LabelType> shape(indexInIterator, indexInIterator + (useSameNumLabels ? 1 : numVariables));
   indexInIterator += (useSameNumLabels ? 1 : numVariables);

   // max num used labels
   const LabelType maxNumUsedLabels = *indexInIterator;

   // value input
   // valid value
   ValueType returnValid = *valueInIterator;
   ++valueInIterator;

   // invalid value
   ValueType returnInvalid = *valueInIterator;

   if(useSameNumLabels) {
      dst = NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(numVariables, shape[0], maxNumUsedLabels, returnValid, returnInvalid);
   } else {
      dst = NumLabelsLimitationFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(shape.begin(), shape.end(), maxNumUsedLabels, returnValid, returnInvalid);
   }
}
/// \endcond

} // namespace opengm

#endif /* OPENGM_NUM_LABELS_LIMITATION_FUNCTION_HXX_ */
