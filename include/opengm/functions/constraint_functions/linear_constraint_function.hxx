#ifndef OPENGM_LINEAR_CONSTRAINT_FUNCTION_HXX_
#define OPENGM_LINEAR_CONSTRAINT_FUNCTION_HXX_

#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

#include <opengm/opengm.hxx>
#include <opengm/functions/function_registration.hxx>

#include <opengm/utilities/subsequence_iterator.hxx>
#include <opengm/datastructures/linear_constraint.hxx>
#include <opengm/functions/constraint_functions/linear_constraint_function_base.hxx>

namespace opengm {

/*********************
 * class definition *
 *********************/
template<class VALUE_TYPE, class INDEX_TYPE = size_t, class LABEL_TYPE = size_t>
class LinearConstraintFunction : public LinearConstraintFunctionBase<LinearConstraintFunction<VALUE_TYPE,INDEX_TYPE, LABEL_TYPE> > {
public:
   // typedefs
   typedef LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>                               LinearConstraintFunctionType;
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
   LinearConstraintFunction();
   template <class SHAPE_ITERATOR_TYPE>
   LinearConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LinearConstraintsContainerType& constraints, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0);
   template <class SHAPE_ITERATOR_TYPE, class CONSTRAINTS_ITERATOR_TYPE>
   LinearConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, CONSTRAINTS_ITERATOR_TYPE constraintsBegin, CONSTRAINTS_ITERATOR_TYPE constraintsEnd, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0);
   ~LinearConstraintFunction();

   // function access
   template<class STATES_ITERATOR_TYPE>
   ValueType   operator()(STATES_ITERATOR_TYPE statesBegin) const;   // function evaluation
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
   size_t                                                size_;
   LinearConstraintsContainerType                        constraints_;
   mutable std::vector<size_t>                           violatedConstraintsIds_;
   mutable ViolatedLinearConstraintsWeightsContainerType violatedConstraintsWeights_;
   ValueType                                             returnValid_;
   ValueType                                             returnInvalid_;
   IndicatorVariablesContainerType                       indicatorVariableList_;
   std::vector<std::vector<size_t> >                     indicatorVariableLookupTable_;

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
   bool checkConstraints() const;

   // helper functions
   void fillIndicatorVariableList();
   void createIndicatorVariableLookupTable();

   // friends
   friend class FunctionSerialization<LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >;
   friend class opengm::LinearConstraintFunctionBase<LinearConstraintFunction<VALUE_TYPE,INDEX_TYPE, LABEL_TYPE> >;
};

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
struct LinearConstraintFunctionTraits<LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
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
struct FunctionRegistration<LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
   enum ID {
      // TODO set final Id
      Id = opengm::FUNCTION_TYPE_ID_OFFSET - 1
   };
};

/// FunctionSerialization
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class FunctionSerialization<LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> > {
public:
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType ValueType;

   static size_t indexSequenceSize(const LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   static size_t valueSequenceSize(const LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
   template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
      static void serialize(const LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&, INDEX_OUTPUT_ITERATOR, VALUE_OUTPUT_ITERATOR);
   template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
      static void deserialize( INDEX_INPUT_ITERATOR, VALUE_INPUT_ITERATOR, LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>&);
};
/// \endcond

/***********************
 * class documentation *
 ***********************/
/*! \file linear_constraint_function.hxx
 *  \brief Provides implementation of a liner constraint function.
 */

/*! \class LinearConstraintFunction
 *  \brief Default implementation of a linear constraint function class.
 *
 *  This class implements a generic linear constraint function. May be slow for
 *  function evaluations etc. but can be used to describe all kinds of linear
 *  constraints.
 *
 *  \tparam VALUE_TYPE The value type used by the linear constraint function.
 *  \tparam INDEX_TYPE The index type used by the linear constraint function.
 *  \tparam LABEL_TYPE The label type used by the linear constraint function.
 *
 *  \ingroup functions
 */

/*! \typedef LinearConstraintFunction::LinearConstraintFunctionType
 *  \brief Typedef of the LinearConstraintFunction class with appropriate
 *         template parameter.
 */

/*! \typedef LinearConstraintFunction::LinearConstraintFunctionBaseType
 *  \brief Typedef of the LinearConstraintFunctionBase class with appropriate
 *         template parameter.
 */

/*! \typedef LinearConstraintFunction::LinearConstraintFunctionTraitsType
 *  \brief Typedef of the LinearConstraintFunctionTraits class with appropriate
 *         template parameter.
 */

/*! \typedef LinearConstraintFunction::ValueType
 *  \brief Typedef of the VALUE_TYPE template parameter type from the class
 *         LinearConstraintFunction.
 */

/*! \typedef LinearConstraintFunction::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         LinearConstraintFunction.
 */

/*! \typedef LinearConstraintFunction::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         LinearConstraintFunction.
 */

/*! \typedef LinearConstraintFunction::LinearConstraintType
 *  \brief Typedef of the LinearConstraint class which is used to represent
 *         linear constraints.
 */

/*! \typedef LinearConstraintFunction::LinearConstraintsContainerType
 *  \brief Defines the linear constraints container type which is used to store
 *         multiple linear constraints.
 */

/*! \typedef LinearConstraintFunction::LinearConstraintsIteratorType
 *  \brief Defines the linear constraints container iterator type which is used
 *         to iterate over the set of linear constraints.
 */

/*! \typedef LinearConstraintFunction::IndicatorVariablesContainerType
 *  \brief Defines the indicator variables container type which is used to store
 *         the indicator variables used by the linear constraint function.
 */

/*! \typedef LinearConstraintFunction::IndicatorVariablesIteratorType
 *  \brief Defines the indicator variables container iterator type which is used
 *         to iterate over the indicator variables used by the linear constraint
 *         function.
 */

/*! \typedef LinearConstraintFunction::VariableLabelPairsIteratorType
 *  \brief Defines the variable label pairs iterator type which is used
 *         to iterate over the variable label pairs of an indicator variable.
 */

/*! \typedef LinearConstraintFunction::ViolatedLinearConstraintsIteratorType
 *  \brief Defines the violated linear constraints iterator type which is used
 *         to iterate over the set of violated linear constraints.
 */

/*! \typedef LinearConstraintFunction::ViolatedLinearConstraintsWeightsContainerType
 *  \brief Defines the violated linear constraints weights container type which
 *         is used to store the weights of the violated linear constraints.
 */

/*! \typedef LinearConstraintFunction::ViolatedLinearConstraintsWeightsIteratorType
 *  \brief Defines the violated linear constraints weights iterator type which
 *         is used to iterate over the weights of the violated linear
 *         constraints.
 */

/*! \fn LinearConstraintFunction::LinearConstraintFunction()
 *  \brief LinearConstraintFunction constructor.
 *
 *  This constructor will create an empty LinearConstraintFunction.
 */

/*! \fn LinearConstraintFunction::LinearConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LinearConstraintsContainerType& constraints, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0)
 *  \brief LinearConstraintFunction constructor.
 *
 *  This constructor will create a LinearConstraintFunction.
 *
 *  \tparam SHAPE_ITERATOR_TYPE Iterator to iterate over the shape of the function
 *
 *  \param[in] shapeBegin Iterator pointing to the begin of the sequence where
 *                        the shape of the function is stored.
 *  \param[in] shapeEnd Iterator pointing to the end of the sequence where the
 *                      shape of the function is stored.
 *  \param[in] constraints The container where the linear constraints for the
 *                         linear constraint function are stored.
 *  \param[in] returnValid The value which will be returned by the function
 *                         evaluation if no constraint is violated.
 *  \param[in] returnInvalid The value which will be returned by the function
 *                           evaluation if at least one constraint is violated.
 */

/*! \fn LinearConstraintFunction::LinearConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, CONSTRAINTS_ITERATOR_TYPE constraintsBegin, CONSTRAINTS_ITERATOR_TYPE constraintsEnd, const ValueType returnValid = 0.0, const ValueType returnInvalid = 1.0)
 *  \brief LinearConstraintFunction constructor.
 *
 *  This constructor will create a LinearConstraintFunction.
 *
 *  \tparam SHAPE_ITERATOR_TYPE Iterator to iterate over the shape of the function.
 *  \tparam CONSTRAINTS_ITERATOR_TYPE Iterator to iterate over the linear constraints
 *                               which shall be added to the function.
 *
 *  \param[in] shapeBegin Iterator pointing to the begin of the sequence where
 *                        the shape of the function is stored.
 *  \param[in] shapeEnd Iterator pointing to the end of the sequence where the
 *                      shape of the function is stored.
 *  \param[in] constraintsBegin Iterator pointing to the begin of the sequence
 *                              where the linear constraints for the linear
 *                              constraint function are stored.
 *  \param[in] constraintsEnd Iterator pointing to the end of the sequence where
 *                            the linear constraints for the linear constraint
 *                            function are stored.
 *  \param[in] returnValid The value which will be returned by the function
 *                         evaluation if no constraint is violated.
 *  \param[in] returnInvalid The value which will be returned by the function
 *                           evaluation if at least one constraint is violated.
 */

/*! \fn LinearConstraintFunction::~LinearConstraintFunction()
 *  \brief LinearConstraintFunction destructor.
 */

/*! \fn LinearConstraintFunction::ValueType LinearConstraintFunction::operator()(STATES_ITERATOR_TYPE statesBegin) const
 *  \brief Function evaluation.
 *
 *  \param[in] statesBegin Iterator pointing to the begin of a sequence of
 *                         labels for the variables of the function.
 *
 *  \return LinearConstraintFunction::returnValid_ if no constraint is violated
 *          by the labeling. LinearConstraintFunction::returnInvalid_ if at
 *          least one constraint is violated by the labeling.
 */

/*! \fn size_t LinearConstraintFunction::shape(const size_t i) const
 *  \brief Number of labels of the indicated input variable.
 *
 *  \param[in] i Index of the variable.
 *
 *  \return Returns the number of labels of the i-th variable.
 */

/*! \fn size_t LinearConstraintFunction::dimension() const
 *  \brief Number of input variables.
 *
 *  \return Returns the number of variables.
 */

/*! \fn size_t LinearConstraintFunction::size() const
 *  \brief Number of parameters.
 *
 *  \return Returns the number of parameters.
 */

/*! \fn LinearConstraintFunction::ValueType LinearConstraintFunction::min() const
 *  \brief Minimum value of the linear constraint function.
 *
 *  \return Returns the minimum value of LinearConstraintFunction::returnValid_
 *          and LinearConstraintFunction::returnInvalid_.
 */

/*! \fn LinearConstraintFunction::ValueType LinearConstraintFunction::max() const
 *  \brief Maximum value of the linear constraint function.
 *
 *  \return Returns the maximum value of LinearConstraintFunction::returnValid_
 *          and LinearConstraintFunction::returnInvalid_.
 */

/*! \fn MinMaxFunctor LinearConstraintFunction::minMax() const
 *  \brief Get minimum and maximum at the same time.
 *
 *  \return Returns a functor containing the minimum and the maximum value of
 *          the linear constraint function.
 */

/*! \var LinearConstraintFunction::shape_
 *  \brief Stores the shape of the linear constraint function.
 */

/*! \var LinearConstraintFunction::size_
 *  \brief Stores the size of the linear constraint function.
 */

/*! \var LinearConstraintFunction::constraints_
 *   \brief Stores the constraints of the linear constraint function.
 */

/*! \var LinearConstraintFunction::violatedConstraintsIds_
 *  \brief Stores the indices of the violated constraints which are detected by
 *         LinearConstraintFunction::challenge and
 *         LinearConstraintFunction::challengeRelaxed.
 */

/*! \var LinearConstraintFunction::violatedConstraintsWeights_
 *  \brief Stores the weights of the violated constraints which are detected by
 *         LinearConstraintFunction::challenge and
 *         LinearConstraintFunction::challengeRelaxed.
 */

/*! \var LinearConstraintFunction::returnValid_
 *  \brief Stores the return value of LinearConstraintFunction::operator() if no
 *         constraint is violated.
 */

/*! \var LinearConstraintFunction::returnInvalid_
 *  \brief Stores the return value of LinearConstraintFunction::operator() if at
 *         least one constraint is violated.
 */

/*! \var LinearConstraintFunction::indicatorVariableList_
 *  \brief A list of all indicator variables present in the linear constraint
 *         function.
 */

/*! \var LinearConstraintFunction::indicatorVariableLookupTable_
 *  \brief Lookup table for fast access of the indicator variable IDs.
 */

/*! \fn LinearConstraintFunction::LinearConstraintsIteratorType LinearConstraintFunction::linearConstraintsBegin_impl() const
 *  \brief Implementation of LinearConstraintFunctionBase::linearConstraintsBegin.
 */

/*! \fn LinearConstraintFunction::LinearConstraintsIteratorType LinearConstraintFunction::linearConstraintsEnd_impl() const
 *  \brief Implementation of LinearConstraintFunctionBase::linearConstraintsEnd.
 */

/*! \fn LinearConstraintFunction::IndicatorVariablesIteratorType LinearConstraintFunction::indicatorVariablesOrderBegin_impl() const
 *   \brief Implementation of LinearConstraintFunctionBase::indicatorVariableOrderBegin.
 */

/*! \fn LinearConstraintFunction::IndicatorVariablesIteratorType LinearConstraintFunction::indicatorVariablesOrderEnd_impl() const
 *  \brief Implementation of LinearConstraintFunctionBase::indicatorVariableOrderEnd.
 */

/*! \fn void LinearConstraintFunction::challenge_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const
 *  \brief Implementation of LinearConstraintFunctionBase::challenge.
 */

/*! \fn void LinearConstraintFunction::challengeRelaxed_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const
 *  \brief Implementation of LinearConstraintFunctionBase::challengeRelaxed.
 */

/*! \fn bool LinearConstraintFunction::checkConstraints() const
 *  \brief Check linear constraints. Only used for assertion in debug mode.
 *
 *  \return Returns true if all linear constraints do not violate the number of
 *          variables and states. False otherwise.
 */

/*! \fn bool LinearConstraintFunction::fillIndicatorVariableList()
 *  \brief Helper function to fill
 *         LinearConstraintFunction::indicatorVariableList_ with all indicator
 *         variables used by the linear constraint function.
 */

/*! \fn bool LinearConstraintFunction::createIndicatorVariableLookupTable()
 *  \brief Helper function to create an indicator variable lookup table. The
 *         table is stored in
 *         LinearConstraintFunction::indicatorVariableLookupTable_.
 */

/******************
 * implementation *
 ******************/
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintFunction() : shape_(),
   size_(), constraints_(), violatedConstraintsIds_(),
   violatedConstraintsWeights_(), returnValid_(), returnInvalid_(),
   indicatorVariableList_(), indicatorVariableLookupTable_() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class SHAPE_ITERATOR_TYPE>
inline LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, const LinearConstraintsContainerType& constraints, const ValueType returnValid, const ValueType returnInvalid)
   : shape_(shapeBegin, shapeEnd),
     size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<SHAPE_ITERATOR_TYPE>::value_type>())),
     constraints_(constraints), violatedConstraintsIds_(constraints_.size()),
     violatedConstraintsWeights_(constraints_.size()),
     returnValid_(returnValid), returnInvalid_(returnInvalid),
     indicatorVariableList_(), indicatorVariableLookupTable_() {
   OPENGM_ASSERT(this->checkConstraints());

   // fill indicatorVariableList_
   fillIndicatorVariableList();

   // create indicatorVariableLookupTable_
   createIndicatorVariableLookupTable();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class SHAPE_ITERATOR_TYPE, class CONSTRAINTS_ITERATOR_TYPE>
inline LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintFunction(SHAPE_ITERATOR_TYPE shapeBegin, SHAPE_ITERATOR_TYPE shapeEnd, CONSTRAINTS_ITERATOR_TYPE constraintsBegin, CONSTRAINTS_ITERATOR_TYPE constraintsEnd, const ValueType returnValid, const ValueType returnInvalid)
   : shape_(shapeBegin, shapeEnd),
     size_(std::accumulate(shapeBegin, shapeEnd, 1, std::multiplies<typename std::iterator_traits<SHAPE_ITERATOR_TYPE>::value_type>())),
     constraints_(constraintsBegin, constraintsEnd),
     violatedConstraintsIds_(constraints_.size()),
     violatedConstraintsWeights_(constraints_.size()),
     returnValid_(returnValid), returnInvalid_(returnInvalid),
     indicatorVariableList_(), indicatorVariableLookupTable_() {
   OPENGM_ASSERT(this->checkConstraints());

   // fill indicatorVariableList_
   fillIndicatorVariableList();

   // create indicatorVariableLookupTable_
   createIndicatorVariableLookupTable();
}
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::~LinearConstraintFunction() {

}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class STATES_ITERATOR_TYPE>
inline typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::operator()(STATES_ITERATOR_TYPE statesBegin) const {
   for(LinearConstraintsIteratorType constraintsIter = constraints_.begin(); constraintsIter != constraints_.end(); ++constraintsIter) {
      // compare result against bound with floating point tolerance
      const ValueType weight = constraintsIter->operator()(statesBegin);
      if(weight > OPENGM_FLOAT_TOL) {
         return returnInvalid_;
      }
   }
   return returnValid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::shape(const size_t i) const {
   OPENGM_ASSERT(i < shape_.size());
   return shape_[i];
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::dimension() const {
   return shape_.size();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::size() const {
   return size_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::min() const {
   return returnValid_ < returnInvalid_ ? returnValid_ : returnInvalid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::max() const {
   return returnValid_ > returnInvalid_ ? returnValid_ : returnInvalid_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline MinMaxFunctor<typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::ValueType> LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::minMax() const {
   if(returnValid_ < returnInvalid_) {
      return MinMaxFunctor<ValueType>(returnValid_, returnInvalid_);
   }
   else {
      return MinMaxFunctor<ValueType>(returnInvalid_, returnValid_);
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::linearConstraintsBegin_impl() const {
   return constraints_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::linearConstraintsEnd_impl() const {
   return constraints_.end();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::indicatorVariablesOrderBegin_impl() const {
   return indicatorVariableList_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::indicatorVariablesOrderEnd_impl() const {
   return indicatorVariableList_.end();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class LABEL_ITERATOR>
inline void LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::challenge_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance) const {
   size_t numViolatedConstraints = 0;
   for(LinearConstraintsIteratorType constraintsIter = constraints_.begin(); constraintsIter != constraints_.end(); ++constraintsIter) {
      // compare result against bound with floating point tolerance
      const double weight = static_cast<double>(constraintsIter->operator()(labelingBegin));
      if(weight > tolerance) {
         violatedConstraintsIds_[numViolatedConstraints] = std::distance(constraints_.begin(), constraintsIter);
         violatedConstraintsWeights_[numViolatedConstraints] = weight;
         ++numViolatedConstraints;
      }
   }
   violatedConstraintsBegin = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
   violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), numViolatedConstraints);
   violatedConstraintsWeightsBegin = violatedConstraintsWeights_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template <class LABEL_ITERATOR>
inline void LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::challengeRelaxed_impl(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance) const {
   size_t numViolatedConstraints = 0;
   std::vector<std::vector<size_t> >::const_iterator indicatorVariableLookupTableIter = indicatorVariableLookupTable_.begin();
   for(LinearConstraintsIteratorType constraintsIter = constraints_.begin(); constraintsIter != constraints_.end(); ++constraintsIter) {
      double result = 0.0;
      typename LinearConstraintType::CoefficientsIteratorType coefficientsIter = constraintsIter->coefficientsBegin();
      for(std::vector<size_t>::const_iterator variableIDsIter = indicatorVariableLookupTableIter->begin(); variableIDsIter != indicatorVariableLookupTableIter->end(); ++variableIDsIter) {
         result += labelingBegin[*variableIDsIter] * (*coefficientsIter);
         ++coefficientsIter;
      }

      // compare result against bound with floating point tolerance
      const double weight = result - constraintsIter->getBound();
      switch(constraintsIter->getConstraintOperator()) {
         case LinearConstraintType::LinearConstraintOperatorType::LessEqual : {
            if(weight > tolerance) {
               violatedConstraintsIds_[numViolatedConstraints] = std::distance(constraints_.begin(), constraintsIter);
               violatedConstraintsWeights_[numViolatedConstraints] = weight;
               ++numViolatedConstraints;
            }
            break;
         }
         case LinearConstraintType::LinearConstraintOperatorType::Equal : {
            if(weight > tolerance) {
               violatedConstraintsIds_[numViolatedConstraints] = std::distance(constraints_.begin(), constraintsIter);
               violatedConstraintsWeights_[numViolatedConstraints] = weight;
               ++numViolatedConstraints;
            } else if(weight < -tolerance) {
               violatedConstraintsIds_[numViolatedConstraints] = std::distance(constraints_.begin(), constraintsIter);
               violatedConstraintsWeights_[numViolatedConstraints] = -weight;
               ++numViolatedConstraints;
            }
            break;
         }
         /*case LinearConstraintType::LinearConstraintOperatorType::GreaterEqual : {
            if(weight < -tolerance) {
               violatedConstraintsIds_[numViolatedConstraints] = std::distance(constraints_.begin(), constraintsIter);
               violatedConstraintsWeights_[numViolatedConstraints] = -weight;
               ++numViolatedConstraints;
            }
            break;
         } */
         default : { // default corresponds to GreaterEqual case
            if(weight < -tolerance) {
               violatedConstraintsIds_[numViolatedConstraints] = std::distance(constraints_.begin(), constraintsIter);
               violatedConstraintsWeights_[numViolatedConstraints] = -weight;
               ++numViolatedConstraints;
            }
         }
      }
      ++indicatorVariableLookupTableIter;
   }
   violatedConstraintsBegin = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), 0);
   violatedConstraintsEnd = ViolatedLinearConstraintsIteratorType(constraints_.begin(), violatedConstraintsIds_.begin(), numViolatedConstraints);
   violatedConstraintsWeightsBegin = violatedConstraintsWeights_.begin();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline bool LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::checkConstraints() const {
   for(LinearConstraintsIteratorType constraintsIter = constraints_.begin(); constraintsIter != constraints_.end(); ++constraintsIter) {
      if(std::distance(constraintsIter->indicatorVariablesBegin(), constraintsIter->indicatorVariablesEnd()) == 0) {
         // no empty constraint allowed
         std::cout << "empty constraint" << std::endl;
         return false;
      }
      for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
         if(std::distance(variablesIter->begin(), variablesIter->end()) == 0) {
            // no empty indicator variable allowed
            std::cout << "empty indicator variable" << std::endl;
            return false;
         }
         for(VariableLabelPairsIteratorType indicatorVariablesIter = variablesIter->begin(); indicatorVariablesIter != variablesIter->end(); ++indicatorVariablesIter) {
            if(indicatorVariablesIter->first >= shape_.size()) {
               return false;
            } else if(indicatorVariablesIter->second >= shape_[indicatorVariablesIter->first]) {
               return false;
            }
         }
      }
   }
   return true;
}


template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::fillIndicatorVariableList() {
   // fill indicatorVariableList_
   for(LinearConstraintsIteratorType constraintsIter = constraints_.begin(); constraintsIter != constraints_.end(); ++constraintsIter) {
      for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
         if(std::find(indicatorVariableList_.begin(), indicatorVariableList_.end(), *variablesIter) == indicatorVariableList_.end()) {
            indicatorVariableList_.push_back(*variablesIter) ;
         }
      }
   }

   // sort indicatorVariableList_
   std::sort(indicatorVariableList_.begin(), indicatorVariableList_.end());
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::createIndicatorVariableLookupTable() {
   // create indicatorVariableLookupTable_
   for(LinearConstraintsIteratorType constraintsIter = constraints_.begin(); constraintsIter != constraints_.end(); ++constraintsIter) {
      std::vector<size_t> currentConstraintLookup;
      for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
         OPENGM_ASSERT(std::find(indicatorVariableList_.begin(), indicatorVariableList_.end(), *variablesIter) != indicatorVariableList_.end());
         currentConstraintLookup.push_back(std::distance(indicatorVariableList_.begin(), std::find(indicatorVariableList_.begin(), indicatorVariableList_.end(), *variablesIter)));
      }
      indicatorVariableLookupTable_.push_back(currentConstraintLookup);
   }
}

/// \cond HIDDEN_SYMBOLS
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::indexSequenceSize(const LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t dimensionSize = 1;
   const size_t shapeSize = src.dimension();
   const size_t numConstraintsSize = 1;
   const size_t operatorTypeSize = src.constraints_.size();
   const size_t numIndicatorVariablesPerConstraintSize = src.constraints_.size();

   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType LinearConstraintsIteratorType;

   size_t numVariablesPerIndicatorVariablePerConstraintSize = 0;
   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      numVariablesPerIndicatorVariablePerConstraintSize += std::distance(constraintsIter->indicatorVariablesBegin(), constraintsIter->indicatorVariablesEnd());
   }
   numVariablesPerIndicatorVariablePerConstraintSize *= 2;

   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType IndicatorVariablesIteratorType;

   size_t variableStatePairSize = 0;
   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
         variableStatePairSize += std::distance(variablesIter->begin(), variablesIter->end());
      }
   }
   variableStatePairSize *= 2;

   const size_t totalIndexSize = dimensionSize + shapeSize + operatorTypeSize + numConstraintsSize + numIndicatorVariablesPerConstraintSize + numVariablesPerIndicatorVariablePerConstraintSize + variableStatePairSize;
   return totalIndexSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline size_t FunctionSerialization<LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::valueSequenceSize(const LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src) {
   const size_t boundsSize = src.constraints_.size();
   const size_t validSize = 1;
   const size_t invalidSize = 1;

   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType LinearConstraintsIteratorType;

   size_t coefficientsSize = 0;
   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      coefficientsSize += std::distance(constraintsIter->coefficientsBegin(), constraintsIter->coefficientsEnd());
   }

   const size_t totalValueSize = boundsSize + validSize + invalidSize + coefficientsSize;
   return totalValueSize;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_OUTPUT_ITERATOR, class VALUE_OUTPUT_ITERATOR>
inline void FunctionSerialization<LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::serialize(const LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& src, INDEX_OUTPUT_ITERATOR indexOutIterator, VALUE_OUTPUT_ITERATOR valueOutIterator) {
   // index output
   // dimension
   *indexOutIterator = src.dimension();
   ++indexOutIterator;

   // shape
   for(size_t i = 0; i < src.dimension(); i++) {
      *indexOutIterator = src.shape_[i];
      ++indexOutIterator;
   }

   // number of constraints
   *indexOutIterator = src.constraints_.size();
   ++indexOutIterator;

   // operator type
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsIteratorType LinearConstraintsIteratorType;

   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      *indexOutIterator = constraintsIter->getConstraintOperator();
      ++indexOutIterator;
   }

   // number of indicator variables per constraint
   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      *indexOutIterator = std::distance(constraintsIter->indicatorVariablesBegin(), constraintsIter->indicatorVariablesEnd());
      ++indexOutIterator;
   }

   // number of variables per indicator variable per constraint
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndicatorVariablesIteratorType IndicatorVariablesIteratorType;

   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
         *indexOutIterator = static_cast<size_t>(variablesIter->getLogicalOperatorType());
         ++indexOutIterator;
         *indexOutIterator = std::distance(variablesIter->begin(), variablesIter->end());
         ++indexOutIterator;
      }
   }

   // variable state pairs
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::VariableLabelPairsIteratorType VariableLabelPairsIteratorType;

   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      for(IndicatorVariablesIteratorType variablesIter = constraintsIter->indicatorVariablesBegin(); variablesIter != constraintsIter->indicatorVariablesEnd(); ++variablesIter) {
         for(VariableLabelPairsIteratorType indicatorVariablesIter = variablesIter->begin(); indicatorVariablesIter != variablesIter->end(); ++indicatorVariablesIter) {
            *indexOutIterator = indicatorVariablesIter->first;
            ++indexOutIterator;
            *indexOutIterator = indicatorVariablesIter->second;
            ++indexOutIterator;
         }
      }
   }

   // value output
   // bound
   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      *valueOutIterator = constraintsIter->getBound();
      ++valueOutIterator;
   }

   // valid value
   *valueOutIterator = src.returnValid_;
   ++valueOutIterator;

   // invalid value
   *valueOutIterator = src.returnInvalid_;
   ++valueOutIterator;

   // coefficients
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintType::CoefficientsIteratorType CoefficientsIteratorType;
   for(LinearConstraintsIteratorType constraintsIter = src.constraints_.begin(); constraintsIter != src.constraints_.end(); ++constraintsIter) {
      for(CoefficientsIteratorType coefficientsIter = constraintsIter->coefficientsBegin(); coefficientsIter != constraintsIter->coefficientsEnd(); ++coefficientsIter) {
         *valueOutIterator = *coefficientsIter;
         ++valueOutIterator;
      }
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class INDEX_INPUT_ITERATOR, class VALUE_INPUT_ITERATOR>
inline void FunctionSerialization<LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> >::deserialize( INDEX_INPUT_ITERATOR indexInIterator, VALUE_INPUT_ITERATOR valueInIterator, LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>& dst) {
   // index input
   // dimension
   const size_t dimension = *indexInIterator;
   ++indexInIterator;
   // shape
   INDEX_INPUT_ITERATOR shapeBegin = indexInIterator;
   INDEX_INPUT_ITERATOR shapeEnd = indexInIterator + dimension;
   indexInIterator += dimension;

   // constraints
   typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsContainerType constraints(*indexInIterator);
   ++indexInIterator;

   // operator type
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintsContainerType::iterator LinearConstraintsIteratorType;

   for(LinearConstraintsIteratorType constraintsIter = constraints.begin(); constraintsIter != constraints.end(); ++constraintsIter) {
      constraintsIter->setConstraintOperator(static_cast<typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintType::LinearConstraintOperatorValueType>(*indexInIterator));
      ++indexInIterator;
   }

   // number of indicator variables per constraint
   std::vector<INDEX_TYPE> numIndicatorariablesPerConstraint;
   for(LinearConstraintsIteratorType constraintsIter = constraints.begin(); constraintsIter != constraints.end(); ++constraintsIter) {
      numIndicatorariablesPerConstraint.push_back(*indexInIterator);
      ++indexInIterator;
   }

   // number of variables per indicator variable per constraint
   std::vector<std::vector<INDEX_TYPE> > numVariablesPerIndicatorVariablePerConstraint;
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintType::IndicatorVariableType::LogicalOperatorType LogicalOperatorType;
   std::vector<std::vector<LogicalOperatorType> > logicalOperatorPerIndicatorVariablePerConstraint;
   for(size_t i = 0; i < numIndicatorariablesPerConstraint.size(); ++i) {
      numVariablesPerIndicatorVariablePerConstraint.push_back(std::vector<INDEX_TYPE>());
      logicalOperatorPerIndicatorVariablePerConstraint.push_back(std::vector<LogicalOperatorType>());
      for(size_t j = 0; j < numIndicatorariablesPerConstraint[i]; ++j) {
         logicalOperatorPerIndicatorVariablePerConstraint[i].push_back(static_cast<LogicalOperatorType>(*indexInIterator));
         ++indexInIterator;
         numVariablesPerIndicatorVariablePerConstraint[i].push_back(*indexInIterator);
         ++indexInIterator;
      }
   }

   // variable state pairs
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintType::IndicatorVariablesContainerType IndicatorVariablesContainerType;
   std::vector<IndicatorVariablesContainerType> variableStatePairs;

   for(size_t i = 0; i < numVariablesPerIndicatorVariablePerConstraint.size(); ++i) {
      variableStatePairs.push_back(IndicatorVariablesContainerType());
      for(size_t j = 0; j < numVariablesPerIndicatorVariablePerConstraint[i].size(); ++j) {
         variableStatePairs[i].push_back(typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintType::IndicatorVariableType());
         for(size_t k = 0; k < numVariablesPerIndicatorVariablePerConstraint[i][j]; ++k) {
            const INDEX_TYPE variable = *indexInIterator;
            ++indexInIterator;
            const LABEL_TYPE label = *indexInIterator;
            ++indexInIterator;
            variableStatePairs[i][j].add(variable, label);
         }

         variableStatePairs[i][j].setLogicalOperatorType(logicalOperatorPerIndicatorVariablePerConstraint[i][j]);
      }
   }

   // value input
   // bound
   for(LinearConstraintsIteratorType constraintsIter = constraints.begin(); constraintsIter != constraints.end(); ++constraintsIter) {
      constraintsIter->setBound(*valueInIterator);
      ++valueInIterator;
   }

   // valid value
   const VALUE_TYPE returnValid = *valueInIterator;
   ++valueInIterator;

   // invalid value
   const VALUE_TYPE returnInvalid = *valueInIterator;
   ++valueInIterator;

   // coefficients
   typedef typename LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::LinearConstraintType::CoefficientsContainerType CoefficientsContainerType;
   std::vector<CoefficientsContainerType> coefficients;
   for(size_t i = 0; i < numIndicatorariablesPerConstraint.size(); ++i) {
      coefficients.push_back(CoefficientsContainerType());
      for(size_t j = 0; j < numIndicatorariablesPerConstraint[i]; ++j) {
         coefficients[i].push_back(*valueInIterator);
         ++valueInIterator;
      }
   }

   // add variables and coefficients to constraints
   for(size_t i = 0; i < constraints.size(); ++i) {
      constraints[i].add(variableStatePairs[i].begin(), variableStatePairs[i].end(), coefficients[i].begin());
   }

   dst = LinearConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>(shapeBegin, shapeEnd, constraints, returnValid, returnInvalid);
}
/// \endcond

} // namespace opengm

#endif /* OPENGM_LINEAR_CONSTRAINT_FUNCTION_HXX_ */
