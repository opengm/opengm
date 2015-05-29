#ifndef OPENGM_LINEAR_CONSTRAINT_FUNCTION_BASE_HXX_
#define OPENGM_LINEAR_CONSTRAINT_FUNCTION_BASE_HXX_

#include <opengm/functions/function_properties_base.hxx>

namespace opengm {

/*********************
 * class definition *
 *********************/
template <typename LINEAR_CONSTRAINT_FUNCTION_TYPE>
struct LinearConstraintFunctionTraits;

template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
class LinearConstraintFunctionBase :  public FunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE, typename LinearConstraintFunctionTraits<LINEAR_CONSTRAINT_FUNCTION_TYPE>::ValueType, typename LinearConstraintFunctionTraits<LINEAR_CONSTRAINT_FUNCTION_TYPE>::IndexType, typename LinearConstraintFunctionTraits<LINEAR_CONSTRAINT_FUNCTION_TYPE>::LabelType> {
public:
   // typedefs
   typedef LINEAR_CONSTRAINT_FUNCTION_TYPE                                                           LinearConstraintFunctionType;
   typedef LinearConstraintFunctionTraits<LinearConstraintFunctionType>                              LinearConstraintFunctionTraitsType;
   typedef typename LinearConstraintFunctionTraitsType::ValueType                                    ValueType;
   typedef typename LinearConstraintFunctionTraitsType::IndexType                                    IndexType;
   typedef typename LinearConstraintFunctionTraitsType::LabelType                                    LabelType;
   typedef typename LinearConstraintFunctionTraitsType::LinearConstraintType                         LinearConstraintType;
   typedef typename LinearConstraintFunctionTraitsType::LinearConstraintsIteratorType                LinearConstraintsIteratorType;
   typedef typename LinearConstraintFunctionTraitsType::ViolatedLinearConstraintsIteratorType        ViolatedLinearConstraintsIteratorType;
   typedef typename LinearConstraintFunctionTraitsType::ViolatedLinearConstraintsWeightsIteratorType ViolatedLinearConstraintsWeightsIteratorType;
   typedef typename LinearConstraintFunctionTraitsType::IndicatorVariablesIteratorType               IndicatorVariablesIteratorType;

   // const access
   LinearConstraintsIteratorType  linearConstraintsBegin() const;
   LinearConstraintsIteratorType  linearConstraintsEnd() const;
   IndicatorVariablesIteratorType indicatorVariablesOrderBegin() const;
   IndicatorVariablesIteratorType indicatorVariablesOrderEnd() const;

   template <class LABEL_ITERATOR>
   void challenge(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const;
   template <class LABEL_ITERATOR>
   void challengeRelaxed(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const;

   // specializations
   bool isLinearConstraint() const;
};

/***********************
 * class documentation *
 ***********************/
/*! \file linear_constraint_function_base.hxx
 *  \brief Provides interface for liner constraint functions.
 */

/*! \struct LinearConstraintFunctionTraits
 *  \brief Traits class for linear constraint functions.
 *
 *  Each linear constraint function has to provide a template specialization of
 *  this function to provide appropriate typedefs. The following types have to
 *  be defined:
 *  -# ValueType
 *  -# IndexType
 *  -# LabelType
 *  -# LinearConstraintType
 *  -# LinearConstraintsIteratorType
 *  -# ViolatedLinearConstraintsIteratorType
 *  -# ViolatedLinearConstraintsWeightsIteratorType
 *  -# IndicatorVariablesIteratorType
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The linear constraint function type.
 *
 */

/*! \class LinearConstraintFunctionBase
 *  \brief Base class for linear constraint functions.
 *
 *  This class defines a base class for all linear constraint functions. It uses
 *  the curiously recurring template pattern (CRTP) to provide static
 *  polymorphism. It defines the interface which can be used to access the
 *  linear constraints which are defined by a linear constraint function.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The child class which inherits from
 *                                          LinearConstraintFunctionBase and
 *                                          thus defines a linear constraint
 *                                          function.
 *
 *  \note A template specialization of class LinearConstraintFunctionTraits has
 *        to be defined for each class which inherits from
 *        LinearConstraintFunctionBase.
 */

/*! \typedef LinearConstraintFunctionBase::LinearConstraintFunctionType
 *   \brief Typedef of the LINEAR_CONSTRAINT_FUNCTION_TYPE template parameter
 *         from the class LinearConstraintFunctionBase.
 */

/*! \typedef LinearConstraintFunctionBase::LinearConstraintFunctionTraitsType
 *   \brief Typedef of the LinearConstraintFunctionTraits class with appropriate
 *          template parameter.
 */

/*! \typedef LinearConstraintFunctionBase::ValueType
 *  \brief Typedef of the value type used by the linear constraint function.
 */

/*! \typedef LinearConstraintFunctionBase::IndexType
 *  \brief Typedef of the index type used by the linear constraint function.
 */

/*! \typedef LinearConstraintFunctionBase::LabelType
 *  \brief Typedef of the label type used by the linear constraint function.
 */

/*! \typedef LinearConstraintFunctionBase::LinearConstraintType
 *   \brief Typedef of the linear constraint type used by the linear constraint
 *          function.
 */

/*! \typedef LinearConstraintFunctionBase::LinearConstraintsIteratorType
 *  \brief Typedef of the linear constraints iterator type used by the linear
 *         constraint function to iterate over the set of linear constraints.
 */

/*! \typedef LinearConstraintFunctionBase::ViolatedLinearConstraintsIteratorType
 *  \brief Typedef of the violated linear constraints iterator type used by the
 *         linear constraint function to iterate over the set of violated linear
 *         constraints.
 */

/*! \typedef LinearConstraintFunctionBase::ViolatedLinearConstraintsWeightsIteratorType
 *  \brief Typedef of the violated linear constraints weights iterator type used
 *         by the linear constraint function to iterate over the weights of the
 *         violated linear constraints.
 */

/*! \typedef LinearConstraintFunctionBase::IndicatorVariablesIteratorType
 *  \brief Typedef of the indicator variables iterator type used by the linear
 *         constraint function to iterate over the indicator variables.
 */

/*! \fn LinearConstraintFunctionBase::LinearConstraintsIteratorType LinearConstraintFunctionBase::linearConstraintsBegin() const
 *  \brief Get the begin iterator to the set of linear constraints represented
 *         by the linear constraint function.
 *
 *  \return The const iterator pointing to the begin of the set of linear
 *          constraints.
 */

/*! \fn LinearConstraintFunctionBase::LinearConstraintsIteratorType LinearConstraintFunctionBase::linearConstraintsEnd() const
 *  \brief Get the end iterator to the set of linear constraints represented by
 *         the linear constraint function.
 *
 *  \return The const iterator pointing to the end of the set of linear
 *          constraints.
 */

/*! \fn LinearConstraintFunctionBase::IndicatorVariablesIteratorType LinearConstraintFunctionBase::indicatorVariablesOrderBegin() const
 *  \brief Get the begin iterator to the set of indicator variables used by the
 *         linear constraint function.
 *
 *  \return The const iterator pointing to the begin of the set of indicator
 *          variables.
 */

/*! \fn LinearConstraintFunctionBase::IndicatorVariablesIteratorType LinearConstraintFunctionBase::indicatorVariablesOrderEnd() const
 *  \brief Get the end iterator to the set of indicator variables used by the
 *         linear constraint function.
 *
 *  \return The const iterator pointing to the end of the set of indicator
 *          variables.
 */

/*! \fn void LinearConstraintFunctionBase::challenge(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const
 *  \brief Challenge the linear constraint function and get all linear
 *         constraints which are violated by a given labeling.
 *
 *  This function returns all linear constraints which are violated by a given
 *  labeling, furthermore it returns the weights telling how much each
 *  constraint is violated. It uses only the labeling for the first order
 *  variables of the function to evaluate the indicator variables of the
 *  function. Hence it is not qualified to challenge the liner function against
 *  a relaxed labeling where each indicator variable can take values in the
 *  range [0.0, 1.0]. Use LinearConstraintFunctionBase::challengeRelaxed for
 *  this case.
 *
 *  \tparam LABEL_ITERATOR Iterator type to iterate over the labels for the
 *                         variables.
 *
 *  \param[out] violatedConstraintsBegin Iterator pointing to the begin of the
 *                                       set of violated constraints.
 *  \param[out] violatedConstraintsEnd Iterator pointing to the end of the set
 *                                     of violated constraints.
 *  \param[out] violatedConstraintsWeightsBegin Iterator pointing to the begin
 *                                              of the weights for the set of
 *                                              violated constraints.
 *  \param[in] labelingBegin Iterator pointing to the begin of the labeling for
 *                           the first order variables.
 *  \param[in] tolerance The tolerance value defines how much a constraint is
 *                       allowed to be violated without returning it as a
 *                       violated constraint.
 *
 *  \warning All iterators returned by this function are only guaranteed to be
 *           valid until the next call to the functions
 *           LinearConstraintFunctionBase::challenge or
 *           LinearConstraintFunctionBase::challengeRelaxed.
 */

/*! \fn void LinearConstraintFunctionBase::challengeRelaxed(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance = 0.0) const
 *  \brief Challenge the linear constraint function and get all linear
 *         constraints which are violated by a given labeling.
 *
 *  This function returns all linear constraints which are violated by a given
 *  labeling, furthermore it returns the weights telling how much each
 *  constraint is violated. Unlike the LinearConstraintFunctionBase::challenge
 *  function it takes the labeling for all indicator variables into account and
 *  therefore is capable of dealing with a relaxed labeling where each indicator
 *  variable can take values in the range [0.0, 1.0]. The order of the relaxed
 *  labeling for the indicator variables has to follow the order which is given
 *  by the iterators returned from indicatorVariablesOrderBegin() and
 *  indicatorVariablesOrderEnd().
 *
 *  \tparam LABEL_ITERATOR Iterator type to iterate over the relaxed labeling
 *                         for the indicator variables.
 *
 *  \param[out] violatedConstraintsBegin Iterator pointing to the begin of the
 *                                       set of violated constraints.
 *  \param[out] violatedConstraintsEnd Iterator pointing to the end of the set
 *                                     of violated constraints.
 *  \param[out] violatedConstraintsWeightsBegin Iterator pointing to the begin
 *                                              of the weights for the set of
 *                                              violated constraints.
 *  \param[in] labelingBegin Iterator pointing to the begin of the relaxed
 *                           labeling for each indicator variable.
 *  \param[in] tolerance The tolerance value defines how much a constraint is
 *                       allowed to be violated without returning it as a
 *                       violated constraint.
 *
 *  \warning All iterators returned by this function are only guaranteed to be
 *           valid until the next call to the functions
 *           LinearConstraintFunctionBase::challenge or
 *           LinearConstraintFunctionBase::challengeRelaxed.
 */

/*! \fn bool LinearConstraintFunctionBase::isLinearConstraint() const
 *  \brief Function specialization for each linear constraint function.
 *
 *  \return Returns always true as every function which inherits from
 *          LinearConstraintFunctionBase is a linear constraint function.
 */

/******************
 * implementation *
 ******************/
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline typename LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::LinearConstraintsIteratorType LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::linearConstraintsBegin() const {
   return static_cast<const LinearConstraintFunctionType*>(this)->linearConstraintsBegin_impl();
}

template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline typename LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::LinearConstraintsIteratorType LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::linearConstraintsEnd() const {
   return static_cast<const LinearConstraintFunctionType*>(this)->linearConstraintsEnd_impl();
}

template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline typename LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::IndicatorVariablesIteratorType LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::indicatorVariablesOrderBegin() const {
   return static_cast<const LinearConstraintFunctionType*>(this)->indicatorVariablesOrderBegin_impl();
}

template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline typename LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::IndicatorVariablesIteratorType LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::indicatorVariablesOrderEnd() const {
   return static_cast<const LinearConstraintFunctionType*>(this)->indicatorVariablesOrderEnd_impl();
}

template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
template <class LABEL_ITERATOR>
inline void LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::challenge(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance) const {
   return static_cast<const LinearConstraintFunctionType*>(this)->challenge_impl(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, labelingBegin, tolerance);
}

template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
template <class LABEL_ITERATOR>
inline void LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::challengeRelaxed(ViolatedLinearConstraintsIteratorType& violatedConstraintsBegin, ViolatedLinearConstraintsIteratorType& violatedConstraintsEnd, ViolatedLinearConstraintsWeightsIteratorType& violatedConstraintsWeightsBegin, LABEL_ITERATOR labelingBegin, const ValueType tolerance) const {
   return static_cast<const LinearConstraintFunctionType*>(this)->challengeRelaxed_impl(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, labelingBegin, tolerance);
}

template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline bool LinearConstraintFunctionBase<LINEAR_CONSTRAINT_FUNCTION_TYPE>::isLinearConstraint() const {
   return true;
}

} // namespace opengm

#endif /* OPENGM_LINEAR_CONSTRAINT_FUNCTION_BASE_HXX_ */
