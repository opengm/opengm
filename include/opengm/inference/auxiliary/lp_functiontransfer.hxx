#ifndef OPENGM_LP_FUNCTIONTRANSFER_HXX_
#define OPENGM_LP_FUNCTIONTRANSFER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/datastructures/linear_constraint.hxx>

#include <opengm/functions/soft_constraint_functions/sum_constraint_function.hxx>
#include <opengm/functions/soft_constraint_functions/label_cost_function.hxx>
namespace opengm {

/*********************
 * class definition *
 *********************/
template<class VALUE_TYPE, class INDEX_TYPE = size_t, class LABEL_TYPE = size_t>
class LPFunctionTransfer {
public:
   // typedefs
   typedef VALUE_TYPE                                           ValueType;
   typedef INDEX_TYPE                                           IndexType;
   typedef LABEL_TYPE                                           LabelType;
   typedef LinearConstraint<ValueType, IndexType, LabelType>    LinearConstraintType;
   typedef std::vector<LinearConstraintType>                    LinearConstraintsContainerType;
   typedef typename LinearConstraintType::IndicatorVariableType IndicatorVariableType;
   typedef std::vector<IndicatorVariableType>                   IndicatorVariablesContainerType;
   typedef std::vector<ValueType>                               SlackVariablesObjectiveCoefficientsContainerType;

   // transfer interface
   template<class FUNCTION_TYPE>
   static bool isTransferable();
   template<class FUNCTION_TYPE>
   static IndexType numSlackVariables(const FUNCTION_TYPE& function);
   template<class FUNCTION_TYPE>
   static void getSlackVariablesOrder(const FUNCTION_TYPE& function, IndicatorVariablesContainerType& order);
   template<class FUNCTION_TYPE>
   static void getSlackVariablesObjectiveCoefficients(const FUNCTION_TYPE& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients);
   template<class FUNCTION_TYPE>
   static void getIndicatorVariables(const FUNCTION_TYPE& function, IndicatorVariablesContainerType& variables);
   template<class FUNCTION_TYPE>
   static void getLinearConstraints(const FUNCTION_TYPE& function, LinearConstraintsContainerType& constraints);

   // functors
   struct IsTransferableFunctor {
      bool isTransferable_;
      template<class FUNCTION_TYPE>
      void operator()(const FUNCTION_TYPE& function);
   };
   struct NumSlackVariablesFunctor {
      IndexType numSlackVariables_;
      template<class FUNCTION_TYPE>
      void operator()(const FUNCTION_TYPE& function);
   };
   struct GetSlackVariablesOrderFunctor {
      IndicatorVariablesContainerType* order_;
      template<class FUNCTION_TYPE>
      void operator()(const FUNCTION_TYPE& function);
   };
   struct GetSlackVariablesObjectiveCoefficientsFunctor {
      SlackVariablesObjectiveCoefficientsContainerType* coefficients_;
      template<class FUNCTION_TYPE>
      void operator()(const FUNCTION_TYPE& function);
   };
   struct GetIndicatorVariablesFunctor {
      IndicatorVariablesContainerType* variables_;
      template<class FUNCTION_TYPE>
      void operator()(const FUNCTION_TYPE& function);
   };
   struct GetLinearConstraintsFunctor {
      LinearConstraintsContainerType* constraints_;
      template<class FUNCTION_TYPE>
      void operator()(const FUNCTION_TYPE& function);
   };
};

template<class FUNCTION_TYPE, class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class LPFunctionTransfer_impl {
public:
   // typedefs
   typedef VALUE_TYPE                                                                         ValueType;
   typedef INDEX_TYPE                                                                         IndexType;
   typedef LABEL_TYPE                                                                         LabelType;
   typedef FUNCTION_TYPE                                                                      FunctionType;
   typedef LPFunctionTransfer<ValueType, IndexType, LabelType>                                LPFunctionTransferType;
   typedef typename LPFunctionTransferType::LinearConstraintType                              LinearConstraintType;
   typedef typename LPFunctionTransferType::LinearConstraintsContainerType                    LinearConstraintsContainerType;
   typedef typename LPFunctionTransferType::IndicatorVariableType                             IndicatorVariableType;
   typedef typename LPFunctionTransferType::IndicatorVariablesContainerType                   IndicatorVariablesContainerType;
   typedef typename LPFunctionTransferType:: SlackVariablesObjectiveCoefficientsContainerType SlackVariablesObjectiveCoefficientsContainerType;

   // transfer interface
   static bool isTransferable();
   static IndexType numSlackVariables(const FunctionType& function);
   static void getSlackVariablesOrder(const FunctionType& function, IndicatorVariablesContainerType& order);
   static void getSlackVariablesObjectiveCoefficients(const FunctionType& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients);
   static void getIndicatorVariables(const FunctionType& function, IndicatorVariablesContainerType& variables);
   static void getLinearConstraints(const FunctionType& function, LinearConstraintsContainerType& constraints);
};

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> {
public:
   // typedefs
   typedef VALUE_TYPE                                                                         ValueType;
   typedef INDEX_TYPE                                                                         IndexType;
   typedef LABEL_TYPE                                                                         LabelType;
   typedef SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>                          FunctionType;
   typedef LPFunctionTransfer<ValueType, IndexType, LabelType>                                LPFunctionTransferType;
   typedef typename LPFunctionTransferType::LinearConstraintType                              LinearConstraintType;
   typedef typename LPFunctionTransferType::LinearConstraintsContainerType                    LinearConstraintsContainerType;
   typedef typename LPFunctionTransferType::IndicatorVariableType                             IndicatorVariableType;
   typedef typename LPFunctionTransferType::IndicatorVariablesContainerType                   IndicatorVariablesContainerType;
   typedef typename LPFunctionTransferType:: SlackVariablesObjectiveCoefficientsContainerType SlackVariablesObjectiveCoefficientsContainerType;

   // transfer interface
   static bool isTransferable();
   static IndexType numSlackVariables(const FunctionType& function);
   static void getSlackVariablesOrder(const FunctionType& function, IndicatorVariablesContainerType& order);
   static void getSlackVariablesObjectiveCoefficients(const FunctionType& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients);
   static void getIndicatorVariables(const FunctionType& function, IndicatorVariablesContainerType& variables);
   static void getLinearConstraints(const FunctionType& function, LinearConstraintsContainerType& constraints);
};

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
class LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE> {
public:
   // typedefs
   typedef VALUE_TYPE                                                                         ValueType;
   typedef INDEX_TYPE                                                                         IndexType;
   typedef LABEL_TYPE                                                                         LabelType;
   typedef LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>                              FunctionType;
   typedef LPFunctionTransfer<ValueType, IndexType, LabelType>                                LPFunctionTransferType;
   typedef typename LPFunctionTransferType::LinearConstraintType                              LinearConstraintType;
   typedef typename LPFunctionTransferType::LinearConstraintsContainerType                    LinearConstraintsContainerType;
   typedef typename LPFunctionTransferType::IndicatorVariableType                             IndicatorVariableType;
   typedef typename LPFunctionTransferType::IndicatorVariablesContainerType                   IndicatorVariablesContainerType;
   typedef typename LPFunctionTransferType:: SlackVariablesObjectiveCoefficientsContainerType SlackVariablesObjectiveCoefficientsContainerType;

   // transfer interface
   static bool isTransferable();
   static IndexType numSlackVariables(const FunctionType& function);
   static void getSlackVariablesOrder(const FunctionType& function, IndicatorVariablesContainerType& order);
   static void getSlackVariablesObjectiveCoefficients(const FunctionType& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients);
   static void getIndicatorVariables(const FunctionType& function, IndicatorVariablesContainerType& variables);
   static void getLinearConstraints(const FunctionType& function, LinearConstraintsContainerType& constraints);
};

/***********************
 * class documentation *
 ***********************/
/*! \class LPFunctionTransfer
 *  \brief Provides transformations for some function types when they are used
 *         in inference algorithms which use linear programming.
 *
 *  In inference algorithms like opengm::LPCplex which rely on opengm::LPBase
 *  the inference task is reformulated as a linear program. Therefore all
 *  factors which are not linear constraint factors are added the same way to
 *  the LP no matter what function type is used for these factors. However some
 *  function types like opengm::LabelCostFunction can be added in a more
 *  efficient formulation to the LP. This class provides an interface to
 *  transform the function behind a factor so opengm::LPBase can benefit from
 *  the more efficient representation of the factor.
 *
 *  \tparam VALUE_TYPE Value type.
 *  \tparam INDEX_TYPE Index type.
 *  \tparam LABEL_TYPE Label type.
 */

/*! \typedef LPFunctionTransfer::ValueType
 *  \brief Typedef of the VALUE_TYPE template parameter type from the class
 *         LPFunctionTransfer.
 */

/*! \typedef LPFunctionTransfer::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         LPFunctionTransfer.
 */

/*! \typedef LPFunctionTransfer::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         LPFunctionTransfer.
 */

/*! \typedef LPFunctionTransfer::LinearConstraintType
 *  \brief Typedef of the LinearConstraint class with appropriate template
 *         parameter.
 */

/*! \typedef LPFunctionTransfer::LinearConstraintsContainerType
 *  \brief Defines the linear constraints container type which is used to store
 *         multiple linear constraints.
 */

/*! \typedef LPFunctionTransfer::IndicatorVariableType
 *  \brief Defines the indicator variable type which is used within linear
 *         constraints.
 */

/*! \typedef LPFunctionTransfer::IndicatorVariablesContainerType
 *  \brief Defines the indicator variables container type which is used to store
 *         multiple indicator variables.
 */

/*! \typedef LPFunctionTransfer::SlackVariablesObjectiveCoefficientsContainerType
 *  \brief Defines the container type which is used to store the coefficients of
 *         the slack variables for the objective function.
 */

/*! \fn bool LPFunctionTransfer::isTransferable()
 *  \brief This function will tell if the function type provided via the
 *         template parameter can be transfered into a more efficient LP
 *         formulation.
 *
 *  \note This function will call LPFunctionTransfer_impl::isTransferable()
 *        which by default will return false unless a partial template
 *        specialization of the class opengm::LPFunctionTransfer_impl is
 *        provided for the FUNCTION_TYPE type.
 *
 *  \tparam FUNCTION_TYPE The type of the function.
 *
 *  \return True if the function can be transfered, false otherwise.
 */

/*! \fn IndexType LPFunctionTransfer::numSlackVariables(const FUNCTION_TYPE& function)
 *  \brief This function will tell the number of required slack variables for
 *         the function transfer.
 *
 *  \note This function will call LPFunctionTransfer_impl::numSlackVariables()
 *        which by default will throw a runtime error for a not supported
 *        function type unless a partial template specialization of the class
 *        opengm::LPFunctionTransfer_impl is provided for the FUNCTION_TYPE
 *        type.
 *
 *  \tparam FUNCTION_TYPE The type of the function.
 *
 *  \param[in] function The function which will be transfered to the LP.
 *
 *  \return The number of required slack variables.
 */

/*! \fn void LPFunctionTransfer::getSlackVariablesOrder(const FUNCTION_TYPE& function, IndicatorVariablesContainerType& order)
 *  \brief This function will tell the order of the slack variables which are
 *         required for the function transfer.
 *
 *  \note This function will call
 *        LPFunctionTransfer_impl::getSlackVariablesOrder() which by default
 *        will throw a runtime error for a not supported function type unless a
 *        partial template specialization of the class
 *        opengm::LPFunctionTransfer_impl is provided for the FUNCTION_TYPE
 *        type.
 *
 *  \tparam FUNCTION_TYPE The type of the function.
 *
 *  \param[in] function The function which will be transfered to the LP.
 *  \param[out] order This container will be filled with the indicator variables
 *                    representing the slack variables.
 */

/*! \fn void LPFunctionTransfer::getSlackVariablesObjectiveCoefficients(const FUNCTION_TYPE& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients)
 *  \brief This function will tell the coefficients of the slack variables for
 *         the objective function of the linear program.
 *
 *  \note This function will call
 *        LPFunctionTransfer_impl::getSlackVariablesObjectiveCoefficients()
 *        which by default will throw a runtime error for a not supported
 *        function type unless a partial template specialization of the class
 *        opengm::LPFunctionTransfer_impl is provided for the FUNCTION_TYPE
 *        type.
 *
 *  \tparam FUNCTION_TYPE The type of the function.
 *
 *  \param[in] function The function which will be transfered to the LP.
 *  \param[out] coefficients This container will be filled with the coefficients
 *              of the slack variables for the objective function of the linear
 *              program.
 */

/*! \fn void LPFunctionTransfer::getIndicatorVariables(const FUNCTION_TYPE& function, IndicatorVariablesContainerType& variables)
 *  \brief This function will tell all the indicator variables which are used in
 *         the linear constraints which are used for the function transfer.
 *
 *  \note This function will call
 *        LPFunctionTransfer_impl::getIndicatorVariables() which by default will
 *        throw a runtime error for a not supported function type unless a
 *        partial template specialization of the class
 *        opengm::LPFunctionTransfer_impl is provided for the FUNCTION_TYPE
 *        type.
 *
 *  \tparam FUNCTION_TYPE The type of the function.
 *
 *  \param[in] function The function which will be transfered to the LP.
 *  \param[out] variables This container will be filled with the indicator
 *                        variables which are used in the linear constraints.
 *
 */

/*! \fn void LPFunctionTransfer::getLinearConstraints(const FUNCTION_TYPE& function, LinearConstraintsContainerType& constraints)
 *  \brief This function will create the necessary linear constraints which are
 *         used to represent the function within the LP.
 *
 *  \note This function will call
 *        LPFunctionTransfer_impl::getLinearConstraints() which by default will
 *        throw a runtime error for a not supported function type unless a
 *        partial template specialization of the class
 *        opengm::LPFunctionTransfer_impl is provided for the FUNCTION_TYPE
 *        type.
 *
 *  \tparam FUNCTION_TYPE The type of the function.
 *
 *  \param[in] function The function which will be transfered to the LP.
 *  \param[out] constraints This container will be filled with the linear
 *                          constraints which are required to add the function
 *                          to the LP. Slack variables which are required for
 *                          the transfer will be represented by
 *                          opengm::IndicatorVariables consisting of exactly one
 *                          variable-label-pair where the variable id is set
 *                          to a value greater or equal to the number of
 *                          variables of the function and the label is set to 0.
 */

/*! \struct LPFunctionTransfer::IsTransferableFunctor
 *  \brief Functor to call LPFunctionTransfer::isTransferable() for a factor of
 *         the graphical model.
 */

/*! \var LPFunctionTransfer::IsTransferableFunctor::isTransferable_
 *  \brief Storage for the return value of the
 *         LPFunctionTransfer::isTransferable() method.
 */

/*! \fn void LPFunctionTransfer::IsTransferableFunctor::operator()(const FUNCTION_TYPE& function)
 *  \brief The operator which implements the access to the function of a factor
 *         of the graphical model.
 *
 *  \tparam FUNCTION_TYPE The function type.
 *
 *  \param[in] function The function.
 */

/*! \struct LPFunctionTransfer::NumSlackVariablesFunctor
 *  \brief Functor to call LPFunctionTransfer::numSlackVariables() for a
 *         graphical model factor.
 */

/*! \var LPFunctionTransfer::NumSlackVariablesFunctor::numSlackVariables_
 *  \brief Storage for the return value of the
 *         LPFunctionTransfer::numSlackVariables() method.
 */

/*! \fn void LPFunctionTransfer::NumSlackVariablesFunctor::operator()(const FUNCTION_TYPE& function)
 *  \brief The operator which implements the access to the function of a
 *         graphical model factor.
 *
 *  \tparam FUNCTION_TYPE The function type.
 *
 *  \param[in] function The function.
 */

/*! \struct LPFunctionTransfer::GetSlackVariablesOrderFunctor
 *  \brief Functor to call LPFunctionTransfer::getSlackVariablesOrder()
 *         for a factor of the graphical model.
 */

/*! \var LPFunctionTransfer::GetSlackVariablesOrderFunctor::order_
 *  \brief Pointer to the storage for the return value of the
 *         LPFunctionTransfer::getSlackVariablesOrder() method.
 */

/*! \fn void LPFunctionTransfer::GetSlackVariablesOrderFunctor::operator()(const FUNCTION_TYPE& function)
 *  \brief The operator which implements the access to the function of a
 *         graphical model factor.
 *
 *  \tparam FUNCTION_TYPE The function type.
 *
 *  \param[in] function The function.
 */

/*! \struct LPFunctionTransfer::GetSlackVariablesObjectiveCoefficientsFunctor
 *  \brief Functor to call
 *         LPFunctionTransfer::getSlackVariablesObjectiveCoefficients()
 *         for a factor of the graphical model.
 */

/*! \var LPFunctionTransfer::GetSlackVariablesObjectiveCoefficientsFunctor::coefficients_
 *  \brief Pointer to the storage for the return value of the
 *         LPFunctionTransfer::getSlackVariablesObjectiveCoefficients() method.
 */

/*! \fn void LPFunctionTransfer::GetSlackVariablesObjectiveCoefficientsFunctor::operator()(const FUNCTION_TYPE& function)
 *   \brief The operator which implements the access to the function of a
 *          graphical model factor.
 *
 *  \tparam FUNCTION_TYPE The function type.
 *
 *  \param[in] function The function.
 */

/*! \struct LPFunctionTransfer::GetIndicatorVariablesFunctor
 *  \brief Functor to call LPFunctionTransfer::getIndicatorVariables() for a
 *         factor of the graphical model.
 */

/*! \var LPFunctionTransfer::GetIndicatorVariablesFunctor::variables_
 *  \brief Pointer to the storage for the return value of the
 *         LPFunctionTransfer::getIndicatorVariables() method.
 */

/*! \fn void LPFunctionTransfer::GetIndicatorVariablesFunctor::operator()(const FUNCTION_TYPE& function)
 *  \brief The operator which implements the access to the function of a
 *         graphical model factor.
 *
 *  \tparam FUNCTION_TYPE The function type.
 *
 *  \param[in] function The function.
 */

/*! \struct LPFunctionTransfer::GetLinearConstraintsFunctor
 *  \brief Functor to call LPFunctionTransfer::getLinearConstraints() for a
 *         factor of the graphical model.
 */

/*! \var LPFunctionTransfer::GetLinearConstraintsFunctor::constraints_
 *  \brief Pointer to the storage for the return value of the
 *         LPFunctionTransfer::getLinearConstraints() method.
 */

/*! \fn void LPFunctionTransfer::GetLinearConstraintsFunctor::operator()(const FUNCTION_TYPE& function)
 *  \brief The operator which implements the access to the function of a
 *         graphical model factor.
 *
 *  \tparam FUNCTION_TYPE The function type.
 *
 *  \param[in] function The function.
 */

/*! \class LPFunctionTransfer_impl
 *  \brief Default implementation for class opengm::LPFunctionTransfer.
 *
 *  \tparam FUNCTION_TYPE Function type,
 *  \tparam VALUE_TYPE Value type.
 *  \tparam INDEX_TYPE Index type.
 *  \tparam LABEL_TYPE Label type.
 *
 *  \note This class has to be overwritten via partial template specialization
 *        to enable support for new function types.
 */

/*! \typedef LPFunctionTransfer_impl::ValueType
 *  \brief Typedef of the VALUE_TYPE template parameter type from the class
 *         LPFunctionTransfer_impl.
 */

/*! \typedef LPFunctionTransfer_impl::IndexType
 *  \brief Typedef of the INDEX_TYPE template parameter type from the class
 *         LPFunctionTransfer_impl.
 */

/*! \typedef LPFunctionTransfer_impl::LabelType
 *  \brief Typedef of the LABEL_TYPE template parameter type from the class
 *         LPFunctionTransfer_impl.
 */

/*! \typedef LPFunctionTransfer_impl::FunctionType
 *  \brief Typedef of the FUNCTION_TYPE template parameter type from the class
 *         LPFunctionTransfer_impl.
 */

/*! \typedef LPFunctionTransfer_impl::LPFunctionTransferType
 *  \brief Typedef of the LPFunctionTransfer class with appropriate template
 *         parameter.
 */

/*! \typedef LPFunctionTransfer_impl::LinearConstraintType
 *  \brief Typedef of the LinearConstraint class with appropriate template
 *         parameter.
 */

/*! \typedef LPFunctionTransfer_impl::LinearConstraintsContainerType
 *  \brief Defines the linear constraints container type which is used to store
 *         multiple linear constraints.
 */

/*! \typedef LPFunctionTransfer_impl::IndicatorVariableType
 *  \brief Defines the indicator variable type which is used within linear
 *         constraints.
 */

/*! \typedef LPFunctionTransfer_impl::IndicatorVariablesContainerType
 *  \brief Defines the indicator variables container type which is used to store
 *         multiple indicator variables.
 */

/*! \typedef LPFunctionTransfer_impl::SlackVariablesObjectiveCoefficientsContainerType
 *  \brief Defines the container type which is used to store the coefficients of
 *         the slack variables for the objective function.
 */

/*! \fn bool LPFunctionTransfer_impl::isTransferable()
 *  \brief This function will tell if the function type provided via the
 *         template parameter can be transfered into a more efficient LP
 *         formulation.
 *
 *  \return This function will always return false as this is the default return
 *          value.
 */

/*! \fn IndexType LPFunctionTransfer_impl::numSlackVariables(const FunctionType& function)
 *  \brief This function will tell the number of required slack variables for
 *         the LP function transfer.
 *
 *  \note This function will throw a runtime error of unsupported function type
 *        as numSlackVariables() must not be called for functions which are not
 *        supported. Supported functions are those for which a partial template
 *        specialization of the class LPFunctionTransfer_impl is provided.
 *
 *  \param[in] function The function which will be transformed.
 *
 *  \return Nothing as this function will throw a runtime error.
 */

/*! \fn void LPFunctionTransfer_impl::getSlackVariablesOrder(const FunctionType& function, IndicatorVariablesContainerType& order)
 *  \brief This function will tell the order of the slack variables for the LP
 *         function transfer.
 *
 *  \note This function will throw a runtime error of unsupported function type
 *        as getSlackVariablesOrder() must not be called for functions which are
 *        not supported. Supported functions are those for which a partial
 *        template specialization of the class LPFunctionTransfer_impl is
 *        provided.
 *
 *  \param[in] function The function which will be transformed.
 *  \param[out] order This container will be filled with the order of the slack
 *                    variables.
 */

/*! \fn void LPFunctionTransfer_impl::getSlackVariablesObjectiveCoefficients(const FunctionType& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients)
 *  \brief This function will tell the coefficients of the slack variables for
 *         the objective function of the linear program.
 *
 *  \note This function will throw a runtime error of unsupported function type
 *        as getSlackVariablesObjectiveCoefficients() must not be called for
 *        functions which are not supported. Supported functions are those for
 *        which a partial template specialization of the class
 *        LPFunctionTransfer_impl is provided.
 *
 *  \param[in] function The function which will be transformed.
 *  \param[out] coefficients This container will be filled with the coefficients
 *              of the slack variables for the objective function of the lLP.
 */

/*! \fn void LPFunctionTransfer_impl::getIndicatorVariables(const FunctionType& function, IndicatorVariablesContainerType& variables)
 *  \brief This function will tell the used indicator variables for the
 *         linear constraints.
 *
 *  \note This function will throw a runtime error of unsupported function type
 *        as getIndicatorVariables() must not be called for functions which are
 *        not supported. Supported functions are those for which a partial
 *        template specialization of the class LPFunctionTransfer_impl is
 *        provided.
 *
 *  \param[in] function The function which will be transformed.
 *  \param[out] variables This container will be filled with the indicator
 *                        variables which are used in the linear constraints.
 */

/*! \fn void LPFunctionTransfer_impl::getLinearConstraints(const FunctionType& function, LinearConstraintsContainerType& constraints)
 *  \brief This function will create the necessary linear constraints to add the
 *         function to the linear program.
 *
 *  \note This function will throw a runtime error of unsupported function type
 *        as getLinearConstraints() must not be called for functions which are
 *        not supported. Supported functions are those for which a partial
 *        template specialization of the class LPFunctionTransfer_impl is
 *        provided.
 *
 *  \param[in] function The function which will be transformed.
 *  \param[out] constraints This container will be filled with the linear
 *                          constraints which are required to add the function
 *                          to the linear program. Slack variables which are
 *                          required for the transformation will be represented
 *                          by opengm::IndicatorVariables consisting of exactly
 *                          one variable-label-pair where the variable id is set
 *                          to a value greater or equal to the number of
 *                          variables of the function and the label is set to 0.
 */

/******************
 * implementation *
 ******************/
template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline bool LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::isTransferable() {
   return LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::isTransferable();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline typename LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndexType LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::numSlackVariables(const FUNCTION_TYPE& function) {
   return LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::numSlackVariables(function);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesOrder(const FUNCTION_TYPE& function, IndicatorVariablesContainerType& order) {
   LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesOrder(function, order);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesObjectiveCoefficients(const FUNCTION_TYPE& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients) {
   LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesObjectiveCoefficients(function, coefficients);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getIndicatorVariables(const FUNCTION_TYPE& function, IndicatorVariablesContainerType& variables) {
   LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getIndicatorVariables(function, variables);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getLinearConstraints(const FUNCTION_TYPE& function, LinearConstraintsContainerType& constraints) {
   LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getLinearConstraints(function, constraints);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IsTransferableFunctor::operator()(const FUNCTION_TYPE& function) {
   isTransferable_ = isTransferable<FUNCTION_TYPE>();
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::NumSlackVariablesFunctor::operator()(const FUNCTION_TYPE& function) {
   numSlackVariables_ = numSlackVariables(function);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::GetSlackVariablesOrderFunctor::operator()(const FUNCTION_TYPE& function) {
   getSlackVariablesOrder(function, *order_);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::GetSlackVariablesObjectiveCoefficientsFunctor::operator()(const FUNCTION_TYPE& function) {
   getSlackVariablesObjectiveCoefficients(function, *coefficients_);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::GetIndicatorVariablesFunctor::operator()(const FUNCTION_TYPE& function) {
   getIndicatorVariables(function, *variables_);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
template<class FUNCTION_TYPE>
inline void LPFunctionTransfer<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::GetLinearConstraintsFunctor::operator()(const FUNCTION_TYPE& function) {
   getLinearConstraints(function, *constraints_);
}

template<class FUNCTION_TYPE, class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline bool LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::isTransferable() {
   // default implementation
   return false;
}

template<class FUNCTION_TYPE, class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndexType LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::numSlackVariables(const FunctionType& function) {
   // default implementation
   throw opengm::RuntimeError("Unsupported Function Type.");
   return 0;
}

template<class FUNCTION_TYPE, class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesOrder(const FunctionType& function, IndicatorVariablesContainerType& order) {
   // default implementation
   throw opengm::RuntimeError("Unsupported Function Type.");
}

template<class FUNCTION_TYPE, class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesObjectiveCoefficients(const FunctionType& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients) {
   // default implementation
   throw opengm::RuntimeError("Unsupported Function Type.");
}

template<class FUNCTION_TYPE, class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getIndicatorVariables(const FunctionType& function, IndicatorVariablesContainerType& variables) {
   // default implementation
   throw opengm::RuntimeError("Unsupported Function Type.");
}

template<class FUNCTION_TYPE, class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<FUNCTION_TYPE, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getLinearConstraints(const FunctionType& function, LinearConstraintsContainerType& constraints) {
   // default implementation
   throw opengm::RuntimeError("Unsupported Function Type.");
}


template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline bool LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::isTransferable() {
   // implementation for SumConstraintFunction
   return true;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndexType LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::numSlackVariables(const FunctionType& function) {
   // implementation for SumConstraintFunction
   return 1;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesOrder(const FunctionType& function, IndicatorVariablesContainerType& order) {
   // implementation for SumConstraintFunction
   order.resize(1);
   order[0] = IndicatorVariableType(function.numVariables_, 0);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesObjectiveCoefficients(const FunctionType& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients) {
   // implementation for SumConstraintFunction
   coefficients.resize(1);
   coefficients[0] = function.lambda_;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getIndicatorVariables(const FunctionType& function, IndicatorVariablesContainerType& variables) {
   // implementation for SumConstraintFunction
   variables.clear();
   for(IndexType i = 0; i < function.numVariables_; ++i) {
      for(LabelType j = 0; j < (function.useSameNumLabels_ ? function.maxNumLabels_ : function.shape_[i]); ++j) {
         const IndicatorVariableType indicatorVariable(i, j);
         variables.push_back(indicatorVariable);
      }
   }
   const IndicatorVariableType slackVariable(function.numVariables_, 0);
   variables.push_back(slackVariable);
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<SumConstraintFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getLinearConstraints(const FunctionType& function, LinearConstraintsContainerType& constraints) {
   // implementation for SumConstraintFunction
   constraints.resize(2);
   LinearConstraintType constraint;
   constraint.setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::LessEqual);
   constraint.setBound(function.bound_);
   for(IndexType i = 0; i < function.numVariables_; ++i) {
      for(LabelType j = 0; j < (function.useSameNumLabels_ ? function.maxNumLabels_ : function.shape_[i]); ++j) {
         const IndicatorVariableType indicatorVariable(i, j);
         constraint.add(indicatorVariable, function.coefficients_[(function.shareCoefficients_ ? j : function.coefficientsOffsets_[i] + j)]);
      }
   }
   const IndicatorVariableType slackVariable(function.numVariables_, 0);
   constraint.add(slackVariable, -1.0);
   constraints[0] = constraint;

   LinearConstraintType constraint2;
   constraint2.setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::LessEqual);
   constraint2.setBound(-function.bound_);
   for(IndexType i = 0; i < function.numVariables_; ++i) {
      for(LabelType j = 0; j < (function.useSameNumLabels_ ? function.maxNumLabels_ : function.shape_[i]); ++j) {
         const IndicatorVariableType indicatorVariable(i, j);
         constraint2.add(indicatorVariable, -function.coefficients_[(function.shareCoefficients_ ? j : function.coefficientsOffsets_[i] + j)]);
      }
   }
   constraint2.add(slackVariable, -1.0);
   constraints[1] = constraint2;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline bool LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::isTransferable() {
   // implementation for LabelCostFunction
   return true;
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline typename LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::IndexType LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::numSlackVariables(const FunctionType& function) {
   // implementation for LabelCostFunction
   // one slack variable for each label cost which is not zero
   if(function.useSingleCost_) {
      return 1;
   } else {
      IndexType numLabelCosts = 0;
      for(LabelType i = 0; i < function.maxNumLabels_; ++i) {
         // sort out unused labels
         if(function.costs_[i] != 0.0) {
            ++numLabelCosts;
         }
      }
      return numLabelCosts;
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesOrder(const FunctionType& function, IndicatorVariablesContainerType& order) {
   // implementation for LabelCostFunction
   const IndexType numLabelCosts = numSlackVariables(function);
   order.resize(numLabelCosts);
   for(IndexType i = 0; i < numLabelCosts; ++i) {
      order[i] = IndicatorVariableType(function.numVariables_ + i, 0);
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getSlackVariablesObjectiveCoefficients(const FunctionType& function, SlackVariablesObjectiveCoefficientsContainerType& coefficients) {
   // implementation for LabelCostFunction
   const IndexType numLabelCosts = numSlackVariables(function);
   coefficients.resize(numLabelCosts);
   if(function.useSingleCost_) {
      OPENGM_ASSERT(numLabelCosts == 1);
      coefficients[0] = function.singleCost_;
   } else {
      LabelType currentCostIndex = 0;
      for(LabelType i = 0; i < function.maxNumLabels_; ++i) {
         // sort out unused label costs
         if(function.costs_[i] != 0.0) {
            coefficients[currentCostIndex] = function.costs_[i];
            ++currentCostIndex;
         }
      }
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getIndicatorVariables(const FunctionType& function, IndicatorVariablesContainerType& variables) {
   // implementation for LabelCostFunction
   const IndexType numLabelCosts = numSlackVariables(function);
   variables.resize(numLabelCosts * 2);
   if(function.useSingleCost_) {
      OPENGM_ASSERT(variables.size() == 2);
      IndicatorVariableType indicatorVar;
      indicatorVar.setLogicalOperatorType(IndicatorVariableType::Or);
      if(function.useSameNumLabels_) {
         for(IndexType i = 0; i < function.numVariables_; ++i) {
            indicatorVar.add(i, function.singleLabel_);
         }
      } else {
         for(IndexType i = 0; i < function.numVariables_; ++i) {
            if(function.shape_[i] > function.singleLabel_) {
               indicatorVar.add(i, function.singleLabel_);
            }
         }
      }
      variables[0] = indicatorVar;

      variables[1] = IndicatorVariableType(function.numVariables_, 0);
   } else {
      LabelType currentNonZeroLabel = 0;
      for(LabelType currentLabel = 0; currentLabel < function.maxNumLabels_; ++currentLabel) {
         if(function.costs_[currentLabel] != 0) {
            IndicatorVariableType indicatorVar;
            indicatorVar.setLogicalOperatorType(IndicatorVariableType::Or);
            if(function.useSameNumLabels_) {
               for(IndexType i = 0; i < function.numVariables_; ++i) {
                  indicatorVar.add(i, currentLabel);
               }
            } else {
               for(IndexType i = 0; i < function.numVariables_; ++i) {
                  if(function.shape_[i] > currentLabel) {
                     indicatorVar.add(i, currentLabel);
                  }
               }
            }
            variables[currentNonZeroLabel * 2] = indicatorVar;

            variables[(currentNonZeroLabel * 2) + 1] = IndicatorVariableType(function.numVariables_ + currentNonZeroLabel, 0);
            ++currentNonZeroLabel;
         }
      }
   }
}

template<class VALUE_TYPE, class INDEX_TYPE, class LABEL_TYPE>
inline void LPFunctionTransfer_impl<LabelCostFunction<VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>, VALUE_TYPE, INDEX_TYPE, LABEL_TYPE>::getLinearConstraints(const FunctionType& function, LinearConstraintsContainerType& constraints) {
   // implementation for LabelCostFunction
   IndicatorVariablesContainerType variables;
   getIndicatorVariables(function, variables);
   OPENGM_ASSERT(variables.size() % 2 == 0);

   constraints.resize(variables.size() / 2);
   for(size_t i = 0; i < constraints.size(); ++i) {
      LinearConstraintType constraint;
      constraint.setConstraintOperator(LinearConstraintType::LinearConstraintOperatorType::Equal);
      constraint.setBound(0.0);
      constraint.add(variables[i * 2], 1.0); // Or indicator variable
      constraint.add(variables[(i * 2) + 1], -1.0); // slack variable
      constraints[i] = constraint;
   }
}

} // namespace opengm

#endif /* OPENGM_LP_FUNCTIONTRANSFER_HXX_ */
