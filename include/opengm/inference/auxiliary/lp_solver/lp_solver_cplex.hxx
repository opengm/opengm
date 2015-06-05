#ifndef OPENGM_LP_SOLVER_CPLEX_HXX_
#define OPENGM_LP_SOLVER_CPLEX_HXX_

#include <iterator>

#include <ilcplex/ilocplex.h>

#include <opengm/inference/auxiliary/lpdef.hxx>
#include <opengm/inference/auxiliary/lp_solver/lp_solver_interface.hxx>

/*********************
 * class definition *
 *********************/
namespace opengm {

class IloNumArrayIterator : public std::iterator<std::random_access_iterator_tag, IloNum> {
public:
   // construction
   IloNumArrayIterator();
   IloNumArrayIterator(const IloNumArray& array, const IloInt position = 0);

   // comparison
   bool operator!=(const IloNumArrayIterator& iter) const;
   bool operator==(const IloNumArrayIterator& iter) const;

   // constant access
   const IloNum& operator*() const;
   const IloNum& operator[](const IloInt n) const;

   // increment
   IloNumArrayIterator& operator++();

   // decrement
   difference_type operator-(const IloNumArrayIterator& iter) const;
protected:
   // storage
   const IloNumArray* array_;
   IloInt             position_;
};

class LPSolverCplex : public LPSolverInterface<LPSolverCplex, IloNum, IloInt, IloNumArrayIterator, IloNum> {
public:
   // typedefs
   typedef IloNum              CplexValueType;
   typedef IloInt              CplexIndexType;
   typedef IloNumArrayIterator CplexSolutionIteratorType;
   typedef IloNum              CplexTimingType;

   typedef LPSolverInterface<LPSolverCplex, CplexValueType, CplexIndexType, CplexSolutionIteratorType, CplexTimingType> LPSolverBaseClass;

   // constructor
   LPSolverCplex(const Parameter& parameter = Parameter());

   // destructor
   ~LPSolverCplex();

protected:
   // Storage for CPLEX variables
   IloEnv         cplexEnvironment_;
   IloModel       cplexModel_;
   IloNumVarArray cplexVariables_;
   IloObjective   cplexObjective_;
   IloRangeArray  cplexConstraints_;
   IloNumArray    cplexSolution_;
   mutable bool   cplexSolutionValid_;
   IloCplex       cplexSolver_;

   // methods for class LPSolverInterface
   // CPLEX infinity value
   static CplexValueType infinity_impl();

   // add Variables
   void addContinuousVariables_impl(const CplexIndexType numVariables, const CplexValueType lowerBound, const CplexValueType upperBound);
   void addIntegerVariables_impl(const CplexIndexType numVariables, const CplexValueType lowerBound, const CplexValueType upperBound);
   void addBinaryVariables_impl(const CplexIndexType numVariables);

   // objective function
   void setObjective_impl(const Objective objective);
   void setObjectiveValue_impl(const CplexIndexType variable, const CplexValueType value);
   template<class ITERATOR_TYPE>
   void setObjectiveValue_impl(ITERATOR_TYPE begin, const ITERATOR_TYPE end);
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void setObjectiveValue_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin);

   // constraints
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addEqualityConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName = "");
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addLessEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName = "");
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addGreaterEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName = "");

   void addConstraintsFinished_impl();
   void addConstraintsFinished_impl(CplexTimingType& timing);

   // parameter
   template <class PARAMETER_TYPE, class PARAMETER_VALUE_TYPE>
   void setParameter_impl(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value);

   // solve
   bool solve_impl();
   bool solve_impl(CplexTimingType& timing);

   // solution
   CplexSolutionIteratorType solutionBegin_impl() const;
   CplexSolutionIteratorType solutionEnd_impl() const;
   CplexValueType solution_impl(const CplexIndexType variable) const;

   CplexValueType objectiveFunctionValue_impl() const;
   CplexValueType objectiveFunctionValueBound_impl() const;

   // model export
   void exportModel_impl(const std::string& filename) const;

   // helper functions
   void updateSolution() const;
   static int getCutLevelValue(const LPDef::MIP_CUT cutLevel);

   // friend
   friend class LPSolverInterface<LPSolverCplex, CplexValueType, CplexIndexType, CplexSolutionIteratorType, CplexTimingType>;
};

} // namespace opengm

/***********************
 * class documentation *
 ***********************/
/*! \file lp_solver_cplex.hxx
 *  \brief Provides wrapper class for LP Solver CPLEX.
 */

/*! \class opengm::IloNumArrayIterator
 *  \brief Iterator to iterate over an array of type IloNumArray.
 *
 *  This class implements an iterator to iterate over the elements of an array
 *  of type IloNumArray. It provides only constant access to the elements.
 */

/*! \fn IloNumArrayIterator::IloNumArrayIterator()
 *  \brief Default constructor to create an empty IloNumArrayIterator.
 */

/*! \fn IloNumArrayIterator::IloNumArrayIterator(const IloNumArray& array, const IloInt position = 0)
 *  \brief Constructor to create an IloNumArrayIterator.
 *
 *  \param[in] array The IloNumArray the iterator will iterate over.
 *  \param[in] position The current index if the element at which the iterator
 *                      points.
 */

/*! \fn bool IloNumArrayIterator::operator!=(IloNumArrayIterator iter) const
 *  \brief Comparison operator to test if two iterators point to different
 *         elements.
 *
 *  \param[in] iter The iterator which will be tested for inequality.
 *
 *  \return True if the two iterators point to different elements, false
 *          otherwise.
 */

/*! \fn bool IloNumArrayIterator::operator==(IloNumArrayIterator iter) const
 *  \brief Comparison operator to test if two iterators point to the same
 *         element.
 *
 *  \param[in] iter The iterator which will be tested for equality.
 *
 *  \return True if the two iterators point to the same element, false
 *          otherwise.
 */

/*! \fn const IloNum& IloNumArrayIterator::operator*() const
 *  \brief The dereference operator provides constant access to the element at
 *         which the iterator is pointing.
 *
 *  \return Constant reference to the element at which the iterator is pointing.
 */

/*! \fn const IloNum& IloNumArrayIterator::operator[](const IloInt n) const
 *  \brief The subscript operator provides constant access to the element in the
 *         sequence which is n elements behind the element at which the iterator
 *         is pointing.
 *
 *  \param[in] n The offset for the element which will be accessed.
 *
 *  \return Constant reference to the element in the sequence which is n
 *          elements behind the element at which the iterator is pointing.
 */

/*! \fn IloNumArrayIterator& IloNumArrayIterator::operator++()
 *  \brief The increment operator increases the position at which the iterator
 *         is pointing by one. Hence the iterator will point to the next element
 *         in the sequence.
 *
 *  \return Reference to the iterator which is now pointing to the next element
 *          in the sequence.
 */

/*! \fn IloNumArrayIterator::difference_type IloNumArrayIterator::operator-(const IloNumArrayIterator& iter) const
 *  \brief The difference operator computes the difference of two
 *         IloNumArrayIterators.
 *
 *  \return Distance between two IloNumArrayIterators.
 */

/*! \var IloNumArrayIterator::array_
 *   \brief Constant pointer to the IloNumArray over which the iterator will
 *          iterate.
 */

/*! \var IloNumArrayIterator::position_
 *   \brief Index of the element in the sequence at which the iterator is
 *          pointing at the moment.
 */

/*! \class opengm::LPSolverCplex
 *  \brief Wrapper class for the IBM ILOG CPLEX optimizer.
 *
 *  \note <a href="http://www.ilog.com/products/cplex/">IBM ILOG CPLEX</a> is a
 *        commercial product that is free for academical use.
 */

/*! \typedef LPSolverCplex::CplexValueType
 *  \brief Defines the value type used by CPLEX.
 */

/*! \typedef LPSolverCplex::CplexIndexType
 *  \brief Defines the index type used by CPLEX.
 */

/*! \typedef LPSolverCplex::CplexSolutionIteratorType
 *  \brief Defines the iterator type which can be used to iterate over the
 *         solution of CPLEX.
 */

/*! \typedef LPSolverCplex::CplexTimingType
 *  \brief Defines the timing type used by CPLEX.
 */

/*! \typedef LPSolverCplex::LPSolverBaseClass
 *  \brief Defines the type of the base class.
 */

/*! \fn LPSolverCplex::LPSolverCplex(const Parameter& parameter = Parameter())
 *  \brief Default constructor for LPSolverCplex.
 *
 *  \param[in] parameter Settings for the CPLEX solver.
 */

/*! \fn LPSolverCplex::~LPSolverCplex()
 *  \brief Destructor for LPSolverCplex.
 */

/*! \var LPSolverCplex::cplexEnvironment_
 *  \brief The CPLEX environment.
 */

/*! \var LPSolverCplex::cplexModel_
 *  \brief The CPLEX model of the LP/MIP problem.
 */

/*! \var LPSolverCplex::cplexVariables_
 *  \brief The variables which are present in the model.
 */

/*! \var LPSolverCplex::cplexObjective_
 *  \brief The objective function.
 */

/*! \var LPSolverCplex::cplexConstraints_
 *  \brief Puffer for the constraints which are added by
 *         CPlexSolver::addEqualityConstraint,
 *         CPlexSolver::addLessEqualConstraint and
 *         CPlexSolver::addGreaterEqualConstraint. Will be flushed into the
 *         CPLEX model when CPlexSolver::addConstraintsFinished is called.
 */

/*! \var LPSolverCplex::cplexSolution_
 *  \brief Storage for the solution computed by CPLEX.
 */

/*! \var LPSolverCplex::cplexSolutionValid_
 *  \brief Tell if the currently stored solution is valid.
 */

/*! \var LPSolverCplex::cplexSolver_
 *  \brief The CPLEX solver.
 */

/*! \fn static LPSolverCplex::CplexValueType LPSolverCplex::infinity_impl()
 *  \brief Get the value which is used by CPLEX to represent infinity.
 *
 *  \note Implementation for base class LPSolverInterface::infinity method.
 */

/*! \fn void LPSolverCplex::addContinuousVariables_impl(const CplexIndexType numVariables, const CplexValueType lowerBound, const CplexValueType upperBound)
 *  \brief Add new continuous variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *  \param[in] lowerBound The lower bound for the new Variables.
 *  \param[in] upperBound The upper bound for the new Variables.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::addContinuousVariables method.
 */

/*! \fn void LPSolverCplex::addIntegerVariables_impl(const CplexIndexType numVariables, const CplexValueType lowerBound, const CplexValueType upperBound)
 *  \brief Add new integer variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *  \param[in] lowerBound The lower bound for the new Variables.
 *  \param[in] upperBound The upper bound for the new Variables.
 *
 *  \note Implementation for base class LPSolverInterface::addIntegerVariables
 *        method.
 */

/*! \fn void LPSolverCplex::addBinaryVariables_impl(const CplexIndexType numVariables)
 *  \brief Add new binary variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *
 *  \note Implementation for base class LPSolverInterface::addBinaryVariables
 *        method.
 */

/*! \fn void LPSolverCplex::setObjective_impl(const Objective objective)
 *  \brief Set objective to minimize or maximize.
 *
 *  \param[in] objective The new objective.
 *
 *  \note Implementation for base class LPSolverInterface::setObjective_impl
 *        method.
 */

/*! \fn void LPSolverCplex::setObjectiveValue_impl(const CplexIndexType variable, const CplexValueType value)
 *  \brief Set the coefficient of a variable in the objective function.
 *
 *  \param[in] variable The index of the variable.
 *  \param[in] value The value which will be set as the coefficient of the
 *             variable in the objective function.
 *
 *  \note Implementation for base class LPSolverInterface::setObjectiveValue
 *        method.
 */

/*! \fn void LPSolverCplex::setObjectiveValue_impl(ITERATOR_TYPE begin, const ITERATOR_TYPE end)
 *  \brief Set values of the coefficients of all variables in the objective
 *         function.
 *  \tparam ITERATOR_TYPE Iterator type used to iterate over the values which
 *                        will be set as the coefficients of the objective
 *                        function.
 *
 *  \param[in] begin Iterator pointing to the begin of the sequence of values
 *                   which will be set as the coefficients of the objective
 *                   function.
 *  \param[in] end Iterator pointing to the end of the sequence of values which
 *                 will be set as the coefficients of the objective function.
 *
 *  \note Implementation for base class LPSolverInterface::setObjectiveValue
 *        method.
 */

/*! \fn void LPSolverCplex::setObjectiveValue_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin)
 *  \brief Set values as the coefficients of selected variables in the objective
 *         function.
 *  \tparam VARIABLES_ITERATOR_TYPE Iterator type used to iterate over the
 *                                  indices of the variables.
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type used to iterate over the
 *                                  coefficients of the variables which will be
 *                                  set in the objective function.
 *
 *  \param[in] variableIDsBegin Iterator pointing to the begin of the sequence
 *                              of indices of the variables.
 *  \param[in] variableIDsEnd Iterator pointing to the end of the sequence of
 *                            indices of the variables.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of the sequence
 *                              of values which will be set as the
 *                              coefficients of the objective function.
 *
 *  \note Implementation for base class LPSolverInterface::setObjectiveValue
 *        method.
 */

/*! \fn void LPSolverCplex::addEqualityConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName = "")
 *  \brief Add a new equality constraint to the model.
 *
 *  \tparam VARIABLES_ITERATOR_TYPE Iterator type to iterate over the variable
 *                                  ids of the constraints.
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type to iterate over the
 *                                     coefficients of the constraints.
 *
 *  \param[in] variableIDsBegin Iterator pointing to the begin of a sequence of
 *                              values defining the variables of the constraint.
 *  \param[in] variableIDsEnd Iterator pointing to the end of a sequence of
 *                            values defining the variables of the constraint.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of a sequence of
 *                               values defining the coefficients for the
 *                               variables.
 *  \param[in] bound The right hand side of the equality constraint.
 *  \param[in] constraintName The name for the equality constraint.
 *
 *  \note
 *        -# Implementation for base class
 *           LPSolverInterface::addEqualityConstraint method.
 *        -# To increase performance for adding multiple constraints to the
 *           model, all constraints added via
 *           LPSolverCplex::addEqualityConstraint,
 *           LPSolverCplex::addLessEqualConstraint and
 *           LPSolverCplex::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverCplex::addConstraintsFinished is called.
 */

/*! \fn void LPSolverCplex::addLessEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName = "");
 *  \brief Add a new less equal constraint to the model.
 *
 *  \tparam VARIABLES_ITERATOR_TYPE Iterator type to iterate over the variable
 *                                  ids of the constraints.
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type to iterate over the
 *                                     coefficients of the constraints.
 *
 *  \param[in] variableIDsBegin Iterator pointing to the begin of a sequence of
 *                              values defining the variables of the constraint.
 *  \param[in] variableIDsEnd Iterator pointing to the end of a sequence of
 *                            values defining the variables of the constraint.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of a sequence of
 *                               values defining the coefficients for the
 *                               variables.
 *  \param[in] bound The right hand side of the less equal constraint.
 *  \param[in] constraintName The name for the less equal constraint.
 *
 *  \note
 *        -# Implementation for base class
 *           LPSolverInterface::addLessEqualConstraint method.
 *        -# To increase performance for adding multiple constraints to the
 *           model, all constraints added via
 *           LPSolverCplex::addEqualityConstraint,
 *           LPSolverCplex::addLessEqualConstraint and
 *           LPSolverCplex::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverCplex::addConstraintsFinished is called.
 */

/*! \fn void LPSolverCplex::addGreaterEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName = "")
 *  \brief Add a new greater equal constraint to the model.
 *
 *  \tparam VARIABLES_ITERATOR_TYPE Iterator type to iterate over the variable
 *                                  ids of the constraints.
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type to iterate over the
 *                                     coefficients of the constraints.
 *
 *  \param[in] variableIDsBegin Iterator pointing to the begin of a sequence of
 *                              values defining the variables of the constraint.
 *  \param[in] variableIDsEnd Iterator pointing to the end of a sequence of
 *                            values defining the variables of the constraint.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of a sequence of
 *                               values defining the coefficients for the
 *                               variables.
 *  \param[in] bound The right hand side of the greater equal constraint.
 *  \param[in] constraintName The name for the greater equal constraint.
 *
 *  \note
 *        -# Implementation for base class
 *           LPSolverInterface::addGreaterEqualConstraint method.
 *        -# To increase performance for adding multiple constraints to the
 *           model, all constraints added via
 *           LPSolverCplex::addEqualityConstraint,
 *           LPSolverCplex::addLessEqualConstraint and
 *           LPSolverCplex::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverCplex::addConstraintsFinished is called.
 */

/*! \fn void LPSolverCplex::addConstraintsFinished_impl()
 *  \brief Join all constraints added via LPSolverCplex::addEqualityConstraint,
 *         LPSolverCplex::addLessEqualConstraint and
 *         LPSolverCplex::addGreaterEqualConstraint to the model.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::addConstraintsFinished method.
 */

/*! \fn void LPSolverCplex::addConstraintsFinished_impl(CplexTimingType& timing)
 *  \brief Join all constraints added via LPSolverCplex::addEqualityConstraint,
 *         LPSolverCplex::addLessEqualConstraint and
 *         LPSolverCplex::addGreaterEqualConstraint to the model.
 *
 *  \param[out] timing Returns the time needed to join all constraints into the
 *                     model.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::addConstraintsFinished method.
 */

/*! \fn void LPSolverCplex::setParameter_impl(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value)
 *  \brief Set CPLEX parameter.
 *
 *  \tparam PARAMETER_VALUE_TYPE The type of the parameter.
 *  \tparam VALUE_TYPE The type of the value.
 *
 *  \param[in] parameter The CPLEX parameter.
 *  \param[in] value The new value to which the parameter will be set.
 *
 *  \note Implementation for base class LPSolverInterface::setParameter method.
 */

/*! \fn bool LPSolverCplex::solve_impl()
 *  \brief Solve the current model.
 *
 *  \return True if solving the model finished successfully, false if CPLEX was
 *          not able to solve the model.
 *
 *  \note Implementation for base class LPSolverInterface::solve method.
 */

/*! \fn bool LPSolverCplex::solve_impl(CplexTimingType& timing)
 *  \brief Solve the current model and measure solving time.
 *
 *  \param[out] timing The time CPLEX needed to solve the problem.
 *
 *  \return True if solving the model finished successfully, false if CPLEX was
 *          not able to solve the model.
 *
 *  \note Implementation for base class LPSolverInterface::solve method.
 */

/*! \fn LPSolverCplex::CplexSolutionIteratorType LPSolverCplex::solutionBegin_impl() const
 *  \brief Get an iterator which is pointing to the begin of the solution
 *         computed by CPLEX.
 *
 *  \return Iterator pointing to the begin of the solution.
 *
 *  \note Implementation for base class LPSolverInterface::solutionBegin method.
 */

/*! \fn LPSolverCplex::CplexSolutionIteratorType LPSolverCplex::solutionEnd_impl() const
 *  \brief Get an iterator which is pointing to the end of the solution computed
 *         by CPLEX.
 *
 *  \return Iterator pointing to the begin of the solution.
 *
 *  \note Implementation for base class LPSolverInterface::solutionEnd method.
 */

/*! \fn LPSolverCplex::CplexValueType LPSolverCplex::solution_impl(const CplexIndexType variable) const;
 *  \brief Get the solution value of a variable computed by CPLEX.
 *
 *  \param[in] variable Index of the variable for which the solution value is
 *                      requested.
 *
 *  \return Solution value of the selected variable.
 *
 *  \note Implementation for base class LPSolverInterface::solution method.
 */

/*! \fn LPSolverCplex::CplexValueType LPSolverCplex::objectiveFunctionValue_impl() const;
 *  \brief Get the objective function value from CPLEX.
 *
 *  \return Objective function value.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::objectiveFunctionValue method.
 */

/*! \fn LPSolverCplex::CplexValueType LPSolverCplex::objectiveFunctionValueBound_impl() const
 *  \brief Get the best known bound for the optimal solution of the current
 *         model.
 *
 *  \return The bound for the current model.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::objectiveFunctionValueBound method.
 */

/*! \fn void LPSolverCplex::exportModel_impl(const std::string& filename) const
 *  \brief Export model to file.
 *
 *  \param[in] filename The name of the file where the model will be stored.
 *
 *  \note Implementation for base class LPSolverInterface::exportModel method.
 */

/*! \fn void LPSolverCplex::updateSolution()
 *  \brief Update solution if required.
 */

/*! \fn static int LPSolverCplex::getCutLevelValue(const LPDef::MIP_CUT cutLevel);
 *  \brief Translate LPDef::MIP_CUT into corresponding CPLEX int value.
 *
 *  \return Integer value corresponding to the LPDef::MIP_CUT parameter.
 */

/******************
 * implementation *
 ******************/
namespace opengm {

inline IloNumArrayIterator::IloNumArrayIterator()
   : array_(), position_() {

}

inline IloNumArrayIterator::IloNumArrayIterator(const IloNumArray& array, const IloInt position)
   : array_(&array), position_(position) {

}

inline bool IloNumArrayIterator::operator!=(const IloNumArrayIterator& iter) const {
   return (array_ != iter.array_) || (position_ != iter.position_);
}

inline bool IloNumArrayIterator::operator==(const IloNumArrayIterator& iter) const {
   return (array_ == iter.array_) && (position_ == iter.position_);
}

inline const IloNum& IloNumArrayIterator::operator*() const {
   return array_->operator [](position_);
}

inline const IloNum& IloNumArrayIterator::operator[](const IloInt n) const {
   return array_->operator [](position_ + n);
}

inline IloNumArrayIterator& IloNumArrayIterator::operator++() {
   ++position_;
   return *this;
}

inline IloNumArrayIterator::difference_type IloNumArrayIterator:: operator-(const IloNumArrayIterator& iter) const {
   return static_cast<difference_type>(position_) - static_cast<difference_type>(iter.position_);
}

inline LPSolverCplex::LPSolverCplex(const Parameter& parameter)
   : LPSolverBaseClass(parameter), cplexEnvironment_(),
   cplexModel_(cplexEnvironment_), cplexVariables_(cplexEnvironment_),
   cplexObjective_(cplexEnvironment_), cplexConstraints_(cplexEnvironment_),
   cplexSolution_(cplexEnvironment_), cplexSolutionValid_(false),
   cplexSolver_() {
   // initialize solver
   try {
      cplexSolver_ = IloCplex(cplexModel_);
      cplexModel_.add(cplexObjective_);
      cplexModel_.add(cplexVariables_);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }

   // set parameter
   try {
      // multi-threading options
      cplexSolver_.setParam(IloCplex::Threads, parameter_.numberOfThreads_);

      // verbose options
      if(!parameter_.verbose_) {
         cplexSolver_.setParam(IloCplex::MIPDisplay,  0);
         cplexSolver_.setParam(IloCplex::BarDisplay,  0);
         cplexSolver_.setParam(IloCplex::SimDisplay,  0);
         cplexSolver_.setParam(IloCplex::NetDisplay,  0);
         cplexSolver_.setParam(IloCplex::SiftDisplay, 0);
      }

      // set hints
      cplexSolver_.setParam(IloCplex::CutUp, parameter_.cutUp_);

      // tolerance settings
      cplexSolver_.setParam(IloCplex::EpOpt,  parameter_.epOpt_);  // Optimality Tolerance
      cplexSolver_.setParam(IloCplex::EpMrk,  parameter_.epMrk_);  // Markowitz tolerance
      cplexSolver_.setParam(IloCplex::EpRHS,  parameter_.epRHS_);  // Feasibility Tolerance
      cplexSolver_.setParam(IloCplex::EpInt,  parameter_.epInt_);  // amount by which an integer variable can differ from an integer
      cplexSolver_.setParam(IloCplex::EpAGap, parameter_.epAGap_); // Absolute MIP gap tolerance
      cplexSolver_.setParam(IloCplex::EpGap,  parameter_.epGap_);  // Relative MIP gap tolerance

      // memory setting
      cplexSolver_.setParam(IloCplex::WorkMem,        parameter_.workMem_);
      cplexSolver_.setParam(IloCplex::TreLim,         parameter_.treeMemoryLimit_);
      cplexSolver_.setParam(IloCplex::MemoryEmphasis, 1);

      // time limit
      cplexSolver_.setParam(IloCplex::ClockType, 2); //wall-clock-time=2 cpu-time=1
      cplexSolver_.setParam(IloCplex::TiLim,     parameter_.timeLimit_);

      // Root Algorithm
      switch(parameter_.rootAlg_) {
         case LPDef::LP_SOLVER_AUTO: {
            cplexSolver_.setParam(IloCplex::RootAlg, 0);
            break;
         }
         case LPDef::LP_SOLVER_PRIMAL_SIMPLEX: {
            cplexSolver_.setParam(IloCplex::RootAlg, 1);
            break;
         }
         case LPDef::LP_SOLVER_DUAL_SIMPLEX: {
            cplexSolver_.setParam(IloCplex::RootAlg, 2);
            break;
         }
         case LPDef::LP_SOLVER_NETWORK_SIMPLEX: {
            cplexSolver_.setParam(IloCplex::RootAlg, 3);
            break;
         }
         case LPDef::LP_SOLVER_BARRIER: {
            cplexSolver_.setParam(IloCplex::RootAlg, 4);
            break;
         }
         case LPDef::LP_SOLVER_SIFTING: {
            cplexSolver_.setParam(IloCplex::RootAlg, 5);
            break;
         }
         case LPDef::LP_SOLVER_CONCURRENT: {
            cplexSolver_.setParam(IloCplex::RootAlg, 6);
            break;
         }
         default: {
            throw std::runtime_error("Unknown Root Algorithm");
         }
      }

      // Node Algorithm
      switch(parameter_.nodeAlg_) {
         case LPDef::LP_SOLVER_AUTO: {
            cplexSolver_.setParam(IloCplex::NodeAlg, 0);
            break;
         }
         case LPDef::LP_SOLVER_PRIMAL_SIMPLEX: {
            cplexSolver_.setParam(IloCplex::NodeAlg, 1);
            break;
         }
         case LPDef::LP_SOLVER_DUAL_SIMPLEX: {
            cplexSolver_.setParam(IloCplex::NodeAlg, 2);
            break;
         }
         case LPDef::LP_SOLVER_NETWORK_SIMPLEX: {
            cplexSolver_.setParam(IloCplex::NodeAlg, 3);
            break;
         }
         case LPDef::LP_SOLVER_BARRIER: {
            cplexSolver_.setParam(IloCplex::NodeAlg, 4);
            break;
         }
         case LPDef::LP_SOLVER_SIFTING: {
            cplexSolver_.setParam(IloCplex::NodeAlg, 5);
            break;
         }
         case LPDef::LP_SOLVER_CONCURRENT: {
            cplexSolver_.setParam(IloCplex::NodeAlg, 6);
            break;
         }
         default: {
            throw std::runtime_error("Unknown Node Algorithm");
         }
      }

      // presolve
      switch(parameter_.presolve_) {
         case LPDef::LP_PRESOLVE_AUTO: {
            cplexSolver_.setParam(IloCplex::PreInd,      CPX_ON);
            cplexSolver_.setParam(IloCplex::RelaxPreInd, -1);
            break;
         }
         case LPDef::LP_PRESOLVE_OFF: {
            cplexSolver_.setParam(IloCplex::PreInd,      CPX_OFF);
            cplexSolver_.setParam(IloCplex::RelaxPreInd, 0);
            break;
         }
         case LPDef::LP_PRESOLVE_CONSERVATIVE: {
            cplexSolver_.setParam(IloCplex::PreInd,      CPX_ON);
            cplexSolver_.setParam(IloCplex::RelaxPreInd, -1);
            break;
         }
         case LPDef::LP_PRESOLVE_AGGRESSIVE: {
            cplexSolver_.setParam(IloCplex::PreInd,      CPX_ON);
            cplexSolver_.setParam(IloCplex::RelaxPreInd, 1);
            break;
         }
         default: {
            throw std::runtime_error("Unknown Presolve Option");
         }
      }

      // MIP EMPHASIS
      switch(parameter_.mipEmphasis_) {
         case LPDef::MIP_EMPHASIS_BALANCED: {
            cplexSolver_.setParam(IloCplex::MIPEmphasis, 0);
            break;
         }
         case LPDef::MIP_EMPHASIS_FEASIBILITY: {
            cplexSolver_.setParam(IloCplex::MIPEmphasis, 1);
            break;
         }
         case LPDef::MIP_EMPHASIS_OPTIMALITY: {
            cplexSolver_.setParam(IloCplex::MIPEmphasis, 2);
            break;
         }
         case LPDef::MIP_EMPHASIS_BESTBOUND: {
            cplexSolver_.setParam(IloCplex::MIPEmphasis, 3);
            break;
         }
         case LPDef::MIP_EMPHASIS_HIDDENFEAS: {
            cplexSolver_.setParam(IloCplex::MIPEmphasis, 4);
            break;
         }
         default: {
            throw std::runtime_error("Unknown MIP Emphasis Option");
         }
      }

      // Tuning
      cplexSolver_.setParam(IloCplex::Probe, parameter_.probingLevel_);

      if(parameter_.cutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.cutLevel_);
         cplexSolver_.setParam(IloCplex::Cliques,    cl);
         cplexSolver_.setParam(IloCplex::Covers,     cl);
         cplexSolver_.setParam(IloCplex::GUBCovers,  cl);
         cplexSolver_.setParam(IloCplex::MIRCuts,    cl);
         cplexSolver_.setParam(IloCplex::ImplBd,     cl);
         cplexSolver_.setParam(IloCplex::FlowCovers, cl);
         cplexSolver_.setParam(IloCplex::FlowPaths,  cl);
         cplexSolver_.setParam(IloCplex::DisjCuts,   cl);
         cplexSolver_.setParam(IloCplex::FracCuts,   cl);
      }

      if(parameter_.cliqueCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.cliqueCutLevel_);
         cplexSolver_.setParam(IloCplex::Cliques, cl);
      }
      if(parameter_.coverCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.coverCutLevel_);
         cplexSolver_.setParam(IloCplex::Covers, cl);
      }
      if(parameter_.gubCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.gubCutLevel_);
         cplexSolver_.setParam(IloCplex::GUBCovers, cl);
      }
      if(parameter_.mirCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.mirCutLevel_);
         cplexSolver_.setParam(IloCplex::MIRCuts, cl);
      }
      if(parameter_.iboundCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.iboundCutLevel_);
         cplexSolver_.setParam(IloCplex::ImplBd, cl);
      }
      if(parameter_.flowcoverCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.flowcoverCutLevel_);
         cplexSolver_.setParam(IloCplex::FlowCovers, cl);
      }
      if(parameter_.flowpathCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.flowpathCutLevel_);
         cplexSolver_.setParam(IloCplex::FlowPaths, cl);
      }
      if(parameter_.disjunctCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.disjunctCutLevel_);
         cplexSolver_.setParam(IloCplex::DisjCuts, cl);
      }
      if(parameter_.gomoryCutLevel_ != LPDef::MIP_CUT_DEFAULT){
         const int cl = getCutLevelValue(parameter_.gomoryCutLevel_);
         cplexSolver_.setParam(IloCplex::FracCuts, cl);
      }
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline LPSolverCplex::~LPSolverCplex() {
   cplexEnvironment_.end();
}

inline LPSolverCplex::CplexValueType LPSolverCplex::infinity_impl() {
   return IloInfinity;
}

inline void LPSolverCplex::addContinuousVariables_impl(const CplexIndexType numVariables, const CplexValueType lowerBound, const CplexValueType upperBound) {
   try {
      cplexVariables_.add(IloNumVarArray(cplexEnvironment_, numVariables, lowerBound, upperBound));
      cplexModel_.add(cplexVariables_);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline void LPSolverCplex::addIntegerVariables_impl(const CplexIndexType numVariables, const CplexValueType lowerBound, const CplexValueType upperBound) {
   try {
      cplexVariables_.add(IloNumVarArray(cplexEnvironment_, numVariables, lowerBound, upperBound, ILOINT));
      cplexModel_.add(cplexVariables_);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline void LPSolverCplex::addBinaryVariables_impl(const CplexIndexType numVariables) {
   try {
      cplexVariables_.add(IloNumVarArray(cplexEnvironment_, numVariables, 0, 1, ILOBOOL));
      cplexModel_.add(cplexVariables_);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline void LPSolverCplex::setObjective_impl(const Objective objective) {
   switch(objective) {
      case Minimize: {
         try {
            cplexObjective_.setSense(IloObjective::Minimize);
         } catch(const IloException& e) {
            std::cout << e << std::endl;
            throw std::runtime_error("CPLEX exception");
         }
         break;
      }
      case Maximize: {
         try {
            cplexObjective_.setSense(IloObjective::Maximize);
         } catch(const IloException& e) {
            std::cout << e << std::endl;
            throw std::runtime_error("CPLEX exception");
         }
         break;
      }
      default: {
         throw std::runtime_error("Unknown Objective");
      }
   }
}

inline void LPSolverCplex::setObjectiveValue_impl(const CplexIndexType variable, const CplexValueType value) {
   try {
      cplexObjective_.setLinearCoef(cplexVariables_[variable], value);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

template<class ITERATOR_TYPE>
inline void LPSolverCplex::setObjectiveValue_impl(ITERATOR_TYPE begin, const ITERATOR_TYPE end) {
   const CplexIndexType numObjectiveVariables = std::distance(begin, end);

   IloNumArray objective(cplexEnvironment_, numObjectiveVariables);

   for(CplexIndexType i = 0; i < numObjectiveVariables; ++i) {
      objective[i] = *begin;
      ++begin;
   }

   try {
      cplexObjective_.setLinearCoefs(cplexVariables_, objective);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverCplex::setObjectiveValue_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin) {
   const CplexIndexType numObjectiveVariables = std::distance(variableIDsBegin, variableIDsEnd);

   IloNumArray objective(cplexEnvironment_, numObjectiveVariables);
   IloNumVarArray variables(cplexEnvironment_, numObjectiveVariables);

   for(CplexIndexType i = 0; i < numObjectiveVariables; ++i) {
      objective[i] = *coefficientsBegin;
      variables[i] = cplexVariables_[*variableIDsBegin];
      ++coefficientsBegin;
      ++variableIDsBegin;
   }

   try {
      cplexObjective_.setLinearCoefs(variables, objective);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverCplex::addEqualityConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName) {
   IloRange constraint(cplexEnvironment_, bound, bound, constraintName.c_str());
   while(variableIDsBegin != variableIDsEnd) {
      constraint.setLinearCoef(cplexVariables_[*variableIDsBegin], *coefficientsBegin);
      ++variableIDsBegin;
      ++coefficientsBegin;
   }

   try {
      cplexConstraints_.add(constraint);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverCplex::addLessEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName) {
   IloRange constraint(cplexEnvironment_, -IloInfinity, bound, constraintName.c_str());
   while(variableIDsBegin != variableIDsEnd) {
      constraint.setLinearCoef(cplexVariables_[*variableIDsBegin], *coefficientsBegin);
      ++variableIDsBegin;
      ++coefficientsBegin;
   }

   try {
      cplexConstraints_.add(constraint);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverCplex::addGreaterEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const CplexValueType bound, const std::string& constraintName) {
   IloRange constraint(cplexEnvironment_, bound, IloInfinity, constraintName.c_str());
   while(variableIDsBegin != variableIDsEnd) {
      constraint.setLinearCoef(cplexVariables_[*variableIDsBegin], *coefficientsBegin);
      ++variableIDsBegin;
      ++coefficientsBegin;
   }

   try {
      cplexConstraints_.add(constraint);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline void LPSolverCplex::addConstraintsFinished_impl() {
   try {
      // add constraints to model
      cplexModel_.add(cplexConstraints_);

      // clear constraints as they are now present in the model
      cplexConstraints_.clear();
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline void LPSolverCplex::addConstraintsFinished_impl(CplexTimingType& timing) {
   try {
      const CplexTimingType begin = cplexSolver_.getCplexTime();
      // add constraints to model
      cplexModel_.add(cplexConstraints_);

      // clear constraints as they are now present in the model
      cplexConstraints_.clear();
      const CplexTimingType end = cplexSolver_.getCplexTime();
      timing = end - begin;
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

template <class PARAMETER_TYPE, class PARAMETER_VALUE_TYPE>
inline void LPSolverCplex::setParameter_impl(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value) {
   try {
      cplexSolver_.setParam(parameter, value);
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline bool LPSolverCplex::solve_impl() {
   cplexSolutionValid_ = false;
   try {
      // solve problem
      if(!cplexSolver_.solve()) {
         IloCplex::CplexStatus status = cplexSolver_.getCplexStatus();
         std::cout << "failed to optimize(CPLEX Status: " << status << ")." << std::endl;
         return false;
      } else {
         return true;
      }
   } catch(const IloException& e) {
      std::cout << "caught CPLEX exception: " << e << std::endl;
      return false;
   }
}

inline bool LPSolverCplex::solve_impl(CplexTimingType& timing) {
   cplexSolutionValid_ = false;
   try {
      // solve problem
      const CplexTimingType begin = cplexSolver_.getCplexTime();
      if(!cplexSolver_.solve()) {
         IloCplex::CplexStatus status = cplexSolver_.getCplexStatus();
         std::cout << "failed to optimize(CPLEX Status: " << status << ")." << std::endl;
         const CplexTimingType end = cplexSolver_.getCplexTime();
         timing = end - begin;
         return false;
      } else {
         const CplexTimingType end = cplexSolver_.getCplexTime();
         timing = end - begin;
         return true;
      }
   } catch(const IloException& e) {
      std::cout << "caught CPLEX exception: " << e << std::endl;
      return false;
   }
}

inline LPSolverCplex::CplexSolutionIteratorType LPSolverCplex::solutionBegin_impl() const {
   updateSolution();
   return IloNumArrayIterator(cplexSolution_, 0);
}

inline LPSolverCplex::CplexSolutionIteratorType LPSolverCplex::solutionEnd_impl() const {
   updateSolution();
   return IloNumArrayIterator(cplexSolution_, cplexVariables_.getSize());
}

inline LPSolverCplex::CplexValueType LPSolverCplex::solution_impl(const CplexIndexType variable) const {
   updateSolution();
   try {
      return cplexSolution_[variable];
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline LPSolverCplex::CplexValueType LPSolverCplex::objectiveFunctionValue_impl() const {
   try {
      return cplexSolver_.getObjValue();
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline LPSolverCplex::CplexValueType LPSolverCplex::objectiveFunctionValueBound_impl() const {
   try {
      if(cplexSolver_.isMIP()) {
         return cplexSolver_.getBestObjValue();
      } else {
         return cplexSolver_.getObjValue();
      }
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline void LPSolverCplex::exportModel_impl(const std::string& filename) const {
   try {
      return cplexSolver_.exportModel(filename.c_str());
   } catch(const IloException& e) {
      std::cout << e << std::endl;
      throw std::runtime_error("CPLEX exception");
   }
}

inline void LPSolverCplex::updateSolution() const {
   if(!cplexSolutionValid_) {
      try {
         cplexSolver_.getValues(cplexSolution_, cplexVariables_);
         cplexSolutionValid_ = true;
      } catch(const IloException& e) {
         std::cout << e << std::endl;
         throw std::runtime_error("CPLEX exception");
      }
   }
}

inline int LPSolverCplex::getCutLevelValue(const LPDef::MIP_CUT cutLevel) {
   switch(cutLevel) {
      case LPDef::MIP_CUT_DEFAULT:
      case LPDef::MIP_CUT_AUTO:
         return 0;
      case LPDef::MIP_CUT_OFF:
         return -1;
      case  LPDef::MIP_CUT_ON:
         return 1;
      case LPDef::MIP_CUT_AGGRESSIVE:
         return 2;
      case LPDef::MIP_CUT_VERYAGGRESSIVE:
         return 3;
      default:
         throw std::runtime_error("Unknown Cut level.");
   }
}

} // namespace opengm

#endif /* OPENGM_LP_SOLVER_CPLEX_HXX_ */
