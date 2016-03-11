#ifndef OPENGM_LP_SOLVER_INTERFACE_HXX_
#define OPENGM_LP_SOLVER_INTERFACE_HXX_

#include <opengm/inference/auxiliary/lpdef.hxx>

/*********************
 * class definition *
 *********************/
namespace opengm {

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
class LPSolverInterface {
public:
   // typedefs
   typedef LP_SOLVER_TYPE         SolverType;
   typedef VALUE_TYPE             SolverValueType;
   typedef INDEX_TYPE             SolverIndexType;
   typedef SOLUTION_ITERATOR_TYPE SolverSolutionIteratorType;
   typedef SOLVER_TIMING_TYPE     SolverTimingType;

   // enums
   enum Objective {Minimize, Maximize};

   // parameter
   struct Parameter {
      // constructor
      Parameter();

      // parameter
      int                 numberOfThreads_;
      bool                verbose_;
      double              cutUp_;
      double              epOpt_;
      double              epMrk_;
      double              epRHS_;
      double              epInt_;
      double              epAGap_;
      double              epGap_;
      double              workMem_;
      double              treeMemoryLimit_;
      double              timeLimit_;
      int                 probingLevel_;
      LPDef::LP_SOLVER    rootAlg_;
      LPDef::LP_SOLVER    nodeAlg_;
      LPDef::MIP_EMPHASIS mipEmphasis_;
      LPDef::LP_PRESOLVE  presolve_;
      LPDef::MIP_CUT      cutLevel_;
      LPDef::MIP_CUT      cliqueCutLevel_;
      LPDef::MIP_CUT      coverCutLevel_;
      LPDef::MIP_CUT      gubCutLevel_;
      LPDef::MIP_CUT      mirCutLevel_;
      LPDef::MIP_CUT      iboundCutLevel_;
      LPDef::MIP_CUT      flowcoverCutLevel_;
      LPDef::MIP_CUT      flowpathCutLevel_;
      LPDef::MIP_CUT      disjunctCutLevel_;
      LPDef::MIP_CUT      gomoryCutLevel_;
   };

   // solver infinity value
   static SolverValueType infinity();

   // constructor
   LPSolverInterface(const Parameter& parameter = Parameter());

   // destructor
   ~LPSolverInterface();

   // add Variables
   void addContinuousVariables(const SolverIndexType numVariables, const SolverValueType lowerBound, const SolverValueType upperBound);
   void addIntegerVariables(const SolverIndexType numVariables, const SolverValueType lowerBound, const SolverValueType upperBound);
   void addBinaryVariables(const SolverIndexType numVariables);

   // objective function
   void setObjective(const Objective objective);
   void setObjectiveValue(const SolverIndexType variable, const SolverValueType value);
   template<class ITERATOR_TYPE>
   void setObjectiveValue(ITERATOR_TYPE begin, const ITERATOR_TYPE end);
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void setObjectiveValue(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin);

   // constraints
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addEqualityConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName = "");
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addLessEqualConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName = "");
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addGreaterEqualConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName = "");

   void addConstraintsFinished();
   void addConstraintsFinished(SolverTimingType& timing);

   // parameter
   template <class PARAMETER_TYPE, class PARAMETER_VALUE_TYPE>
   void setParameter(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value);

   // solve
   bool solve();
   bool solve(SolverTimingType& timing);

   // solution
   SolverSolutionIteratorType solutionBegin() const;
   SolverSolutionIteratorType solutionEnd() const;
   SolverValueType solution(const SolverIndexType variable) const;

   SolverValueType objectiveFunctionValue() const;
   SolverValueType objectiveFunctionValueBound() const;

   // model export
   void exportModel(const std::string& filename) const;
protected:
   // storage
   const Parameter parameter_;
};

} // namespace opengm

/***********************
 * class documentation *
 ***********************/
/*! \file lp_solver_interface.hxx
 *  \brief Provides Interface definition for wrapper of LP Solvers like CPLEX
 *         and Gurobi.
 */

/*! \class opengm::LPSolverInterface
 *  \brief Interface definition for wrapper of LP Solvers like CPLEX and Gurobi.
 *
 *  \tparam LP_SOLVER_TYPE The type of the child class which inherits from
 *                         LPSolverInterface.
 *  \tparam VALUE_TYPE The value type used by the LP Solver.
 *  \tparam INDEX_TYPE The index type used by the LP Solver.
 *  \tparam SOLUTION_ITERATOR_TYPE The iterator type which can be used to
 *                                 iterate over the solution of the LP Solver.
 *  \tparam SOLVER_TIMING_TYPE The timing type used by the LP Solver.
 *
 *  \note The interface uses the curiously recurring template pattern (CRTP) to
 *        provide static polymorphism. Hence a child class which inherits from
 *        LPSolverInterface has to provide itself as a template parameter to
 *        LPSolverInterface.
 */

/*! \typedef LPSolverInterface::SolverType
 *  \brief Defines the type of the child class which inherits from
 *         LPSolverInterface.
 */

/*! \typedef LPSolverInterface::SolverValueType
 *  \brief Defines the value type used by the LP Solver.
 */

/*! \typedef LPSolverInterface::SolverIndexType
 *  \brief Defines the index type used by the LP Solver.
 */

/*! \typedef LPSolverInterface::SolverSolutionIteratorType
 *  \brief Defines the iterator type which can be used to iterate over the
 *         solution of the LP Solver.
 */

/*! \typedef LPSolverInterface::SolverTimingType
 *  \brief Defines the timing type used by the LP Solver.
 */

/*! \enum opengm::LPSolverInterface::Objective
 *  \brief This enum defines the type of the objective. It is used to select
 *   either to minimize or to maxime the objective function.
 */

/*! \var LPSolverInterface::Objective LPSolverInterface::Minimize
 *  \brief Objective function will be minimized.
 */

/*! \var LPSolverInterface::Objective LPSolverInterface::Maximize
 *  \brief Objective function will be maximized.
 */

/*! \class opengm::LPSolverInterface::Parameter
 *  \brief Parameter class provides options to modify LP Solver behavior.
 *
 *  \note
 *        -# Not all LP Solver might provide support for all parameter options.
 *        -# Default values are taken from class LPDef.
 */

/*! \fn LPSolverInterface::Parameter::Parameter()
 *  \brief Default constructor of class Parameter. Sets default values provided
 *         by class LPDef for all options.
 */

/*! \var LPSolverInterface::Parameter::numberOfThreads_
 *  \brief The number of threads used for Optimization (0 = autoselect).
 */

/*! \var LPSolverInterface::Parameter::verbose_
 *  \brief Enable verbose output if set to true.
 */

/*! \var LPSolverInterface::Parameter::cutUp_
 *  \brief Upper cutoff tolerance.
 */

/*! \var LPSolverInterface::Parameter::epOpt_
 *  \brief Optimality tolerance.
 */

/*! \var LPSolverInterface::Parameter::epMrk_
 *  \brief Markowitz tolerance.
 */

/*! \var LPSolverInterface::Parameter::epRHS_
 *  \brief Feasibility tolerance.
 */

/*! \var LPSolverInterface::Parameter::epInt_
 *  \brief Amount by which an integer variable can differ from an integer.
 */

/*! \var LPSolverInterface::Parameter::epAGap_
 *  \brief Absolute MIP gap tolerance.
 */

/*! \var LPSolverInterface::Parameter::epGap_
 *  \brief Relative MIP gap tolerance.
 */

/*! \var LPSolverInterface::Parameter::workMem_
 *  \brief Maximal amount of memory in MB used for workspace.
 */

/*! \var LPSolverInterface::Parameter::treeMemoryLimit_
 *  \brief Maximal amount of memory in MB used for tree.
 */

/*! \var LPSolverInterface::Parameter::timeLimit_
 *  \brief Maximal time in seconds the solver has.
 */

/*! \var LPSolverInterface::Parameter::probingLevel_
 *  \brief Amount of probing on variables to be performed before MIP branching.
 */

/*! \var LPSolverInterface::Parameter::rootAlg_
 *  \brief Select which algorithm is used to solve continuous models or to solve
 *         the root relaxation of a MIP.
 */

/*! \var LPSolverInterface::Parameter::nodeAlg_
 *  \brief Select which algorithm is used to solve the subproblems in a MIP
 *         after the initial relaxation has been solved.
 */

/*! \var LPSolverInterface::Parameter::mipEmphasis_
 *  \brief Controls trade-offs between speed, feasibility, optimality,
 *         and moving bounds in a MIP.
 */

/*! \var LPSolverInterface::Parameter::presolve_
 *  \brief Controls how aggressive presolve is performed during preprocessing.
 */

/*! \var LPSolverInterface::Parameter::cutLevel_
 *  \brief Determines whether or not to generate cuts for the problem and how
 *         aggressively (will be overruled by specific ones).
 */

/*! \var LPSolverInterface::Parameter::cliqueCutLevel_
 *  \brief Determines whether or not to generate clique cuts for the problem and
 *         how aggressively.
 */

/*! \var LPSolverInterface::Parameter::coverCutLevel_
 *  \brief Determines whether or not to generate cover cuts for the problem and
 *         how aggressively.
 */

/*! \var LPSolverInterface::Parameter::gubCutLevel_
 *  \brief Determines whether or not to generate generalized upper bound (GUB)
 *         cuts for the problem and how aggressively.
 */

/*! \var LPSolverInterface::Parameter::mirCutLevel_
 *  \brief Determines whether or not mixed integer rounding (MIR) cuts should be
 *         generated for the problem and how aggressively.
 */

/*! \var LPSolverInterface::Parameter::iboundCutLevel_
 *  \brief Determines whether or not to generate implied bound cuts for the
 *         problem and how aggressively.
 */

/*! \var LPSolverInterface::Parameter::flowcoverCutLevel_
 *  \brief Determines whether or not to generate flow cover cuts for the problem
 *         and how aggressively.
 */

/*! \var LPSolverInterface::Parameter::flowpathCutLevel_
 *  \brief Determines whether or not to generate flow path cuts for the problem
 *         and how aggressively.
 */

/*! \var LPSolverInterface::Parameter::disjunctCutLevel_
 *  \brief Determines whether or not to generate disjunctive cuts for the
 *         problem and how aggressively.
 */

/*! \var LPSolverInterface::Parameter::gomoryCutLevel_
 *  \brief Determines whether or not to generate gomory fractional cuts for the
 *         problem and how aggressively.
 */

/*! \fn static LPSolverInterface::SolverValueType LPSolverInterface::infinity()
 *  \brief Get the value which is used by the LP Solver to represent infinity.
 *
 *  \note The Solver class has to provide the corresponding infinity_impl()
 *        method.
 */

/*! \fn LPSolverInterface::LPSolverInterface()
 *  \brief Default constructor of class LPSolverInterface.
 */

/*! \fn LPSolverInterface::~LPSolverInterface()
 *  \brief Default destructor of class LPSolverInterface.
 */

/*! \fn void LPSolverInterface::addContinuousVariables(const SolverIndexType numVariables, const SolverValueType lowerBound, const SolverValueType upperBound)
 *  \brief Add new continuous variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *  \param[in] lowerBound The lower bound for the new Variables.
 *  \param[in] upperBound The upper bound for the new Variables.
 *
 *  \note The Solver class has to provide the corresponding
 *        addContinuousVariables_impl() method.
 */

/*! \fn void LPSolverInterface::addIntegerVariables(const SolverIndexType numVariables, const SolverValueType lowerBound, const SolverValueType upperBound)
 *  \brief Add new integer variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *  \param[in] lowerBound The lower bound for the new Variables.
 *  \param[in] upperBound The upper bound for the new Variables.
 *
 *  \note The Solver class has to provide the corresponding
 *        addIntegerVariables_impl() method.
 */

/*! \fn void LPSolverInterface::addBinaryVariables(const SolverIndexType numVariables)
 *  \brief Add new binary variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *
 *  \note The Solver class has to provide the corresponding
 *        addBinaryVariables_impl() method.
 */

/*! \fn void LPSolverInterface::setObjective(const Objective objective)
 *  \brief Set objective to minimize or maximize.
 *
 *  \param[in] objective The new objective.
 *
 *  \note The Solver class has to provide the corresponding setObjective_impl()
 *        method.
 */

/*! \fn void LPSolverInterface::setObjectiveValue(const SolverIndexType variable, const SolverValueType value)
 *  \brief Set the coefficient of a variable in the objective function.
 *
 *  \param[in] variable The index of the variable.
 *  \param[in] value The value which will be added to the coefficient of the
 *             variable in the objective function.
 *
 *  \note The Solver class has to provide the corresponding
 *        setObjectiveValue_impl() method.
 */

/*! \fn void LPSolverInterface::setObjectiveValue(ITERATOR_TYPE begin, const ITERATOR_TYPE end)
 *  \brief Set the coefficients of all variables in the objective
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
 *  \note The Solver class has to provide the corresponding
 *        setObjectiveValue_impl() method.
 */

/*! \fn void LPSolverInterface::setObjectiveValue(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin)
 *  \brief Set the coefficients of selected variables in the objective
 *         function.
 *  \tparam VARIABLES_ITERATOR_TYPE Iterator type used to iterate over the
 *                                  indices of the variables.
 *  \tparam COEFFICIENTS_ITERATOR_TYPE Iterator type used to iterate over the
 *                                  coefficients of the variables which will be
 *                                  set for the objective function.
 *
 *  \param[in] variableIDsBegin Iterator pointing to the begin of the sequence
 *                              of indices of the variables.
 *  \param[in] variableIDsEnd Iterator pointing to the end of the sequence of
 *                            indices of the variables.
 *  \param[in] coefficientsBegin Iterator pointing to the begin of the sequence
 *                              of values which will be set as the
 *                              coefficients of the objective function.
 *
 *  \note The Solver class has to provide the corresponding
 *        setObjectiveValue_impl() method.
 */

/*! \fn void LPSolverInterface::addEqualityConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName = "")
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
 *        -# The Solver class has to provide the corresponding
 *           addEqualityConstraint_impl() method.
 *        -# To increase performance for adding multiple constraints to the
 *           model, all constraints added via
 *           LPSolverInterface::addEqualityConstraint,
 *           LPSolverInterface::addLessEqualConstraint and
 *           LPSolverInterface::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverInterface::addConstraintsFinished is called.
 */

/*! \fn void LPSolverInterface::addLessEqualConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName = "");
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
 *        -# The Solver class has to provide the corresponding
 *           addLessEqualConstraint_impl() method.
 *        -# To increase performance for adding multiple constraints to the
 *           model, all constraints added via
 *           LPSolverInterface::addEqualityConstraint,
 *           LPSolverInterface::addLessEqualConstraint and
 *           LPSolverInterface::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverInterface::addConstraintsFinished is called.
 */

/*! \fn void LPSolverInterface::addGreaterEqualConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName = "")
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
 *        -# The Solver class has to provide the corresponding
 *           addGreaterEqualConstraint_impl() method.
 *        -# To increase performance for adding multiple constraints to the
 *           model, all constraints added via
 *           LPSolverInterface::addEqualityConstraint,
 *           LPSolverInterface::addLessEqualConstraint and
 *           LPSolverInterface::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverInterface::addConstraintsFinished is called.
 */

/*! \fn void LPSolverInterface::addConstraintsFinished()
 *  \brief Join all constraints added via
 *         LPSolverInterface::addEqualityConstraint,
 *         LPSolverInterface::addLessEqualConstraint and
 *         LPSolverInterface::addGreaterEqualConstraint to the model.
 *
 *  \note The Solver class has to provide the corresponding
 *        addConstraintsFinished_impl() method.
 */

/*! \fn void LPSolverInterface::addConstraintsFinished(SolverTimingType& timing)
 *  \brief Join all constraints added via
 *         LPSolverInterface::addEqualityConstraint,
 *         LPSolverInterface::addLessEqualConstraint and
 *         LPSolverInterface::addGreaterEqualConstraint to the model.
 *
 *  \param[out] timing Returns the time needed to join all constraints to the
 *                     model.
 *
 *  \note The Solver class has to provide the corresponding
 *        addConstraintsFinished_impl() method.
 */

/*! \fn void LPSolverInterface::setParameter(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value)
 *  \brief Set Solver parameter.
 *
 *  \tparam PARAMETER_VALUE_TYPE The type of the parameter.
 *  \tparam VALUE_TYPE The type of the value.
 *
 *  \param[in] parameter The Solver parameter.
 *  \param[in] value The new value to which the parameter will be set.
 *
 *  \note The Solver class has to provide the corresponding setParameter_impl()
 *        method.
 */

/*! \fn bool LPSolverInterface::solve()
 *  \brief Solve the current model.
 *
 *  \return True if solving the model finished successfully, false if the Solver
 *          was not able to solve the model.
 *
 *  \note The Solver class has to provide the corresponding solve_impl()
 *        method.
 */

/*! \fn bool LPSolverInterface::solve(SolverTimingType& timing)
 *  \brief Solve the current model and measure solving time.
 *
 *  \param[out] timing The time the solver needed to solve the problem.
 *
 *  \return True if solving the model finished successfully, false if the solver
 *          was not able to solve the model.
 *
 *  \note The Solver class has to provide the corresponding solve_impl()
 *        method.
 */

/*! \fn LPSolverInterface::SolverSolutionIteratorType LPSolverInterface::solutionBegin() const
 *  \brief Get an iterator which is pointing to the begin of the solution
 *         computed by the Solver.
 *
 *  \return Iterator pointing to the begin of the solution.
 *
 *  \note The Solver class has to provide the corresponding solutionBegin_impl()
 *        method.
 */

/*! \fn LPSolverInterface::SolverSolutionIteratorType LPSolverInterface::solutionEnd() const
 *  \brief Get an iterator which is pointing to the end of the solution computed
 *         by the Solver.
 *
 *  \return Iterator pointing to the begin of the solution.
 *
 *  \note The Solver class has to provide the corresponding solutionEnd_impl()
 *        method.
 */

/*! \fn LPSolverInterface::SolverValueType LPSolverInterface::solution(const SolverIndexType variable) const;
 *  \brief Get the solution value of a variable computed by the Solver.
 *
 *  \param[in] variable Index of the variable for which the solution value is
 *                      requested.
 *
 *  \return Solution value of the selected variable.
 *
 *  \note The Solver class has to provide the corresponding solution_impl()
 *        method.
 */

/*! \fn LPSolverInterface::SolverValueType LPSolverInterface::objectiveFunctionValue() const;
 *  \brief Get the objective function value from the Solver.
 *
 *  \return Objective function value.
 *
 *  \note The Solver class has to provide the corresponding objectiveFunctionValue_impl()
 *        method.
 */

/*! \fn LPSolverInterface::SolverValueType LPSolverInterface::objectiveFunctionValueBound() const
 *  \brief Get the best known bound for the optimal solution of the current
 *         model.
 *
 *  \return The bound for the current model.
 *
 *  \note The Solver class has to provide the corresponding
 *        objectiveFunctionValueBound_impl() method.
 */

/*! \fn void LPSolverInterface::exportModel(const std::string& filename) const
 *  \brief Export model to file.
 *
 *  \param[in] filename The name of the file where the model will be stored.
 *
 *  \note The Solver class has to provide the corresponding exportModel_impl()
 *        method.
 */

/*! \var LPSolverInterface::parameter_
 *  \brief Storage for parameter.
 */
/******************
 * implementation *
 ******************/
namespace opengm {

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::Parameter::Parameter()
   : numberOfThreads_(LPDef::default_numberOfThreads_),
     verbose_(LPDef::default_verbose_), 
     cutUp_(LPDef::default_cutUp_),
     epOpt_(LPDef::default_epOpt_), 
     epMrk_(LPDef::default_epMrk_),
     epRHS_(LPDef::default_epRHS_), 
     epInt_(LPDef::default_epInt_),
     epAGap_(LPDef::default_epAGap_), 
     epGap_(LPDef::default_epGap_),
     workMem_(LPDef::default_workMem_),
     treeMemoryLimit_(LPDef::default_treeMemoryLimit_),
     timeLimit_(LPDef::default_timeLimit_),
     probingLevel_(LPDef::default_probingLevel_),
     rootAlg_(LPDef::default_rootAlg_), nodeAlg_(LPDef::default_nodeAlg_),
     mipEmphasis_(LPDef::default_mipEmphasis_),
     presolve_(LPDef::default_presolve_), cutLevel_(LPDef::default_cutLevel_),
     cliqueCutLevel_(LPDef::default_cliqueCutLevel_),
     coverCutLevel_(LPDef::default_coverCutLevel_),
     gubCutLevel_(LPDef::default_gubCutLevel_),
     mirCutLevel_(LPDef::default_mirCutLevel_),
     iboundCutLevel_(LPDef::default_iboundCutLevel_),
     flowcoverCutLevel_(LPDef::default_flowcoverCutLevel_),
     flowpathCutLevel_(LPDef::default_flowpathCutLevel_),
     disjunctCutLevel_(LPDef::default_disjunctCutLevel_),
     gomoryCutLevel_(LPDef::default_gomoryCutLevel_) {

}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline typename LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::SolverValueType LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::infinity() {
   return SolverType::infinity_impl();
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::LPSolverInterface(const Parameter& parameter)
   : parameter_(parameter) {

}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::~LPSolverInterface() {

}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::addContinuousVariables(const SolverIndexType numVariables, const SolverValueType lowerBound, const SolverValueType upperBound) {
   static_cast<SolverType*>(this)->addContinuousVariables_impl(numVariables, lowerBound, upperBound);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::addIntegerVariables(const SolverIndexType numVariables, const SolverValueType lowerBound, const SolverValueType upperBound) {
   static_cast<SolverType*>(this)->addIntegerVariables_impl(numVariables, lowerBound, upperBound);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::addBinaryVariables(const SolverIndexType numVariables) {
   static_cast<SolverType*>(this)->addBinaryVariables_impl(numVariables);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::setObjective(const Objective objective) {
   static_cast<SolverType*>(this)->setObjective_impl(objective);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::setObjectiveValue(const SolverIndexType variable, const SolverValueType value) {
   static_cast<SolverType*>(this)->setObjectiveValue_impl(variable, value);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
template<class ITERATOR_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::setObjectiveValue(ITERATOR_TYPE begin, const ITERATOR_TYPE end) {
   static_cast<SolverType*>(this)->setObjectiveValue_impl(begin, end);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::setObjectiveValue(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin) {
   static_cast<SolverType*>(this)->setObjectiveValue_impl(variableIDsBegin, variableIDsEnd, coefficientsBegin);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::addEqualityConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName) {
   static_cast<SolverType*>(this)->addEqualityConstraint_impl(variableIDsBegin, variableIDsEnd, coefficientsBegin, bound, constraintName);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::addLessEqualConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName) {
   static_cast<SolverType*>(this)->addLessEqualConstraint_impl(variableIDsBegin, variableIDsEnd, coefficientsBegin, bound, constraintName);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::addGreaterEqualConstraint(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const SolverValueType bound, const std::string& constraintName) {
   static_cast<SolverType*>(this)->addGreaterEqualConstraint_impl(variableIDsBegin, variableIDsEnd, coefficientsBegin, bound, constraintName);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::addConstraintsFinished() {
   static_cast<SolverType*>(this)->addConstraintsFinished_impl();
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::addConstraintsFinished(SolverTimingType& timing) {
   static_cast<SolverType*>(this)->addConstraintsFinished_impl(timing);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
template <class PARAMETER_TYPE, class PARAMETER_VALUE_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::setParameter(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value) {
   static_cast<SolverType*>(this)->setParameter_impl(parameter, value);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline bool LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::solve() {
   return static_cast<SolverType*>(this)->solve_impl();
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline bool LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::solve(SolverTimingType& timing) {
   return static_cast<SolverType*>(this)->solve_impl(timing);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline typename LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::SolverSolutionIteratorType LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::solutionBegin() const {
   return static_cast<const SolverType*>(this)->solutionBegin_impl();
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline typename LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::SolverSolutionIteratorType LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::solutionEnd() const {
   return static_cast<const SolverType*>(this)->solutionEnd_impl();
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline typename LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::SolverValueType LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::solution(const SolverIndexType variable) const {
   return static_cast<const SolverType*>(this)->solution_impl(variable);
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline typename LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::SolverValueType LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::objectiveFunctionValue() const {
   return static_cast<const SolverType*>(this)->objectiveFunctionValue_impl();
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline typename LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::SolverValueType LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::objectiveFunctionValueBound() const {
   return static_cast<const SolverType*>(this)->objectiveFunctionValueBound_impl();
}

template <class LP_SOLVER_TYPE, class VALUE_TYPE, class INDEX_TYPE, class SOLUTION_ITERATOR_TYPE, class SOLVER_TIMING_TYPE>
inline void LPSolverInterface<LP_SOLVER_TYPE, VALUE_TYPE, INDEX_TYPE, SOLUTION_ITERATOR_TYPE, SOLVER_TIMING_TYPE>::exportModel(const std::string& filename) const {
   static_cast<const SolverType*>(this)->exportModel_impl(filename);
}

} // namespace opengm

#endif /* OPENGM_LP_SOLVER_INTERFACE_HXX_ */
