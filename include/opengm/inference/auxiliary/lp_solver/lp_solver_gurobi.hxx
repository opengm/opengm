#ifndef OPENGM_LP_SOLVER_GUROBI_HXX_
#define OPENGM_LP_SOLVER_GUROBI_HXX_

#include <gurobi_c++.h>

#include <opengm/inference/auxiliary/lpdef.hxx>
#include <opengm/inference/auxiliary/lp_solver/lp_solver_interface.hxx>
#include <opengm/utilities/timer.hxx>

/*********************
 * class definition *
 *********************/
namespace opengm {

class LPSolverGurobi : public LPSolverInterface<LPSolverGurobi, double, int, std::vector<double>::const_iterator, double> {
public:
   // typedefs
   typedef double                                       GurobiValueType;
   typedef int                                          GurobiIndexType;
   typedef std::vector<GurobiValueType>::const_iterator GurobiSolutionIteratorType;
   typedef double                                       GurobiTimingType;

   typedef LPSolverInterface<LPSolverGurobi, GurobiValueType, GurobiIndexType, GurobiSolutionIteratorType, GurobiTimingType> LPSolverBaseClass;

   // constructor
   LPSolverGurobi(const Parameter& parameter = Parameter());

   // destructor
   ~LPSolverGurobi();

protected:
   // Storage for Gurobi variables
   GRBEnv                               gurobiEnvironment_;
   mutable GRBModel                     gurobiModel_;         // model is mutable as GRBModel::write() is not marked as const. However exporting a Model to file is expected to not change the model itself. Is the missing const intended by Gurobi?
   std::vector<GRBVar>                  gurobiVariables_;
   mutable std::vector<GurobiValueType> gurobiSolution_;
   mutable bool                         gurobiSolutionValid_;

   // methods for class LPSolverInterface
   // Gurobi infinity value
   static GurobiValueType infinity_impl();

   // add Variables
   void addContinuousVariables_impl(const GurobiIndexType numVariables, const GurobiValueType lowerBound, const GurobiValueType upperBound);
   void addIntegerVariables_impl(const GurobiIndexType numVariables, const GurobiValueType lowerBound, const GurobiValueType upperBound);
   void addBinaryVariables_impl(const GurobiIndexType numVariables);

   // objective function
   void setObjective_impl(const Objective objective);
   void setObjectiveValue_impl(const GurobiIndexType variable, const GurobiValueType value);
   template<class ITERATOR_TYPE>
   void setObjectiveValue_impl(ITERATOR_TYPE begin, const ITERATOR_TYPE end);
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void setObjectiveValue_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin);

   // constraints
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addEqualityConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName = "");
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addLessEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName = "");
   template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
   void addGreaterEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName = "");

   void addConstraintsFinished_impl();
   void addConstraintsFinished_impl(GurobiTimingType& timing);

   // parameter
   template <class PARAMETER_TYPE, class PARAMETER_VALUE_TYPE>
   void setParameter_impl(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value);

   // solve
   bool solve_impl();
   bool solve_impl(GurobiTimingType& timing);

   // solution
   GurobiSolutionIteratorType solutionBegin_impl() const;
   GurobiSolutionIteratorType solutionEnd_impl() const;
   GurobiValueType solution_impl(const GurobiIndexType variable) const;

   GurobiValueType objectiveFunctionValue_impl() const;
   GurobiValueType objectiveFunctionValueBound_impl() const;

   // model export
   void exportModel_impl(const std::string& filename) const;

   // helper functions
   void updateSolution() const;
   static int getCutLevelValue(const LPDef::MIP_CUT cutLevel);

   // friend
   friend class LPSolverInterface<LPSolverGurobi, GurobiValueType, GurobiIndexType, GurobiSolutionIteratorType, GurobiTimingType>;
};

} // namespace opengm

/***********************
 * class documentation *
 ***********************/
/*! \file lp_solver_gurobi.hxx
 *  \brief Provides wrapper class for LP Solver Gurobi.
 */

/*! \class opengm::LPSolverGurobi
 *  \brief Wrapper class for the Gurobi optimizer.
 *
 *  \note <a href="http://www.gurobi.com/index">Gurobi</a> is a
 *        commercial product that is free for academical use.
 */

/*! \typedef LPSolverGurobi::GurobiValueType
 *  \brief Defines the value type used by Gurobi.
 */

/*! \typedef LPSolverGurobi::GurobiIndexType
 *  \brief Defines the index type used by Gurobi.
 */

/*! \typedef LPSolverGurobi::GurobiSolutionIteratorType
 *  \brief Defines the iterator type which can be used to iterate over the
 *         solution of Gurobi.
 */

/*! \typedef LPSolverGurobi::GurobiTimingType
 *  \brief Defines the timing type used by Gurobi.
 */

/*! \typedef LPSolverGurobi::LPSolverBaseClass
 *  \brief Defines the type of the base class.
 */

/*! \fn LPSolverGurobi::LPSolverGurobi(const Parameter& parameter = Parameter())
 *  \brief Default constructor for LPSolverGurobi.
 *
 *  \param[in] parameter Settings for the Gurobi solver.
 */

/*! \fn LPSolverGurobi::~LPSolverGurobi()
 *  \brief Destructor for LPSolverGurobi.
 */

/*! \var LPSolverGurobi::gurobiEnvironment_
 *  \brief The Gurobi environment.
 */

/*! \var LPSolverGurobi::gurobiModel_
 *  \brief The Gurobi model of the LP/MIP problem.
 */

/*! \var LPSolverGurobi::gurobiVariables_
 *  \brief The variables which are present in the model.
 */

/*! \var LPSolverGurobi::gurobiSolution_
 *  \brief Storage for the solution computed by Gurobi.
 */

/*! \var LPSolverGurobi::gurobiSolutionValid_
 *  \brief Tell if the currently stored solution is valid.
 */

/*! \fn static LPSolverGurobi::GurobiValueType LPSolverGurobi::infinity_impl()
 *  \brief Get the value which is used by Gurobi to represent infinity.
 *
 *  \note Implementation for base class LPSolverInterface::infinity method.
 */

/*! \fn void LPSolverGurobi::addContinuousVariables_impl(const GurobiIndexType numVariables, const GurobiValueType lowerBound, const GurobiValueType upperBound)
 *  \brief Add new continuous variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *  \param[in] lowerBound The lower bound for the new Variables.
 *  \param[in] upperBound The upper bound for the new Variables.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::addContinuousVariables method.
 */

/*! \fn void LPSolverGurobi::addIntegerVariables_impl(const GurobiIndexType numVariables, const GurobiValueType lowerBound, const GurobiValueType upperBound)
 *  \brief Add new integer variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *  \param[in] lowerBound The lower bound for the new Variables.
 *  \param[in] upperBound The upper bound for the new Variables.
 *
 *  \note Implementation for base class LPSolverInterface::addIntegerVariables
 *        method.
 */

/*! \fn void LPSolverGurobi::addBinaryVariables_impl(const GurobiIndexType numVariables)
 *  \brief Add new binary variables to the model.
 *
 *  \param[in] numVariables The number of new Variables.
 *
 *  \note Implementation for base class LPSolverInterface::addBinaryVariables
 *        method.
 */

/*! \fn void LPSolverGurobi::setObjective_impl(const Objective objective)
 *  \brief Set objective to minimize or maximize.
 *
 *  \param[in] objective The new objective.
 *
 *  \note Implementation for base class LPSolverInterface::setObjective_impl
 *        method.
 */

/*! \fn void LPSolverGurobi::setObjectiveValue_impl(const GurobiIndexType variable, const GurobiValueType value)
 *  \brief Set the coefficient of a variable in the objective function.
 *
 *  \param[in] variable The index of the variable.
 *  \param[in] value The value which will be set as the coefficient of the
 *             variable in the objective function.
 *
 *  \note Implementation for base class LPSolverInterface::setObjectiveValue
 *        method.
 */

/*! \fn void LPSolverGurobi::setObjectiveValue_impl(ITERATOR_TYPE begin, const ITERATOR_TYPE end)
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

/*! \fn void LPSolverGurobi::setObjectiveValue_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin)
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

/*! \fn void LPSolverGurobi::addEqualityConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName = "")
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
 *           LPSolverGurobi::addEqualityConstraint,
 *           LPSolverGurobi::addLessEqualConstraint and
 *           LPSolverGurobi::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverGurobi::addConstraintsFinished is called.
 */

/*! \fn void LPSolverGurobi::addLessEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName = "");
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
 *           LPSolverGurobi::addEqualityConstraint,
 *           LPSolverGurobi::addLessEqualConstraint and
 *           LPSolverGurobi::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverGurobi::addConstraintsFinished is called.
 */

/*! \fn void LPSolverGurobi::addGreaterEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName = "")
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
 *           LPSolverGurobi::addEqualityConstraint,
 *           LPSolverGurobi::addLessEqualConstraint and
 *           LPSolverGurobi::addGreaterEqualConstraint are stored in a puffer
 *           and will be added to the model all at once when the function
 *           LPSolverGurobi::addConstraintsFinished is called.
 */

/*! \fn void LPSolverGurobi::addConstraintsFinished_impl()
 *  \brief Join all constraints added via LPSolverGurobi::addEqualityConstraint,
 *         LPSolverGurobi::addLessEqualConstraint and
 *         LPSolverGurobi::addGreaterEqualConstraint to the model.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::addConstraintsFinished method.
 */

/*! \fn void LPSolverGurobi::addConstraintsFinished_impl(GurobiTimingType& timing)
 *  \brief Join all constraints added via LPSolverGurobi::addEqualityConstraint,
 *         LPSolverGurobi::addLessEqualConstraint and
 *         LPSolverGurobi::addGreaterEqualConstraint to the model.
 *
 *  \param[out] timing Returns the time needed to join all constraints into the
 *                     model.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::addConstraintsFinished method.
 */

/*! \fn void LPSolverGurobi::setParameter_impl(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value)
 *  \brief Set Gurobi parameter.
 *
 *  \tparam PARAMETER_VALUE_TYPE The type of the parameter.
 *  \tparam VALUE_TYPE The type of the value.
 *
 *  \param[in] parameter The Gurobi parameter.
 *  \param[in] value The new value to which the parameter will be set.
 *
 *  \note Implementation for base class LPSolverInterface::setParameter method.
 */

/*! \fn bool LPSolverGurobi::solve_impl()
 *  \brief Solve the current model.
 *
 *  \return True if solving the model finished successfully, false if Gurobi was
 *          not able to solve the model.
 *
 *  \note Implementation for base class LPSolverInterface::solve method.
 */

/*! \fn bool LPSolverGurobi::solve_impl(GurobiTimingType& timing)
 *  \brief Solve the current model and measure solving time.
 *
 *  \param[out] timing The time Gurobi needed to solve the problem.
 *
 *  \return True if solving the model finished successfully, false if Gurobi was
 *          not able to solve the model.
 *
 *  \note Implementation for base class LPSolverInterface::solve method.
 */

/*! \fn LPSolverGurobi::GurobiSolutionIteratorType LPSolverGurobi::solutionBegin_impl() const
 *  \brief Get an iterator which is pointing to the begin of the solution
 *         computed by Gurobi.
 *
 *  \return Iterator pointing to the begin of the solution.
 *
 *  \note Implementation for base class LPSolverInterface::solutionBegin method.
 */

/*! \fn LPSolverGurobi::GurobiSolutionIteratorType LPSolverGurobi::solutionEnd_impl() const
 *  \brief Get an iterator which is pointing to the end of the solution computed
 *         by Gurobi.
 *
 *  \return Iterator pointing to the begin of the solution.
 *
 *  \note Implementation for base class LPSolverInterface::solutionEnd method.
 */

/*! \fn LPSolverGurobi::GurobiValueType LPSolverGurobi::solution_impl(const GurobiIndexType variable) const;
 *  \brief Get the solution value of a variable computed by Gurobi.
 *
 *  \param[in] variable Index of the variable for which the solution value is
 *                      requested.
 *
 *  \return Solution value of the selected variable.
 *
 *  \note Implementation for base class LPSolverInterface::solution method.
 */

/*! \fn LPSolverGurobi::GurobiValueType LPSolverGurobi::objectiveFunctionValue_impl() const;
 *  \brief Get the objective function value from Gurobi.
 *
 *  \return Objective function value.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::objectiveFunctionValue method.
 */

/*! \fn LPSolverGurobi::GurobiValueType LPSolverGurobi::objectiveFunctionValueBound_impl() const
 *  \brief Get the best known bound for the optimal solution of the current
 *         model.
 *
 *  \return The bound for the current model.
 *
 *  \note Implementation for base class
 *        LPSolverInterface::objectiveFunctionValueBound method.
 */

/*! \fn void LPSolverGurobi::exportModel_impl(const std::string& filename) const
 *  \brief Export model to file.
 *
 *  \param[in] filename The name of the file where the model will be stored.
 *
 *  \note Implementation for base class LPSolverInterface::exportModel method.
 */

/*! \fn void LPSolverGurobi::updateSolution()
 *  \brief Update solution if required.
 */

/*! \fn static int LPSolverGurobi::getCutLevelValue(const LPDef::MIP_CUT cutLevel);
 *  \brief Translate LPDef::MIP_CUT into corresponding Gurobi int value.
 *
 *  \return Integer value corresponding to the LPDef::MIP_CUT parameter.
 */


/******************
 * implementation *
 ******************/
namespace opengm {

inline LPSolverGurobi::LPSolverGurobi(const Parameter& parameter)
   : LPSolverBaseClass(parameter), gurobiEnvironment_(),
     gurobiModel_(gurobiEnvironment_), gurobiVariables_(), gurobiSolution_(),
     gurobiSolutionValid_(false) {
   // set parameter
   try {
      // multi-threading options
      gurobiModel_.getEnv().set(GRB_IntParam_Threads, parameter_.numberOfThreads_);

      // verbose options
      if(!parameter_.verbose_) {
         gurobiModel_.getEnv().set(GRB_IntParam_OutputFlag, 0);
         gurobiModel_.getEnv().set(GRB_IntParam_TuneOutput, 0);
         gurobiModel_.getEnv().set(GRB_IntParam_LogToConsole, 0);
      }

      // set hints
      // CutUp is missing http://www.gurobi.com/resources/switching-to-gurobi/switching-from-cplex#setting

      // tolerance settings
      //gurobiModel_.getEnv().set(GRB_DoubleParam_Cutoff,         parameter_.cutUp_);  // Optimality Tolerance
      gurobiModel_.getEnv().set(GRB_DoubleParam_OptimalityTol,  parameter_.epOpt_);  // Optimality Tolerance
      gurobiModel_.getEnv().set(GRB_DoubleParam_IntFeasTol,     parameter_.epInt_);  // amount by which an integer variable can differ from an integer
      gurobiModel_.getEnv().set(GRB_DoubleParam_MIPGapAbs,      parameter_.epAGap_); // Absolute MIP gap tolerance
      gurobiModel_.getEnv().set(GRB_DoubleParam_MIPGap,         parameter_.epGap_);  // Relative MIP gap tolerance
      gurobiModel_.getEnv().set(GRB_DoubleParam_FeasibilityTol, parameter_.epRHS_);
      gurobiModel_.getEnv().set(GRB_DoubleParam_MarkowitzTol,   parameter_.epMrk_);

      // memory setting
      // missing

      // time limit
      gurobiModel_.getEnv().set(GRB_DoubleParam_TimeLimit, parameter_.timeLimit_);

      // Root Algorithm
      switch(parameter_.rootAlg_) {
         case LPDef::LP_SOLVER_AUTO: {
            gurobiModel_.getEnv().set(GRB_IntParam_Method, -1);
            break;
         }
         case LPDef::LP_SOLVER_PRIMAL_SIMPLEX: {
            gurobiModel_.getEnv().set(GRB_IntParam_Method, 0);
            break;
         }
         case LPDef::LP_SOLVER_DUAL_SIMPLEX: {
            gurobiModel_.getEnv().set(GRB_IntParam_Method, 1);
            break;
         }
         case LPDef::LP_SOLVER_NETWORK_SIMPLEX: {
            throw std::runtime_error("Gurobi does not support Network Simplex");
            break;
         }
         case LPDef::LP_SOLVER_BARRIER: {
            gurobiModel_.getEnv().set(GRB_IntParam_Method, 2);
            break;
         }
         case LPDef::LP_SOLVER_SIFTING: {
            gurobiModel_.getEnv().set(GRB_IntParam_Method, 1);
            gurobiModel_.getEnv().set(GRB_IntParam_SiftMethod, 1);
            break;
         }
         case LPDef::LP_SOLVER_CONCURRENT: {
            gurobiModel_.getEnv().set(GRB_IntParam_Method, 4);
            break;
         }
         default: {
            throw std::runtime_error("Unknown Root Algorithm");
         }
      }

      // Node Algorithm
      switch(parameter_.nodeAlg_) {
         case LPDef::LP_SOLVER_AUTO: {
            gurobiModel_.getEnv().set(GRB_IntParam_NodeMethod, 1);
            break;
         }
         case LPDef::LP_SOLVER_PRIMAL_SIMPLEX: {
            gurobiModel_.getEnv().set(GRB_IntParam_NodeMethod, 0);
            break;
         }
         case LPDef::LP_SOLVER_DUAL_SIMPLEX: {
            gurobiModel_.getEnv().set(GRB_IntParam_NodeMethod, 1);
            break;
         }
         case LPDef::LP_SOLVER_NETWORK_SIMPLEX: {
            throw std::runtime_error("Gurobi does not support Network Simplex");
            break;
         }
         case LPDef::LP_SOLVER_BARRIER: {
            gurobiModel_.getEnv().set(GRB_IntParam_NodeMethod, 2);
            break;
         }
         case LPDef::LP_SOLVER_SIFTING: {
            throw std::runtime_error("Gurobi does not support Sifting as node algorithm");
            break;
         }
         case LPDef::LP_SOLVER_CONCURRENT: {
            throw std::runtime_error("Gurobi does not support concurrent solvers as node algorithm");
            break;
         }
         default: {
            throw std::runtime_error("Unknown Node Algorithm");
         }
      }

      // presolve
      switch(parameter_.presolve_) {
         case LPDef::LP_PRESOLVE_AUTO: {
            gurobiModel_.getEnv().set(GRB_IntParam_Presolve, -1);
            break;
         }
         case LPDef::LP_PRESOLVE_OFF: {
            gurobiModel_.getEnv().set(GRB_IntParam_Presolve, 0);
            break;
         }
         case LPDef::LP_PRESOLVE_CONSERVATIVE: {
            gurobiModel_.getEnv().set(GRB_IntParam_Presolve, 1);
            break;
         }
         case LPDef::LP_PRESOLVE_AGGRESSIVE: {
            gurobiModel_.getEnv().set(GRB_IntParam_Presolve, 2);
            break;
         }
         default: {
            throw std::runtime_error("Unknown Presolve Option");
         }
      }

      // MIP EMPHASIS
      switch(parameter_.mipEmphasis_) {
         case LPDef::MIP_EMPHASIS_BALANCED: {
            gurobiModel_.getEnv().set(GRB_IntParam_MIPFocus, 0);
            break;
         }
         case LPDef::MIP_EMPHASIS_FEASIBILITY: {
            gurobiModel_.getEnv().set(GRB_IntParam_MIPFocus, 1);
            break;
         }
         case LPDef::MIP_EMPHASIS_OPTIMALITY: {
            gurobiModel_.getEnv().set(GRB_IntParam_MIPFocus, 2);
            break;
         }
         case LPDef::MIP_EMPHASIS_BESTBOUND: {
            gurobiModel_.getEnv().set(GRB_IntParam_MIPFocus, 3);
            break;
         }
         case LPDef::MIP_EMPHASIS_HIDDENFEAS: {
            throw std::runtime_error("Gurobi does not support hidden feasibility as MIP-focus");
            break;
         }
         default: {
            throw std::runtime_error("Unknown MIP Emphasis Option");
         }
      }

      // Tuning
      // Probing missing
      if(parameter_.cutLevel_ != LPDef::MIP_CUT_DEFAULT) {
         gurobiModel_.getEnv().set(GRB_IntParam_Cuts,          getCutLevelValue(parameter_.cutLevel_));
      }
      if(parameter_.cliqueCutLevel_ != LPDef::MIP_CUT_DEFAULT) {
         gurobiModel_.getEnv().set(GRB_IntParam_CliqueCuts,    getCutLevelValue(parameter_.cliqueCutLevel_));
      }
      if(parameter_.coverCutLevel_ != LPDef::MIP_CUT_DEFAULT) {
         gurobiModel_.getEnv().set(GRB_IntParam_CoverCuts,     getCutLevelValue(parameter_.coverCutLevel_));
      }
      if(parameter_.gubCutLevel_ != LPDef::MIP_CUT_DEFAULT) {
         gurobiModel_.getEnv().set(GRB_IntParam_GUBCoverCuts,  getCutLevelValue(parameter_.gubCutLevel_));
      }
      if(parameter_.mirCutLevel_ != LPDef::MIP_CUT_DEFAULT) {
         gurobiModel_.getEnv().set(GRB_IntParam_MIRCuts,       getCutLevelValue(parameter_.mirCutLevel_));
      }
      if(parameter_.iboundCutLevel_ != LPDef::MIP_CUT_DEFAULT) {
         gurobiModel_.getEnv().set(GRB_IntParam_ImpliedCuts,   getCutLevelValue(parameter_.iboundCutLevel_));
      }
      if(parameter_.flowcoverCutLevel_ != LPDef::MIP_CUT_DEFAULT) {
         gurobiModel_.getEnv().set(GRB_IntParam_FlowCoverCuts, getCutLevelValue(parameter_.flowcoverCutLevel_));
      }
      if(parameter_.flowpathCutLevel_ != LPDef::MIP_CUT_DEFAULT) {
         gurobiModel_.getEnv().set(GRB_IntParam_FlowPathCuts,  getCutLevelValue(parameter_.flowpathCutLevel_));
      }
      // DisjCuts missing
      // Gomory missing
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while setting parameter for Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while setting parameter for Gurobi model.");
   }
}

inline LPSolverGurobi::~LPSolverGurobi() {

}

inline typename LPSolverGurobi::GurobiValueType LPSolverGurobi::infinity_impl() {
   return GRB_INFINITY;
}

inline void LPSolverGurobi::addContinuousVariables_impl(const GurobiIndexType numVariables, const GurobiValueType lowerBound, const GurobiValueType upperBound) {
   gurobiVariables_.reserve(numVariables + gurobiVariables_.size());
   // according to the Gurobi documentation, adding variables separately does not have any performance impact
   try {
      for(GurobiIndexType i = 0; i < numVariables; ++i) {
         gurobiVariables_.push_back(gurobiModel_.addVar(lowerBound, upperBound, 0.0, GRB_CONTINUOUS));
      }
      gurobiSolution_.resize(gurobiVariables_.size());
      gurobiSolutionValid_ = false;
      gurobiModel_.update();
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while adding continuous variables to Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while adding continuous variables to Gurobi model.");
   }
}

inline void LPSolverGurobi::addIntegerVariables_impl(const GurobiIndexType numVariables, const GurobiValueType lowerBound, const GurobiValueType upperBound) {
   gurobiVariables_.reserve(numVariables + gurobiVariables_.size());
   // according to the Gurobi documentation, adding variables separately does not have any performance impact
   try {
      for(GurobiIndexType i = 0; i < numVariables; ++i) {
         gurobiVariables_.push_back(gurobiModel_.addVar(lowerBound, upperBound, 0.0, GRB_INTEGER));
      }
      gurobiSolution_.resize(gurobiVariables_.size());
      gurobiSolutionValid_ = false;
      gurobiModel_.update();
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while adding integer variables to Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while adding integer variables to Gurobi model.");
   }
}

inline void LPSolverGurobi::addBinaryVariables_impl(const GurobiIndexType numVariables) {
   gurobiVariables_.reserve(numVariables + gurobiVariables_.size());
   // according to the Gurobi documentation, adding variables separately does not have any performance impact
   try {
      for(GurobiIndexType i = 0; i < numVariables; ++i) {
         gurobiVariables_.push_back(gurobiModel_.addVar(0.0, 1.0, 0.0, GRB_BINARY));
      }
      gurobiSolution_.resize(gurobiVariables_.size());
      gurobiSolutionValid_ = false;
      gurobiModel_.update();
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while adding binary variables to Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while adding binary variables to Gurobi model.");
   }
}

inline void LPSolverGurobi::setObjective_impl(const Objective objective) {
   switch(objective) {
      case Minimize: {
         try {
            gurobiModel_.set(GRB_IntAttr_ModelSense, 1);
         } catch(const GRBException& e) {
            std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
            std::cout << e.getMessage() << std::endl;
            throw  std::runtime_error(e.getMessage());
         } catch(...) {
            std::cout << "Exception while setting objective of Gurobi model." << std::endl;
            throw  std::runtime_error("Exception while setting objective of Gurobi model.");
         }
         break;
      }
      case Maximize: {
         try {
            gurobiModel_.set(GRB_IntAttr_ModelSense, -1);
         } catch(const GRBException& e) {
            std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
            std::cout << e.getMessage() << std::endl;
            throw  std::runtime_error(e.getMessage());
         } catch(...) {
            std::cout << "Exception while setting objective of Gurobi model." << std::endl;
            throw  std::runtime_error("Exception while setting objective of Gurobi model.");
         }
         break;
      }
      default: {
         throw std::runtime_error("Unknown Objective");
      }
   }
}

inline void LPSolverGurobi::setObjectiveValue_impl(const GurobiIndexType variable, const GurobiValueType value) {
   try {
      gurobiVariables_[variable].set(GRB_DoubleAttr_Obj, value);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while setting objective value of Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while setting objective value of Gurobi model.");
   }
}

template<class ITERATOR_TYPE>
inline void LPSolverGurobi::setObjectiveValue_impl(ITERATOR_TYPE begin, const ITERATOR_TYPE end) {
   try {
      GRBLinExpr objective;
      objective.addTerms(&(*begin), &gurobiVariables_[0], gurobiVariables_.size());
      gurobiModel_.setObjective(objective);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while setting objective value of Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while setting objective value of Gurobi model.");
   }
}

template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverGurobi::setObjectiveValue_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin) {
   try {
      while(variableIDsBegin != variableIDsEnd) {
         gurobiVariables_[*variableIDsBegin].set(GRB_DoubleAttr_Obj, *coefficientsBegin);
         ++variableIDsBegin;
         ++coefficientsBegin;
      }
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while setting objective value of Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while setting objective value of Gurobi model.");
   }
}

template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverGurobi::addEqualityConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName) {
   const GurobiIndexType numConstraintVariables = std::distance(variableIDsBegin, variableIDsEnd);
   std::vector<GRBVar> constraintVariables;
   constraintVariables.reserve(numConstraintVariables);
   while(variableIDsBegin != variableIDsEnd) {
      constraintVariables.push_back(gurobiVariables_[*variableIDsBegin]);
      ++variableIDsBegin;
   }

   try {
      GRBLinExpr constraint;
      constraint.addTerms(&(*coefficientsBegin), &constraintVariables[0], numConstraintVariables);
      gurobiModel_.addConstr(constraint, GRB_EQUAL, bound, constraintName);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while adding equality constraint to Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while adding equality constraint to Gurobi model.");
   }
}

template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverGurobi::addLessEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName) {
   const GurobiIndexType numConstraintVariables = std::distance(variableIDsBegin, variableIDsEnd);
   std::vector<GRBVar> constraintVariables;
   constraintVariables.reserve(numConstraintVariables);
   while(variableIDsBegin != variableIDsEnd) {
      constraintVariables.push_back(gurobiVariables_[*variableIDsBegin]);
      ++variableIDsBegin;
   }

   try {
      GRBLinExpr constraint;
      constraint.addTerms(&(*coefficientsBegin), &constraintVariables[0], numConstraintVariables);
      gurobiModel_.addConstr(constraint, GRB_LESS_EQUAL, bound, constraintName);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while adding less equal constraint to Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while adding less equal constraint to Gurobi model.");
   }
}

template<class VARIABLES_ITERATOR_TYPE, class COEFFICIENTS_ITERATOR_TYPE>
inline void LPSolverGurobi::addGreaterEqualConstraint_impl(VARIABLES_ITERATOR_TYPE variableIDsBegin, const VARIABLES_ITERATOR_TYPE variableIDsEnd, COEFFICIENTS_ITERATOR_TYPE coefficientsBegin, const GurobiValueType bound, const std::string& constraintName) {
   const GurobiIndexType numConstraintVariables = std::distance(variableIDsBegin, variableIDsEnd);
   std::vector<GRBVar> constraintVariables;
   constraintVariables.reserve(numConstraintVariables);
   while(variableIDsBegin != variableIDsEnd) {
      constraintVariables.push_back(gurobiVariables_[*variableIDsBegin]);
      ++variableIDsBegin;
   }

   try {
      GRBLinExpr constraint;
      constraint.addTerms(&(*coefficientsBegin), &constraintVariables[0], numConstraintVariables);
      gurobiModel_.addConstr(constraint, GRB_GREATER_EQUAL, bound, constraintName);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while adding greater equal constraint to Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while adding greater equal constraint to Gurobi model.");
   }
}

inline void LPSolverGurobi::addConstraintsFinished_impl() {
   try {
      gurobiModel_.update();
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while incorporating constraints into Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while incorporating constraints into Gurobi model.");
   }
}

inline void LPSolverGurobi::addConstraintsFinished_impl(GurobiTimingType& timing) {
   try {
      Timer timer;
      timer.tic();
      gurobiModel_.update();
      timer.toc();
      timing = timer.elapsedTime();
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while incorporating constraints into Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while incorporating constraints into Gurobi model.");
   }
}

template <class PARAMETER_TYPE, class PARAMETER_VALUE_TYPE>
inline void LPSolverGurobi::setParameter_impl(const PARAMETER_TYPE parameter, const PARAMETER_VALUE_TYPE value) {
   try {
      gurobiModel_.getEnv().set(parameter, value);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while setting parameter for Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while setting parameter for Gurobi model.");
   }
}

inline bool LPSolverGurobi::solve_impl() {
   gurobiSolutionValid_ = false;
   try {
      gurobiModel_.optimize();
      if(gurobiModel_.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
         return true;
      } else {
         return false;
      }
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while solving Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while solving Gurobi model.");
   }
}

inline bool LPSolverGurobi::solve_impl(GurobiTimingType& timing) {
   gurobiSolutionValid_ = false;
   try {
      gurobiModel_.optimize();
      timing = gurobiModel_.get(GRB_DoubleAttr_Runtime);
      if(gurobiModel_.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
         return true;
      } else {
         return false;
      }
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while solving Gurobi model." << std::endl;
      throw  std::runtime_error("Exception while solving Gurobi model.");
   }
}

inline typename LPSolverGurobi::GurobiSolutionIteratorType LPSolverGurobi::solutionBegin_impl() const {
   updateSolution();
   return gurobiSolution_.begin();
}

inline typename LPSolverGurobi::GurobiSolutionIteratorType LPSolverGurobi::solutionEnd_impl() const {
   updateSolution();
   return gurobiSolution_.end();
}

inline typename LPSolverGurobi::GurobiValueType LPSolverGurobi::solution_impl(const GurobiIndexType variable) const {
   try {
      return gurobiVariables_[variable].get(GRB_DoubleAttr_X);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while accessing Gurobi solution of variable." << std::endl;
      throw  std::runtime_error("Exception while accessing Gurobi solution of variable.");
   }
}

inline typename LPSolverGurobi::GurobiValueType LPSolverGurobi::objectiveFunctionValue_impl() const {
   try {
      return gurobiModel_.get(GRB_DoubleAttr_ObjVal);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while accessing Gurobi solution for objective function value." << std::endl;
      throw  std::runtime_error("Exception while accessing Gurobi solution for objective function value.");
   }
}

inline typename LPSolverGurobi::GurobiValueType LPSolverGurobi::objectiveFunctionValueBound_impl() const {
   try {
      if(gurobiModel_.get(GRB_IntAttr_IsMIP)) {
         return gurobiModel_.get(GRB_DoubleAttr_ObjBound);
      } else {
         return gurobiModel_.get(GRB_DoubleAttr_ObjVal);
      }
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while accessing Gurobi bound for objective function value." << std::endl;
      throw  std::runtime_error("Exception while accessing Gurobi bound for objective function value.");
   }
}

inline void LPSolverGurobi::exportModel_impl(const std::string& filename) const {
   try {
      gurobiModel_.write(filename);
   } catch(const GRBException& e) {
      std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
      throw  std::runtime_error(e.getMessage());
   } catch(...) {
      std::cout << "Exception while writing Gurobi model to file." << std::endl;
      throw  std::runtime_error("Exception while writing Gurobi model to file.");
   }
}

inline void LPSolverGurobi::updateSolution() const {
   if(!gurobiSolutionValid_) {
      try {
         for(GurobiIndexType i = 0; i < static_cast<GurobiIndexType>(gurobiVariables_.size()); ++i) {
            gurobiSolution_[i] = gurobiVariables_[i].get(GRB_DoubleAttr_X);
         }
         gurobiSolutionValid_ = true;
      } catch(const GRBException& e) {
         std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
         std::cout << e.getMessage() << std::endl;
         throw  std::runtime_error(e.getMessage());
      } catch(...) {
         std::cout << "Exception while updating Gurobi solution." << std::endl;
         throw  std::runtime_error("Exception while updating Gurobi solution.");
      }
   }
}

inline int LPSolverGurobi::getCutLevelValue(const LPDef::MIP_CUT cutLevel) {
   switch(cutLevel){
      case LPDef::MIP_CUT_DEFAULT:
      case LPDef::MIP_CUT_AUTO:
         return -1;
      case LPDef::MIP_CUT_OFF:
         return 0;
      case LPDef::MIP_CUT_ON:
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

#endif /* OPENGM_LP_SOLVER_GUROBI_HXX_ */
