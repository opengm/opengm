#ifndef OPENGM_LP_INFERENCE_BASE_HXX_
#define OPENGM_LP_INFERENCE_BASE_HXX_

#include <utility>
#include <vector>
#include <map>
#include <list>
#include <typeinfo>
#include <limits>

#include <opengm/datastructures/marray/marray.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/datastructures/linear_constraint.hxx>
#include <opengm/inference/auxiliary/lp_functiontransfer.hxx>
#include <opengm/functions/constraint_functions/linear_constraint_function_base.hxx>
#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/utilities/subsequence_iterator.hxx>

namespace opengm {

/********************
 * class definition *
 *******************/
template <class LP_INFERENCE_TYPE>
struct LPInferenceTraits;

template <class LP_INFERENCE_TYPE>
class LPInferenceBase : public Inference<typename LPInferenceTraits<LP_INFERENCE_TYPE>::GraphicalModelType, typename LPInferenceTraits<LP_INFERENCE_TYPE>::AccumulationType> {
public:
   // typedefs
   typedef LP_INFERENCE_TYPE                                  LPInferenceType;
   typedef LPInferenceTraits<LPInferenceType>                 LPInferenceTraitsType;
   typedef LPInferenceBase<LPInferenceType>                   LPInferenceBaseType;
   typedef typename LPInferenceTraitsType::AccumulationType   AccumulationType;
   typedef typename LPInferenceTraitsType::GraphicalModelType GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;

   typedef visitors::VerboseVisitor<LPInferenceBaseType> VerboseVisitorType;
   typedef visitors::EmptyVisitor<LPInferenceBaseType>   EmptyVisitorType;
   typedef visitors::TimingVisitor<LPInferenceBaseType>  TimingVisitorType;

   typedef LinearConstraint<ValueType, IndexType, LabelType>              LinearConstraintType;
   typedef std::vector<LinearConstraintType>                              LinearConstraintsContainerType;
   typedef typename LinearConstraintsContainerType::const_iterator        LinearConstraintsIteratorType;
   typedef typename LinearConstraintType::IndicatorVariableType           IndicatorVariableType;
   typedef typename LinearConstraintType::IndicatorVariablesContainerType IndicatorVariablesContainerType;
   typedef typename LinearConstraintType::IndicatorVariablesIteratorType  IndicatorVariablesIteratorType;
   typedef typename LinearConstraintType::VariableLabelPairsIteratorType  VariableLabelPairsIteratorType;

   typedef LPFunctionTransfer<ValueType, IndexType, LabelType> LPFunctionTransferType;

   typedef typename LPInferenceTraitsType::SolverType                 SolverType;
   typedef typename LPInferenceTraitsType::SolverIndexType            SolverIndexType;
   typedef typename LPInferenceTraitsType::SolverValueType            SolverValueType;
   typedef typename LPInferenceTraitsType::SolverSolutionIteratorType SolverSolutionIteratorType;
   typedef typename LPInferenceTraitsType::SolverTimingType           SolverTimingType;
   typedef typename LPInferenceTraitsType::SolverParameterType        SolverParameterType;

   typedef SubsequenceIterator<typename std::vector<LabelType>::const_iterator, typename std::vector<size_t>::const_iterator> IntegerSolutionSubsequenceIterator;
   typedef SubsequenceIterator<SolverSolutionIteratorType, typename std::vector<size_t>::const_iterator>                      RelaxedSolutionSubsequenceIterator;

   struct Parameter : public SolverParameterType {
      // LocalPolytope will add a first order local polytope approximation of the marginal polytope, i.e. the affine instead of the convex hull.
      // LoosePolytope will add no constraints at all. All linear constraints will be added iteratively only if they are violated.
      // TightPolytope will add all constraints of the LocalPolytope relaxation and furthermore all constraints that are present in the model via constraint functions. Thus all constraints will be added before the first run of lp inference which leads to solving the problem in only one iteration.
      enum Relaxation {LocalPolytope, LoosePolytope, TightPolytope};
      enum ChallengeHeuristic{Random, Weighted};

      Parameter();

      // general options
      bool               integerConstraintNodeVar_;    // use integer constraints for node variables
      bool               integerConstraintFactorVar_;  // use integer constraints for factor variables
      bool               useSoftConstraints_;          // if constraint factors are present in the model add them as soft constraints e.g. treat them as normal factors
      bool               useFunctionTransfer_;         // use function transfer if available to generate more efficient lp models
      bool               mergeParallelFactors_;        // merge factors which are connected to the same set of variables
      bool               nameConstraints_;             // create unique names for the linear constraints added to the model (might be helpful for debugging models)
      Relaxation         relaxation_;                  // relaxation method
      size_t             maxNumIterations_;            // maximum number of tightening iterations (infinite if set to 0)
      size_t             maxNumConstraintsPerIter_;    // maximum number of added constraints per tightening iteration (all if set to 0)
      ChallengeHeuristic challengeHeuristic_;          // heuristic on how to select violated constraints
      ValueType          tolerance_;                   // tolerance for violation of linear constraints
   };

   // public member functions
   virtual const GraphicalModelType& graphicalModel() const;
   virtual InferenceTermination infer();
   template<class VISITOR_TYPE>
   InferenceTermination infer(VISITOR_TYPE& visitor);
   virtual ValueType bound() const;
   virtual ValueType value() const;
   virtual InferenceTermination arg(std::vector<LabelType>& x, const size_t N = 1) const;

protected:
   // structs
   struct ConstraintStorage {
      std::vector<SolverIndexType>                                     variableIDs_;
      std::vector<SolverValueType>                                     coefficients_;
      SolverValueType                                                  bound_;
      typename LinearConstraintType::LinearConstraintOperatorValueType operator_;
      std::string                                                      name_;
   };

   // protected typedefs
   typedef typename std::list<ConstraintStorage>                                                InactiveConstraintsListType;
   typedef typename InactiveConstraintsListType::iterator                                       InactiveConstraintsListIteratorType;
   typedef std::pair<IndexType, const LinearConstraintType*>                                    FactorIndexConstraintPointerPairType;
   typedef std::pair<InactiveConstraintsListIteratorType, FactorIndexConstraintPointerPairType> InactiveConstraintFactorConstraintPairType;
   typedef std::multimap<double, InactiveConstraintFactorConstraintPairType>                    SortedViolatedConstraintsListType;

   // functors
   struct GetIndicatorVariablesOrderBeginFunctor {
      // storage
      IndicatorVariablesIteratorType indicatorVariablesOrderBegin_;
      // operator()
      template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
      void operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction);
      // helper
      template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
      struct GetIndicatorVariablesOrderBeginFunctor_impl {
         static void getIndicatorVariablesOrderBeginFunctor_impl(GetIndicatorVariablesOrderBeginFunctor& myself, const FUNCTION_TYPE& function);
      };
      template<class FUNCTION_TYPE>
      struct GetIndicatorVariablesOrderBeginFunctor_impl<FUNCTION_TYPE, true> {
         static void getIndicatorVariablesOrderBeginFunctor_impl(GetIndicatorVariablesOrderBeginFunctor& myself, const FUNCTION_TYPE& function);
      };
   };
   struct GetIndicatorVariablesOrderEndFunctor {
      // storage
      IndicatorVariablesIteratorType indicatorVariablesOrderEnd_;
      // operator()
      template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
      void operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction);
      // helper
      template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
      struct GetIndicatorVariablesOrderEndFunctor_impl {
         static void getIndicatorVariablesOrderEndFunctor_impl(GetIndicatorVariablesOrderEndFunctor& myself, const FUNCTION_TYPE& function);
      };
      template<class FUNCTION_TYPE>
      struct GetIndicatorVariablesOrderEndFunctor_impl<FUNCTION_TYPE, true> {
         static void getIndicatorVariablesOrderEndFunctor_impl(GetIndicatorVariablesOrderEndFunctor& myself, const FUNCTION_TYPE& function);
      };
   };
   struct GetLinearConstraintsBeginFunctor {
      // storage
      LinearConstraintsIteratorType linearConstraintsBegin_;
      // operator()
      template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
      void operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction);
      // helper
      template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
      struct GetLinearConstraintsBeginFunctor_impl {
         static void getLinearConstraintsBeginFunctor_impl(GetLinearConstraintsBeginFunctor& myself, const FUNCTION_TYPE& function);
      };
      template<class FUNCTION_TYPE>
      struct GetLinearConstraintsBeginFunctor_impl<FUNCTION_TYPE, true> {
         static void getLinearConstraintsBeginFunctor_impl(GetLinearConstraintsBeginFunctor& myself, const FUNCTION_TYPE& function);
      };
   };
   struct GetLinearConstraintsEndFunctor {
      // storage
      LinearConstraintsIteratorType linearConstraintsEnd_;
      // operator()
      template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
      void operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction);
      // helper
      template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
      struct GetLinearConstraintsEndFunctor_impl {
         static void getLinearConstraintsEndFunctor_impl(GetLinearConstraintsEndFunctor& myself, const FUNCTION_TYPE& function);
      };
      template<class FUNCTION_TYPE>
      struct GetLinearConstraintsEndFunctor_impl<FUNCTION_TYPE, true> {
         static void getLinearConstraintsEndFunctor_impl(GetLinearConstraintsEndFunctor& myself, const FUNCTION_TYPE& function);
      };
   };
   struct AddAllViolatedLinearConstraintsFunctor {
      // storage
      ValueType                          tolerance_;
      IntegerSolutionSubsequenceIterator labelingBegin_;
      bool                               violatedConstraintAdded_;
      LPInferenceBaseType*               lpInference_;
      IndexType                          linearConstraintID_;
      // operator()
      template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
      void operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction);
      // helper
      template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
      struct AddAllViolatedLinearConstraintsFunctor_impl {
         static void addAllViolatedLinearConstraintsFunctor_impl(AddAllViolatedLinearConstraintsFunctor& myself, const FUNCTION_TYPE& function);
      };
      template<class FUNCTION_TYPE>
      struct AddAllViolatedLinearConstraintsFunctor_impl<FUNCTION_TYPE, true> {
         static void addAllViolatedLinearConstraintsFunctor_impl(AddAllViolatedLinearConstraintsFunctor& myself, const FUNCTION_TYPE& function);
      };
   };
   struct AddAllViolatedLinearConstraintsRelaxedFunctor {
      // storage
      ValueType                          tolerance_;
      RelaxedSolutionSubsequenceIterator labelingBegin_;
      bool                               violatedConstraintAdded_;
      LPInferenceBaseType*               lpInference_;
      IndexType                          linearConstraintID_;
      // operator()
      template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
      void operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction);
      // helper
      template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
      struct AddAllViolatedLinearConstraintsRelaxedFunctor_impl {
         static void addAllViolatedLinearConstraintsRelaxedFunctor_impl(AddAllViolatedLinearConstraintsRelaxedFunctor& myself, const FUNCTION_TYPE& function);
      };
      template<class FUNCTION_TYPE>
      struct AddAllViolatedLinearConstraintsRelaxedFunctor_impl<FUNCTION_TYPE, true> {
         static void addAllViolatedLinearConstraintsRelaxedFunctor_impl(AddAllViolatedLinearConstraintsRelaxedFunctor& myself, const FUNCTION_TYPE& function);
      };
   };

   // meta
   // note: LinearConstraintFunctionTypeList might be an empty type list containing only meta::ListEnd elements.
   // This happens if GM::FunctionTypeList does not contain any linear constraint function
   typedef typename meta::GetLinearConstraintFunctionTypeList<typename GraphicalModelType::FunctionTypeList>::type LinearConstraintFunctionTypeList;

   // storage
   const GraphicalModelType&    gm_;
   const Parameter              parameter_;
   ValueType                    constValue_;
   std::vector<IndexType>       unaryFactors_;
   std::vector<IndexType>       higherOrderFactors_;
   std::vector<IndexType>       linearConstraintFactors_;
   std::vector<IndexType>       transferableFactors_;
   bool                         inferenceStarted_;
   SolverIndexType              numLPVariables_;
   SolverIndexType              numNodesLPVariables_;
   SolverIndexType              numFactorsLPVariables_;
   SolverIndexType              numLinearConstraintsLPVariables_;
   SolverIndexType              numTransferedFactorsLPVariables;
   SolverIndexType              numSlackVariables_;

   // lookup tables
   std::vector<SolverIndexType>                                         nodesLPVariablesOffset_;
   std::vector<SolverIndexType>                                         factorsLPVariablesOffset_;
   // TODO The lookups might be faster by using hashmaps instead of std::map (requires C++11)
   std::vector<std::map<const IndicatorVariableType, SolverIndexType> > linearConstraintsLPVariablesIndicesLookupTable_;
   std::vector<std::map<const IndicatorVariableType, SolverIndexType> > transferedFactorsLPVariablesIndicesLookupTable_;
   std::vector<std::vector<size_t> >                                    linearConstraintLPVariablesSubsequenceIndices_;

   // cache
   IndexType                       addLocalPolytopeFactorConstraintCachePreviousFactorID_;
   marray::Marray<SolverIndexType> addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_;
   InactiveConstraintsListType     inactiveConstraints_;

   // protected member functions
   // construction
   LPInferenceBase(const GraphicalModelType& gm, const Parameter& parameter = Parameter());  // no instance of LPInferenceBase is allowed
   virtual ~LPInferenceBase();

   // initialization
   void sortFactors();
   void countLPVariables();
   void fillLinearConstraintLPVariablesSubsequenceIndices();
   void setAccumulation();
   void addLPVariables();
   void createObjectiveFunction();
   void addLocalPolytopeConstraints();
   void addLoosePolytopeConstraints();
   void addTightPolytopeConstraints();

   // LP variables mapping functions
   SolverIndexType nodeLPVariableIndex(const IndexType nodeID, const LabelType label) const;
   SolverIndexType factorLPVariableIndex(const IndexType factorID, const size_t labelingIndex) const;
   template<class LABELING_ITERATOR_TYPE>
   SolverIndexType factorLPVariableIndex(const IndexType factorID, LABELING_ITERATOR_TYPE labelingBegin, const LABELING_ITERATOR_TYPE labelingEnd) const;
   template <class HIGHER_ORDER_FACTORS_MAP_TYPE, class INDICATOR_VARIABLES_MAP_TYPE>
   bool getLPVariableIndexFromIndicatorVariable(const HIGHER_ORDER_FACTORS_MAP_TYPE& higherOrderFactorVariablesLookupTable, const INDICATOR_VARIABLES_MAP_TYPE& indicatorVariablesLookupTable, const IndicatorVariableType& indicatorVariable, const IndexType linearConstraintFactorIndex, SolverIndexType& lpVariableIndex) const;

   // constraints creation helper
   void addLocalPolytopeVariableConstraint(const IndexType variableID, const bool addToModel);
   void addLocalPolytopeFactorConstraint(const IndexType factor, const IndexType variable, const LabelType label, const bool addToModel);
   void addIndicatorVariableConstraints(const IndexType factor, const IndicatorVariableType& indicatorVariable, const SolverIndexType indicatorVariableLPVariable, const bool addToModel);
   void addLinearConstraint(const IndexType linearConstraintFactor, const LinearConstraintType& constraint);

   // inference helper
   template <class VISITOR_TYPE>
   InferenceTermination infer_impl_selectRelaxation(VISITOR_TYPE& visitor);
   template <class VISITOR_TYPE, typename Parameter::Relaxation RELAXATION>
   InferenceTermination infer_impl_selectHeuristic(VISITOR_TYPE& visitor);
   template <class VISITOR_TYPE, typename Parameter::Relaxation RELAXATION, typename Parameter::ChallengeHeuristic HEURISTIC>
   InferenceTermination infer_impl_selectIterations(VISITOR_TYPE& visitor);
   template <class VISITOR_TYPE, typename Parameter::Relaxation RELAXATION, typename Parameter::ChallengeHeuristic HEURISTIC, bool USE_INFINITE_ITERATIONS>
   InferenceTermination infer_impl_selectViolatedConstraints(VISITOR_TYPE& visitor);
   template <class VISITOR_TYPE, typename Parameter::Relaxation RELAXATION, typename Parameter::ChallengeHeuristic HEURISTIC, bool USE_INFINITE_ITERATIONS, bool ADD_ALL_VIOLATED_CONSTRAINTS>
   InferenceTermination infer_impl_selectLPType(VISITOR_TYPE& visitor);
   template <class VISITOR_TYPE, typename Parameter::Relaxation RELAXATION, typename Parameter::ChallengeHeuristic HEURISTIC, bool USE_INFINITE_ITERATIONS, bool ADD_ALL_VIOLATED_CONSTRAINTS, bool USE_INTEGER_CONSTRAINTS>
   InferenceTermination infer_impl(VISITOR_TYPE& visitor);
   template <typename Parameter::Relaxation RELAXATION, typename Parameter::ChallengeHeuristic HEURISTIC, bool ADD_ALL_VIOLATED_CONSTRAINTS>
   bool tightenPolytope();
   template <typename Parameter::Relaxation RELAXATION, typename Parameter::ChallengeHeuristic HEURISTIC, bool ADD_ALL_VIOLATED_CONSTRAINTS>
   bool tightenPolytopeRelaxed();
   void checkInactiveConstraint(const ConstraintStorage& constraint, double& weight) const;
   void addInactiveConstraint(const ConstraintStorage& constraint);

   // friends
   template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
   friend struct AddViolatedLinearConstraintsFunctor;
   template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
   friend struct AddViolatedLinearConstraintsRelaxedFunctor;
};

template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
struct AddViolatedLinearConstraintsFunctor {
   // storage
   typename LP_INFERENCE_BASE_TYPE::ValueType                          tolerance_;
   typename LP_INFERENCE_BASE_TYPE::IntegerSolutionSubsequenceIterator labelingBegin_;
   size_t                                                              numConstraintsAdded_;
   LP_INFERENCE_BASE_TYPE*                                             lpInference_;
   typename LP_INFERENCE_BASE_TYPE::IndexType                          linearConstraintID_;
   typename LP_INFERENCE_BASE_TYPE::SortedViolatedConstraintsListType* sortedViolatedConstraintsList_;
   // operator()
   template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
   void operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction);
   // helper
   template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
   struct AddViolatedLinearConstraintsFunctor_impl {
      static void addViolatedLinearConstraintsFunctor_impl(AddViolatedLinearConstraintsFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>& myself, const FUNCTION_TYPE& function);
   };
   template<class FUNCTION_TYPE>
   struct AddViolatedLinearConstraintsFunctor_impl<FUNCTION_TYPE, true> {
      static void addViolatedLinearConstraintsFunctor_impl(AddViolatedLinearConstraintsFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>& myself, const FUNCTION_TYPE& function);
   };
};
template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
struct AddViolatedLinearConstraintsRelaxedFunctor {
   // storage
   typename LP_INFERENCE_BASE_TYPE::ValueType                          tolerance_;
   typename LP_INFERENCE_BASE_TYPE::RelaxedSolutionSubsequenceIterator labelingBegin_;
   size_t                                                              numConstraintsAdded_;
   LP_INFERENCE_BASE_TYPE*                                             lpInference_;
   typename LP_INFERENCE_BASE_TYPE::IndexType                          linearConstraintID_;
   typename LP_INFERENCE_BASE_TYPE::SortedViolatedConstraintsListType* sortedViolatedConstraintsList_;
   // operator()
   template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
   void operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction);
   // helper
   template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
   struct AddViolatedLinearConstraintsRelaxedFunctor_impl {
      static void addViolatedLinearConstraintsRelaxedFunctor_impl(AddViolatedLinearConstraintsRelaxedFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>& myself, const FUNCTION_TYPE& function);
   };
   template<class FUNCTION_TYPE>
   struct AddViolatedLinearConstraintsRelaxedFunctor_impl<FUNCTION_TYPE, true> {
      static void addViolatedLinearConstraintsRelaxedFunctor_impl(AddViolatedLinearConstraintsRelaxedFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>& myself, const FUNCTION_TYPE& function);
   };
};

/***********************
 * class documentation *
 **********************/
/*! \file lp_inference_base.hxx
 *  \brief Provides implementation of a base class for LP inference.
 */

/*! \struct LPInferenceTraits
 *  \brief Traits class for lp inference classes.
 *
 *  Each lp inference class which uses opengm::LPInferenceBase as a base class
 *  has to provide a template specialization of this class to provide
 *  appropriate typedefs. The following types have to be defined:
 *  -# AccumulationType
 *  -# GraphicalModelType
 *  -# SolverType
 *  -# SolverIndexType
 *  -# SolverValueType
 *  -# SolverSolutionIteratorType
 *  -# SolverTimingType
 *  -# SolverParameterType
 *
 *  \tparam LP_INFERENCE_TYPE The lp inference class.
 */

/*! \class LPInferenceBase
 *  \brief Base class for Linear Programming based inference.
 *
 *  This base class provides optimization by Linear Programming (LP) or Mixed
 *  Integer Programming (MIP) it can be used with different LP and MIP solvers
 *  (e.g. IBM ILOG CPLEX). Only a small interface class has to be written for
 *  each desired LP and MIP solver. It uses the curiously recurring template
 *  pattern (CRTP) to provide static polymorphism.
 *
 *  \tparam LP_INFERENCE_TYPE The lp inference class.
 *
 *  \note
 *        -# A template specialization of class opengm::LPInferenceTraits has to
 *             be defined for each class which inherits from
 *             opengm::LPInferenceBase.
 *        -# The child class which inherits from opengm::LPInferenceBase has to
 *           provide the same interface as it is defined by
 *           opengm::LPSolverInterface.
 */

/*! \typedef LPInferenceBase::LPInferenceType
 *  \brief Typedef of the child class which inherits from
 *         opengm::LPInferenceBase.
 */

/*! \typedef LPInferenceBase::LPInferenceTraitsType
 *  \brief Typedef of the opengm::LPInferenceTraits class with appropriate
 *         template parameter.
 */

/*! \typedef LPInferenceBase::LPInferenceBaseType
 *  \brief Typedef of the opengm::LPInferenceBase class with appropriate
 *         template parameter.
 */

/*! \typedef LPInferenceBase::AccumulationType
 *  \brief Typedef of the Accumulation type.
 */

/*! \typedef LPInferenceBase::GraphicalModelType
 *  \brief Typedef of the graphical model type.
 */

/*! \typedef LPInferenceBase::VerboseVisitorType
 *  \brief Typedef of the opengm::visitors::VerboseVisitor class with
 *         appropriate template parameter.
 */

/*! \typedef LPInferenceBase::EmptyVisitorType
 *  \brief Typedef of the opengm::visitors::EmptyVisitor class with appropriate
 *         template parameter.
 */

/*! \typedef LPInferenceBase::TimingVisitorType
 *  \brief Typedef of the opengm::visitors::TimingVisitor class with appropriate
 *         template parameter.
 */

/*! \typedef LPInferenceBase::LinearConstraintType
 *  \brief Typedef of the opengm::LinearConstraint class with appropriate
 *         template parameter. Used to represent linear constraints.
 */

/*! \typedef LPInferenceBase::LinearConstraintsContainerType
 *  \brief Typedef of the container type used to store a set of linear
 *         constraints.
 */

/*! \typedef LPInferenceBase::LinearConstraintsIteratorType
 *  \brief Typedef of the iterator type used to iterate over a set of linear
 *         constraints.
 */

/*! \typedef LPInferenceBase::IndicatorVariableType
 *  \brief Typedef of the indicator variable type used within linear
 *         constraints.
 */

/*! \typedef LPInferenceBase::IndicatorVariablesContainerType
 *  \brief Typedef of the container type used to store a set of indicator
 *         variables.
 */

/*! \typedef LPInferenceBase::IndicatorVariablesIteratorType
 *  \brief Typedef of the iterator type used to iterate over a set of indicator
 *         variables.
 */

/*! \typedef LPInferenceBase::VariableLabelPairsIteratorType
 *  \brief Typedef of the iterator type used to iterate over a set of varible
 *         label pairs used within an indicator variable.
 */

/*! \typedef LPInferenceBase::LPFunctionTransferType
 *  \brief Typedef of the opengm::LPFunctionTransfer class with appropriate
 *         template parameter.
 */

/*! \typedef LPInferenceBase::SolverType
 *  \brief Typedef of the solver type used to solve the LP or MIP which is
 *         generated from the graphical model.
 */

/*! \typedef LPInferenceBase::SolverIndexType
 *  \brief Typedef of the index type used by the LP/MIP solver.
 */

/*! \typedef LPInferenceBase::SolverValueType
 *  \brief Typedef of the value type used by the LP/MIP solver.
 */

/*! \typedef LPInferenceBase::SolverSolutionIteratorType
 *  \brief Typedef of the iterator type used to iterate over the computed
 *         solution from the LP/MIP solver.
 */

/*! \typedef LPInferenceBase::SolverTimingType
 *  \brief Typedef of the type used by the LP/MIP solver to measure timings.
 */

/*! \typedef LPInferenceBase::SolverParameterType
 *  \brief Typedef of the parameter class used by the LP/MIP solver.
 */

/*! \typedef LPInferenceBase::IntegerSolutionSubsequenceIterator
 *  \brief Typedef of the iterator type used to iterate over a subset of the
 *         computed solution. This iterator type is used to challenge the linear
 *         constraint functions present in the model. Only used when the problem
 *         is solved as a MIP.
 */

/*! \typedef LPInferenceBase::RelaxedSolutionSubsequenceIterator
 *  \brief Typedef of the iterator type used to iterate over a subset of the
 *         computed solution. This iterator type is used to challenge the linear
 *         constraint functions present in the model. Only used when the problem
 *         is solved as a LP.
 */

/*! \struct LPInferenceBase::Parameter
 *  \brief Parameter class for opengm::LPInferenceBase.
 */

/*! \enum LPInferenceBase::Parameter::Relaxation
 *  \brief This enum defines the type of the linear programming model which is
 *         used for inference.
 */

/*! \var LPInferenceBase::Parameter::Relaxation LPInferenceBase::Parameter::LocalPolytope
 *  \brief LocalPolytope will use a first order local polytope approximation of
 *         the marginal polytope, i.e. the affine instead of the convex hull.
 *         All linear constraints given by linear constraint functions will be
 *         added iteratively only if they are violated.
 */

/*! \var LPInferenceBase::Parameter::LPInferenceBase LPInferenceBase::Parameter::LoosePolytope
 *  \brief LoosePolytope will add no constraints at all. All linear constraints
 *         will be added iteratively only if they are violated.
 */

/*! \var LPInferenceBase::Parameter::Relaxation LPInferenceBase::Parameter::TightPolytope
 *  \brief TightPolytope will add all constraints of the LocalPolytope
 *         relaxation and furthermore all constraints that are present in the
 *         model via constraint functions. Thus all constraints will be added
 *         before the first run of inference which leads to solving the problem
 *         in only one iteration.
 */

/*! \enum LPInferenceBase::Parameter::ChallengeHeuristic
 *  \brief This enum defines the heuristic by which the violated constraints are
 *         added to the LP/MIP model.
 */

/*! \var LPInferenceBase::Parameter::ChallengeHeuristic LPInferenceBase::Parameter::Random
 *  \brief Random will add violated constraints in a random order.
 */

/*! \var LPInferenceBase::Parameter::ChallengeHeuristic LPInferenceBase::Parameter::Weighted
 *  \brief Weighted will add constraints sorted by their weights. This is only
 *         meaningful if the maximum number of violated constraints added per
 *         iteration is limited.
 */

/*! \fn LPInferenceBase::Parameter::Parameter()
 *  \brief Parameter constructor setting default value for all options.
 */

/*! \var LPInferenceBase::Parameter::integerConstraintNodeVar_
 *  \brief Use integer constraints for node variables.
 */

/*! \var LPInferenceBase::Parameter::integerConstraintFactorVar_
 *  \brief Use integer constraints for factor variables.
 */

/*! \var LPInferenceBase::Parameter::useSoftConstraints_
 *  \brief If constraint factors are present in the model add them as soft
 *         constraints e.g. treat them as normal factors.
 */

/*! \var LPInferenceBase::Parameter::useFunctionTransfer_
 *  \brief Use function transfer if available to generate more efficient LP/MIP
 *         models.
 */

/*! \var LPInferenceBase::Parameter::mergeParallelFactors_
 *  \brief Merge factors which are connected to the same set of variables. Might
 *         increase construction time but will result in a smaller LP/MIP model
 *         if parallel factors are present in the graphical model.
 */

/*! \var LPInferenceBase::Parameter::nameConstraints_
 *  \brief Create unique names for the linear constraints added to the model
 *         (might be helpful for debugging models).
 */

/*! \var LPInferenceBase::Parameter::relaxation_
 *  \brief Selected relaxation method.
 */

/*! \var LPInferenceBase::Parameter::maxNumIterations_
 *  \brief Maximum number of tightening iterations (infinite if set to 0).
 */

/*! \var LPInferenceBase::Parameter::maxNumConstraintsPerIter_
 *  \brief Maximum number of violated constraints which are added per tightening
 *         iteration (all if set to 0).
 */

 /*! \var LPInferenceBase::Parameter::challengeHeuristic_
  *  \brief Heuristic on how to select violated constraints.
  */

/*! \var LPInferenceBase::Parameter::tolerance_
  *  \brief Tolerance for violation of linear constraints
  */

/*! \fn const GraphicalModelType& LPInferenceBase::graphicalModel() const
 *  \brief Get graphical model.
 *
 *  \return Reference to the graphical model.
 */

/*! \fn InferenceTermination LPInferenceBase::infer()
 *  \brief Run inference with empty visitor.
 *
 *  \return Termination code of the inference.
 */

/*! \fn InferenceTermination LPInferenceBase::infer(VISITOR_TYPE& visitor)
 *  \brief Run inference with provided visitor.
 *
 *  \tparam VISITOR_TYPE The visitor type used to log inference.
 *
 *  \param[in,out] visitor The provided inference visitor.
 *
 *  \return Termination code of the inference.
 *
 *  \note If timing visitor is used an additional field for the timings
 *        required by the LP / ILP solver to solve the current model in each
 *        iteration is added.
 */

/*! \fn ValueType LPInferenceBase::bound() const
 *  \brief Get the current bound.
 *
 *  \return The current bound.
 */

/*! \fn ValueType LPInferenceBase::value()const
 *  \brief Get the current value.
 *
 *  \return The current value.
 */

/*! \fn InferenceTermination LPInferenceBase::arg(std::vector<LabelType>& x, const size_t N = 1) const
 *  \brief Get the current argument.
 *
 *  \param[out] x Will be filled with the current argument.
 *  \param[in] N UNKNOWN????
 *
 *  \return The status code of the current argument.
 */

/*! \struct LPInferenceBase::ConstraintStorage
 *  \brief Storage class for linear constraints representing the local polytope
 *         constraints. They are generated and stored for later use if
 *         LPInferenceBase::Parameter::LoosePolytope is selected as relaxation
 *         method.
 */

/*! \var LPInferenceBase::ConstraintStorage::variableIDs_
 *  \brief The variables of the LP/MIP model which are used in the constraint.
 */

/*! \var LPInferenceBase::ConstraintStorage::coefficients_
 *  \brief The coefficients for the variables of the LP/MIP model which are used
 *         in the constraint.
 */

/*! \var LPInferenceBase::ConstraintStorage::bound_
 *  \brief The value for the right hand side of the constraint.
 */

/*! \var LPInferenceBase::ConstraintStorage::operator_
 *  \brief The operator type used to compare the left hand side of the
 *         constraint against the right hand side (<=, ==, >=).
 */

/*! \var LPInferenceBase::ConstraintStorage::name_
 *  \brief The name of the constraint.
 */

/*! \typedef LPInferenceBase::InactiveConstraintsListType
 *  \brief Typedef of the container type used to sore a set of
 *         LPInferenceBase::ConstraintStorage objects.
 */

/*! \typedef LPInferenceBase::InactiveConstraintsListIteratorType
 *  \brief Typedef of the iterator type used to iterate over a set of
 *         LPInferenceBase::ConstraintStorage objects.
 */

/*! \typedef LPInferenceBase::FactorIndexConstraintPointerPairType
 *  \brief Typedef of the pair type used to store a pointer to a linear
 *         constraint in combination with the linear constraint factor index it
 *         belongs to.
 */

/*! \typedef LPInferenceBase::InactiveConstraintFactorConstraintPairType
 *  \brief Typedef of the pair type used to store a pointer to an inactive
 *         constraint of the local polytope constraints and a
 *         LPInferenceBase::FactorIndexConstraintPointerPairType.
 */

/*! \typedef LPInferenceBase::SortedViolatedConstraintsListType
 *  \brief Typedef of the map type used to store a set of violated constraints
 *         sorted by their weights. Is only used when
 *         LPInferenceBase::Parameter::Weighted is selected as challenge
 *         heuristic.
 *
 *  \note
 *        -# The weights of the violated constraints are used as the key of the
 *           map thus using a reverse iterator to access the violated
 *           constraints will result in accessing the violated constraints with
 *           the highest weight first.
 *        -# As there are two possible sources for linear constraints, the local
 *           polytope constraints and the constraints from the linear constraint
 *           functions, the value of the map stores an iterator pointer pair
 *           where either the iterator points to an element of
 *           LPInferenceBase::InactiveConstraintsListType and the pointer is set
 *           to NULL or the iterator points to
 *           LPInferenceBase::InactiveConstraintsListType.end() and the pointer
 *           points to an violated constraint of a linear constraint function.
 */

/*! \struct LPInferenceBase::GetIndicatorVariablesOrderBeginFunctor
 *  \brief Functor used to access the method indicatorVariablesOrderBegin() of
 *         the underlying linear constraint function of a graphical model
 *         factor.
 */

/*! \var LPInferenceBase::GetIndicatorVariablesOrderBeginFunctor::indicatorVariablesOrderBegin_
 *  \brief Storage for the iterator returned by the method
 *         indicatorVariablesOrderBegin().
 */

/*! \fn void LPInferenceBase::GetIndicatorVariablesOrderBeginFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction)
 *  \brief Operator used to access the method indicatorVariablesOrderBegin() of
 *         the underlying linear constraint function of a graphical model
 *         factor.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The underlying linear constraint
 *          function type of a graphical model factor.
 *
 *  \param[in] linearConstraintFunction The underlying linear constraint
 *             function of a graphical model factor.
 */

/*! \struct LPInferenceBase::GetIndicatorVariablesOrderBeginFunctor::GetIndicatorVariablesOrderBeginFunctor_impl
 *  \brief Helper struct to distinguish between linear constraint functions and
 *         other function types. This is necessary as only linear constraint
 *         functions provide the method indicatorVariablesOrderBegin().
 *
 *  \tparam FUNCTION_TYPE The function type used with
 *                        LPInferenceBase::GetIndicatorVariablesOrderBeginFunctor.
 *  \tparam IS_LINEAR_CONSTRAINT_FUNCTION Indicator to tell if FUNCTION_TYPE is
 *                                        a linear constraint function type.
 */

/*! \fn void LPInferenceBase::GetIndicatorVariablesOrderBeginFunctor::GetIndicatorVariablesOrderBeginFunctor_impl::getIndicatorVariablesOrderBeginFunctor_impl(GetIndicatorVariablesOrderBeginFunctor& myself, const FUNCTION_TYPE& function)
 *  \brief Actual access to the method indicatorVariablesOrderBegin() of the
 *         underlying linear constraint function of a graphical model factor.
 *
 *  \param[out] myself Reference to the functor
 *                     LPInferenceBase::GetIndicatorVariablesOrderBeginFunctor
 *                     to access the variable
 *                     LPInferenceBase::GetIndicatorVariablesOrderBeginFunctor::indicatorVariablesOrderBegin_.
 *  \param[in] function The function which will be accessed.
 */

/*! \struct LPInferenceBase::GetIndicatorVariablesOrderEndFunctor
 *  \brief Functor used to access the method indicatorVariablesOrderEnd() of the
 *         underlying linear constraint function of a graphical model factor.
 */

/*! \var LPInferenceBase::GetIndicatorVariablesOrderEndFunctor::indicatorVariablesOrderEnd_
 *  \brief Storage for the iterator returned by the method
 *         indicatorVariablesOrderEnd().
 */

/*! \fn void LPInferenceBase::GetIndicatorVariablesOrderEndFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction)
 *  \brief Operator used to access the method indicatorVariablesOrderEnd() of
 *         the underlying linear constraint function of a graphical model
 *         factor.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The underlying linear constraint
 *          function type of a graphical model factor.
 *
 *  \param[in] linearConstraintFunction The underlying linear constraint
 *             function of a graphical model factor.
 */

/*! \struct LPInferenceBase::GetIndicatorVariablesOrderEndFunctor::GetIndicatorVariablesOrderEndFunctor_impl
 *  \brief Helper struct to distinguish between linear constraint functions and
 *         other function types. This is necessary as only linear constraint
 *         functions provide the method indicatorVariablesOrderEnd().
 *
 *  \tparam FUNCTION_TYPE The function type used with
 *                        LPInferenceBase::GetIndicatorVariablesOrderEndFunctor.
 *  \tparam IS_LINEAR_CONSTRAINT_FUNCTION Indicator to tell if FUNCTION_TYPE is
 *                                        a linear constraint function type.
 */

/*! \fn void LPInferenceBase::GetIndicatorVariablesOrderEndFunctor::GetIndicatorVariablesOrderEndFunctor_impl::getIndicatorVariablesOrderEndFunctor_impl(GetIndicatorVariablesOrderEndFunctor& myself, const FUNCTION_TYPE& function)
 *  \brief Actual access to the method indicatorVariablesOrderEnd() of the
 *         underlying linear constraint function of a graphical model factor.
 *
 *  \param[out] myself Reference to the functor
 *                     LPInferenceBase::GetIndicatorVariablesOrderEndFunctor to
 *                     access the variable
 *                     LPInferenceBase::GetIndicatorVariablesOrderEndFunctor::indicatorVariablesOrderEnd_.
 *  \param[in] function The function which will be accessed.
 */

/*! \struct LPInferenceBase::GetLinearConstraintsBeginFunctor
 *  \brief Functor used to access the method linearConstraintsBegin() of the
 *         underlying linear constraint function of a graphical model factor.
 */

/*! \var LPInferenceBase::GetLinearConstraintsBeginFunctor::linearConstraintsBegin_
 *  \brief Storage for the iterator returned by the method
 *         linearConstraintsBegin().
 */

/*! \fn void LPInferenceBase::GetLinearConstraintsBeginFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction)
 *  \brief Operator used to access the method linearConstraintsBegin() of the
 *         underlying linear constraint function of a graphical model factor.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The underlying linear constraint
 *          function type of a graphical model factor.
 *
 *  \param[in] linearConstraintFunction The underlying linear constraint
 *             function of a graphical model factor.
 */

/*! \struct LPInferenceBase::GetLinearConstraintsBeginFunctor::GetLinearConstraintsBeginFunctor_impl
 *  \brief Helper struct to distinguish between linear constraint functions and
 *         other function types. This is necessary as only linear constraint
 *         functions provide the method linearConstraintsBegin().
 *
 *  \tparam FUNCTION_TYPE The function type used with
 *                        LPInferenceBase::GetLinearConstraintsBeginFunctor.
 *  \tparam IS_LINEAR_CONSTRAINT_FUNCTION Indicator to tell if FUNCTION_TYPE is
 *                                        a linear constraint function type.
 */

/*! \fn void LPInferenceBase::GetLinearConstraintsBeginFunctor::GetLinearConstraintsBeginFunctor_impl::getLinearConstraintsBeginFunctor_impl(GetLinearConstraintsBeginFunctor& myself, const FUNCTION_TYPE& function)
 *  \brief Actual access to the method linearConstraintsBegin() of the
 *         underlying linear constraint function of a graphical model factor.
 *
 *  \param[out] myself Reference to the functor
 *                     LPInferenceBase::GetLinearConstraintsBeginFunctor to
 *                     access the variable
 *                     LPInferenceBase::GetLinearConstraintsBeginFunctor::linearConstraintsBegin_.
 *  \param[in] function The function which will be accessed.
 */

/*! \struct LPInferenceBase::GetLinearConstraintsEndFunctor
 *  \brief Functor used to access the method linearConstraintsEnd() of the
 *         underlying linear constraint function of a graphical model factor.
 */

/*! \var LPInferenceBase::GetLinearConstraintsEndFunctor::linearConstraintsEnd_
 *  \brief Storage for the iterator returned by the method
 *         linearConstraintsEnd().
 */

/*! \fn void LPInferenceBase::GetLinearConstraintsEndFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction)
 *  \brief Operator used to access the method linearConstraintsEnd() of the
 *         underlying linear constraint function of a graphical model factor.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The underlying linear constraint
 *          function type of a graphical model factor.
 *
 *  \param[in] linearConstraintFunction The underlying linear constraint
 *             function of a graphical model factor.
 */

/*! \struct LPInferenceBase::GetLinearConstraintsEndFunctor::GetLinearConstraintsEndFunctor_impl
 *  \brief Helper struct to distinguish between linear constraint functions and
 *         other function types. This is necessary as only linear constraint
 *         functions provide the method linearConstraintsEnd().
 *
 *  \tparam FUNCTION_TYPE The function type used with
 *                        LPInferenceBase::GetLinearConstraintsEndFunctor
 *  \tparam IS_LINEAR_CONSTRAINT_FUNCTION Indicator to tell if FUNCTION_TYPE is
 *                                        a linear constraint function type.
 */

/*! \fn void LPInferenceBase::GetLinearConstraintsEndFunctor::GetLinearConstraintsEndFunctor_impl::getLinearConstraintsEndFunctor_impl(GetLinearConstraintsEndFunctor& myself, const FUNCTION_TYPE& function)
 *  \brief Actual access to the method linearConstraintsEnd() of the
 *         underlying linear constraint function of a graphical model factor.
 *
 *  \param[out] myself Reference to the functor
 *                     LPInferenceBase::GetLinearConstraintsEndFunctor to access
 *                     the variable
 *                     LPInferenceBase::GetLinearConstraintsEndFunctor::linearConstraintsEnd_.
 *  \param[in] function The function which will be accessed.
 */

/*! \struct LPInferenceBase::AddAllViolatedLinearConstraintsFunctor
 *  \brief Functor used to access the method challenge() of the underlying
 *         linear constraint function of a graphical model factor and to add all
 *         violated constraints to the LP/MIP model.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsFunctor::tolerance_
 *  \brief The tolerance used for the method challenge() of the underlying
 *         linear constraint function of a graphical model factor.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsFunctor::labelingBegin_
 *  \brief Iterator used to iterate over the current solution.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsFunctor::violatedConstraintAdded_
 *  \brief Indicator used to tell if at least one constraint was added to the
 *         LP/MIP model.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsFunctor::lpInference_
 *  \brief Pointer pointing to the instance of opengm::LPInferenceBase to get
 *         access to the LP/MIP model.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsFunctor::linearConstraintID_
 *  \brief Index of the linear constraint factor.
 */

/*! \fn void LPInferenceBase::AddAllViolatedLinearConstraintsFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction)
 *  \brief Operator used to access the method challenge() of the underlying
 *         linear constraint function of a graphical model factor and to add all
 *         violated constraints to the LP/MIP model.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The underlying linear constraint
 *          function type of a graphical model factor.
 *
 *  \param[in] linearConstraintFunction The underlying linear constraint
 *             function of a graphical model factor.
 */

/*! \struct LPInferenceBase::AddAllViolatedLinearConstraintsFunctor::AddAllViolatedLinearConstraintsFunctor_impl
 *  \brief Helper struct to distinguish between linear constraint functions and
 *         other function types. This is necessary as only linear constraint
 *         functions provide the method challenge().
 *
 *  \tparam FUNCTION_TYPE The function type used with
 *                        LPInferenceBase::AddAllViolatedLinearConstraintsFunctor.
 *  \tparam IS_LINEAR_CONSTRAINT_FUNCTION Indicator to tell if FUNCTION_TYPE is
 *                                        a linear constraint function type.
 */

/*! \fn void LPInferenceBase::AddAllViolatedLinearConstraintsFunctor::AddAllViolatedLinearConstraintsFunctor_impl::addAllViolatedLinearConstraintsFunctor_impl(AddAllViolatedLinearConstraintsFunctor& myself, const FUNCTION_TYPE& function)
 *  \brief Actual access to the method challenge() of the underlying linear
 *         constraint function of a graphical model factor.
 *
 *  \param[out] myself Reference to the functor
 *                     LPInferenceBase::AddAllViolatedLinearConstraintsFunctor
 *                     to access the variables of the functor.
 *  \param[in] function The function which will be accessed.
 */

/*! \struct LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor
 *  \brief Functor used to access the method challengeRelaxed() of the
 *         underlying linear constraint function of a graphical model factor and
 *         to add all violated constraints to the LP/MIP model.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor::tolerance_
 *  \brief The tolerance used for the method challengeRelaxed() of the
 *         underlying linear constraint function of a graphical model factor.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor::labelingBegin_
 *  \brief Iterator used to iterate over the current solution.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor::violatedConstraintAdded_
 *  \brief Indicator used to tell if at least one constraint was added to the
 *         LP/MIP model.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor::lpInference_
 *  \brief Pointer pointing to the instance of opengm::LPInferenceBase to get
 *         access to the LP/MIP model.
 */

/*! \var LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor::linearConstraintID_
 *  \brief Index of the linear constraint factor.
 */

/*! \fn void LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction)
 *  \brief Operator used to access the method challengeRelaxed() of the
 *         underlying linear constraint function of a graphical model factor and
 *         to add all violated constraints to the LP/MIP model.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The underlying linear constraint
 *          function type of a graphical model factor.
 *
 *  \param[in] linearConstraintFunction The underlying linear constraint
 *             function of a graphical model factor.
 */

/*! \struct LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor::AddAllViolatedLinearConstraintsRelaxedFunctor_impl
 *  \brief Helper struct to distinguish between linear constraint functions and
 *         other function types. This is necessary as only linear constraint
 *         functions provide the method challengeRelaxed().
 *
 *  \tparam FUNCTION_TYPE The function type used with
 *                        LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor.
 *  \tparam IS_LINEAR_CONSTRAINT_FUNCTION Indicator to tell if FUNCTION_TYPE is
 *                                        a linear constraint function type.
 */

/*! \fn void LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor::AddAllViolatedLinearConstraintsRelaxedFunctor_impl::addAllViolatedLinearConstraintsRelaxedFunctor_impl(AddAllViolatedLinearConstraintsRelaxedFunctor& myself, const FUNCTION_TYPE& function)
 *  \brief Actual access to the method challengeRelaxed() of the underlying
 *         linear constraint function of a graphical model factor.
 *
 *  \param[out] myself Reference to the functor
 *                     LPInferenceBase::AddAllViolatedLinearConstraintsRelaxedFunctor
 *                     to access the variables of the functor.
 *  \param[in] function The function which will be accessed.
 */

/*! \typedef LPInferenceBase::LinearConstraintFunctionTypeList
 *  \brief Typelist of all linear constraint function types which are present in
 *         the graphical model type.
 *
 *  \note LinearConstraintFunctionTypeList might be an empty type list
 *        containing only meta::ListEnd elements. This happens if
 *        GraphicalModelType::FunctionTypeList does not contain any linear
 *        constraint function.
 */

/*! \var LPInferenceBase::gm_
 *  \brief Reference to the graphical model.
 */

/*! \var LPInferenceBase::parameter_
 *  \brief Parameter which stores the settings for the inference.
 */

/*! \var LPInferenceBase::constValue_
 *  \brief Constant value offset.
 */

/*! \var LPInferenceBase::unaryFactors_
 *  \brief List of all unary factors.
 */

/*! \var LPInferenceBase::higherOrderFactors_
 *  \brief List of all higher order factors.
 */

/*! \var LPInferenceBase::linearConstraintFactors_
 *  \brief List of all linear constraint factors.
 */

/*! \var LPInferenceBase::transferableFactors_
 *  \brief List of all transferable factors.
 */

/*! \var LPInferenceBase::inferenceStarted_
 *  \brief Tell if inference was already started.
 */

/*! \var LPInferenceBase::numLPVariables_
 *  \brief The total number of lp variables except slack variables.
 */

/*! \var LPInferenceBase::numNodesLPVariables_
 *  \brief The number of lp variables for the nodes of the graphical model.
 */

/*! \var LPInferenceBase::numFactorsLPVariables_
 *  \brief The number of lp variables for the factors of the graphical model.
 */

/*! \var LPInferenceBase::numLinearConstraintsLPVariables_
 *  \brief The number of lp variables for the linear constraint factors of the
 *         graphical model.
 */

/*! \var LPInferenceBase::numTransferedFactorsLPVariables
 *  \brief The number of lp variables for the transferable factors of the
 *         graphical model.
 */

/*! \var LPInferenceBase::numSlackVariables_
 *  \brief The number of slack variables for the transferable factors of the
 *         graphical model.
 */

/*! \var LPInferenceBase::nodesLPVariablesOffset_
 *  \brief The offsets for the indices of the lp variables for each node of the
 *         graphical model.
 */

/*! \var LPInferenceBase::factorsLPVariablesOffset_
 *  \brief The offsets for the indices of the lp variables for each factor of
 *         the graphical model.
 */

/*! \var LPInferenceBase::linearConstraintsLPVariablesIndicesLookupTable_
 *  \brief Lookup table for the lp variable indices of each linear constraint.
 */

/*! \var LPInferenceBase::transferedFactorsLPVariablesIndicesLookupTable_
 *  \brief Lookup table for the lp variable indices of each transferable factor.
 */

/*! \var LPInferenceBase::linearConstraintLPVariablesSubsequenceIndices_
 *  \brief The indices of the subset of the solution variables which are
 *         relevant for each linear constraint.
 */

/*! \var LPInferenceBase::addLocalPolytopeFactorConstraintCachePreviousFactorID_
 *  \brief Cache for the function
 *         LPInferenceBase::addLocalPolytopeFactorConstraint. It is used to
 *         store the factor id used for the last call of this function. If the
 *         previous factor id and the factor id of the current call to
 *         LPInferenceBase::addLocalPolytopeFactorConstraint are the same, the
 *         variable
 *         LPInferenceBase::addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_
 *         does not have to be updated.
 */

/*! \var LPInferenceBase::addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_
 *  \brief Lookup table for the factor lp variable ids required by the
 *         LPInferenceBase::addLocalPolytopeFactorConstraint function. This
 *         lookup table is cached as it may not be necessary to create a new
 *         lookup table at each function call. This is only required if the
 *         factor id of the previous function call and the factor id of the
 *         current function call differ. The factor id of the previous function
 *         call is stored in the variable
 *         LPInferenceBase::addLocalPolytopeFactorConstraintCachePreviousFactorID_.
 */

/*! \var LPInferenceBase::inactiveConstraints_
 *  \brief Storage for all linear constraints representing the local polytope
 *         constraints. They are generated and stored for later use if
 *         LPInferenceBase::Parameter::LoosePolytope is selected as relaxation
 *         method. The constraints are removed from this list when they are
 *         added to the LP/MIP model.
 */

/*! \fn LPInferenceBase::LPInferenceBase(const GraphicalModelType& gm, const Parameter& parameter = Parameter())
 *  \brief LPInferenceBase constructor.
 *
 *  \param[in] gm The graphical model which will be solved.
 *  \param[in] parameter Parameter providing the settings for the inference.
 *
 *  \note Constructor is protected as no instance of LPInferenceBase is allowed.
 */

/*! \fn LPInferenceBase::~LPInferenceBase()
 *  \brief LPInferenceBase destructor.
 *
 *  \note Destructor is protected as no instance of LPInferenceBase is allowed.
 */

/*! \fn void LPInferenceBase::sortFactors()
 *  \brief Sorts the factors of the graphical model into the lists
 *         unaryFactors_, higherOrderFactors_, linearConstraintFactors_ and
 *         transferableFactors_.
 */

/*! \fn void LPInferenceBase::countLPVariables()
 *  \brief Count the number of lp variables required to build a lp model for
 *         inference of the graphical model.
 */

/*! \fn void LPInferenceBase::fillLinearConstraintLPVariablesSubsequenceIndices()
 *  \brief Fill the variable
 *         LPInferenceBase::linearConstraintLPVariablesSubsequenceIndices_ with
 *         the appropriate indices of the subset of the solution variables which
 *         are relevant for each linear constraint.
 */

/*! \fn void LPInferenceBase::setAccumulation()
 *  \brief Set the accumulation for the lp solver.
 */

/*! \fn void LPInferenceBase::addLPVariables()
 *  \brief Add the number of lp variables computed by
 *         LPInferenceBase::countLPVariables to the lp model.
 */

/*! \fn void LPInferenceBase::createObjectiveFunction()
 *  \brief Create the objective function for the lp model.
 */

/*! \fn void LPInferenceBase::addLocalPolytopeConstraints()
 *  \brief Add all constraints to the lp model which are required by the local
 *         polytope relaxation.
 */

/*! \fn void LPInferenceBase::addLoosePolytopeConstraints()
 *  \brief Add all constraints to the lp model which are required by the loose
 *         polytope relaxation.
 */

/*! \fn void LPInferenceBase::addTightPolytopeConstraints()
 *  \brief Add all constraints to the lp model which are required by the tight
 *         polytope relaxation.
 */

/*! \fn LPInferenceBase::SolverIndexType LPInferenceBase::nodeLPVariableIndex(const IndexType nodeID, const LabelType label) const
 *  \brief Get the lp variable which corresponds to the variable label pair of
 *         the graphical model.
 *
 *  \param[in] nodeID The variable of the graphical model.
 *  \param[in] label The label of the variable.
 *
 *  \return The index of the requested lp variable.
 */

/*! \fn LPInferenceBase::SolverIndexType LPInferenceBase::factorLPVariableIndex(const IndexType factorID, const size_t labelingIndex) const;
 *  \brief Get the lp variable which corresponds to the labeling of the factor.
 *
 *  \param[in] factorID The index of the factor.
 *  \param[in] labelingIndex The index describing the selected labeling.
 *
 *  \return The index of the requested lp variable.
 */

/*! \fn LPInferenceBase::SolverIndexType LPInferenceBase::factorLPVariableIndex(const IndexType factorID, LABELING_ITERATOR_TYPE labelingBegin, const LABELING_ITERATOR_TYPE labelingEnd) const
 *  \brief Get the lp variable which corresponds to the labeling of the factor.
 *
 *  \tparam LABELING_ITERATOR_TYPE Iterator type to iterate over the selected
 *                                 labeling.
 *  \param[in] factorID The index of the factor.
 *  \param[in] labelingBegin Iterator pointing to the begin of the selected
 *                           labeling.
 *  \param[in] labelingEnd Iterator pointing to the end of the selected
 *                         labeling.
 *
 *  \return The index of the requested lp variable.
 */

/*! \fn bool LPInferenceBase::getLPVariableIndexFromIndicatorVariable(const HIGHER_ORDER_FACTORS_MAP_TYPE& higherOrderFactorVariablesLookupTable, const INDICATOR_VARIABLES_MAP_TYPE& indicatorVariablesLookupTable, const IndicatorVariableType& indicatorVariable, const IndexType linearConstraintFactorIndex, SolverIndexType& lpVariableIndex) const
 *  \brief Get the index of the lp variable associated with an indicator
 *         variable.
 *
 *  \tparam HIGHER_ORDER_FACTORS_MAP_TYPE The type of the map which is used as a
 *                                        lookup table for the higher order
 *                                        factors.
 *  \tparam INDICATOR_VARIABLES_MAP_TYPE The type of the map which is used as a
 *                                       lookup table for the indicator
 *                                       variables.
 *
 *  \param[in] higherOrderFactorVariablesLookupTable The lookup table which is
 *                                                   used to check if another
 *                                                   factor is connected to the
 *                                                   exact same variables of the
 *                                                   graphical model as the
 *                                                   indicator variable.
 *  \param[in] indicatorVariablesLookupTable The lookup table which is used to
 *                                           check if another indicator variable
 *                                           with the same set of variable label
 *                                           pairs exists.
 *  \param[in] indicatorVariable The indicator variable for which the
 *                               corresponding lp variable is searched.
 *  \param[in] linearConstraintFactorIndex The index of the linear constraint
 *                                         factor to which the indicator
 *                                         variable belongs.
 *  \param[out] lpVariableIndex If a lp variable which is associated with the
 *                              indicator variable is found the corresponding
 *                              index will be stored in this variable.
 *
 *  \return True if a lp variable which is associated with the indicator
 *          variable is found, false otherwise.
 */

/*! \fn void LPInferenceBase::addLocalPolytopeVariableConstraint(const IndexType variableID, const bool addToModel);
 *  \brief Add a new variable constraint to the lp model.
 *
 *  The new variable constraint has the form \f$\sum_i \mu_i = 1\f$ where
 *  \f$\mu_i\f$ is the lp variable corresponding to the \f$i\f$-th label of the
 *  graphical model variable.
 *
 *  \param[in] variableID The variable for which the new constraint will be
 *                        added.
 *  \param[in] addToModel Indicator to tell if the constraint shall be added
 *                        directly to the LP/MIP model. If set to false the
 *                        constraint will be added to
 *                        LPInferenceBase::inactiveConstraints_.
 */

/*! \fn void LPInferenceBase::addLocalPolytopeFactorConstraint(const IndexType factor, const IndexType variable, const LabelType label, const bool addToModel)
 *  \brief Add a new factor constraint to the lp model.
 *
 *  The new factor constraint has the form
 *  \f$\sum_i \mu\{f;i_1,...,i_n\} - \mu\{b;j\} = 0\f$ where
 *  \f$\mu\{b;j\}\f$ is the lp variable corresponding to the \f$j\f$-th label of
 *  variable \f$b\f$ from the graphical model and \f$\mu\{f;i_1,...,i_n\}\f$ is
 *  the lp variable which corresponds to the \f$i\f$-th variable of the factor
 *  where variable \f$b\f$ is fixed to the label \f$j\f$.
 *
 *  \param[in] factor The factor for which the constraint will be added to the
 *                    lp model.
 *  \param[in] variable The variable of the factor for which the constraint will
 *                      be added to the lp model.
 *  \param[in] label The label of the variable of the factor for which the
 *                   constraint will be added to the lp model.
 *  \param[in] addToModel Indicator to tell if the constraint shall be added
 *                        directly to the LP/MIP model. If set to false the
 *                        constraint will be added to
 *                        LPInferenceBase::inactiveConstraints_.
 */

/*! \fn void LPInferenceBase::addIndicatorVariableConstraints(const IndexType factor, const IndicatorVariableType& indicatorVariable, const SolverIndexType indicatorVariableLPVariable, const bool addToModel)
 *  \brief Add constraints for an indicator variable to the lp model.
 *
 *  Depending on the selected logical ooperator type there are three
 *  possibilities for an indicator variable:
 *  -# An indicator variable corresponds to the logical conjunction of several
 *     other lp variables \f$(v_1 \wedge ... \wedge v_n)\f$. Hence the indicator
 *     variable \f$I\f$ is one if all lp variables \f$\{v_1, ..., v_n\}\f$ are
 *     one and zero if at least one lp variable \f$v_i\f$ is zero. The
 *     constraints which are added to achieve the logical conjunction are:
 *     -# \f[I - v_i \leq 0 \quad \forall i\f]
 *     -# \f[\sum_i v_i - I \leq n - 1\f]
 *  -# An indicator variable corresponds to the logical disjunction of several
 *     other lp variables \f$(v_1 \wedge ... \wedge v_n)\f$. Hence the indicator
 *     variable \f$I\f$ is one if at least one lp variable \f$v_i \in
 *     \{v_1, ..., v_n\}\f$ is one and zero if all lp variables are zero. The
 *     constraints which are added to achieve the logical disjunction are:
 *     -# \f[I - v_i \geq 0 \quad \forall i\f]
 *     -# \f[\sum_i v_i - I \geq 0\f]
 *  -# An indicator variable corresponds to the logical not of several other lp
 *     variables \f$(v_1 \wedge ... \wedge v_n)\f$. Hence the indicator variable
 *     \f$I\f$ is one if all lp variables \f$\{v_1, ..., v_n\}\f$ are zero and
 *     zero if at least one lp variable \f$v_i\f$ is one. The constraints which
 *     are added to achieve the logical not are:
 *     -# \f[I + v_i \leq 1 \quad \forall i\f]
 *     -# \f[\sum_i v_i + I \geq 1\f]
 *
 *  \param[in] factor The index of the factor for which the indicator variable
 *                                    constraints will be added.
 *  \param[in] indicatorVariable The index of the indicator variable from the
 *                               factor for which the constraints will be added.
 *  \param[in] indicatorVariableLPVariable The index of the lp variable which
 *                                         represents the indicator variable.
 *  \param[in] addToModel Indicator to tell if the constraints shall be added
 *                        directly to the LP/MIP model. If set to false the
 *                        constraints will be added to
 *                        LPInferenceBase::inactiveConstraints_.
 */

/*! \fn void LPInferenceBase::addLinearConstraint(const IndexType linearConstraintFactor, const LinearConstraintType& constraint)
 *  \brief Add a new linear constraint from a linear constraint function to the
 *         lp model.
 *
 *  \param[in] linearConstraintFactor The index of the linear constraint factor
 *                                    for which the linear constraint will be
 *                                    added.
 *  \param[in] constraint The new linear constraint factor.
 */

/*! \fn InferenceTermination LPInferenceBase::infer_impl_selectRelaxation(VISITOR_TYPE& visitor)
 *  \brief Helper function for LPInferenceBase::infer_impl to select the
 *         relaxation template parameter.
 *
 *  \tparam VISITOR_TYPE The type of the visitor.
 *
 *  \param[in,out] visitor The visitor which will be passed to
 *                         LPInferenceBase::infer_impl.
 *
 *  \return The inference termination code of LPInferenceBase::infer_impl.
 *
 *  \note The large amount of parameters which can affect how inference is
 *        performed can lead to large if ... else statement blocks which would
 *        result in unreadable code. Therefore the parameters are evaluated in a
 *        chain of template functions where each function evaluates a specific
 *        parameter and tells the result of this evaluation to the next function
 *        in the chain via a template parameter. The chain which leads to the
 *        call of LPInferenceBase::infer_impl is:
 *        -# LPInferenceBase::infer_impl_selectRelaxation
 *        -# LPInferenceBase::infer_impl_selectHeuristic
 *        -# LPInferenceBase::infer_impl_selectIterations
 *        -# LPInferenceBase::infer_impl_selectViolatedConstraints
 *        -# LPInferenceBase::infer_impl_selectLPType
 *        -# LPInferenceBase::infer_impl.
 */

/*! \fn InferenceTermination LPInferenceBase::infer_impl_selectHeuristic(VISITOR_TYPE& visitor)
 *  \brief Helper function for LPInferenceBase::infer_impl to select the
 *         challenge heuristic template parameter.
 *
 *  \tparam VISITOR_TYPE The type of the visitor.
 *  \tparam RELAXATION The selected relaxation type.
 *
 *  \param[in,out] visitor The visitor which will be passed to
 *                         LPInferenceBase::infer_impl.
 *
 *  \return The inference termination code of LPInferenceBase::infer_impl.
 *
 *  \note The large amount of parameters which can affect how inference is
 *        performed can lead to large if ... else statement blocks which would
 *        result in unreadable code. Therefore the parameters are evaluated in a
 *        chain of template functions where each function evaluates a specific
 *        parameter and tells the result of this evaluation to the next function
 *        in the chain via a template parameter. The chain which leads to the
 *        call of LPInferenceBase::infer_impl is:
 *        -# LPInferenceBase::infer_impl_selectRelaxation
 *        -# LPInferenceBase::infer_impl_selectHeuristic
 *        -# LPInferenceBase::infer_impl_selectIterations
 *        -# LPInferenceBase::infer_impl_selectViolatedConstraints
 *        -# LPInferenceBase::infer_impl_selectLPType
 *        -# LPInferenceBase::infer_impl.
 */

/*! \fn InferenceTermination LPInferenceBase::infer_impl_selectIterations(VISITOR_TYPE& visitor)
 *  \brief Helper function for LPInferenceBase::infer_impl to select the use
 *         infinite iterations template parameter.
 *
 *  \tparam VISITOR_TYPE The type of the visitor.
 *  \tparam RELAXATION The selected relaxation type.
 *  \tparam HEURISTIC The selected Heuristic type.
 *
 *  \param[in,out] visitor The visitor which will be passed to
 *                         LPInferenceBase::infer_impl.
 *
 *  \return The inference termination code of LPInferenceBase::infer_impl.
 *
 *  \note The large amount of parameters which can affect how inference is
 *        performed can lead to large if ... else statement blocks which would
 *        result in unreadable code. Therefore the parameters are evaluated in a
 *        chain of template functions where each function evaluates a specific
 *        parameter and tells the result of this evaluation to the next function
 *        in the chain via a template parameter. The chain which leads to the
 *        call of LPInferenceBase::infer_impl is:
 *        -# LPInferenceBase::infer_impl_selectRelaxation
 *        -# LPInferenceBase::infer_impl_selectHeuristic
 *        -# LPInferenceBase::infer_impl_selectIterations
 *        -# LPInferenceBase::infer_impl_selectViolatedConstraints
 *        -# LPInferenceBase::infer_impl_selectLPType
 *        -# LPInferenceBase::infer_impl.
 */

/*! \fn InferenceTermination LPInferenceBase::infer_impl_selectViolatedConstraints(VISITOR_TYPE& visitor)
 *  \brief Helper function for LPInferenceBase::infer_impl to select the add all
 *         violated constraints template parameter.
 *
 *  \tparam VISITOR_TYPE The type of the visitor.
 *  \tparam RELAXATION The selected relaxation type.
 *  \tparam HEURISTIC The selected Heuristic type.
 *  \tparam USE_INFINITE_ITERATIONS Tell if inference is performed until no more
 *                                  violated constraints are found.
 *
 *  \param[in,out] visitor The visitor which will be passed to
 *                         LPInferenceBase::infer_impl.
 *
 *  \return The inference termination code of LPInferenceBase::infer_impl.
 *
 *  \note The large amount of parameters which can affect how inference is
 *        performed can lead to large if ... else statement blocks which would
 *        result in unreadable code. Therefore the parameters are evaluated in a
 *        chain of template functions where each function evaluates a specific
 *        parameter and tells the result of this evaluation to the next function
 *        in the chain via a template parameter. The chain which leads to the
 *        call of LPInferenceBase::infer_impl is:
 *        -# LPInferenceBase::infer_impl_selectRelaxation
 *        -# LPInferenceBase::infer_impl_selectHeuristic
 *        -# LPInferenceBase::infer_impl_selectIterations
 *        -# LPInferenceBase::infer_impl_selectViolatedConstraints
 *        -# LPInferenceBase::infer_impl_selectLPType
 *        -# LPInferenceBase::infer_impl.
 */

/*! \fn InferenceTermination LPInferenceBase::infer_impl_selectLPType(VISITOR_TYPE& visitor)
 *  \brief Helper function for LPInferenceBase::infer_impl to select the use
 *         integer constraints template parameter.
 *
 *  \tparam VISITOR_TYPE The type of the visitor.
 *  \tparam RELAXATION The selected relaxation type.
 *  \tparam HEURISTIC The selected Heuristic type.
 *  \tparam USE_INFINITE_ITERATIONS Tell if inference is performed until no more
 *                                  violated constraints are found.
 *  \tparam ADD_ALL_VIOLATED_CONSTRAINTS Tell if all violated constraints which
 *                                       were found during one iteration are
 *                                       added to the lp model.
 *
 *  \param[in,out] visitor The visitor which will be passed to
 *                         LPInferenceBase::infer_impl.
 *
 *  \return The inference termination code of LPInferenceBase::infer_impl.
 *
 *  \note The large amount of parameters which can affect how inference is
 *        performed can lead to large if ... else statement blocks which would
 *        result in unreadable code. Therefore the parameters are evaluated in a
 *        chain of template functions where each function evaluates a specific
 *        parameter and tells the result of this evaluation to the next function
 *        in the chain via a template parameter. The chain which leads to the
 *        call of LPInferenceBase::infer_impl is:
 *        -# LPInferenceBase::infer_impl_selectRelaxation
 *        -# LPInferenceBase::infer_impl_selectHeuristic
 *        -# LPInferenceBase::infer_impl_selectIterations
 *        -# LPInferenceBase::infer_impl_selectViolatedConstraints
 *        -# LPInferenceBase::infer_impl_selectLPType
 *        -# LPInferenceBase::infer_impl.
 */

/*! \fn InferenceTermination LPInferenceBase::infer_impl(VISITOR_TYPE& visitor)
 *  \brief The implementation of the inference method.
 *
 *  \tparam VISITOR_TYPE The type of the visitor.
 *  \tparam RELAXATION The selected relaxation type.
 *  \tparam HEURISTIC The selected Heuristic type.
 *  \tparam USE_INFINITE_ITERATIONS Tell if inference is performed until no more
 *                                  violated constraints are found.
 *  \tparam ADD_ALL_VIOLATED_CONSTRAINTS Tell if all violated constraints which
 *                                       were found during one iteration are
 *                                       added to the lp model.
 *  \tparam USE_INTEGER_CONSTRAINTS Tell if the current model is a LP or a MIP
 *                                  model.
 *
 *  \param[in,out] visitor The visitor used during inference.
 *
 *  \return The inference termination code.
 *
 *  \note The large amount of parameters which can affect how inference is
 *        performed can lead to large if ... else statement blocks which would
 *        result in unreadable code. Therefore the parameters are evaluated in a
 *        chain of template functions where each function evaluates a specific
 *        parameter and tells the result of this evaluation to the next function
 *        in the chain via a template parameter. The chain which leads to the
 *        call of LPInferenceBase::infer_impl is:
 *        -# LPInferenceBase::infer_impl_selectRelaxation
 *        -# LPInferenceBase::infer_impl_selectHeuristic
 *        -# LPInferenceBase::infer_impl_selectIterations
 *        -# LPInferenceBase::infer_impl_selectViolatedConstraints
 *        -# LPInferenceBase::infer_impl_selectLPType
 *        -# LPInferenceBase::infer_impl.
 */

/*! \fn bool LPInferenceBase::tightenPolytope()
 *  \brief Search for linear constraints which are violated by the current
 *         integer solution and add them to the MIP model.
 *
 *  \tparam RELAXATION Tell which relaxation is used and therefore which
 *                     constraints have to be evaluated to check if they are
 *                     violated.
 *  \tparam HEURISTIC Tell which challenge heuristic will be used to add
 *                    violated constraints.
 *  \tparam ADD_ALL_VIOLATED_CONSTRAINTS If set to true all violated constraints
 *                                       will be added to the MIP model.
 *                                       Otherwise only the number of violated
 *                                       constraints specified by the
 *                                       LPInferenceBase::parameter_ will be
 *                                       added.
 */

/*! \fn bool LPInferenceBase::tightenPolytopeRelaxed()
 *  \brief Search for linear constraints which are violated by the current
 *         relaxed solution and add them to the LP model.
 *
 *  \tparam RELAXATION Tell which relaxation is used and therefore which
 *                     constraints have to be evaluated to check if they are
 *                     violated.
 *  \tparam HEURISTIC Tell which challenge heuristic will be used to add
 *                    violated constraints.
 *  \tparam ADD_ALL_VIOLATED_CONSTRAINTS If set to true all violated constraints
 *                                       will be added to the LP model.
 *                                       Otherwise only the number of violated
 *                                       constraints specified by the
 *                                       LPInferenceBase::parameter_ will be
 *                                       added.
 */

/*! \fn void LPInferenceBase::checkInactiveConstraint(const ConstraintStorage& constraint, double& weight) const
 *  \brief Check if a given linear constraint from the local polytope
 *         constraints is violated.
 *
 *  \param[in] constraint The linear constraint which will be checked for
 *                        violation.
 *  \param[out] weight The weight by which the constraint is violated. Will be
 *                     set to 0.0 if the constraint is not violated.
 */

/*! \fn void LPInferenceBase::addInactiveConstraint(const ConstraintStorage& constraint)
 *  \brief Add a linear constraint from the local polytope constraint to the
 *         LP/MIP model.
 *
 *  \param[in] constraint The linear constraint which will be added to the
 *                        LP/MIP model.
 */

/*! \struct AddViolatedLinearConstraintsFunctor
 *  \brief Functor used to access the method challenge() of the underlying
 *         linear constraint function of a graphical model factor and to add a
 *         limited number of violated constraints to the LP/MIP model.
 *
 *  \tparam LP_INFERENCE_BASE_TYPE The type of opengm::LPInferenceBase for which
 *                                 the functor is used.
 *  \tparam HEURISTIC Tell which challenge heuristic will be used to add
 *                    violated constraints.
 */

/*! \var AddViolatedLinearConstraintsFunctor::tolerance_
 *  \brief The tolerance used for the method challenge() of the underlying
 *         linear constraint function of a graphical model factor.
 */

/*! \var AddViolatedLinearConstraintsFunctor::labelingBegin_
 *  \brief Iterator used to iterate over the current solution.
 */

/*! \var AddViolatedLinearConstraintsFunctor::numConstraintsAdded_
 *  \brief Indicator used to tell how many constraints were added to the
 *         LP/MIP model.
 */

/*! \var AddViolatedLinearConstraintsFunctor::lpInference_
 *  \brief Pointer pointing to the instance of opengm::LPInferenceBase to get
 *         access to the LP/MIP model.
 */

/*! \var AddViolatedLinearConstraintsFunctor::linearConstraintID_
 *  \brief Index of the linear constraint factor.
 */

/*! \var AddViolatedLinearConstraintsFunctor::sortedViolatedConstraintsList_
 *  \brief Storage for the violated linear constraints sorted by their weights.
 *         Only used when LPInferenceBase::Parameter::Weighted is used as
 *         challenge heuristic.
 */

/*! \fn void AddViolatedLinearConstraintsFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction)
 *  \brief Operator used to access the method challenge() of the underlying
 *         linear constraint function of a graphical model factor and to add a
 *         limited number of violated constraints to the LP/MIP model.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The underlying linear constraint
 *          function type of a graphical model factor.
 *
 *  \param[in] linearConstraintFunction The underlying linear constraint
 *             function of a graphical model factor.
 */

/*! \struct AddViolatedLinearConstraintsFunctor::AddViolatedLinearConstraintsFunctor_impl
 *  \brief Helper struct to distinguish between linear constraint functions and
 *         other function types. This is necessary as only linear constraint
 *         functions provide the method challenge().
 *
 *  \tparam FUNCTION_TYPE The function type used with
 *                        AddViolatedLinearConstraintsFunctor.
 *  \tparam IS_LINEAR_CONSTRAINT_FUNCTION Indicator to tell if FUNCTION_TYPE is
 *                                        a linear constraint function type.
 */

/*! \fn void AddViolatedLinearConstraintsFunctor::AddViolatedLinearConstraintsFunctor_impl::addViolatedLinearConstraintsFunctor_impl(AddViolatedLinearConstraintsFunctor& myself, const FUNCTION_TYPE& function)
 *  \brief Actual access to the method challenge() of the underlying linear
 *         constraint function of a graphical model factor.
 *
 *  \param[out] myself Reference to the functor
 *                     AddViolatedLinearConstraintsFunctor to access the
 *                     variables of the functor.
 *  \param[in] function The function which will be accessed.
 */




/*! \struct AddViolatedLinearConstraintsRelaxedFunctor
 *  \brief Functor used to access the method challengeRelaxed() of the
 *         underlying linear constraint function of a graphical model factor and
 *         to add a limited number of violated constraints to the LP/MIP model.
 *
 *  \tparam LP_INFERENCE_BASE_TYPE The type of opengm::LPInferenceBase for which
 *                                 the functor is used.
 *  \tparam HEURISTIC Tell which challenge heuristic will be used to add
 *                    violated constraints.
 */

/*! \var AddViolatedLinearConstraintsRelaxedFunctor::tolerance_
 *  \brief The tolerance used for the method challengeRelaxed() of the
 *         underlying linear constraint function of a graphical model factor.
 */

/*! \var AddViolatedLinearConstraintsRelaxedFunctor::labelingBegin_
 *  \brief Iterator used to iterate over the current solution.
 */

/*! \var AddViolatedLinearConstraintsRelaxedFunctor::numConstraintsAdded_
 *  \brief Indicator used to tell how many constraints were added to the
 *         LP/MIP model.
 */

/*! \var AddViolatedLinearConstraintsRelaxedFunctor::lpInference_
 *  \brief Pointer pointing to the instance of opengm::LPInferenceBase to get
 *         access to the LP/MIP model.
 */

/*! \var AddViolatedLinearConstraintsRelaxedFunctor::linearConstraintID_
 *  \brief Index of the linear constraint factor.
 */

/*! \var AddViolatedLinearConstraintsRelaxedFunctor::sortedViolatedConstraintsList_
 *  \brief Storage for the violated linear constraints sorted by their weights.
 *         Only used when LPInferenceBase::Parameter::Weighted is used as
 *         challenge heuristic.
 */

/*! \fn void AddViolatedLinearConstraintsRelaxedFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction)
 *  \brief Operator used to access the method challengeRelaxed() of the
 *         underlying linear constraint function of a graphical model factor
 *         and to add a limited number of violated constraints to the LP/MIP
 *         model.
 *
 *  \tparam LINEAR_CONSTRAINT_FUNCTION_TYPE The underlying linear constraint
 *          function type of a graphical model factor.
 *
 *  \param[in] linearConstraintFunction The underlying linear constraint
 *             function of a graphical model factor.
 */

/*! \struct AddViolatedLinearConstraintsRelaxedFunctor::AddViolatedLinearConstraintsRelaxedFunctor_impl
 *  \brief Helper struct to distinguish between linear constraint functions and
 *         other function types. This is necessary as only linear constraint
 *         functions provide the method challengeRelaxed().
 *
 *  \tparam FUNCTION_TYPE The function type used with
 *                        AddViolatedLinearConstraintsRelaxedFunctor.
 *  \tparam IS_LINEAR_CONSTRAINT_FUNCTION Indicator to tell if FUNCTION_TYPE is
 *                                        a linear constraint function type.
 */

/*! \fn void AddViolatedLinearConstraintsRelaxedFunctor::AddViolatedLinearConstraintsRelaxedFunctor_impl::addViolatedLinearConstraintsRelaxedFunctor_impl(AddViolatedLinearConstraintsRelaxedFunctor& myself, const FUNCTION_TYPE& function)
 *  \brief Actual access to the method challengeRelaxed() of the underlying
 *         linear constraint function of a graphical model factor.
 *
 *  \param[out] myself Reference to the functor
 *                     AddViolatedLinearConstraintsRelaxedFunctor to access the
 *                     variables of the functor.
 *  \param[in] function The function which will be accessed.
 */

/******************
 * implementation *
 *****************/
template <class LP_INFERENCE_TYPE>
inline LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::Parameter()
   : SolverParameterType(), integerConstraintNodeVar_(false),
     integerConstraintFactorVar_(false), useSoftConstraints_(false),
     useFunctionTransfer_(false), mergeParallelFactors_(false),
     nameConstraints_(false), relaxation_(LocalPolytope),
     maxNumIterations_(1000), maxNumConstraintsPerIter_(0),
     challengeHeuristic_(Random), tolerance_(OPENGM_FLOAT_TOL) {

}

template <class LP_INFERENCE_TYPE>
inline const typename LPInferenceBase<LP_INFERENCE_TYPE>::GraphicalModelType& LPInferenceBase<LP_INFERENCE_TYPE>::graphicalModel() const {
   return gm_;
}

template <class LP_INFERENCE_TYPE>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::infer() {
   EmptyVisitorType visitor;
   return infer(visitor);
}

template <class LP_INFERENCE_TYPE>
template<class VISITOR_TYPE>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::infer(VISITOR_TYPE& visitor) {
   // Inference is performed in the method infer_impl(). Therefore appropriate
   // template parameters have to be selected. This is done by the
   // infer_impl_select... methods.
   return infer_impl_selectRelaxation(visitor);
}

template <class LP_INFERENCE_TYPE>
inline typename LPInferenceBase<LP_INFERENCE_TYPE>::ValueType LPInferenceBase<LP_INFERENCE_TYPE>::bound() const {
   if(inferenceStarted_) {
      return static_cast<const LPInferenceType*>(this)->objectiveFunctionValueBound() + constValue_;
   }
   else{
      return AccumulationType::template ineutral<ValueType>();
   }
}

template <class LP_INFERENCE_TYPE>
inline typename LPInferenceBase<LP_INFERENCE_TYPE>::ValueType LPInferenceBase<LP_INFERENCE_TYPE>::value() const {
   std::vector<LabelType> states;
   arg(states);
   return gm_.evaluate(states);
}

template <class LP_INFERENCE_TYPE>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::arg(std::vector<LabelType>& x, const size_t N) const {
   x.resize(gm_.numberOfVariables());
   if(inferenceStarted_) {
      SolverSolutionIteratorType solutionIterator = static_cast<const LPInferenceType*>(this)->solutionBegin();
      for(IndexType node = 0; node < gm_.numberOfVariables(); ++node) {
         SolverIndexType currentNodeLPVariable = nodesLPVariablesOffset_[node];
         SolverValueType bestValue = solutionIterator[currentNodeLPVariable];
         LabelType state = 0;
         for(LabelType i = 1; i < gm_.numberOfLabels(node); ++i) {
            ++currentNodeLPVariable;
            const SolverValueType currentValue = solutionIterator[currentNodeLPVariable];
            if(currentValue > bestValue) {
               bestValue = currentValue;
               state = i;
            }
         }
         x[node] = state;
      }
      return NORMAL;
   } else {
      for(IndexType node = 0; node < gm_.numberOfVariables(); ++node) {
         x[node] = 0;
      }
      return UNKNOWN;
   }
}

template <class LP_INFERENCE_TYPE>
inline LPInferenceBase<LP_INFERENCE_TYPE>::LPInferenceBase(const GraphicalModelType& gm, const Parameter& parameter)
   : gm_(gm), parameter_(parameter), constValue_(0.0), unaryFactors_(),
     higherOrderFactors_(), linearConstraintFactors_(), transferableFactors_(),
     inferenceStarted_(false), numLPVariables_(0), numNodesLPVariables_(0),
     numFactorsLPVariables_(0), numLinearConstraintsLPVariables_(0),
     numTransferedFactorsLPVariables(0), numSlackVariables_(0),
     nodesLPVariablesOffset_(gm_.numberOfVariables()),
     factorsLPVariablesOffset_(gm_.numberOfFactors()),
     linearConstraintsLPVariablesIndicesLookupTable_(),
     transferedFactorsLPVariablesIndicesLookupTable_(),
     linearConstraintLPVariablesSubsequenceIndices_(),
     addLocalPolytopeFactorConstraintCachePreviousFactorID_(gm_.numberOfFactors()),
     addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_(),
     inactiveConstraints_() {
   if(!opengm::meta::Compare<OperatorType, opengm::Adder>::value) {
      throw RuntimeError("This implementation does only supports Min-Sum-Semiring and Max-Sum-Semiring.");
   }
   // sort factors
   sortFactors();

   // count number of required LP variables
   countLPVariables();

   // fill subsequence look up table for linear constraints
   fillLinearConstraintLPVariablesSubsequenceIndices();

   // set accumulation
   setAccumulation();

   // add variables
   addLPVariables();

   // create objective function
   createObjectiveFunction();

   // add constraints
   switch(parameter_.relaxation_){
      case Parameter::LocalPolytope : {
         addLocalPolytopeConstraints();
         break;
      }
      case Parameter::LoosePolytope : {
         addLoosePolytopeConstraints();
         break;
      }
      case Parameter::TightPolytope: {
         addTightPolytopeConstraints();
         break;
      }
      default : {
         throw RuntimeError("Unknown Relaxation");
      }
   }

   // Tell child class we are finished with adding constraints
   static_cast<LPInferenceType*>(this)->addConstraintsFinished();

   // clear cache (only needed durig construction)
   addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_ = marray::Marray<SolverIndexType>();
   addLocalPolytopeFactorConstraintCachePreviousFactorID_ = gm_.numberOfFactors();
}

template <class LP_INFERENCE_TYPE>
inline LPInferenceBase<LP_INFERENCE_TYPE>::~LPInferenceBase() {

}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::sortFactors() {
   typename LPFunctionTransferType::IsTransferableFunctor isTransferableFunctor;
   for(IndexType factorIndex = 0; factorIndex < gm_.numberOfFactors(); ++factorIndex){
      gm_[factorIndex].callFunctor(isTransferableFunctor);
      if((!parameter_.useSoftConstraints_) && gm_[factorIndex].isLinearConstraint()) {
         linearConstraintFactors_.push_back(factorIndex);
      } else if(parameter_.useFunctionTransfer_ && isTransferableFunctor.isTransferable_) {
         transferableFactors_.push_back(factorIndex);
      } else if(gm_[factorIndex].numberOfVariables() == 0) {
         const LabelType l = 0;
         constValue_ += gm_[factorIndex](&l);
      } else if(gm_[factorIndex].numberOfVariables() == 1) {
         unaryFactors_.push_back(factorIndex);
      } else if(gm_[factorIndex].numberOfVariables() > 1) {
         higherOrderFactors_.push_back(factorIndex);
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::countLPVariables() {
   // number of node LP variables
   for(IndexType node = 0; node < gm_.numberOfVariables(); ++node){
      numNodesLPVariables_ += gm_.numberOfLabels(node);
      nodesLPVariablesOffset_[node] = numLPVariables_;
      numLPVariables_ += gm_.numberOfLabels(node);
   }

   // set unary factors offset
   for(IndexType i = 0; i < unaryFactors_.size(); ++i) {
      const IndexType factorIndex = unaryFactors_[i];
      const IndexType node = gm_[factorIndex].variableIndex(0);
      factorsLPVariablesOffset_[factorIndex] = nodesLPVariablesOffset_[node];
   }

   // number of factor LP variables
   // lookup table to search for parallel factors
   // TODO The lookup might be faster by using a hashmap (requires C++11)
   std::map<std::vector<IndexType>, IndexType> higherOrderFactorVariablesLookupTable;
   for(IndexType i = 0; i < higherOrderFactors_.size(); ++i) {
      const IndexType factorIndex = higherOrderFactors_[i];
      bool duplicate = false;
      if(parameter_.mergeParallelFactors_) {
         // check if factor with same variables is already present in the model
         std::vector<IndexType> currentFactorVariables(gm_[factorIndex].numberOfVariables());
         for(IndexType j = 0; j < gm_[factorIndex].numberOfVariables(); ++j) {
            currentFactorVariables[j] = gm_[factorIndex].variableIndex(j);
         }
         const typename std::map<std::vector<IndexType>, IndexType>::const_iterator iter = higherOrderFactorVariablesLookupTable.find(currentFactorVariables);
         if(iter != higherOrderFactorVariablesLookupTable.end()) {
            // parallel factor found
            factorsLPVariablesOffset_[factorIndex] = factorsLPVariablesOffset_[iter->second];
            duplicate = true;
         } else {
            higherOrderFactorVariablesLookupTable[currentFactorVariables] = factorIndex;
         }
      }
      if(!duplicate) {
         const size_t currentSize = gm_[factorIndex].size();
         numFactorsLPVariables_ += currentSize;
         factorsLPVariablesOffset_[factorIndex] = numLPVariables_;
         numLPVariables_ += currentSize;
      }
   }

   OPENGM_ASSERT(numLPVariables_ == numNodesLPVariables_ + numFactorsLPVariables_);

   // count linear constraint variables
   // lookup table to search for parallel indicator variables
   // TODO The lookup might be faster by using a hashmap (requires C++11)
   std::map<std::pair<typename IndicatorVariableType::LogicalOperatorType, std::vector<std::pair<IndexType, LabelType> > >, SolverIndexType> linearConstraintIndicatorVariablesLookupTable;
   GetIndicatorVariablesOrderBeginFunctor getIndicatorVariablesOrderBegin;
   GetIndicatorVariablesOrderEndFunctor getIndicatorVariablesOrderEnd;
   for(IndexType i = 0; i < linearConstraintFactors_.size(); ++i) {
      const IndexType currentLinearConstraintFactorIndex = linearConstraintFactors_[i];
      factorsLPVariablesOffset_[currentLinearConstraintFactorIndex] = numLPVariables_;
      gm_[currentLinearConstraintFactorIndex].callFunctor(getIndicatorVariablesOrderBegin);
      IndicatorVariablesIteratorType currentIndicatorVariablesBegin = getIndicatorVariablesOrderBegin.indicatorVariablesOrderBegin_;
      gm_[currentLinearConstraintFactorIndex].callFunctor(getIndicatorVariablesOrderEnd);
      const IndicatorVariablesIteratorType currentIndicatorVariablesEnd = getIndicatorVariablesOrderEnd.indicatorVariablesOrderEnd_;

      linearConstraintsLPVariablesIndicesLookupTable_.push_back(std::map<const IndicatorVariableType, SolverIndexType>());
      const size_t numIndicatorVariables = std::distance(currentIndicatorVariablesBegin, currentIndicatorVariablesEnd);
      for(size_t j = 0; j < numIndicatorVariables; ++j) {
         const IndicatorVariableType& currentIndicatorVariable = *currentIndicatorVariablesBegin;

         SolverIndexType lpVariableIndex = std::numeric_limits<SolverIndexType>::max();
         const bool matchingLPVariableIndexFound = getLPVariableIndexFromIndicatorVariable(higherOrderFactorVariablesLookupTable, linearConstraintIndicatorVariablesLookupTable, currentIndicatorVariable, currentLinearConstraintFactorIndex, lpVariableIndex);

         if(matchingLPVariableIndexFound) {
            linearConstraintsLPVariablesIndicesLookupTable_.back()[currentIndicatorVariable] = lpVariableIndex;
         } else {
            // new LP variable required
            linearConstraintsLPVariablesIndicesLookupTable_.back()[currentIndicatorVariable] = numLPVariables_;
            const size_t currentIndicatorVariableSize = std::distance(currentIndicatorVariable.begin(), currentIndicatorVariable.end());
            std::vector<std::pair<IndexType, LabelType> > currentVariableLabelPairs(currentIndicatorVariableSize);
            for(size_t k = 0; k < currentIndicatorVariableSize; ++k) {
               currentVariableLabelPairs[k] = std::pair<IndexType, LabelType>(gm_[currentLinearConstraintFactorIndex].variableIndex((currentIndicatorVariable.begin() + k)->first), (currentIndicatorVariable.begin() + k)->second);
            }
            linearConstraintIndicatorVariablesLookupTable[make_pair(currentIndicatorVariable.getLogicalOperatorType(), currentVariableLabelPairs)] = numLPVariables_;
            ++numLinearConstraintsLPVariables_;
            ++numLPVariables_;
         }
         ++currentIndicatorVariablesBegin;
      }
      OPENGM_ASSERT(currentIndicatorVariablesBegin == currentIndicatorVariablesEnd);
   }

   OPENGM_ASSERT(linearConstraintFactors_.size() == linearConstraintsLPVariablesIndicesLookupTable_.size());
   OPENGM_ASSERT(numLPVariables_ == numNodesLPVariables_ + numFactorsLPVariables_ + numLinearConstraintsLPVariables_);

   // count lp variables for transferable factors
   typename LPFunctionTransferType::NumSlackVariablesFunctor numSlackVariablesFunctor;
   typename LPFunctionTransferType::GetIndicatorVariablesFunctor getIndicatorVariablesFunctor;
   for(IndexType i = 0; i < transferableFactors_.size(); ++i) {
      const IndexType currentTransferableFactorIndex = transferableFactors_[i];
      factorsLPVariablesOffset_[currentTransferableFactorIndex] = numLPVariables_;

      // get number of slack variables
      gm_[currentTransferableFactorIndex].callFunctor(numSlackVariablesFunctor);

      IndexType currentNumSlackVariables = 0;

      transferedFactorsLPVariablesIndicesLookupTable_.push_back(std::map<const IndicatorVariableType, SolverIndexType>());

      // get indicator variables of
      IndicatorVariablesContainerType currentIndicatorVariables;
      getIndicatorVariablesFunctor.variables_ = &currentIndicatorVariables;
      gm_[currentTransferableFactorIndex].callFunctor(getIndicatorVariablesFunctor);

      // fill transferedFactorsLPVariablesIndicesLookupTable_
      for(IndicatorVariablesIteratorType iter = currentIndicatorVariables.begin(); iter != currentIndicatorVariables.end(); ++iter) {
         const IndicatorVariableType& currentIndicatorVariable = *iter;
         const size_t currentIndicatorVariableSize = std::distance(currentIndicatorVariable.begin(), currentIndicatorVariable.end());

         if(currentIndicatorVariableSize == 1) {
            if(currentIndicatorVariable.begin()->first >= gm_[currentTransferableFactorIndex].numberOfVariables()) {
               // slack variable
               transferedFactorsLPVariablesIndicesLookupTable_.back()[currentIndicatorVariable] = numSlackVariables_ + currentNumSlackVariables; // note: slack variables indices will be shifted to the end of the indices later
               ++currentNumSlackVariables;
               continue;
            }
         }

         SolverIndexType lpVariableIndex;
         const bool matchingLPVariableIndexFound = getLPVariableIndexFromIndicatorVariable(higherOrderFactorVariablesLookupTable, linearConstraintIndicatorVariablesLookupTable, currentIndicatorVariable, currentTransferableFactorIndex, lpVariableIndex);

         if(matchingLPVariableIndexFound) {
            transferedFactorsLPVariablesIndicesLookupTable_.back()[currentIndicatorVariable] = lpVariableIndex;
         } else {
            // new LP variable required
            transferedFactorsLPVariablesIndicesLookupTable_.back()[currentIndicatorVariable] = numLPVariables_;
            const size_t currentIndicatorVariableSize = std::distance(currentIndicatorVariable.begin(), currentIndicatorVariable.end());
            std::vector<std::pair<IndexType, LabelType> > currentVariableLabelPairs(currentIndicatorVariableSize);
            for(size_t j = 0; j < currentIndicatorVariableSize; ++j) {
               currentVariableLabelPairs[j] = std::pair<IndexType, LabelType>(gm_[currentTransferableFactorIndex].variableIndex((currentIndicatorVariable.begin() + j)->first), (currentIndicatorVariable.begin() + j)->second);
            }
            linearConstraintIndicatorVariablesLookupTable[make_pair(currentIndicatorVariable.getLogicalOperatorType(), currentVariableLabelPairs)] = numLPVariables_;
            ++numTransferedFactorsLPVariables;
            ++numLPVariables_;
         }
      }

      OPENGM_ASSERT(currentNumSlackVariables == numSlackVariablesFunctor.numSlackVariables_);
      numSlackVariables_ += numSlackVariablesFunctor.numSlackVariables_;
   }

   // update slack variables indices to shift their indices to the end (indices >= numLPVariables_)
   typename LPFunctionTransferType::GetSlackVariablesOrderFunctor getSlackVariablesOrderFunctor;
   for(IndexType i = 0; i < transferableFactors_.size(); ++i) {
      const IndexType currentTransferableFactorIndex = transferableFactors_[i];

      // get slack variables
      IndicatorVariablesContainerType slackVariables;
      getSlackVariablesOrderFunctor.order_ = &slackVariables;
      gm_[currentTransferableFactorIndex].callFunctor(getSlackVariablesOrderFunctor);

      // shift indices
      for(IndicatorVariablesIteratorType iter = slackVariables.begin(); iter != slackVariables.end(); ++iter) {
         const IndicatorVariableType& currentSlackVariable = *iter;
         transferedFactorsLPVariablesIndicesLookupTable_[i][currentSlackVariable] += numLPVariables_;
      }
   }

   OPENGM_ASSERT(transferableFactors_.size() == transferedFactorsLPVariablesIndicesLookupTable_.size());
   OPENGM_ASSERT(numLPVariables_ == numNodesLPVariables_ + numFactorsLPVariables_ + numLinearConstraintsLPVariables_ + numTransferedFactorsLPVariables);
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::fillLinearConstraintLPVariablesSubsequenceIndices() {
   linearConstraintLPVariablesSubsequenceIndices_.resize(linearConstraintFactors_.size());
   if(parameter_.integerConstraintNodeVar_ || parameter_.integerConstraintFactorVar_) {
      for(size_t i = 0; i < linearConstraintFactors_.size(); ++i) {
         const IndexType currentFactor = linearConstraintFactors_[i];
         const size_t numVariables = gm_[currentFactor].numberOfVariables();
         linearConstraintLPVariablesSubsequenceIndices_[i].resize(numVariables);
         for(size_t j = 0; j < numVariables; ++j) {
            linearConstraintLPVariablesSubsequenceIndices_[i][j] = gm_[currentFactor].variableIndex(j);
         }
      }
   } else {
      GetIndicatorVariablesOrderBeginFunctor getIndicatorVariablesOrderBegin;
      GetIndicatorVariablesOrderEndFunctor getIndicatorVariablesOrderEnd;
      for(size_t i = 0; i < linearConstraintFactors_.size(); ++i) {
         const IndexType currentFactor = linearConstraintFactors_[i];
         gm_[currentFactor].callFunctor(getIndicatorVariablesOrderBegin);
         gm_[currentFactor].callFunctor(getIndicatorVariablesOrderEnd);
         const IndicatorVariablesIteratorType currentIndicatorVariablesEnd = getIndicatorVariablesOrderEnd.indicatorVariablesOrderEnd_;
         for(IndicatorVariablesIteratorType iter = getIndicatorVariablesOrderBegin.indicatorVariablesOrderBegin_; iter != currentIndicatorVariablesEnd; ++iter) {
            linearConstraintLPVariablesSubsequenceIndices_[i].push_back(linearConstraintsLPVariablesIndicesLookupTable_[i][*iter]);
         }
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::setAccumulation() {
   if(meta::Compare<AccumulationType, Minimizer>::value) {
      static_cast<LPInferenceType*>(this)->setObjective(LPInferenceType::Minimize);
   } else if(meta::Compare<AccumulationType, Maximizer>::value) {
      static_cast<LPInferenceType*>(this)->setObjective(LPInferenceType::Maximize);
   } else {
      throw RuntimeError("This implementation of lp inference does only support Minimizer or Maximizer accumulators");
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addLPVariables() {
   if(parameter_.integerConstraintNodeVar_) {
      static_cast<LPInferenceType*>(this)->addBinaryVariables(numNodesLPVariables_);
   } else {
      static_cast<LPInferenceType*>(this)->addContinuousVariables(numNodesLPVariables_, 0.0, 1.0);
   }

   if(parameter_.integerConstraintFactorVar_) {
      static_cast<LPInferenceType*>(this)->addBinaryVariables(numFactorsLPVariables_);
      static_cast<LPInferenceType*>(this)->addBinaryVariables(numLinearConstraintsLPVariables_);
      static_cast<LPInferenceType*>(this)->addBinaryVariables(numTransferedFactorsLPVariables);
   } else {
      static_cast<LPInferenceType*>(this)->addContinuousVariables(numFactorsLPVariables_, 0.0, 1.0);
      static_cast<LPInferenceType*>(this)->addContinuousVariables(numLinearConstraintsLPVariables_, 0.0, 1.0);
      static_cast<LPInferenceType*>(this)->addContinuousVariables(numTransferedFactorsLPVariables, 0.0, 1.0);
   }

   // add slack variables
   static_cast<LPInferenceType*>(this)->addContinuousVariables(numSlackVariables_, 0.0, LPInferenceType::infinity());
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::createObjectiveFunction() {
   std::vector<SolverValueType> objective(numNodesLPVariables_ + numFactorsLPVariables_ + numLinearConstraintsLPVariables_ + numTransferedFactorsLPVariables + numSlackVariables_, 0.0);

   // node lp variables coefficients
   for(IndexType i = 0; i < unaryFactors_.size(); i++) {
      const IndexType factorIndex = unaryFactors_[i];
      const IndexType node = gm_[factorIndex].variableIndex(0);
      for(LabelType j = 0; j < gm_.numberOfLabels(node); j++) {
         objective[nodeLPVariableIndex(node, j)] += static_cast<SolverValueType>(gm_[factorIndex](&j));
      }
   }

   // factor lp variables coefficients
   for(IndexType i = 0; i < higherOrderFactors_.size(); i++) {
      const IndexType factorIndex = higherOrderFactors_[i];
      // copy values
      std::vector<ValueType> tempValues(gm_[factorIndex].size());
      gm_[factorIndex].copyValues(tempValues.begin());
      for(size_t j = 0; j < tempValues.size(); ++j) {
         objective[factorLPVariableIndex(factorIndex, j)] += static_cast<SolverValueType>(tempValues[j]);
      }
   }

   // slack variables of transformed factors
   typename LPFunctionTransferType::GetSlackVariablesOrderFunctor getSlackVariablesOrderFunctor;
   typename LPFunctionTransferType::GetSlackVariablesObjectiveCoefficientsFunctor getSlackVariablesObjectiveCoefficientsFunctor;
   for(IndexType i = 0; i < transferableFactors_.size(); ++i) {
      const IndexType currentTransferableFactorIndex = transferableFactors_[i];

      // get slack variables
      IndicatorVariablesContainerType slackVariables;
      getSlackVariablesOrderFunctor.order_ = &slackVariables;
      gm_[currentTransferableFactorIndex].callFunctor(getSlackVariablesOrderFunctor);

      // get coefficients
      typename LPFunctionTransferType::SlackVariablesObjectiveCoefficientsContainerType coefficients;
      getSlackVariablesObjectiveCoefficientsFunctor.coefficients_ = &coefficients;
      gm_[currentTransferableFactorIndex].callFunctor(getSlackVariablesObjectiveCoefficientsFunctor);

      OPENGM_ASSERT(coefficients.size() == slackVariables.size());

      // add coefficients
      for(size_t j = 0; j < slackVariables.size(); ++j) {
         const IndicatorVariableType& currentSlackVariable = slackVariables[j];
         const ValueType currentCoefficient = coefficients[j];
         const SolverIndexType currentSlackVariableLPVariableIndex = transferedFactorsLPVariablesIndicesLookupTable_[i][currentSlackVariable];
         objective[currentSlackVariableLPVariableIndex] += static_cast<SolverValueType>(currentCoefficient);
      }
   }
   static_cast<LPInferenceType*>(this)->setObjectiveValue(objective.begin(), objective.end());
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addLocalPolytopeConstraints() {
   // \sum_i \mu_i = 1
   for(IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
      addLocalPolytopeVariableConstraint(i, true);
   }

   // \sum_i \mu_{f;i_1,...,i_n} - \mu{b;j}= 0
   for(IndexType i = 0; i < higherOrderFactors_.size(); ++i) {
      const IndexType factorIndex = higherOrderFactors_[i];
      for(IndexType j = 0; j < gm_[factorIndex].numberOfVariables(); ++j) {
         const IndexType node = gm_[factorIndex].variableIndex(j);
         for(LabelType k = 0; k < gm_.numberOfLabels(node); k++) {
            addLocalPolytopeFactorConstraint(i, j, k, true);
         }
      }
   }

   // add constraints for linear constraint factor variables
   GetIndicatorVariablesOrderBeginFunctor getIndicatorVariablesOrderBegin;
   GetIndicatorVariablesOrderEndFunctor getIndicatorVariablesOrderEnd;
   for(IndexType i = 0; i < linearConstraintFactors_.size(); ++i) {
      gm_[linearConstraintFactors_[i]].callFunctor(getIndicatorVariablesOrderBegin);
      gm_[linearConstraintFactors_[i]].callFunctor(getIndicatorVariablesOrderEnd);
      const size_t linearConstraintNumIndicatorVariables = std::distance(getIndicatorVariablesOrderBegin.indicatorVariablesOrderBegin_, getIndicatorVariablesOrderEnd.indicatorVariablesOrderEnd_);
      for(size_t j = 0; j < linearConstraintNumIndicatorVariables; ++j) {
         const IndexType currentFactor = linearConstraintFactors_[i];
         const IndicatorVariableType& currentIndicatorVariable = getIndicatorVariablesOrderBegin.indicatorVariablesOrderBegin_[j];
         const SolverIndexType currentLPVariable = linearConstraintsLPVariablesIndicesLookupTable_[i][currentIndicatorVariable];
         addIndicatorVariableConstraints(currentFactor, currentIndicatorVariable, currentLPVariable, true);
      }
   }

   // add constraints for transfered factor variables
   typename LPFunctionTransferType::GetIndicatorVariablesFunctor getIndicatorVariablesFunctor;
   for(IndexType i = 0; i < transferableFactors_.size(); ++i) {
      typename LPFunctionTransferType::IndicatorVariablesContainerType currentIndicatorVariables;
      getIndicatorVariablesFunctor.variables_ = &currentIndicatorVariables;
      gm_[transferableFactors_[i]].callFunctor(getIndicatorVariablesFunctor);
      const size_t transformedFactorsNumIndicatorVariables = currentIndicatorVariables.size();
      for(size_t j = 0; j < transformedFactorsNumIndicatorVariables; ++j) {
         const IndexType currentFactor = transferableFactors_[i];
         const IndicatorVariableType& currentIndicatorVariable = currentIndicatorVariables[j];
         const SolverIndexType currentLPVariable = transferedFactorsLPVariablesIndicesLookupTable_[i][currentIndicatorVariable];
         addIndicatorVariableConstraints(currentFactor, currentIndicatorVariable, currentLPVariable, true);
      }
   }

   // add constraints for transfered factors
   typename LPFunctionTransferType::GetLinearConstraintsFunctor getLinearConstraintsFunctor;
   for(IndexType i = 0; i < transferableFactors_.size(); ++i) {
      const IndexType currentTransferableFactorIndex = transferableFactors_[i];
      // get constraints
      typename LPFunctionTransferType::LinearConstraintsContainerType constraints;
      getLinearConstraintsFunctor.constraints_ = &constraints;
      gm_[currentTransferableFactorIndex].callFunctor(getLinearConstraintsFunctor);
      for(size_t j = 0; j < constraints.size(); ++j) {
         const LinearConstraintType& currentConstraint = constraints[j];
         std::vector<SolverIndexType> lpVariables(std::distance(currentConstraint.indicatorVariablesBegin(), currentConstraint.indicatorVariablesEnd()));
         for(size_t k = 0; k < lpVariables.size(); ++k) {
            lpVariables[k] = transferedFactorsLPVariablesIndicesLookupTable_[i][currentConstraint.indicatorVariablesBegin()[k]];
         }
         switch(currentConstraint.getConstraintOperator()) {
            case LinearConstraintType::LinearConstraintOperatorType::LessEqual : {
               static_cast<LPInferenceType*>(this)->addLessEqualConstraint(lpVariables.begin(), lpVariables.end(), currentConstraint.coefficientsBegin(), static_cast<const SolverValueType>(currentConstraint.getBound()));
               break;
            }
            case LinearConstraintType::LinearConstraintOperatorType::Equal : {
               static_cast<LPInferenceType*>(this)->addEqualityConstraint(lpVariables.begin(), lpVariables.end(), currentConstraint.coefficientsBegin(), static_cast<const SolverValueType>(currentConstraint.getBound()));
               break;
            }
            default: {
               // default corresponds to LinearConstraintType::LinearConstraintOperatorType::GreaterEqual case
               static_cast<LPInferenceType*>(this)->addGreaterEqualConstraint(lpVariables.begin(), lpVariables.end(), currentConstraint.coefficientsBegin(), static_cast<const SolverValueType>(currentConstraint.getBound()));
            }
         }
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addLoosePolytopeConstraints() {
   // \sum_i \mu_i = 1
   for(IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
      addLocalPolytopeVariableConstraint(i, true);
   }

   // \sum_i \mu_{f;i_1,...,i_n} - \mu{b;j}= 0
   for(IndexType i = 0; i < higherOrderFactors_.size(); ++i) {
      const IndexType factorIndex = higherOrderFactors_[i];
      for(IndexType j = 0; j < gm_[factorIndex].numberOfVariables(); ++j) {
         const IndexType node = gm_[factorIndex].variableIndex(j);
         for(LabelType k = 0; k < gm_.numberOfLabels(node); k++) {
            addLocalPolytopeFactorConstraint(i, j, k, false);
         }
      }
   }

   // add constraints for linear constraint factor variables
   GetIndicatorVariablesOrderBeginFunctor getIndicatorVariablesOrderBegin;
   GetIndicatorVariablesOrderEndFunctor getIndicatorVariablesOrderEnd;
   for(IndexType i = 0; i < linearConstraintFactors_.size(); ++i) {
      gm_[linearConstraintFactors_[i]].callFunctor(getIndicatorVariablesOrderBegin);
      gm_[linearConstraintFactors_[i]].callFunctor(getIndicatorVariablesOrderEnd);
      const size_t linearConstraintNumIndicatorVariables = std::distance(getIndicatorVariablesOrderBegin.indicatorVariablesOrderBegin_, getIndicatorVariablesOrderEnd.indicatorVariablesOrderEnd_);
      for(size_t j = 0; j < linearConstraintNumIndicatorVariables; ++j) {
         const IndexType currentFactor = linearConstraintFactors_[i];
         const IndicatorVariableType& currentIndicatorVariable = getIndicatorVariablesOrderBegin.indicatorVariablesOrderBegin_[j];
         const SolverIndexType currentLPVariable = linearConstraintsLPVariablesIndicesLookupTable_[i][currentIndicatorVariable];
         addIndicatorVariableConstraints(currentFactor, currentIndicatorVariable, currentLPVariable, false);
      }
   }

   // add constraints for transfered factor variables
   typename LPFunctionTransferType::GetIndicatorVariablesFunctor getIndicatorVariablesFunctor;
   for(IndexType i = 0; i < transferableFactors_.size(); ++i) {
      typename LPFunctionTransferType::IndicatorVariablesContainerType currentIndicatorVariables;
      getIndicatorVariablesFunctor.variables_ = &currentIndicatorVariables;
      gm_[transferableFactors_[i]].callFunctor(getIndicatorVariablesFunctor);
      const size_t transformedFactorsNumIndicatorVariables = currentIndicatorVariables.size();
      for(size_t j = 0; j < transformedFactorsNumIndicatorVariables; ++j) {
         const IndexType currentFactor = transferableFactors_[i];
         const IndicatorVariableType& currentIndicatorVariable = currentIndicatorVariables[j];
         const SolverIndexType currentLPVariable = transferedFactorsLPVariablesIndicesLookupTable_[i][currentIndicatorVariable];
         addIndicatorVariableConstraints(currentFactor, currentIndicatorVariable, currentLPVariable, false);
      }
   }

   // add constraints for transfered factors
   typename LPFunctionTransferType::GetLinearConstraintsFunctor getLinearConstraintsFunctor;
   for(IndexType i = 0; i < transferableFactors_.size(); ++i) {
      const IndexType currentTransferableFactorIndex = transferableFactors_[i];
      // get constraints
      typename LPFunctionTransferType::LinearConstraintsContainerType constraints;
      getLinearConstraintsFunctor.constraints_ = &constraints;
      gm_[currentTransferableFactorIndex].callFunctor(getLinearConstraintsFunctor);
      for(size_t j = 0; j < constraints.size(); ++j) {
         const LinearConstraintType& currentConstraint = constraints[j];
         std::vector<SolverIndexType> lpVariables(std::distance(currentConstraint.indicatorVariablesBegin(), currentConstraint.indicatorVariablesEnd()));
         for(size_t k = 0; k < lpVariables.size(); ++k) {
            lpVariables[k] = transferedFactorsLPVariablesIndicesLookupTable_[i][currentConstraint.indicatorVariablesBegin()[k]];
         }

         std::stringstream constraintName;
         if(parameter_.nameConstraints_) {
            constraintName << "transfered factor " << currentTransferableFactorIndex << " constraint " << j << " of " << constraints.size();
         }
         ConstraintStorage constraint;
         constraint.variableIDs_ = lpVariables;
         constraint.coefficients_ = std::vector<SolverValueType>(currentConstraint.coefficientsBegin(), currentConstraint.coefficientsEnd());
         constraint.bound_ = currentConstraint.getBound();
         constraint.operator_ = currentConstraint.getConstraintOperator();
         constraint.name_ = constraintName.str();
         inactiveConstraints_.push_back(constraint);
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addTightPolytopeConstraints() {
   addLocalPolytopeConstraints();

   // Add all linear constraints from all linear constraint functions
   GetLinearConstraintsBeginFunctor getLinearConstraintsBegin;
   GetLinearConstraintsEndFunctor getLinearConstraintsEnd;
   for(IndexType i = 0; i < linearConstraintFactors_.size(); ++i) {
      const IndexType currentLinearConstraintFactorIndex = linearConstraintFactors_[i];
      gm_[currentLinearConstraintFactorIndex].callFunctor(getLinearConstraintsBegin);
      LinearConstraintsIteratorType currentLinearConstraintsBegin = getLinearConstraintsBegin.linearConstraintsBegin_;
      gm_[currentLinearConstraintFactorIndex].callFunctor(getLinearConstraintsEnd);
      const LinearConstraintsIteratorType currentLinearConstraintsEnd = getLinearConstraintsEnd.linearConstraintsEnd_;
      while(currentLinearConstraintsBegin != currentLinearConstraintsEnd) {
         addLinearConstraint(i, *currentLinearConstraintsBegin);
         ++currentLinearConstraintsBegin;
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline typename LPInferenceBase<LP_INFERENCE_TYPE>::SolverIndexType LPInferenceBase<LP_INFERENCE_TYPE>::nodeLPVariableIndex(const IndexType nodeID, const LabelType label) const {
   OPENGM_ASSERT(nodeID < gm_.numberOfVariables());
   OPENGM_ASSERT(label < gm_.numberOfLabels(nodeID));
   return nodesLPVariablesOffset_[nodeID] + label;
}

template <class LP_INFERENCE_TYPE>
inline typename LPInferenceBase<LP_INFERENCE_TYPE>::SolverIndexType LPInferenceBase<LP_INFERENCE_TYPE>::factorLPVariableIndex (const IndexType factorID, const size_t labelingIndex) const {
   OPENGM_ASSERT(factorID < gm_.numberOfFactors());
   OPENGM_ASSERT(labelingIndex < gm_[factorID].size());

   return factorsLPVariablesOffset_[factorID] + labelingIndex;
}

template <class LP_INFERENCE_TYPE>
template<class LABELING_ITERATOR_TYPE>
inline typename LPInferenceBase<LP_INFERENCE_TYPE>::SolverIndexType LPInferenceBase<LP_INFERENCE_TYPE>::factorLPVariableIndex(const IndexType factorID, LABELING_ITERATOR_TYPE labelingBegin, const LABELING_ITERATOR_TYPE labelingEnd) const {
   OPENGM_ASSERT(factorID < gm_.numberOfFactors());
   OPENGM_ASSERT(static_cast<IndexType>(std::distance(labelingBegin, labelingEnd)) == gm_[factorID].numberOfVariables());

   const size_t numVar = gm_[factorID].numberOfVariables();
   size_t labelingIndex = *labelingBegin;
   labelingBegin++;
   size_t strides = gm_[factorID].numberOfLabels(0);
   for(size_t i = 1; i < numVar; i++){
      OPENGM_ASSERT(*labelingBegin < gm_[factorID].numberOfLabels(i));
      labelingIndex += strides * (*labelingBegin);
      strides *= gm_[factorID].numberOfLabels(i);
      labelingBegin++;
   }

   OPENGM_ASSERT(labelingBegin == labelingEnd);

   return factorLPVariableIndex(factorID, labelingIndex);
}

template <class LP_INFERENCE_TYPE>
template <class HIGHER_ORDER_FACTORS_MAP_TYPE, class INDICATOR_VARIABLES_MAP_TYPE>
inline bool LPInferenceBase<LP_INFERENCE_TYPE>::getLPVariableIndexFromIndicatorVariable(const HIGHER_ORDER_FACTORS_MAP_TYPE& higherOrderFactorVariablesLookupTable, const INDICATOR_VARIABLES_MAP_TYPE& indicatorVariablesLookupTable, const IndicatorVariableType& indicatorVariable, const IndexType linearConstraintFactorIndex, SolverIndexType& lpVariableIndex) const {
   const size_t indicatorVariableSize = std::distance(indicatorVariable.begin(), indicatorVariable.end());
   OPENGM_ASSERT(indicatorVariableSize > 0);

   if(indicatorVariableSize == 1) {
      const IndexType currentNode = gm_[linearConstraintFactorIndex].variableIndex(indicatorVariable.begin()->first);
      const LabelType currentLabel = indicatorVariable.begin()->second;
      if(indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::Not) {
         if(gm_.numberOfLabels(currentNode) == 2) {
            OPENGM_ASSERT(currentLabel < 2);
            // use second label as not variable
            lpVariableIndex = nodeLPVariableIndex(currentNode, currentLabel == 0 ? LabelType(1) : LabelType(0));
            return true;
         } else {
            return false;
         }
      } else {
         lpVariableIndex = nodeLPVariableIndex(currentNode, currentLabel);
         return true;
      }
   } else {
      // search if any factor has the same variable combination
      if(indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::And) {
         std::vector<IndexType> currentVariables(indicatorVariableSize);
         std::vector<LabelType> currentLabeling(indicatorVariableSize);
         for(size_t i = 0; i < indicatorVariableSize; ++i) {
            currentVariables[i] = gm_[linearConstraintFactorIndex].variableIndex((indicatorVariable.begin() + i)->first);
            currentLabeling[i] = (indicatorVariable.begin() + i)->second;
         }
         const typename HIGHER_ORDER_FACTORS_MAP_TYPE::const_iterator iter = higherOrderFactorVariablesLookupTable.find(currentVariables);
         if(iter != higherOrderFactorVariablesLookupTable.end()) {
            // matching factor found
            lpVariableIndex = factorLPVariableIndex(iter->second, currentLabeling.begin(), currentLabeling.end());
            return true;
         }
      }
      // search if any previous linear constraint has the same variable combination
      std::vector<std::pair<IndexType, LabelType> > currentVariableLabelPairs(indicatorVariableSize);
      for(size_t i = 0; i < indicatorVariableSize; ++i) {
         currentVariableLabelPairs[i] = std::pair<IndexType, LabelType>(gm_[linearConstraintFactorIndex].variableIndex((indicatorVariable.begin() + i)->first), (indicatorVariable.begin() + i)->second);
      }
      const typename INDICATOR_VARIABLES_MAP_TYPE::const_iterator iter = indicatorVariablesLookupTable.find(make_pair(indicatorVariable.getLogicalOperatorType(), currentVariableLabelPairs));
      if(iter != indicatorVariablesLookupTable.end()) {
         // indicator variable with same variable label combination found
         lpVariableIndex = iter->second;
         return true;
      } else {
         return false;
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addLocalPolytopeVariableConstraint(const IndexType variableID, const bool addToModel) {
   OPENGM_ASSERT(variableID < gm_.numberOfVariables());
   static std::vector<SolverIndexType> variableIDs;
   static std::vector<SolverValueType> coefficients;

   // \sum_i \mu_i = 1
   const LabelType size = gm_.numberOfLabels(variableID);
   if(variableIDs.size() != size) {
      variableIDs.resize(size);
      coefficients.resize(size, 1.0);
   }
   for(LabelType j = 0; j < size; j++) {
      variableIDs[j] = nodeLPVariableIndex(variableID, j);
   }

   std::stringstream constraintName;
   if(parameter_.nameConstraints_) {
      constraintName << "local polytope variable constraint of variable " << variableID;
   }
   if(addToModel) {
      static_cast<LPInferenceType*>(this)->addEqualityConstraint(variableIDs.begin(), variableIDs.end(), coefficients.begin(), 1.0, constraintName.str());
   } else {
      ConstraintStorage constraint;
      constraint.variableIDs_ = variableIDs;
      constraint.coefficients_ = coefficients;
      constraint.bound_ = 1.0;
      constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::Equal;
      constraint.name_ = constraintName.str();
      inactiveConstraints_.push_back(constraint);
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addLocalPolytopeFactorConstraint(const IndexType factor, const IndexType variable, const LabelType label, const bool addToModel) {
   OPENGM_ASSERT(factor < higherOrderFactors_.size());
   OPENGM_ASSERT(variable < gm_[higherOrderFactors_[factor]].numberOfVariables());
   OPENGM_ASSERT(label < gm_[higherOrderFactors_[factor]].shape(variable));

   static std::vector<SolverIndexType> variableIDs;
   static std::vector<SolverValueType> coefficients;
   if(addLocalPolytopeFactorConstraintCachePreviousFactorID_ != higherOrderFactors_[factor]) {
      // update lookup table
      addLocalPolytopeFactorConstraintCachePreviousFactorID_ = higherOrderFactors_[factor];
      addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_.resize(gm_[higherOrderFactors_[factor]].shapeBegin(), gm_[higherOrderFactors_[factor]].shapeEnd());
      SolverIndexType counter = factorLPVariableIndex(higherOrderFactors_[factor], 0);
      for(typename marray::Marray<SolverIndexType>::iterator factorLPVariableIDsIter = addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_.begin(); factorLPVariableIDsIter != addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_.end(); ++factorLPVariableIDsIter) {
         *factorLPVariableIDsIter = counter++;
      }
   }

   marray::View<SolverIndexType> view = addLocalPolytopeFactorConstraintCacheFactorLPVariableIDs_.boundView(variable, label);
   const IndexType node = gm_[higherOrderFactors_[factor]].variableIndex(variable);
   const size_t size = view.size() + 1;
   const size_t containerSize = variableIDs.size();
   if(containerSize != size) {
      variableIDs.resize(size);
      // reset coefficients
      if(containerSize > 0) {
         coefficients[containerSize - 1] = 1.0;
      }
      coefficients.resize(size, 1.0);
      coefficients[size - 1] = -1.0;
   }
   SolverIndexType currentVariableIDsIndex = 0;
   for(typename marray::View<SolverIndexType>::iterator viewIter = view.begin(); viewIter != view.end(); ++viewIter) {
      variableIDs[currentVariableIDsIndex] = *viewIter;
      currentVariableIDsIndex++;
   }
   OPENGM_ASSERT(static_cast<size_t>(currentVariableIDsIndex) == size - 1);
   variableIDs[size - 1] = nodeLPVariableIndex(node, label);

   std::stringstream constraintName;
   if(parameter_.nameConstraints_) {
      constraintName << "local polytope factor constraint of higher order factor " << factor << " variable " << variable << " and label " << label;
   }
   if(addToModel) {
      static_cast<LPInferenceType*>(this)->addEqualityConstraint(variableIDs.begin(), variableIDs.end(), coefficients.begin(), 0.0, constraintName.str());
   } else {
      ConstraintStorage constraint;
      constraint.variableIDs_ = variableIDs;
      constraint.coefficients_ = coefficients;
      constraint.bound_ = 0.0;
      constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::Equal;
      constraint.name_ = constraintName.str();
      inactiveConstraints_.push_back(constraint);
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addIndicatorVariableConstraints(const IndexType factor, const IndicatorVariableType& indicatorVariable, const SolverIndexType indicatorVariableLPVariable, const bool addToModel) {
   OPENGM_ASSERT(factor < gm_.numberOfFactors());
   OPENGM_ASSERT(indicatorVariableLPVariable < numLPVariables_ + numSlackVariables_);
   if(indicatorVariableLPVariable >= numLPVariables_) {
      // slack variable nothing to do.
   } else {
      if(indicatorVariableLPVariable >= factorsLPVariablesOffset_[factor]) {
         // new constraints needed
         OPENGM_ASSERT(std::distance(indicatorVariable.begin(), indicatorVariable.end()) > 0);

         const SolverIndexType numVariables = static_cast<const SolverIndexType>(std::distance(indicatorVariable.begin(), indicatorVariable.end()));
         if(numVariables == 1) {
            // Only Not requires a new variable if the IndicatorVariable has size 1
            OPENGM_ASSERT(indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::Not);
            // Not: currentLPVariable + lpNodeVar(node; label) == 1.0
            std::vector<SolverValueType> coefficients(2, 1.0);
            std::vector<SolverIndexType> lpVariableIDs(2, indicatorVariableLPVariable);
            lpVariableIDs[0] = nodeLPVariableIndex(gm_[factor].variableIndex(indicatorVariable.begin()->first), indicatorVariable.begin()->second);
            std::stringstream constraintName;
            if(parameter_.nameConstraints_) {
               constraintName << "indicator variable constraint of factor " << factor << "of type Not for node" << indicatorVariable.begin()->first << " and label " << indicatorVariable.begin()->second;
            }
            if(addToModel) {
               static_cast<LPInferenceType*>(this)->addEqualityConstraint(lpVariableIDs.begin(), lpVariableIDs.end(), coefficients.begin(), 1.0, constraintName.str());
            } else {
               ConstraintStorage constraint;
               constraint.variableIDs_ = lpVariableIDs;
               constraint.coefficients_ = coefficients;
               constraint.bound_ = 1.0;
               constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::Equal;
               constraint.name_ = constraintName.str();
               inactiveConstraints_.push_back(constraint);
            }
         } else {
            OPENGM_ASSERT((indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::And) || (indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::Or) || (indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::Not) )

            // And: currentLPVariable - lpNodeVar(node; label) <= 0.0 for all node label pairs of the indicator variable
            // Or:  currentLPVariable - lpNodeVar(node; label) >= 0.0 for all node label pairs of the indicator variable
            // Not: currentLPVariable + lpNodeVar(node; label) <= 1.0 for all node label pairs of the indicator variable
            std::vector<SolverValueType> coefficients(2, 1.0);
            if((indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::And) || (indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::Or)) {
               coefficients.back() = -1.0;
            }
            std::vector<SolverIndexType> lpVariableIDs(2, indicatorVariableLPVariable);
            for(VariableLabelPairsIteratorType iter = indicatorVariable.begin(); iter != indicatorVariable.end(); ++iter) {
               lpVariableIDs[1] = nodeLPVariableIndex(gm_[factor].variableIndex(iter->first), iter->second);
               std::stringstream constraintName;
               if(parameter_.nameConstraints_) {
                  constraintName << "indicator variable constraint of factor " << factor << "of type ";
                  switch(indicatorVariable.getLogicalOperatorType()) {
                     case IndicatorVariableType::And : {
                        constraintName << "And";
                        break;
                     }
                     case IndicatorVariableType::Or : {
                        constraintName << "Or";
                        break;
                     }
                     default : {
                        // default corresponds to IndicatorVariableType::Not
                        constraintName << "Not";
                     }
                  }
                  constraintName << " for node" << iter->first << " and label " << iter->second;
               }

               if(addToModel) {
                  switch(indicatorVariable.getLogicalOperatorType()) {
                     case IndicatorVariableType::And : {
                        static_cast<LPInferenceType*>(this)->addLessEqualConstraint(lpVariableIDs.begin(), lpVariableIDs.end(), coefficients.begin(), 0.0, constraintName.str());
                        break;
                     }
                     case IndicatorVariableType::Or : {
                        static_cast<LPInferenceType*>(this)->addGreaterEqualConstraint(lpVariableIDs.begin(), lpVariableIDs.end(), coefficients.begin(), 0.0, constraintName.str());
                        break;
                     }
                     default : {
                        // default corresponds to IndicatorVariableType::Not
                        static_cast<LPInferenceType*>(this)->addLessEqualConstraint(lpVariableIDs.begin(), lpVariableIDs.end(), coefficients.begin(), 1.0, constraintName.str());
                     }
                  }
               } else {
                  ConstraintStorage constraint;
                  constraint.variableIDs_ = lpVariableIDs;
                  constraint.coefficients_ = coefficients;
                  switch(indicatorVariable.getLogicalOperatorType()) {
                     case IndicatorVariableType::And : {
                        constraint.bound_ = 0.0;
                        constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::LessEqual;
                        break;
                     }
                     case IndicatorVariableType::Or : {
                        constraint.bound_ = 0.0;
                        constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::GreaterEqual;
                        break;
                     }
                     default : {
                        // default corresponds to IndicatorVariableType::Not
                        constraint.bound_ = 1.0;
                        constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::LessEqual;
                     }
                  }
                  constraint.name_ = constraintName.str();
                  inactiveConstraints_.push_back(constraint);
               }
            }

            // And: \sum_i lpNodeVar(node_i; label_i) - currentLPVariable <= n - 1.0
            // Or:  \sum_i lpNodeVar(node_i; label_i) - currentLPVariable >= 0.0
            // Not: \sum_i lpNodeVar(node_i; label_i) + currentLPVariable >= 1.0
            if((indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::And) || (indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::Or)) {
               coefficients.back() = 1.0;
            }
            coefficients.resize(numVariables + 1, 1.0);
            if((indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::And) || (indicatorVariable.getLogicalOperatorType() == IndicatorVariableType::Or)) {
               coefficients.back() = -1.0;
            }
            lpVariableIDs.clear();
            for(VariableLabelPairsIteratorType iter = indicatorVariable.begin(); iter != indicatorVariable.end(); ++iter) {
               lpVariableIDs.push_back(nodeLPVariableIndex(gm_[factor].variableIndex(iter->first), iter->second));
            }
            lpVariableIDs.push_back(indicatorVariableLPVariable);
            std::stringstream constraintName;
            if(parameter_.nameConstraints_) {
               constraintName << "indicator variable sum constraint of factor " << factor << "of type ";
               switch(indicatorVariable.getLogicalOperatorType()) {
                  case IndicatorVariableType::And : {
                     constraintName << "And";
                     break;
                  }
                  case IndicatorVariableType::Or : {
                     constraintName << "Or";
                     break;
                  }
                  default : {
                     // default corresponds to IndicatorVariableType::Not
                     constraintName << "Not";
                  }
               }
               constraintName << " for lp variable" << indicatorVariableLPVariable;
            }
            if(addToModel) {
               switch(indicatorVariable.getLogicalOperatorType()) {
                  case IndicatorVariableType::And : {
                     static_cast<LPInferenceType*>(this)->addLessEqualConstraint(lpVariableIDs.begin(), lpVariableIDs.end(), coefficients.begin(), static_cast<const SolverValueType>(numVariables - 1.0), constraintName.str());
                     break;
                  }
                  case IndicatorVariableType::Or : {
                     static_cast<LPInferenceType*>(this)->addGreaterEqualConstraint(lpVariableIDs.begin(), lpVariableIDs.end(), coefficients.begin(), static_cast<const SolverValueType>(0.0), constraintName.str());
                     break;
                  }
                  default : {
                     // default corresponds to IndicatorVariableType::Not
                     static_cast<LPInferenceType*>(this)->addGreaterEqualConstraint(lpVariableIDs.begin(), lpVariableIDs.end(), coefficients.begin(), static_cast<const SolverValueType>(1.0), constraintName.str());
                  }
               }
            } else {
               ConstraintStorage constraint;
               constraint.variableIDs_ = lpVariableIDs;
               constraint.coefficients_ = coefficients;
               switch(indicatorVariable.getLogicalOperatorType()) {
                  case IndicatorVariableType::And : {
                     constraint.bound_ = static_cast<const SolverValueType>(numVariables - 1.0);
                     constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::LessEqual;
                     break;
                  }
                  case IndicatorVariableType::Or : {
                     constraint.bound_ = static_cast<const SolverValueType>(0.0);
                     constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::GreaterEqual;
                     break;
                  }
                  default : {
                     // default corresponds to IndicatorVariableType::Not
                     constraint.bound_ = static_cast<const SolverValueType>(1.0);
                     constraint.operator_ = LinearConstraintType::LinearConstraintOperatorType::GreaterEqual;
                  }
               }
               constraint.name_ = constraintName.str();
               inactiveConstraints_.push_back(constraint);
            }
         }
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addLinearConstraint(const IndexType linearConstraintFactor, const LinearConstraintType& constraint) {
   OPENGM_ASSERT(linearConstraintFactor < linearConstraintsLPVariablesIndicesLookupTable_.size());
   std::vector<SolverIndexType> lpVariables(std::distance(constraint.indicatorVariablesBegin(), constraint.indicatorVariablesEnd()));
   for(size_t i = 0; i < lpVariables.size(); ++i) {
      lpVariables[i] = linearConstraintsLPVariablesIndicesLookupTable_[linearConstraintFactor][constraint.indicatorVariablesBegin()[i]];
   }
   switch(constraint.getConstraintOperator()) {
      case LinearConstraintType::LinearConstraintOperatorType::LessEqual : {
         static_cast<LPInferenceType*>(this)->addLessEqualConstraint(lpVariables.begin(), lpVariables.end(), constraint.coefficientsBegin(), static_cast<const SolverValueType>(constraint.getBound()));
         break;
      }
      case LinearConstraintType::LinearConstraintOperatorType::Equal : {
         static_cast<LPInferenceType*>(this)->addEqualityConstraint(lpVariables.begin(), lpVariables.end(), constraint.coefficientsBegin(), static_cast<const SolverValueType>(constraint.getBound()));
         break;
      }
      default: {
         // default corresponds to LinearConstraintOperatorType::LinearConstraintOperator::GreaterEqual case
         static_cast<LPInferenceType*>(this)->addGreaterEqualConstraint(lpVariables.begin(), lpVariables.end(), constraint.coefficientsBegin(), static_cast<const SolverValueType>(constraint.getBound()));
      }
   }
}

template <class LP_INFERENCE_TYPE>
template <class VISITOR_TYPE>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::infer_impl_selectRelaxation(VISITOR_TYPE& visitor) {
   switch(parameter_.relaxation_){
      case Parameter::LocalPolytope : {
         return infer_impl_selectHeuristic<VISITOR_TYPE, Parameter::LocalPolytope>(visitor);
      }
      case Parameter::LoosePolytope : {
         return infer_impl_selectHeuristic<VISITOR_TYPE, Parameter::LoosePolytope>(visitor);
      }
      case Parameter::TightPolytope: {
         return infer_impl_selectHeuristic<VISITOR_TYPE, Parameter::TightPolytope>(visitor);
      }
      default : {
         throw RuntimeError("Unknown Relaxation");
      }
   }
}

template <class LP_INFERENCE_TYPE>
template <class VISITOR_TYPE, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::Relaxation RELAXATION>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::infer_impl_selectHeuristic(VISITOR_TYPE& visitor) {
   switch(parameter_.challengeHeuristic_){
      case Parameter::Random : {
         return infer_impl_selectIterations<VISITOR_TYPE, RELAXATION, Parameter::Random>(visitor);
      }
      case Parameter::Weighted : {
         return infer_impl_selectIterations<VISITOR_TYPE, RELAXATION, Parameter::Weighted>(visitor);
      }
      default : {
         throw RuntimeError("Unknown Heuristic");
      }
   }
}

template <class LP_INFERENCE_TYPE>
template <class VISITOR_TYPE, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::Relaxation RELAXATION, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::ChallengeHeuristic HEURISTIC>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::infer_impl_selectIterations(VISITOR_TYPE& visitor) {
   if(parameter_.maxNumIterations_ == 0) {
      return infer_impl_selectViolatedConstraints<VISITOR_TYPE, RELAXATION, HEURISTIC, true>(visitor);
   } else {
      return infer_impl_selectViolatedConstraints<VISITOR_TYPE, RELAXATION, HEURISTIC, false>(visitor);
   }
}

template <class LP_INFERENCE_TYPE>
template <class VISITOR_TYPE, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::Relaxation RELAXATION, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::ChallengeHeuristic HEURISTIC, bool USE_INFINITE_ITERATIONS>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::infer_impl_selectViolatedConstraints(VISITOR_TYPE& visitor) {
   if(parameter_.maxNumConstraintsPerIter_ == 0) {
      return infer_impl_selectLPType<VISITOR_TYPE, RELAXATION, HEURISTIC, USE_INFINITE_ITERATIONS, true>(visitor);
   } else {
      return infer_impl_selectLPType<VISITOR_TYPE, RELAXATION, HEURISTIC, USE_INFINITE_ITERATIONS, false>(visitor);
   }
}

template <class LP_INFERENCE_TYPE>
template <class VISITOR_TYPE, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::Relaxation RELAXATION, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::ChallengeHeuristic HEURISTIC, bool USE_INFINITE_ITERATIONS, bool ADD_ALL_VIOLATED_CONSTRAINTS>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::infer_impl_selectLPType(VISITOR_TYPE& visitor) {
   if(parameter_.integerConstraintNodeVar_ || parameter_.integerConstraintFactorVar_) {
      return infer_impl<VISITOR_TYPE, RELAXATION, HEURISTIC, USE_INFINITE_ITERATIONS, ADD_ALL_VIOLATED_CONSTRAINTS, true>(visitor);
   } else {
      return infer_impl<VISITOR_TYPE, RELAXATION, HEURISTIC, USE_INFINITE_ITERATIONS, ADD_ALL_VIOLATED_CONSTRAINTS, false>(visitor);
   }
}

template <class LP_INFERENCE_TYPE>
template <class VISITOR_TYPE, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::Relaxation RELAXATION, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::ChallengeHeuristic HEURISTIC, bool USE_INFINITE_ITERATIONS, bool ADD_ALL_VIOLATED_CONSTRAINTS, bool USE_INTEGER_CONSTRAINTS>
inline InferenceTermination LPInferenceBase<LP_INFERENCE_TYPE>::infer_impl(VISITOR_TYPE& visitor) {
   if(meta::Compare<VISITOR_TYPE, TimingVisitorType>::value) {
       visitor.addLog("LP Solver Time");
       visitor.addLog("Search Violated Constraints Time");
       visitor.addLog("Add Violated Constraints Time");
    }

    visitor.begin(*this);
    inferenceStarted_ = true;
    for(size_t i = 0; USE_INFINITE_ITERATIONS || (i < parameter_.maxNumIterations_);) {
       // solve problem
       if(meta::Compare<VISITOR_TYPE, TimingVisitorType>::value) {
          SolverTimingType solverTime;
          const bool solveSuccess = static_cast<LPInferenceType*>(this)->solve(solverTime);
          if(!solveSuccess) {
             // LPSOLVER failed to optimize
             return UNKNOWN;
          }
          const size_t visitorReturnFlag = visitor(*this);
          visitor.log("LP Solver Time", solverTime);
          if(visitorReturnFlag != visitors::VisitorReturnFlag::ContinueInf) {
             // timeout or bound reached
             break;
          }
       } else {
          if(!static_cast<LPInferenceType*>(this)->solve()) {
             // LPSOLVER failed to optimize
             return UNKNOWN;
          }
          if(visitor(*this) != visitors::VisitorReturnFlag::ContinueInf) {
             // timeout or bound reached
             break;
          }
       }
       // search violated constraints
       if(meta::Compare<VISITOR_TYPE, TimingVisitorType>::value) {
          static Timer searchViolatedConstraintsTimer;
          searchViolatedConstraintsTimer.reset();
          searchViolatedConstraintsTimer.tic();
          const bool newViolatedConstraintsFound = USE_INTEGER_CONSTRAINTS ? tightenPolytope<RELAXATION, HEURISTIC, ADD_ALL_VIOLATED_CONSTRAINTS>() : tightenPolytopeRelaxed<RELAXATION, HEURISTIC, ADD_ALL_VIOLATED_CONSTRAINTS>();
          searchViolatedConstraintsTimer.toc();
          visitor.log("Search Violated Constraints Time", searchViolatedConstraintsTimer.elapsedTime());
          if(newViolatedConstraintsFound){
             SolverTimingType addConstraintsTime;
             static_cast<LPInferenceType*>(this)->addConstraintsFinished(addConstraintsTime);
             visitor.log("Add Violated Constraints Time", addConstraintsTime);
          } else {
             // all constraints are satisfied
             break;
          }
       } else {
          if(USE_INTEGER_CONSTRAINTS ? tightenPolytope<RELAXATION, HEURISTIC, ADD_ALL_VIOLATED_CONSTRAINTS>() : tightenPolytopeRelaxed<RELAXATION, HEURISTIC, ADD_ALL_VIOLATED_CONSTRAINTS>()){
             static_cast<LPInferenceType*>(this)->addConstraintsFinished();
          } else {
             // all constraints are satisfied
             break;
          }
       }
       if(!USE_INFINITE_ITERATIONS) {
          ++i;
       }
    }
    visitor.end(*this);
    return NORMAL;
}

template <class LP_INFERENCE_TYPE>
template <typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::Relaxation RELAXATION, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::ChallengeHeuristic HEURISTIC, bool ADD_ALL_VIOLATED_CONSTRAINTS>
inline bool LPInferenceBase<LP_INFERENCE_TYPE>::tightenPolytope() {
   if(RELAXATION == Parameter::TightPolytope) {
      // nothing to tighten!
      return false;
   }

   // get current solution
   static std::vector<LabelType> currentArg;
   arg(currentArg);

   if(ADD_ALL_VIOLATED_CONSTRAINTS) {
      bool violatedConstraintAdded = false;
      if(RELAXATION == Parameter::LoosePolytope) {
         double currentWeight;
         InactiveConstraintsListIteratorType inactiveConstraintsBegin = inactiveConstraints_.begin();
         InactiveConstraintsListIteratorType inactiveConstraintsEnd = inactiveConstraints_.end();
         while(inactiveConstraintsBegin != inactiveConstraintsEnd) {
            checkInactiveConstraint(*inactiveConstraintsBegin, currentWeight);
            if(currentWeight > parameter_.tolerance_) {
               addInactiveConstraint(*inactiveConstraintsBegin);
               violatedConstraintAdded = true;
               const InactiveConstraintsListIteratorType removeInactiveConstraintIterator = inactiveConstraintsBegin;
               ++inactiveConstraintsBegin;
               inactiveConstraints_.erase(removeInactiveConstraintIterator);
               break;
            } else {
               ++inactiveConstraintsBegin;
            }
         }
         while(inactiveConstraintsBegin != inactiveConstraintsEnd) {
            checkInactiveConstraint(*inactiveConstraintsBegin, currentWeight);
            if(currentWeight > parameter_.tolerance_) {
               addInactiveConstraint(*inactiveConstraintsBegin);
               const InactiveConstraintsListIteratorType removeInactiveConstraintIterator = inactiveConstraintsBegin;
               ++inactiveConstraintsBegin;
               inactiveConstraints_.erase(removeInactiveConstraintIterator);
            } else {
               ++inactiveConstraintsBegin;
            }
         }
      }

      // add violated linear constraints from linear constraint factors
      size_t i = 0;
      AddAllViolatedLinearConstraintsFunctor addAllViolatedLinearConstraintsFunctor;
      addAllViolatedLinearConstraintsFunctor.tolerance_ = parameter_.tolerance_;
      addAllViolatedLinearConstraintsFunctor.lpInference_ = this;
      addAllViolatedLinearConstraintsFunctor.violatedConstraintAdded_ = false;
      if(!violatedConstraintAdded) {
         for(; i < linearConstraintFactors_.size(); ++i) {
            addAllViolatedLinearConstraintsFunctor.labelingBegin_ = IntegerSolutionSubsequenceIterator(currentArg.begin(), linearConstraintLPVariablesSubsequenceIndices_[i].begin());
            addAllViolatedLinearConstraintsFunctor.linearConstraintID_ = i;
            const IndexType currentFactor = linearConstraintFactors_[i];
            gm_[currentFactor].callFunctor(addAllViolatedLinearConstraintsFunctor);
            if(addAllViolatedLinearConstraintsFunctor.violatedConstraintAdded_) {
               violatedConstraintAdded = true;
               break;
            }
         }
      }
      for(; i < linearConstraintFactors_.size(); ++i) {
         for(; i < linearConstraintFactors_.size(); ++i) {
            addAllViolatedLinearConstraintsFunctor.labelingBegin_ = IntegerSolutionSubsequenceIterator(currentArg.begin(), linearConstraintLPVariablesSubsequenceIndices_[i].begin());
            addAllViolatedLinearConstraintsFunctor.linearConstraintID_ = i;
            const IndexType currentFactor = linearConstraintFactors_[i];
            gm_[currentFactor].callFunctor(addAllViolatedLinearConstraintsFunctor);
         }
      }
      return violatedConstraintAdded;
   } else {
      size_t numConstraintsAdded = 0;
      SortedViolatedConstraintsListType sortedViolatedConstraints;

      if(RELAXATION == Parameter::LoosePolytope) {
         double currentWeight;
         InactiveConstraintsListIteratorType inactiveConstraintsBegin = inactiveConstraints_.begin();
         InactiveConstraintsListIteratorType inactiveConstraintsEnd = inactiveConstraints_.end();
         while(inactiveConstraintsBegin != inactiveConstraintsEnd) {
            checkInactiveConstraint(*inactiveConstraintsBegin, currentWeight);
            if(currentWeight > parameter_.tolerance_) {
               if(HEURISTIC == Parameter::Random) {
                  addInactiveConstraint(*inactiveConstraintsBegin);
                  ++numConstraintsAdded;
                  const InactiveConstraintsListIteratorType removeInactiveConstraintIterator = inactiveConstraintsBegin;
                  ++inactiveConstraintsBegin;
                  inactiveConstraints_.erase(removeInactiveConstraintIterator);
                  if(numConstraintsAdded == parameter_.maxNumConstraintsPerIter_) {
                     break;
                  }
               } else {
                  sortedViolatedConstraints.insert(typename SortedViolatedConstraintsListType::value_type(currentWeight, std::make_pair(inactiveConstraintsBegin, std::make_pair(linearConstraintFactors_.size(), static_cast<const LinearConstraintType*>(NULL)))));
                  if(sortedViolatedConstraints.size() > parameter_.maxNumConstraintsPerIter_) {
                     // remove constraints with to small weight
                     sortedViolatedConstraints.erase(sortedViolatedConstraints.begin());
                     OPENGM_ASSERT(sortedViolatedConstraints.size() == parameter_.maxNumConstraintsPerIter_);
                  }
                  ++inactiveConstraintsBegin;
               }
            } else {
               ++inactiveConstraintsBegin;
            }
         }
      }

      // add violated linear constraints from linear constraint factors
      AddViolatedLinearConstraintsFunctor<LPInferenceBaseType, HEURISTIC> addViolatedLinearConstraintsFunctor;
      addViolatedLinearConstraintsFunctor.tolerance_ = parameter_.tolerance_;
      addViolatedLinearConstraintsFunctor.lpInference_ = this;
      addViolatedLinearConstraintsFunctor.numConstraintsAdded_ = numConstraintsAdded;
      addViolatedLinearConstraintsFunctor.sortedViolatedConstraintsList_ = &sortedViolatedConstraints;
      for(size_t i = 0; i < linearConstraintFactors_.size(); ++i) {
         addViolatedLinearConstraintsFunctor.labelingBegin_ = IntegerSolutionSubsequenceIterator(currentArg.begin(), linearConstraintLPVariablesSubsequenceIndices_[i].begin());
         addViolatedLinearConstraintsFunctor.linearConstraintID_ = i;
         const IndexType currentFactor = linearConstraintFactors_[i];
         gm_[currentFactor].callFunctor(addViolatedLinearConstraintsFunctor);
         if(addViolatedLinearConstraintsFunctor.numConstraintsAdded_ == parameter_.maxNumConstraintsPerIter_) {
            break;
         }
      }

      numConstraintsAdded = addViolatedLinearConstraintsFunctor.numConstraintsAdded_;
      typename SortedViolatedConstraintsListType::reverse_iterator sortedViolatedConstraintsListRBegin = sortedViolatedConstraints.rbegin();
      const typename SortedViolatedConstraintsListType::reverse_iterator sortedViolatedConstraintsListREnd = sortedViolatedConstraints.rend();
      OPENGM_ASSERT(sortedViolatedConstraints.size() <= parameter_.maxNumConstraintsPerIter_);
      while(sortedViolatedConstraintsListRBegin != sortedViolatedConstraintsListREnd) {
         if(sortedViolatedConstraintsListRBegin->second.first == inactiveConstraints_.end()) {
            addLinearConstraint(sortedViolatedConstraintsListRBegin->second.second.first, *(sortedViolatedConstraintsListRBegin->second.second.second));
         } else {
            addInactiveConstraint(*(sortedViolatedConstraintsListRBegin->second.first));
            inactiveConstraints_.erase(sortedViolatedConstraintsListRBegin->second.first);
         }
         ++numConstraintsAdded;
         ++sortedViolatedConstraintsListRBegin;
      }
      if(numConstraintsAdded == 0) {
         return false;
      } else {
         return true;
      }
   }
}

template <class LP_INFERENCE_TYPE>
template <typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::Relaxation RELAXATION, typename LPInferenceBase<LP_INFERENCE_TYPE>::Parameter::ChallengeHeuristic HEURISTIC, bool ADD_ALL_VIOLATED_CONSTRAINTS>
inline bool LPInferenceBase<LP_INFERENCE_TYPE>::tightenPolytopeRelaxed() {
   if(RELAXATION == Parameter::TightPolytope) {
      // nothing to tighten!
      return false;
   }

   // get current solution
   SolverSolutionIteratorType relaxedArgBegin = static_cast<const LPInferenceType*>(this)->solutionBegin();

   if(ADD_ALL_VIOLATED_CONSTRAINTS) {
      bool violatedConstraintAdded = false;
      if(RELAXATION == Parameter::LoosePolytope) {
         double currentWeight;
         InactiveConstraintsListIteratorType inactiveConstraintsBegin = inactiveConstraints_.begin();
         InactiveConstraintsListIteratorType inactiveConstraintsEnd = inactiveConstraints_.end();
         while(inactiveConstraintsBegin != inactiveConstraintsEnd) {
            checkInactiveConstraint(*inactiveConstraintsBegin, currentWeight);
            if(currentWeight > parameter_.tolerance_) {
               addInactiveConstraint(*inactiveConstraintsBegin);
               violatedConstraintAdded = true;
               const InactiveConstraintsListIteratorType removeInactiveConstraintIterator = inactiveConstraintsBegin;
               ++inactiveConstraintsBegin;
               inactiveConstraints_.erase(removeInactiveConstraintIterator);
               break;
            } else {
               ++inactiveConstraintsBegin;
            }
         }
         while(inactiveConstraintsBegin != inactiveConstraintsEnd) {
            checkInactiveConstraint(*inactiveConstraintsBegin, currentWeight);
            if(currentWeight > parameter_.tolerance_) {
               addInactiveConstraint(*inactiveConstraintsBegin);
               const InactiveConstraintsListIteratorType removeInactiveConstraintIterator = inactiveConstraintsBegin;
               ++inactiveConstraintsBegin;
               inactiveConstraints_.erase(removeInactiveConstraintIterator);
            } else {
               ++inactiveConstraintsBegin;
            }
         }
      }

      // add violated linear constraints from linear constraint factors
      size_t i = 0;
      AddAllViolatedLinearConstraintsRelaxedFunctor addAllViolatedLinearConstraintsRelaxedFunctor;
      addAllViolatedLinearConstraintsRelaxedFunctor.tolerance_ = parameter_.tolerance_;
      addAllViolatedLinearConstraintsRelaxedFunctor.lpInference_ = this;
      addAllViolatedLinearConstraintsRelaxedFunctor.violatedConstraintAdded_ = false;
      if(!violatedConstraintAdded) {
         for(; i < linearConstraintFactors_.size(); ++i) {
            addAllViolatedLinearConstraintsRelaxedFunctor.labelingBegin_ = RelaxedSolutionSubsequenceIterator(relaxedArgBegin, linearConstraintLPVariablesSubsequenceIndices_[i].begin());
            addAllViolatedLinearConstraintsRelaxedFunctor.linearConstraintID_ = i;
            const IndexType currentFactor = linearConstraintFactors_[i];
            gm_[currentFactor].callFunctor(addAllViolatedLinearConstraintsRelaxedFunctor);
            if(addAllViolatedLinearConstraintsRelaxedFunctor.violatedConstraintAdded_) {
               violatedConstraintAdded = true;
               break;
            }
         }
      }
      for(; i < linearConstraintFactors_.size(); ++i) {
         for(; i < linearConstraintFactors_.size(); ++i) {
            addAllViolatedLinearConstraintsRelaxedFunctor.labelingBegin_ = RelaxedSolutionSubsequenceIterator(relaxedArgBegin, linearConstraintLPVariablesSubsequenceIndices_[i].begin());
            addAllViolatedLinearConstraintsRelaxedFunctor.linearConstraintID_ = i;
            const IndexType currentFactor = linearConstraintFactors_[i];
            gm_[currentFactor].callFunctor(addAllViolatedLinearConstraintsRelaxedFunctor);
         }
      }
      return violatedConstraintAdded;
   } else {
      size_t numConstraintsAdded = 0;
      SortedViolatedConstraintsListType sortedViolatedConstraints;

      if(RELAXATION == Parameter::LoosePolytope) {
         double currentWeight;
         InactiveConstraintsListIteratorType inactiveConstraintsBegin = inactiveConstraints_.begin();
         InactiveConstraintsListIteratorType inactiveConstraintsEnd = inactiveConstraints_.end();
         while(inactiveConstraintsBegin != inactiveConstraintsEnd) {
            checkInactiveConstraint(*inactiveConstraintsBegin, currentWeight);
            if(currentWeight > parameter_.tolerance_) {
               if(HEURISTIC == Parameter::Random) {
                  addInactiveConstraint(*inactiveConstraintsBegin);
                  ++numConstraintsAdded;
                  const InactiveConstraintsListIteratorType removeInactiveConstraintIterator = inactiveConstraintsBegin;
                  ++inactiveConstraintsBegin;
                  inactiveConstraints_.erase(removeInactiveConstraintIterator);
                  if(numConstraintsAdded == parameter_.maxNumConstraintsPerIter_) {
                     break;
                  }
               } else {
                  sortedViolatedConstraints.insert(typename SortedViolatedConstraintsListType::value_type(currentWeight, std::make_pair(inactiveConstraintsBegin, std::make_pair(linearConstraintFactors_.size(), static_cast<LinearConstraintType*>(NULL)))));
                  if(sortedViolatedConstraints.size() > parameter_.maxNumConstraintsPerIter_) {
                     // remove constraints with to small weight
                     sortedViolatedConstraints.erase(sortedViolatedConstraints.begin());
                     OPENGM_ASSERT(sortedViolatedConstraints.size() == parameter_.maxNumConstraintsPerIter_);
                  }
                  ++inactiveConstraintsBegin;
               }
            } else {
               ++inactiveConstraintsBegin;
            }
         }
      }

      // add violated linear constraints from linear constraint factors
      AddViolatedLinearConstraintsRelaxedFunctor<LPInferenceBaseType, HEURISTIC> addViolatedLinearConstraintsRelaxedFunctor;
      addViolatedLinearConstraintsRelaxedFunctor.tolerance_ = parameter_.tolerance_;
      addViolatedLinearConstraintsRelaxedFunctor.lpInference_ = this;
      addViolatedLinearConstraintsRelaxedFunctor.numConstraintsAdded_ = numConstraintsAdded;
      addViolatedLinearConstraintsRelaxedFunctor.sortedViolatedConstraintsList_ = &sortedViolatedConstraints;
      for(size_t i = 0; i < linearConstraintFactors_.size(); ++i) {
         addViolatedLinearConstraintsRelaxedFunctor.labelingBegin_ = RelaxedSolutionSubsequenceIterator(relaxedArgBegin, linearConstraintLPVariablesSubsequenceIndices_[i].begin());
         addViolatedLinearConstraintsRelaxedFunctor.linearConstraintID_ = i;
         const IndexType currentFactor = linearConstraintFactors_[i];
         gm_[currentFactor].callFunctor(addViolatedLinearConstraintsRelaxedFunctor);
         if(addViolatedLinearConstraintsRelaxedFunctor.numConstraintsAdded_ == parameter_.maxNumConstraintsPerIter_) {
            break;
         }
      }

      numConstraintsAdded = addViolatedLinearConstraintsRelaxedFunctor.numConstraintsAdded_;
      typename SortedViolatedConstraintsListType::reverse_iterator sortedViolatedConstraintsListRBegin = sortedViolatedConstraints.rbegin();
      const typename SortedViolatedConstraintsListType::reverse_iterator sortedViolatedConstraintsListREnd = sortedViolatedConstraints.rend();
      OPENGM_ASSERT(sortedViolatedConstraints.size() <= parameter_.maxNumConstraintsPerIter_);
      while(sortedViolatedConstraintsListRBegin != sortedViolatedConstraintsListREnd) {
         if(sortedViolatedConstraintsListRBegin->second.first == inactiveConstraints_.end()) {
            addLinearConstraint(sortedViolatedConstraintsListRBegin->second.second.first, *(sortedViolatedConstraintsListRBegin->second.second.second));
         } else {
            addInactiveConstraint(*(sortedViolatedConstraintsListRBegin->second.first));
            inactiveConstraints_.erase(sortedViolatedConstraintsListRBegin->second.first);
         }
         ++numConstraintsAdded;
         if(numConstraintsAdded == parameter_.maxNumConstraintsPerIter_) {
            break;
         }
         ++sortedViolatedConstraintsListRBegin;
      }
      if(numConstraintsAdded == 0) {
         return false;
      } else {
         return true;
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::checkInactiveConstraint(const ConstraintStorage& constraint, double& weight) const {
   const SolverSolutionIteratorType currentSolution = static_cast<const LPInferenceType*>(this)->solutionBegin();
   double sum = 0.0;
   for(size_t i = 0; i < constraint.variableIDs_.size(); ++i) {
      sum += constraint.coefficients_[i] * currentSolution[constraint.variableIDs_[i]];
   }
   switch(constraint.operator_) {
      case LinearConstraintType::LinearConstraintOperatorType::LessEqual : {
         if(sum <= constraint.bound_) {
            weight = 0.0;
         } else {
            weight = sum - constraint.bound_;
         }
         break;
      }
      case LinearConstraintType::LinearConstraintOperatorType::Equal : {
         if(sum == constraint.bound_) {
            weight = 0.0;
         } else {
            weight = std::abs(sum - constraint.bound_);
         }
         break;
      }
      default: {
         // default corresponds to LinearConstraintType::LinearConstraintOperatorType::GreaterEqual case
         if(sum >= constraint.bound_) {
            weight = 0.0;
         } else {
            weight = constraint.bound_ - sum;
         }
         break;
      }
   }
}

template <class LP_INFERENCE_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::addInactiveConstraint(const ConstraintStorage& constraint) {
   switch(constraint.operator_) {
      case LinearConstraintType::LinearConstraintOperatorType::LessEqual : {
         static_cast<LPInferenceType*>(this)->addLessEqualConstraint(constraint.variableIDs_.begin(), constraint.variableIDs_.end(), constraint.coefficients_.begin(), constraint.bound_, constraint.name_);
         break;
      }
      case LinearConstraintType::LinearConstraintOperatorType::Equal : {
         static_cast<LPInferenceType*>(this)->addEqualityConstraint(constraint.variableIDs_.begin(), constraint.variableIDs_.end(), constraint.coefficients_.begin(), constraint.bound_, constraint.name_);
         break;
      }
      default: {
         // default corresponds to LinearConstraintType::LinearConstraintOperatorType::GreaterEqual case
         static_cast<LPInferenceType*>(this)->addGreaterEqualConstraint(constraint.variableIDs_.begin(), constraint.variableIDs_.end(), constraint.coefficients_.begin(), constraint.bound_, constraint.name_);
         break;
      }
   }
}

template <class LP_INFERENCE_TYPE>
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetIndicatorVariablesOrderBeginFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction) {
   GetIndicatorVariablesOrderBeginFunctor_impl<LINEAR_CONSTRAINT_FUNCTION_TYPE, meta::HasTypeInTypeList<LinearConstraintFunctionTypeList, LINEAR_CONSTRAINT_FUNCTION_TYPE>::value>::getIndicatorVariablesOrderBeginFunctor_impl(*this, linearConstraintFunction);
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetIndicatorVariablesOrderBeginFunctor::GetIndicatorVariablesOrderBeginFunctor_impl<FUNCTION_TYPE, IS_LINEAR_CONSTRAINT_FUNCTION>::getIndicatorVariablesOrderBeginFunctor_impl(GetIndicatorVariablesOrderBeginFunctor& myself, const FUNCTION_TYPE& function) {
   throw RuntimeError(std::string("GetIndicatorVariablesOrderBeginFunctor: Unsupported linear constraint function type") + typeid(FUNCTION_TYPE).name());
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetIndicatorVariablesOrderBeginFunctor::GetIndicatorVariablesOrderBeginFunctor_impl<FUNCTION_TYPE, true>::getIndicatorVariablesOrderBeginFunctor_impl(GetIndicatorVariablesOrderBeginFunctor& myself, const FUNCTION_TYPE& function) {
   myself.indicatorVariablesOrderBegin_ = function.indicatorVariablesOrderBegin();
}

template <class LP_INFERENCE_TYPE>
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetIndicatorVariablesOrderEndFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction) {
   GetIndicatorVariablesOrderEndFunctor_impl<LINEAR_CONSTRAINT_FUNCTION_TYPE, meta::HasTypeInTypeList<LinearConstraintFunctionTypeList, LINEAR_CONSTRAINT_FUNCTION_TYPE>::value>::getIndicatorVariablesOrderEndFunctor_impl(*this, linearConstraintFunction);
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetIndicatorVariablesOrderEndFunctor::GetIndicatorVariablesOrderEndFunctor_impl<FUNCTION_TYPE, IS_LINEAR_CONSTRAINT_FUNCTION>::getIndicatorVariablesOrderEndFunctor_impl(GetIndicatorVariablesOrderEndFunctor& myself, const FUNCTION_TYPE& function) {
   throw RuntimeError(std::string("GetIndicatorVariablesOrderEnd: Unsupported linear constraint function type") + typeid(FUNCTION_TYPE).name());
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetIndicatorVariablesOrderEndFunctor::GetIndicatorVariablesOrderEndFunctor_impl<FUNCTION_TYPE, true>::getIndicatorVariablesOrderEndFunctor_impl(GetIndicatorVariablesOrderEndFunctor& myself, const FUNCTION_TYPE& function) {
   myself.indicatorVariablesOrderEnd_ = function.indicatorVariablesOrderEnd();
}

template <class LP_INFERENCE_TYPE>
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetLinearConstraintsBeginFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction) {
   GetLinearConstraintsBeginFunctor_impl<LINEAR_CONSTRAINT_FUNCTION_TYPE, meta::HasTypeInTypeList<LinearConstraintFunctionTypeList, LINEAR_CONSTRAINT_FUNCTION_TYPE>::value>::getLinearConstraintsBeginFunctor_impl(*this, linearConstraintFunction);
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetLinearConstraintsBeginFunctor::GetLinearConstraintsBeginFunctor_impl<FUNCTION_TYPE, IS_LINEAR_CONSTRAINT_FUNCTION>::getLinearConstraintsBeginFunctor_impl(GetLinearConstraintsBeginFunctor& myself, const FUNCTION_TYPE& function) {
   throw RuntimeError(std::string("GetLinearConstraintsBeginFunctor: Unsupported linear constraint function type") + typeid(FUNCTION_TYPE).name());
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetLinearConstraintsBeginFunctor::GetLinearConstraintsBeginFunctor_impl<FUNCTION_TYPE, true>::getLinearConstraintsBeginFunctor_impl(GetLinearConstraintsBeginFunctor& myself, const FUNCTION_TYPE& function) {
   myself.linearConstraintsBegin_ = function.linearConstraintsBegin();
}

template <class LP_INFERENCE_TYPE>
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetLinearConstraintsEndFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction) {
   GetLinearConstraintsEndFunctor_impl<LINEAR_CONSTRAINT_FUNCTION_TYPE, meta::HasTypeInTypeList<LinearConstraintFunctionTypeList, LINEAR_CONSTRAINT_FUNCTION_TYPE>::value>::getLinearConstraintsEndFunctor_impl(*this, linearConstraintFunction);
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetLinearConstraintsEndFunctor::GetLinearConstraintsEndFunctor_impl<FUNCTION_TYPE, IS_LINEAR_CONSTRAINT_FUNCTION>::getLinearConstraintsEndFunctor_impl(GetLinearConstraintsEndFunctor& myself, const FUNCTION_TYPE& function) {
   throw RuntimeError(std::string("GetLinearConstraintsEndFunctor: Unsupported linear constraint function type") + typeid(FUNCTION_TYPE).name());
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::GetLinearConstraintsEndFunctor::GetLinearConstraintsEndFunctor_impl<FUNCTION_TYPE, true>::getLinearConstraintsEndFunctor_impl(GetLinearConstraintsEndFunctor& myself, const FUNCTION_TYPE& function) {
   myself.linearConstraintsEnd_ = function.linearConstraintsEnd();
}

template <class LP_INFERENCE_TYPE>
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::AddAllViolatedLinearConstraintsFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction) {
   AddAllViolatedLinearConstraintsFunctor_impl<LINEAR_CONSTRAINT_FUNCTION_TYPE, meta::HasTypeInTypeList<LinearConstraintFunctionTypeList, LINEAR_CONSTRAINT_FUNCTION_TYPE>::value>::addAllViolatedLinearConstraintsFunctor_impl(*this, linearConstraintFunction);
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::AddAllViolatedLinearConstraintsFunctor::AddAllViolatedLinearConstraintsFunctor_impl<FUNCTION_TYPE, IS_LINEAR_CONSTRAINT_FUNCTION>::addAllViolatedLinearConstraintsFunctor_impl(AddAllViolatedLinearConstraintsFunctor& myself, const FUNCTION_TYPE& function) {
   throw RuntimeError(std::string("AddAllViolatedLinearConstraintsFunctor: Unsupported linear constraint function type") + typeid(FUNCTION_TYPE).name());
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::AddAllViolatedLinearConstraintsFunctor::AddAllViolatedLinearConstraintsFunctor_impl<FUNCTION_TYPE, true>::addAllViolatedLinearConstraintsFunctor_impl(AddAllViolatedLinearConstraintsFunctor& myself, const FUNCTION_TYPE& function) {
   typename FUNCTION_TYPE::ViolatedLinearConstraintsIteratorType        violatedConstraintsBegin;
   typename FUNCTION_TYPE::ViolatedLinearConstraintsIteratorType        violatedConstraintsEnd;
   typename FUNCTION_TYPE::ViolatedLinearConstraintsWeightsIteratorType violatedConstraintsWeightsBegin;
   function.challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, myself.labelingBegin_, myself.tolerance_);
   if(std::distance(violatedConstraintsBegin, violatedConstraintsEnd) > 0) {
      while(violatedConstraintsBegin != violatedConstraintsEnd) {
         myself.lpInference_->addLinearConstraint(myself.linearConstraintID_, *violatedConstraintsBegin);
         ++violatedConstraintsBegin;
      }
      myself.violatedConstraintAdded_ = true;
   }
}

template <class LP_INFERENCE_TYPE>
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::AddAllViolatedLinearConstraintsRelaxedFunctor::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction) {
   AddAllViolatedLinearConstraintsRelaxedFunctor_impl<LINEAR_CONSTRAINT_FUNCTION_TYPE, meta::HasTypeInTypeList<LinearConstraintFunctionTypeList, LINEAR_CONSTRAINT_FUNCTION_TYPE>::value>::addAllViolatedLinearConstraintsRelaxedFunctor_impl(*this, linearConstraintFunction);
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::AddAllViolatedLinearConstraintsRelaxedFunctor::AddAllViolatedLinearConstraintsRelaxedFunctor_impl<FUNCTION_TYPE, IS_LINEAR_CONSTRAINT_FUNCTION>::addAllViolatedLinearConstraintsRelaxedFunctor_impl(AddAllViolatedLinearConstraintsRelaxedFunctor& myself, const FUNCTION_TYPE& function) {
   throw RuntimeError(std::string("AddAllViolatedLinearConstraintsRelaxedFunctor: Unsupported linear constraint function type") + typeid(FUNCTION_TYPE).name());
}

template <class LP_INFERENCE_TYPE>
template<class FUNCTION_TYPE>
inline void LPInferenceBase<LP_INFERENCE_TYPE>::AddAllViolatedLinearConstraintsRelaxedFunctor::AddAllViolatedLinearConstraintsRelaxedFunctor_impl<FUNCTION_TYPE, true>::addAllViolatedLinearConstraintsRelaxedFunctor_impl(AddAllViolatedLinearConstraintsRelaxedFunctor& myself, const FUNCTION_TYPE& function) {
   typename FUNCTION_TYPE::ViolatedLinearConstraintsIteratorType        violatedConstraintsBegin;
   typename FUNCTION_TYPE::ViolatedLinearConstraintsIteratorType        violatedConstraintsEnd;
   typename FUNCTION_TYPE::ViolatedLinearConstraintsWeightsIteratorType violatedConstraintsWeightsBegin;
   function.challengeRelaxed(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, myself.labelingBegin_, myself.tolerance_);
   if(std::distance(violatedConstraintsBegin, violatedConstraintsEnd) > 0) {
      while(violatedConstraintsBegin != violatedConstraintsEnd) {
         myself.lpInference_->addLinearConstraint(myself.linearConstraintID_, *violatedConstraintsBegin);
         ++violatedConstraintsBegin;
      }
      myself.violatedConstraintAdded_ = true;
   }
}

template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline void AddViolatedLinearConstraintsFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction) {
   AddViolatedLinearConstraintsFunctor_impl<LINEAR_CONSTRAINT_FUNCTION_TYPE, meta::HasTypeInTypeList<typename LP_INFERENCE_BASE_TYPE::LinearConstraintFunctionTypeList, LINEAR_CONSTRAINT_FUNCTION_TYPE>::value>::addViolatedLinearConstraintsFunctor_impl(*this, linearConstraintFunction);
}

template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
inline void AddViolatedLinearConstraintsFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>::AddViolatedLinearConstraintsFunctor_impl<FUNCTION_TYPE, IS_LINEAR_CONSTRAINT_FUNCTION>::addViolatedLinearConstraintsFunctor_impl(AddViolatedLinearConstraintsFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>& myself, const FUNCTION_TYPE& function) {
   throw RuntimeError(std::string("AddViolatedLinearConstraintsFunctor: Unsupported linear constraint function type") + typeid(FUNCTION_TYPE).name());
}

template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
template<class FUNCTION_TYPE>
inline void AddViolatedLinearConstraintsFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>::AddViolatedLinearConstraintsFunctor_impl<FUNCTION_TYPE, true>::addViolatedLinearConstraintsFunctor_impl(AddViolatedLinearConstraintsFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>& myself, const FUNCTION_TYPE& function) {
   typename FUNCTION_TYPE::ViolatedLinearConstraintsIteratorType        violatedConstraintsBegin;
   typename FUNCTION_TYPE::ViolatedLinearConstraintsIteratorType        violatedConstraintsEnd;
   typename FUNCTION_TYPE::ViolatedLinearConstraintsWeightsIteratorType violatedConstraintsWeightsBegin;
   function.challenge(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, myself.labelingBegin_, myself.tolerance_);
   if(std::distance(violatedConstraintsBegin, violatedConstraintsEnd) > 0) {
      while(violatedConstraintsBegin != violatedConstraintsEnd) {
         if(HEURISTIC == LP_INFERENCE_BASE_TYPE::Parameter::Random) {
            myself.lpInference_->addLinearConstraint(myself.linearConstraintID_, *violatedConstraintsBegin);
            ++violatedConstraintsBegin;
            ++myself.numConstraintsAdded_;
            if(myself.numConstraintsAdded_ == myself.lpInference_->parameter_.maxNumConstraintsPerIter_) {
               break;
            }
         } else {
            myself.sortedViolatedConstraintsList_->insert(typename LP_INFERENCE_BASE_TYPE::SortedViolatedConstraintsListType::value_type(*violatedConstraintsWeightsBegin, std::make_pair(myself.lpInference_->inactiveConstraints_.end(), std::make_pair(myself.linearConstraintID_, &(*violatedConstraintsBegin)))));
            if(myself.sortedViolatedConstraintsList_->size() > myself.lpInference_->parameter_.maxNumConstraintsPerIter_) {
               // remove constraints with to small weight
               myself.sortedViolatedConstraintsList_->erase(myself.sortedViolatedConstraintsList_->begin());
               OPENGM_ASSERT(myself.sortedViolatedConstraintsList_->size() == myself.lpInference_->parameter_.maxNumConstraintsPerIter_);
            }
            ++violatedConstraintsBegin;
            ++violatedConstraintsWeightsBegin;
         }
      }
   }
}

template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
template<class LINEAR_CONSTRAINT_FUNCTION_TYPE>
inline void AddViolatedLinearConstraintsRelaxedFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>::operator()(const LINEAR_CONSTRAINT_FUNCTION_TYPE& linearConstraintFunction) {
   AddViolatedLinearConstraintsRelaxedFunctor_impl<LINEAR_CONSTRAINT_FUNCTION_TYPE, meta::HasTypeInTypeList<typename LP_INFERENCE_BASE_TYPE::LinearConstraintFunctionTypeList, LINEAR_CONSTRAINT_FUNCTION_TYPE>::value>::addViolatedLinearConstraintsRelaxedFunctor_impl(*this, linearConstraintFunction);
}

template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
template<class FUNCTION_TYPE, bool IS_LINEAR_CONSTRAINT_FUNCTION>
inline void AddViolatedLinearConstraintsRelaxedFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>::AddViolatedLinearConstraintsRelaxedFunctor_impl<FUNCTION_TYPE, IS_LINEAR_CONSTRAINT_FUNCTION>::addViolatedLinearConstraintsRelaxedFunctor_impl(AddViolatedLinearConstraintsRelaxedFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>& myself, const FUNCTION_TYPE& function) {
   throw RuntimeError(std::string("AddViolatedLinearConstraintsRelaxedFunctor: Unsupported linear constraint function type") + typeid(FUNCTION_TYPE).name());
}

template <class LP_INFERENCE_BASE_TYPE, typename LP_INFERENCE_BASE_TYPE::Parameter::ChallengeHeuristic HEURISTIC>
template<class FUNCTION_TYPE>
inline void AddViolatedLinearConstraintsRelaxedFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>::AddViolatedLinearConstraintsRelaxedFunctor_impl<FUNCTION_TYPE, true>::addViolatedLinearConstraintsRelaxedFunctor_impl(AddViolatedLinearConstraintsRelaxedFunctor<LP_INFERENCE_BASE_TYPE, HEURISTIC>& myself, const FUNCTION_TYPE& function) {
   typename FUNCTION_TYPE::ViolatedLinearConstraintsIteratorType        violatedConstraintsBegin;
   typename FUNCTION_TYPE::ViolatedLinearConstraintsIteratorType        violatedConstraintsEnd;
   typename FUNCTION_TYPE::ViolatedLinearConstraintsWeightsIteratorType violatedConstraintsWeightsBegin;
   function.challengeRelaxed(violatedConstraintsBegin, violatedConstraintsEnd, violatedConstraintsWeightsBegin, myself.labelingBegin_, myself.tolerance_);
   if(std::distance(violatedConstraintsBegin, violatedConstraintsEnd) > 0) {
      while(violatedConstraintsBegin != violatedConstraintsEnd) {
         if(HEURISTIC == LP_INFERENCE_BASE_TYPE::Parameter::Random) {
            myself.lpInference_->addLinearConstraint(myself.linearConstraintID_, *violatedConstraintsBegin);
            ++violatedConstraintsBegin;
            ++myself.numConstraintsAdded_;
            if(myself.numConstraintsAdded_ == myself.lpInference_->parameter_.maxNumConstraintsPerIter_) {
               break;
            }
         } else {
            myself.sortedViolatedConstraintsList_->insert(typename LP_INFERENCE_BASE_TYPE::SortedViolatedConstraintsListType::value_type(*violatedConstraintsWeightsBegin, std::make_pair(myself.lpInference_->inactiveConstraints_.end(), std::make_pair(myself.linearConstraintID_, &(*violatedConstraintsBegin)))));
            if(myself.sortedViolatedConstraintsList_->size() > myself.lpInference_->parameter_.maxNumConstraintsPerIter_) {
               // remove constraints with to small weight
               myself.sortedViolatedConstraintsList_->erase(myself.sortedViolatedConstraintsList_->begin());
               OPENGM_ASSERT(myself.sortedViolatedConstraintsList_->size() == myself.lpInference_->parameter_.maxNumConstraintsPerIter_);
            }
            ++violatedConstraintsBegin;
            ++violatedConstraintsWeightsBegin;
         }
      }
   }
}

} // namespace opengm

#endif /* OPENGM_LP_INFERENCE_BASE_HXX_ */
