#ifndef OPENGM_LPCPLEX2_HXX_
#define OPENGM_LPCPLEX2_HXX_

#include <opengm/inference/auxiliary/lp_solver/lp_solver_cplex.hxx>
#include <opengm/inference/lp_inference_base.hxx>

namespace opengm {

/********************
 * class definition *
 *******************/
template<class GM_TYPE, class ACC_TYPE>
class LPCplex2 : public LPSolverCplex, public LPInferenceBase<LPCplex2<GM_TYPE, ACC_TYPE> > {
public:
   // typedefs
   typedef ACC_TYPE                                                         AccumulationType;
   typedef GM_TYPE                                                          GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef LPInferenceBase<LPCplex2<GraphicalModelType, AccumulationType> > LPInferenceBaseType;
   typedef typename LPInferenceBaseType::Parameter                          Parameter;

   // construction
   LPCplex2(const GraphicalModelType& gm, const Parameter& parameter = Parameter());
   virtual ~LPCplex2();

   // public member functions
   virtual std::string name() const;

   template<class _GM>
   struct RebindGm{
       typedef LPCplex2<_GM, ACC_TYPE> type;
   };

   template<class _GM,class _ACC>
   struct RebindGmAndAcc{
       typedef LPCplex2<_GM, _ACC > type;
   };
};

template<class GM_TYPE, class ACC_TYPE>
struct LPInferenceTraits<LPCplex2<GM_TYPE, ACC_TYPE> > {
   // typedefs
   typedef ACC_TYPE                                          AccumulationType;
   typedef GM_TYPE                                           GraphicalModelType;
   typedef LPSolverCplex                                     SolverType;
   typedef typename LPSolverCplex::CplexIndexType            SolverIndexType;
   typedef typename LPSolverCplex::CplexValueType            SolverValueType;
   typedef typename LPSolverCplex::CplexSolutionIteratorType SolverSolutionIteratorType;
   typedef typename LPSolverCplex::CplexTimingType           SolverTimingType;
   typedef typename LPSolverCplex::Parameter                 SolverParameterType;
};

/***********************
 * class documentation *
 **********************/
/*! \file lpcplex2.hxx
 *  \brief Provides implementation for LP inference with CPLEX.
 */

/*! \class LPCplex2
 *  \brief LP inference with CPLEX.
 *
 *  This class combines opengm::LPSolverCplex and opengm::LPInferenceBase to
 *  provide inference for graphical models using CPLEX.
 *
 *  \tparam GM_TYPE Graphical Model type.
 *  \tparam ACC_TYPE Accumulation type.
 *
 *  \ingroup inference
 */

/*! \typedef LPCplex2::AccumulationType
 *  \brief Typedef of the Accumulation type.
 */

/*! \typedef LPCplex2::GraphicalModelType
 *  \brief Typedef of the graphical model type.
 */

/*! \typedef LPCplex2::LPInferenceBaseType
 *  \brief Typedef of class opengm::LPInferenceBase with appropriate template
 *         parameter.
 */

/*! \typedef LPCplex2::Parameter
 *  \brief Typedef of the parameter type defined by class
 *         opengm::LPInferenceBase.
 */

/*! \fn LPCplex2::LPCplex2(const GraphicalModelType& gm, const Parameter& parameter = Parameter())
 *  \brief LPCplex2 constructor.
 *
 *  \param[in] gm The graphical model for inference.
 *  \param[in] parameter The parameter defining the settings for inference. See
 *                       opengm::LPSolverInterface::Parameter and
 *                       opengm::LPInferenceBase::Parameter for possible
 *                       settings.
 */

/*! \fn LPCplex2::~LPCplex2()
 *  \brief LPCplex2 destructor.
 */

/*! \fn std::string LPCplex2::name() const
 *  \brief Name of the inference method.
 */

/******************
 * implementation *
 *****************/
template<class GM_TYPE, class ACC_TYPE>
inline LPCplex2<GM_TYPE, ACC_TYPE>::LPCplex2(const GraphicalModelType& gm, const Parameter& parameter)
   : LPSolverCplex(parameter), LPInferenceBaseType(gm, parameter) {

}

template<class GM_TYPE, class ACC_TYPE>
inline LPCplex2<GM_TYPE, ACC_TYPE>::~LPCplex2() {

}

template<class GM_TYPE, class ACC_TYPE>
inline std::string LPCplex2<GM_TYPE, ACC_TYPE>::name() const {
   return "LPCplex2";
}

} // namespace opengm

#endif /* OPENGM_LPCPLEX2_HXX_ */
