#ifndef OPENGM_LPGUROBI2_HXX_
#define OPENGM_LPGUROBI2_HXX_

#include <opengm/inference/auxiliary/lp_solver/lp_solver_gurobi.hxx>
#include <opengm/inference/lp_inference_base.hxx>

namespace opengm {

/********************
 * class definition *
 *******************/
template<class GM_TYPE, class ACC_TYPE>
class LPGurobi2 : public LPSolverGurobi, public LPInferenceBase<LPGurobi2<GM_TYPE, ACC_TYPE> > {
public:
   // typedefs
   typedef ACC_TYPE                                                          AccumulationType;
   typedef GM_TYPE                                                           GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef LPInferenceBase<LPGurobi2<GraphicalModelType, AccumulationType> > LPInferenceBaseType;
   typedef typename LPInferenceBaseType::Parameter                           Parameter;

   // construction
   LPGurobi2(const GraphicalModelType& gm, const Parameter& parameter = Parameter());
   virtual ~LPGurobi2();

   // public member functions
   virtual std::string name() const;

   template<class _GM>
   struct RebindGm{
       typedef LPGurobi2<_GM, ACC_TYPE> type;
   };

   template<class _GM,class _ACC>
   struct RebindGmAndAcc{
       typedef LPGurobi2<_GM, _ACC > type;
   };
};

template<class GM_TYPE, class ACC_TYPE>
struct LPInferenceTraits<LPGurobi2<GM_TYPE, ACC_TYPE> > {
   // typedefs
   typedef ACC_TYPE                                          AccumulationType;
   typedef GM_TYPE                                           GraphicalModelType;
   typedef LPSolverGurobi                                     SolverType;
   typedef typename LPSolverGurobi::GurobiIndexType            SolverIndexType;
   typedef typename LPSolverGurobi::GurobiValueType            SolverValueType;
   typedef typename LPSolverGurobi::GurobiSolutionIteratorType SolverSolutionIteratorType;
   typedef typename LPSolverGurobi::GurobiTimingType           SolverTimingType;
   typedef typename LPSolverGurobi::Parameter                 SolverParameterType;
};

/***********************
 * class documentation *
 **********************/
/*! \file lpgurobi2.hxx
 *  \brief Provides implementation for LP inference with Gurobi.
 */

/*! \class LPGurobi2
 *  \brief LP inference with Gurobi.
 *
 *  This class combines opengm::LPSolverGurobi and opengm::LPInferenceBase to
 *  provide inference for graphical models using Gurobi.
 *
 *  \tparam GM_TYPE Graphical Model type.
 *  \tparam ACC_TYPE Accumulation type.
 *
 *  \ingroup inference
 */

/*! \typedef LPGurobi2::AccumulationType
 *  \brief Typedef of the Accumulation type.
 */

/*! \typedef LPGurobi2::GraphicalModelType
 *  \brief Typedef of the graphical model type.
 */

/*! \typedef LPGurobi2::LPInferenceBaseType
 *  \brief Typedef of class opengm::LPInferenceBase with appropriate template
 *         parameter.
 */

/*! \typedef LPGurobi2::Parameter
 *  \brief Typedef of the parameter type defined by class
 *         opengm::LPInferenceBase.
 */

/*! \fn LPGurobi2::LPGurobi2(const GraphicalModelType& gm, const Parameter& parameter = Parameter())
 *  \brief LPGurobi2 constructor.
 *
 *  \param[in] gm The graphical model for inference.
 *  \param[in] parameter The parameter defining the settings for inference. See
 *                       opengm::LPSolverInterface::Parameter and
 *                       opengm::LPInferenceBase::Parameter for possible
 *                       settings.
 */

/*! \fn LPGurobi2::~LPGurobi2()
 *  \brief LPGurobi2 destructor.
 */

/*! \fn std::string LPGurobi2::name() const
 *  \brief Name of the inference method.
 */

/******************
 * implementation *
 *****************/
template<class GM_TYPE, class ACC_TYPE>
inline LPGurobi2<GM_TYPE, ACC_TYPE>::LPGurobi2(const GraphicalModelType& gm, const Parameter& parameter)
   : LPSolverGurobi(parameter), LPInferenceBaseType(gm, parameter) {

}

template<class GM_TYPE, class ACC_TYPE>
inline LPGurobi2<GM_TYPE, ACC_TYPE>::~LPGurobi2() {

}

template<class GM_TYPE, class ACC_TYPE>
inline std::string LPGurobi2<GM_TYPE, ACC_TYPE>::name() const {
   return "LPGurobi2";
}

} // namespace opengm

#endif /* OPENGM_LPGUROBI2_HXX_ */
