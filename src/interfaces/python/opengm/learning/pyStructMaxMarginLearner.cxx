#if defined(WITH_CPLEX) || defined(WITH_GUROBI)

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/learning/struct-max-margin.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/learning/gridsearch-learning.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>

namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{


    template<class PARAM>
    PARAM * pyStructMaxMarginBundleParamConstructor(
        double regularizerWeight,
        op::GmValueType minGap,
        unsigned int steps
    ){
        PARAM * p  = new PARAM();
        p->optimizerParameter_.lambda  = regularizerWeight;
        p->optimizerParameter_.min_gap = minGap;
        p->optimizerParameter_.steps   = steps;
        return p;
    }

    template<class LEARNER, class INF>
    void pyLearnWithInf(LEARNER & learner, const typename INF::Parameter & param){
        learner. template learn<INF>(param);
    }

    template<class DATASET, class OPTIMIZER>
    void export_struct_max_margin_bundle_learner(const std::string & clsName){
        typedef learning::StructMaxMargin<DATASET, OPTIMIZER> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;
        typedef typename PyLearner::GMType GMType;
        typedef typename PyLearner::DatasetType DatasetType;

        const std::string paramClsName = clsName + std::string("Parameter");


        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
            .def("__init__", make_constructor(&pyStructMaxMarginBundleParamConstructor<PyLearnerParam> ,boost::python::default_call_policies()))
        ;

        // SOME INFERENCE METHODS
        typedef typename  PyLearner::GMType GMType;
        typedef opengm::Minimizer ACC;

        typedef opengm::ICM<GMType, ACC> IcmInf;
        typedef opengm::BeliefPropagationUpdateRules<GMType, ACC> UpdateRulesType;
        typedef opengm::MessagePassing<GMType, ACC, UpdateRulesType, opengm::MaxDistance> BpInf;

        bp::class_<PyLearner>( clsName.c_str(), bp::init<DatasetType &, const PyLearnerParam &>() )
            .def("_learn",&pyLearnWithInf<PyLearner, IcmInf>)
            .def("_learn",&pyLearnWithInf<PyLearner, BpInf>)
        ;
    }

    template void
    export_struct_max_margin_bundle_learner<op::GmAdderHammingLossDataset, ol::BundleOptimizer<op::GmValueType> > (const std::string& className);

    template void
    export_struct_max_margin_bundle_learner<op::GmAdderGeneralizedHammingLossDataset, ol::BundleOptimizer<op::GmValueType> > (const std::string& className);
}



#endif
