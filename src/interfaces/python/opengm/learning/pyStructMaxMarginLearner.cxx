#if defined(WITH_CPLEX) || defined(WITH_GUROBI)

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/learning/struct-max-margin.hxx>

#define DefaultErrorFn DefaultErrorFn_TrwsExternal_SMM
#include "helper.hxx"

namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{


    template<class PARAM>
    PARAM * pyStructMaxMarginBundleParamConstructor(
        double regularizerWeight,
        op::GmValueType minEps,
        unsigned int steps,
        bool eps_from_gap = true
    ){
        PARAM * p  = new PARAM();
        p->optimizerParameter_.lambda  = regularizerWeight;
        p->optimizerParameter_.min_eps = minEps;
        p->optimizerParameter_.steps   = steps;
        if(eps_from_gap)
            p->optimizerParameter_.epsStrategy = ol::BundleOptimizer<op::GmValueType>::EpsFromGap;
        else
            p->optimizerParameter_.epsStrategy = ol::BundleOptimizer<op::GmValueType>::EpsFromChange;
        return p;
    }

    template<class DATASET, class OPTIMIZER>
    void export_struct_max_margin_bundle_learner(const std::string & clsName){
        typedef learning::StructMaxMargin<DATASET, OPTIMIZER> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;
        typedef typename PyLearner::DatasetType DatasetType;

        const std::string paramClsName = clsName + std::string("Parameter");

        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
            .def("__init__", make_constructor(&pyStructMaxMarginBundleParamConstructor<PyLearnerParam> ,boost::python::default_call_policies()))
        ;

        boost::python::class_<PyLearner>( clsName.c_str(), boost::python::init<DatasetType &, const PyLearnerParam &>() )
            .def(LearnerInferenceSuite<PyLearner>())
        ;
    }

    template void
    export_struct_max_margin_bundle_learner<op::GmAdderFlexibleLossDataset, ol::BundleOptimizer<op::GmValueType> > (const std::string& className);

}



#endif
