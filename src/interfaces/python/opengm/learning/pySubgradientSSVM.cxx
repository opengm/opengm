#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/learning/subgradient_ssvm.hxx>

#define DefaultErrorFn DefaultErrorFn_TrwsExternalSubgradientSSVM
#include "helper.hxx"

namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{


    template<class PARAM>
    PARAM * pyStructuredPerceptronParamConstructor(
    ){
        PARAM * p  = new PARAM();
        return p;
    }

    template<class L >
    L * pyStructuredPerceptronConstructor(
        typename L::DatasetType & dataset,
        const typename L::Parameter & param
    ){
        L * l  = new L(dataset, param);
        return l;
    }

    template<class DATASET>
    void export_subgradient_ssvm_learner(const std::string & clsName){
        typedef learning::SubgradientSSVM<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;

        const std::string paramClsName = clsName + std::string("Parameter");

        const std::string paramEnumLearningModeName = clsName + std::string("Parameter_LearningMode");

        // learner param enum
        bp::enum_<typename PyLearnerParam::LearningMode>(paramEnumLearningModeName.c_str())
            .value("online", PyLearnerParam::Online)
            .value("batch", PyLearnerParam::Batch)
        ;

        // learner param
        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
            .def("__init__", make_constructor(&pyStructuredPerceptronParamConstructor<PyLearnerParam> ,boost::python::default_call_policies()))
            .def_readwrite("eps",  &PyLearnerParam::eps_)
            .def_readwrite("maxIterations", &PyLearnerParam::maxIterations_)
            .def_readwrite("stopLoss", &PyLearnerParam::stopLoss_)
            .def_readwrite("learningRate", &PyLearnerParam::learningRate_)
            .def_readwrite("C", &PyLearnerParam::C_)
            .def_readwrite("learningMode", &PyLearnerParam::learningMode_)
            .def_readwrite("averaging", &PyLearnerParam::averaging_)
            .def_readwrite("nConf", &PyLearnerParam::nConf_)
        ;


        // learner
        bp::class_<PyLearner>( clsName.c_str(), bp::no_init )
        .def("__init__", make_constructor(&pyStructuredPerceptronConstructor<PyLearner> ,boost::python::default_call_policies()))
        .def(LearnerInferenceSuite<PyLearner>())
        ;
    }

    // template void 
    // export_subgradient_ssvm_learner<op::GmAdderHammingLossDataset> (const std::string& className);

    // template void 
    // export_subgradient_ssvm_learner<op::GmAdderGeneralizedHammingLossDataset> (const std::string& className);

    template void 
    export_subgradient_ssvm_learner<op::GmAdderFlexibleLossDataset> (const std::string& className);
}


