#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/learning/rws.hxx>

#define DefaultErrorFn DefaultErrorFn_TrwsExternalRws
#include "helper.hxx"

namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{


    template<class PARAM>
    PARAM * pyRwsParamConstructor(
    ){
        PARAM * p  = new PARAM();
        return p;
    }

    template<class L >
    L * pyRwsConstructor(
        typename L::DatasetType & dataset,
        const typename L::Parameter & param
    ){
        L * l  = new L(dataset, param);
        return l;
    }

    template<class DATASET>
    void export_rws_learner(const std::string & clsName){
        typedef learning::Rws<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;

        const std::string paramClsName = clsName + std::string("Parameter");


        // learner param
        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
            .def("__init__", make_constructor(&pyRwsParamConstructor<PyLearnerParam> ,boost::python::default_call_policies()))
            .def_readwrite("eps",  &PyLearnerParam::eps_)
            .def_readwrite("maxIterations", &PyLearnerParam::maxIterations_)
            .def_readwrite("stopLoss", &PyLearnerParam::stopLoss_)
            .def_readwrite("learningRate", &PyLearnerParam::learningRate_)
            .def_readwrite("C", &PyLearnerParam::C_)
            .def_readwrite("p", &PyLearnerParam::p_)
            .def_readwrite("sigma", &PyLearnerParam::sigma_)
        ;


        // learner
        bp::class_<PyLearner>( clsName.c_str(), bp::no_init )
        .def("__init__", make_constructor(&pyRwsConstructor<PyLearner> ,boost::python::default_call_policies()))
        .def(LearnerInferenceSuite<PyLearner>())
        ;
    }

    // template void 
    // export_subgradient_ssvm_learner<op::GmAdderHammingLossDataset> (const std::string& className);

    // template void 
    // export_subgradient_ssvm_learner<op::GmAdderGeneralizedHammingLossDataset> (const std::string& className);

    template void 
    export_rws_learner<op::GmAdderFlexibleLossDataset> (const std::string& className);
}


