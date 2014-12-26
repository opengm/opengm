#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/learning/structured_perceptron.hxx>

#define DefaultErrorFn DefaultErrorFn_TrwsExternalSPerceptron
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
    void export_struct_perceptron_learner(const std::string & clsName){
        typedef learning::StructuredPerceptron<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;

        const std::string paramClsName = clsName + std::string("Parameter");


        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
            .def("__init__", make_constructor(&pyStructuredPerceptronParamConstructor<PyLearnerParam> ,boost::python::default_call_policies()))
            .def_readwrite("eps",  &PyLearnerParam::eps_)
            .def_readwrite("maxIterations", &PyLearnerParam::maxIterations_)
            .def_readwrite("stopLoss", &PyLearnerParam::stopLoss_)
            .def_readwrite("kappa", &PyLearnerParam::kappa_)
        ;

        bp::class_<PyLearner>( clsName.c_str(), bp::no_init )
        .def("__init__", make_constructor(&pyStructuredPerceptronConstructor<PyLearner> ,boost::python::default_call_policies()))
        .def(LearnerInferenceSuite<PyLearner>())
        ;
    }

    template void 
    export_struct_perceptron_learner<op::GmAdderHammingLossDataset> (const std::string& className);

    template void 
    export_struct_perceptron_learner<op::GmAdderGeneralizedHammingLossDataset> (const std::string& className);
}


