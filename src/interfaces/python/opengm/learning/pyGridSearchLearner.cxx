#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#define DefaultErrorFn DefaultErrorFn_TrwsExternalB
#include "helper.hxx"

namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{


    template<class PARAM>
    PARAM * pyGridSearchParamConstructor(
        op::NumpyView<double> lowerBound,
        op::NumpyView<double> upperBound,
        op::NumpyView<size_t> nTestPoints
    ){
        PARAM * p  = new PARAM();
        p->parameterUpperbound_.assign(lowerBound.begin(), lowerBound.end());
        p->parameterLowerbound_.assign(upperBound.begin(), upperBound.end());
        p->testingPoints_.assign(nTestPoints.begin(), nTestPoints.end());
        return p;
    }

    template<class L >
    L * pyGridSearchConstructor(
        typename L::DatasetType & dataset,
        const typename L::Parameter & param
    ){
        L * l  = new L(dataset, param);
        return l;
    }

    template<class DATASET>
    void export_grid_search_learner(const std::string & clsName){
        typedef learning::GridSearchLearner<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;

        const std::string paramClsName = clsName + std::string("Parameter");


        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
            .def("__init__", make_constructor(&pyGridSearchParamConstructor<PyLearnerParam> ,boost::python::default_call_policies()))
        ;

        bp::class_<PyLearner>( clsName.c_str(), bp::no_init )
        .def("__init__", make_constructor(&pyGridSearchConstructor<PyLearner> ,boost::python::default_call_policies()))
        .def(LearnerInferenceSuite<PyLearner>())
        ;
    }

    template void 
    export_grid_search_learner<op::GmAdderFlexibleLossDataset> (const std::string& className);

    //template void 
    //export_grid_search_learner<op::GmAdderGeneralizedHammingLossDataset> (const std::string& className);
}


