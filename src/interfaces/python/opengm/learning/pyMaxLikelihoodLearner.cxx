#if defined(WITH_CPLEX) || defined(WITH_GUROBI)

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/learning/maximum_likelihood_learning.hxx>

#define DefaultErrorFn DefaultErrorFn_TrwsExternal_ML
#include "helper.hxx"

namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{




    template<class DATASET>
    void export_max_likelihood_learner(const std::string & clsName){
        typedef learning::MaximumLikelihoodLearner<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;
        typedef typename PyLearner::DatasetType DatasetType;

        const std::string paramClsName = clsName + std::string("Parameter");

        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
            .def_readwrite("maxIterations", &PyLearnerParam::maxNumSteps_)
            .def_readwrite("reg", &PyLearnerParam::reg_)
            .def_readwrite("temperature", &PyLearnerParam::temperature_)
        ;

        boost::python::class_<PyLearner>( clsName.c_str(), boost::python::init<DatasetType &, const PyLearnerParam &>() )
            //.def("learn",&PyLearner::learn)
        ;
    }

    //template void
    //export_max_likelihood_learner<op::GmAdderHammingLossDataset> (const std::string& className);

    template void
    export_max_likelihood_learner<op::GmAdderFlexibleLossDataset> (const std::string& className);
}



#endif

