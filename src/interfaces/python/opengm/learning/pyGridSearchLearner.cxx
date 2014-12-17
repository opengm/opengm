#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/learning/gridsearch-learning.hxx>


namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{

    template<class LEARNER>
    void pyLearnDummy(LEARNER & learner){

        typedef typename  LEARNER::GMType GMType;
        typedef opengm::ICM<GMType, opengm::Minimizer> Inf;
        typedef typename Inf::Parameter InfParam;
        InfParam infParam;
        learner. template learn<Inf>(infParam);
    }

    template<class DATASET>
    void export_grid_search_learner(const std::string & clsName){
        typedef learning::GridSearchLearner<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;
        typedef typename  PyLearner::GMType GMType;
        typedef typename PyLearner::DatasetType DatasetType;

        const std::string paramClsName = clsName + std::string("Parameter");


        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
        ;

        bp::class_<PyLearner>( clsName.c_str(), bp::init<DatasetType &, const PyLearnerParam &>() )
        .def("learn",&pyLearnDummy<PyLearner>)
        ;
    }

    template void 
    export_grid_search_learner<op::GmAdderHammingLossDataset> (const std::string& className);

    template void 
    export_grid_search_learner<op::GmAdderGeneralizedHammingLossDataset> (const std::string& className);
}


