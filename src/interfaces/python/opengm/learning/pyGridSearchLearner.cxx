#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>


#include <opengm/learning/gridsearch-learning.hxx>


namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{




    template<class DATASET>
    void export_grid_search_learner(const std::string & clsName){
        typedef learning::GridSearchLearner<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;

        const std::string paramClsName = clsName + std::string("Parameter");


        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
        ;
    }

    template void 
    export_grid_search_learner<op::GmAdderHammingLossDataset> (const std::string& className);

    template void 
    export_grid_search_learner<op::GmAdderGeneralizedHammingLossDataset> (const std::string& className);
}


