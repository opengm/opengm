#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>



namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{




    template<class DATASET>
    void export_grid_search_learner(const std::string & clsName){
        typedef learning::GridSearchLearner<DATASET> PyLearner;

        bp::class_<PyLearner>(clsName.c_str(),)
    }

    template void 
    export_dataset<op::GmAdderHammingLossDataset> (const std::string& className);
}


