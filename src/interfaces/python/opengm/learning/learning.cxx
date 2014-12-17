#include <boost/python.hpp>
#include <stddef.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>

#include <opengm/learning/loss/hammingloss.hxx>
#include <opengm/learning/loss/generalized-hammingloss.hxx>
#include <opengm/learning/loss/noloss.hxx>

namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{

    void export_weights();

    template<class GM, class LOSS>
    void export_dataset(const std::string& className);

    template<class GM>
    void export_loss();

    template<class DATASET>
    void export_grid_search_learner(const std::string & clsName);
}



BOOST_PYTHON_MODULE_INIT(_learning) {


    Py_Initialize();
    PyEval_InitThreads();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options doc_options(true,true,false);


    opengm::export_weights();

    // export loss
    opengm::export_loss<op::GmAdder>();

    // templated datasets
    opengm::export_dataset<op::GmAdder, ol::HammingLoss >("DatasetWithHammingLoss");
    opengm::export_dataset<op::GmAdder, ol::NoLoss >("DatasetWithNoLoss");


    opengm::export_grid_search_learner<GmAdderHammingLossDataset>("GridSearch_HammingLoss");
}
