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
#include <opengm/learning/loss/flexibleloss.hxx>

#if defined(WITH_CPLEX) || defined(WITH_GUROBI)
#include <opengm/learning/bundle-optimizer.hxx>
#endif


namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{

    void export_weights();
    void export_weight_constraints();

    template<class GM, class LOSS>
    void export_dataset(const std::string& className);

    template<class GM>
    void export_loss();

    template<class DATASET>
    void export_grid_search_learner(const std::string & clsName);

    template<class DATASET, class OPTIMIZER>
    void export_struct_max_margin_bundle_learner(const std::string & clsName);

    template<class DATASET>
    void export_max_likelihood_learner(const std::string & clsName);

    template<class DATASET>
    void export_struct_perceptron_learner(const std::string & clsName);

    template<class DATASET>
    void export_subgradient_ssvm_learner(const std::string & clsName);

    template<class DATASET>
    void export_rws_learner(const std::string & clsName);

    template<class GM_ADDER,class GM_MULT>  
    void export_lfunction_generator();


}



BOOST_PYTHON_MODULE_INIT(_learning) {


    Py_Initialize();
    PyEval_InitThreads();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options doc_options(true,true,false);


    opengm::export_weights();
    opengm::export_weight_constraints();
    // function exporter
    opengm::export_lfunction_generator<op::GmAdder,op::GmMultiplier>();

    // export loss
    opengm::export_loss<op::GmAdder>();

    // templated datasets
    opengm::export_dataset<op::GmAdder, ol::FlexibleLoss >("DatasetWithFlexibleLoss");



    opengm::export_grid_search_learner<op::GmAdderFlexibleLossDataset>("GridSearch_FlexibleLoss");
    opengm::export_struct_perceptron_learner<op::GmAdderFlexibleLossDataset>("StructPerceptron_FlexibleLoss");
    opengm::export_subgradient_ssvm_learner<op::GmAdderFlexibleLossDataset>("SubgradientSSVM_FlexibleLoss");
    opengm::export_max_likelihood_learner<op::GmAdderFlexibleLossDataset>("MaxLikelihood_FlexibleLoss");
    opengm::export_rws_learner<op::GmAdderFlexibleLossDataset>("Rws_FlexibleLoss");
    
    #if defined(WITH_CPLEX) || defined(WITH_GUROBI)
        opengm::export_struct_max_margin_bundle_learner< op::GmAdderFlexibleLossDataset, ol::BundleOptimizer<op::GmValueType> >("StructMaxMargin_Bundle_FlexibleLoss");
    #endif
}
