#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>



namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{

    template<class V>
    learning::Weights<V>  * pyWeightsConstructor(
        python::NumpyView<V, 1> values                                           
    ){
        learning::Weights<V>   * f = new learning::Weights<V> (values.shape(0));
        for(size_t i=0; i<values.shape(0); ++i){
            f->setWeight(i, values(i));
        }
        return f;
    }


    template<class DATASET>
    void export_grid_search_learner(const std::string & clsName){
        
    }


}
