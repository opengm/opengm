#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>



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



    void export_weights(){
        typedef  python::GmValueType V;
        typedef learning::Weights<V> Weights;
        boost::python::class_<Weights>("Weights",boost::python::init<const size_t >())
            .def("__init__", make_constructor(&pyWeightsConstructor<V> ,boost::python::default_call_policies()))
            .def("__getitem__", &Weights::getWeight)
            .def("__setitem__", &Weights::setWeight)
            .def("__len__", &Weights::numberOfWeights)
        ;
    }

    void export_weight_constraints(){
        typedef  python::GmValueType V;
        typedef learning::WeightConstraints<V> Weights;
        boost::python::class_<Weights>("WeightConstraints",boost::python::init<const size_t >())
            //.def("__init__", make_constructor(&pyWeightsConstructor<V> ,boost::python::default_call_policies()))
            //.def("__getitem__", &Weights::getWeight)
            //.def("__setitem__", &Weights::setWeight)
        ;
    }


}
