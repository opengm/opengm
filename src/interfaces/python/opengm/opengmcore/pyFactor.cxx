         
#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include "opengm_helpers.hxx"
#include "copyhelper.hxx"
#include "nifty_iterator.hxx"
#include "iteratorToTuple.hxx"
#include "export_typedes.hxx"
#include "utilities/shapeHolder.hxx"
#include "copyhelper.hxx"
#include "factorhelper.hxx"
#include "../converter.hxx"

using namespace boost::python;

template <class Base = default_call_policies>
struct incref_return_value_policy : Base
{
    static PyObject *postcall(PyObject *args, PyObject *result)
    {
        PyObject *self = PyTuple_GET_ITEM(args, 0);
        Py_INCREF(self);
        return result;
    }
};
   



template<class GM>
void export_factor(){
   typedef GM PyGm;
   typedef typename PyGm::ValueType ValueType;
   typedef typename PyGm::IndexType IndexType;
   typedef typename PyGm::LabelType LabelType;
   typedef typename opengm::ExplicitFunction<ValueType,IndexType,LabelType> PyExplicitFunction;
   
   
   typedef typename PyGm::FunctionIdentifier PyFid;
   typedef typename PyGm::FactorType PyFactor;
   typedef typename PyGm::IndependentFactorType PyIndependentFactor;
   typedef typename PyFid::FunctionIndexType FunctionIndexType;
   typedef typename PyFid::FunctionTypeIndexType FunctionTypeIndexType;
   import_array();
   
   typedef FactorShapeHolder<PyFactor> ShapeHolder;
   typedef FactorViHolder<PyFactor> ViHolder;
   class_<ShapeHolder > ("FactorShape", init<const  PyFactor &>() )
   .def(init< >())
   .def("__len__", &ShapeHolder::size)
   .def("__str__",&ShapeHolder::asString)
   .def("asNumpy", &ShapeHolder::toNumpy)
   .def("asList", &ShapeHolder::toList)
   .def("asTuple",&ShapeHolder::toTuple)
   .def("__getitem__", &ShapeHolder::operator[], return_value_policy<return_by_value>())
   .def("__copy__", &generic__copy__< ShapeHolder >)
   ;
   
   class_<ViHolder > ("FactorVariableIndices", init<const  PyFactor &>() )
   .def(init< >())
   .def("__len__", &ViHolder::size)
   .def("__str__",&ViHolder::asString)
   .def("asNumpy", &ViHolder::toNumpy)
   .def("asList", &ViHolder::toList)
   .def("asTuple",&ViHolder::toTuple)
   .def("__getitem__", &ViHolder::operator[], return_value_policy<return_by_value>())
   .def("__copy__", &generic__copy__< ViHolder >)
   //.def("__deepcopy__", &generic__deepcopy__< ShapeHolder >)
   ;
   
   
   
//------------------------------------------------------------------------------------
   // factor
   //------------------------------------------------------------------------------------
   class_<PyFactor > ("Factor", init< >())
   .add_property("size", &PyFactor::size)
   .add_property("numberOfVariables", &PyFactor::numberOfVariables)
   .add_property("numberOfLabels", &PyFactor::numberOfLabels)
   //.def_readonly("variableIndices", &PyFactor::variableIndices_   )
   //.add_property("shape2", &pyfactor::getShapeCallByReturnPyTuple<PyGm,int>  )
   .add_property("variableIndices", &pyfactor::getViHolder<PyFactor>  )
   .add_property("shape", &pyfactor::getShapeHolder<PyFactor>  )
   .add_property("functionType", &PyFactor::functionType)
   .add_property("functionIndex", &PyFactor::functionIndex)
   .def("__getitem__", &pyfactor::getValuePyNumpy<PyFactor>, return_value_policy<return_by_value>())
   .def("__getitem__", &pyfactor::getValuePyTuple<PyFactor,int>, return_value_policy<return_by_value>())
   .def("__getitem__", &pyfactor::getValuePyList<PyFactor,int>, return_value_policy<return_by_value>())
   .def("__str__", &pyfactor::printFactorPy<PyFactor>)
   .def("asIndependentFactor", &pyfactor::iFactorFromFactor<PyFactor,PyIndependentFactor> ,return_value_policy<manage_new_object>())
   .def("copyValues", &pyfactor::copyValuesCallByReturnPy<PyFactor>)
   .def("copyValuesSwitchedOrder", &pyfactor::copyValuesSwitchedOrderCallByReturnPy<PyFactor>)
   .def("isPotts", &PyFactor::isPotts)
   .def("isGeneralizedPotts", &PyFactor::isGeneralizedPotts)
   .def("isSubmodular", &PyFactor::isSubmodular)
   .def("isSquaredDifference", &PyFactor::isSquaredDifference)
   .def("isTruncatedSquaredDifference", &PyFactor::isTruncatedSquaredDifference)
   .def("isAbsoluteDifference", &PyFactor::isAbsoluteDifference)
   .def("isTruncatedAbsoluteDifference", &PyFactor::isTruncatedAbsoluteDifference)
   // min
   .def("min", &pyacc::accSomeInplacePyNumpy<PyFactor,opengm::Minimizer>)
   .def("min", &pyacc::accSomeCopyPyNumpy<PyFactor,opengm::Minimizer>)
   .def("min", &pyacc::accSomeInplacePyTuple<PyFactor,opengm::Minimizer,int>)
   .def("min", &pyacc::accSomeCopyPyTuple<PyFactor,opengm::Minimizer,int>)
   .def("min", &pyacc::accSomeInplacePyList<PyFactor,opengm::Minimizer,int>)
   .def("min", &pyacc::accSomeCopyPyList<PyFactor,opengm::Minimizer,int>)
   .def("min", &PyFactor::min)
   // max
   .def("max", &pyacc::accSomeInplacePyNumpy<PyFactor,opengm::Maximizer>)
   .def("max", &pyacc::accSomeCopyPyNumpy<PyFactor,opengm::Maximizer>)
   .def("max", &pyacc::accSomeInplacePyTuple<PyFactor,opengm::Maximizer,int>)
   .def("max", &pyacc::accSomeCopyPyTuple<PyFactor,opengm::Maximizer,int>)
   .def("max", &pyacc::accSomeInplacePyList<PyFactor,opengm::Maximizer,int>)
   .def("max", &pyacc::accSomeCopyPyList<PyFactor,opengm::Maximizer,int>)
   .def("max", &PyFactor::max)
   //sum
   .def("sum", &pyacc::accSomeInplacePyNumpy<PyFactor,opengm::Integrator>)
   .def("sum", &pyacc::accSomeCopyPyNumpy<PyFactor,opengm::Integrator>)
   .def("sum", &pyacc::accSomeInplacePyTuple<PyFactor,opengm::Integrator,int>)
   .def("sum", &pyacc::accSomeCopyPyTuple<PyFactor,opengm::Integrator,int>)
   .def("sum", &pyacc::accSomeInplacePyList<PyFactor,opengm::Integrator,int>)
   .def("sum", &pyacc::accSomeCopyPyList<PyFactor,opengm::Integrator,int>)
   .def("sum", &PyFactor::sum)
   // product
   .def("product", &pyacc::accSomeInplacePyNumpy<PyFactor,opengm::Multiplier>)
   .def("product", &pyacc::accSomeCopyPyNumpy<PyFactor,opengm::Multiplier>)
   .def("product", &pyacc::accSomeInplacePyTuple<PyFactor,opengm::Multiplier,int>)
   .def("product", &pyacc::accSomeCopyPyTuple<PyFactor,opengm::Multiplier,int>)
   .def("product", &pyacc::accSomeInplacePyList<PyFactor,opengm::Multiplier,int>)
   .def("product", &pyacc::accSomeCopyPyList<PyFactor,opengm::Multiplier,int>)
   .def("product", &PyFactor::product)
   // interoperate with self
   .def(self + self)
   .def(self - self)
   .def(self * self)
   .def(self / self)
   //interoperate with ValueType 
   .def(self + ValueType())
   .def(self - ValueType())
   .def(self * ValueType())
   .def(self / ValueType())
   .def(ValueType() + self)
   .def(ValueType() - self)
   .def(ValueType() * self)
   .def(ValueType() / self)
   //interoperate with IndependentFactor
   .def(self + GmIndependentFactor())
   .def(self - GmIndependentFactor())
   .def(self * GmIndependentFactor())
   .def(self / GmIndependentFactor())
   .def(GmIndependentFactor() + self)
   .def(GmIndependentFactor() - self)
   .def(GmIndependentFactor() * self)
   .def(GmIndependentFactor() / self)
   ;
}


template void export_factor<GmAdder>();
template void export_factor<GmMultiplier>();   
