         
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
#include "../export_typedes.hxx"

using namespace boost::python;


   

template<class V,class I>
void export_ifactor(){
   typedef V ValueType;
   typedef I IndexType;
   typedef IndexType LabelType;
   typedef opengm::IndependentFactor<ValueType,IndexType,LabelType> PyIndependentFactor;
   
   import_array();
   typedef FactorShapeHolder<PyIndependentFactor> ShapeHolder;
   
   class_<ShapeHolder > ("IndependentFactorShape", init<const  PyIndependentFactor &>() )
   .def(init< >())
   .def("__init__", make_constructor(&pyfactor::iFactorFromFactor<FactorGmAdder,PyIndependentFactor> ))
   .def("__init__", make_constructor(&pyfactor::iFactorFromFactor<FactorGmMultiplier,PyIndependentFactor> ))
   .def("__len__", &ShapeHolder::size)
   .def("__str__",&ShapeHolder::asString)
   .def("asNumpy", &ShapeHolder::toNumpy)
   .def("asList", &ShapeHolder::toList)
   .def("asTuple",&ShapeHolder::toTuple)
   .def("__getitem__", &ShapeHolder::operator[], return_value_policy<return_by_value>())
   .def("__copy__", &generic__copy__< ShapeHolder >)
   //.def("__deepcopy__", &generic__deepcopy__< ShapeHolder >)
   ;
   class_<PyIndependentFactor > ("IndependentFactor", init< >())
   .add_property("PyIndependentFactor", &PyIndependentFactor::size)
   .add_property("numberOfVariables", &PyIndependentFactor::numberOfVariables)
   .add_property("numberOfLabels", &PyIndependentFactor::numberOfLabels)
   //.add_property("shape2", &pyfactor::getShapeCallByReturnPyTuple<PyGm,int>  )
   .add_property("shape", &pyfactor::getShapeHolder<PyIndependentFactor>  )
   .def("__getitem__", &pyfactor::getValuePyTuple<PyIndependentFactor,int>, return_value_policy<return_by_value>())
   .def("__getitem__", &pyfactor::getValuePyList<PyIndependentFactor,int>, return_value_policy<return_by_value>())
   .def("__getitem__", &pyfactor::getValuePyNumpy<PyIndependentFactor>, return_value_policy<return_by_value>())
   .def("__str__", &pyfactor::printFactorPy<PyIndependentFactor>)
   // copy values to numpy order
   .def("copyValuesSwitchedOrder", &pyfactor::ifactorToNumpy<PyIndependentFactor>)
   // min
   .def("minInplace", &pyacc::accSomeIFactorInplacePyNumpy<PyIndependentFactor,opengm::Minimizer>)
   .def("minInplace", &pyacc::accSomeIFactorInplacePyTuple<PyIndependentFactor,opengm::Minimizer,int>)
   .def("minInplace", &pyacc::accSomeIFactorInplacePyList<PyIndependentFactor,opengm::Minimizer,int>)
   .def("min", &pyacc::accSomeInplacePyNumpy<PyIndependentFactor,opengm::Minimizer>)
   .def("min", &pyacc::accSomeCopyPyNumpy<PyIndependentFactor,opengm::Minimizer>)
   .def("min", &pyacc::accSomeInplacePyTuple<PyIndependentFactor,opengm::Minimizer,int>)
   .def("min", &pyacc::accSomeCopyPyTuple<PyIndependentFactor,opengm::Minimizer,int>)
   .def("min", &pyacc::accSomeInplacePyList<PyIndependentFactor,opengm::Minimizer,int>)
   .def("min", &pyacc::accSomeCopyPyList<PyIndependentFactor,opengm::Minimizer,int>)
   .def("min", &PyIndependentFactor::min)
    // max
   .def("maxInplace", &pyacc::accSomeIFactorInplacePyNumpy<PyIndependentFactor,opengm::Maximizer>)
   .def("maxInplace", &pyacc::accSomeIFactorInplacePyTuple<PyIndependentFactor,opengm::Maximizer,int>)
   .def("maxInplace", &pyacc::accSomeIFactorInplacePyList<PyIndependentFactor,opengm::Maximizer,int>)
   .def("max", &pyacc::accSomeInplacePyNumpy<PyIndependentFactor,opengm::Maximizer>)
   .def("max", &pyacc::accSomeCopyPyNumpy<PyIndependentFactor,opengm::Maximizer>)
   .def("max", &pyacc::accSomeInplacePyTuple<PyIndependentFactor,opengm::Maximizer,int>)
   .def("max", &pyacc::accSomeCopyPyTuple<PyIndependentFactor,opengm::Maximizer,int>)
   .def("max", &pyacc::accSomeInplacePyList<PyIndependentFactor,opengm::Maximizer,int>)
   .def("max", &pyacc::accSomeCopyPyList<PyIndependentFactor,opengm::Maximizer,int>)
   .def("max", &PyIndependentFactor::max)
   //sum
   .def("sumInplace", &pyacc::accSomeIFactorInplacePyNumpy<PyIndependentFactor,opengm::Integrator>)
   .def("sumInplace", &pyacc::accSomeIFactorInplacePyTuple<PyIndependentFactor,opengm::Integrator,int>)
   .def("sumInplace", &pyacc::accSomeIFactorInplacePyList<PyIndependentFactor,opengm::Integrator,int>)
   .def("sum", &pyacc::accSomeInplacePyNumpy<PyIndependentFactor,opengm::Integrator>)
   .def("sum", &pyacc::accSomeCopyPyNumpy<PyIndependentFactor,opengm::Integrator>)
   .def("sum", &pyacc::accSomeInplacePyTuple<PyIndependentFactor,opengm::Integrator,int>)
   .def("sum", &pyacc::accSomeCopyPyTuple<PyIndependentFactor,opengm::Integrator,int>)
   .def("sum", &pyacc::accSomeInplacePyList<PyIndependentFactor,opengm::Integrator,int>)
   .def("sum", &pyacc::accSomeCopyPyList<PyIndependentFactor,opengm::Integrator,int>)
   .def("sum", &PyIndependentFactor::sum)
   // product
   .def("productInplace", &pyacc::accSomeIFactorInplacePyNumpy<PyIndependentFactor,opengm::Multiplier>)
   .def("productInplace", &pyacc::accSomeIFactorInplacePyTuple<PyIndependentFactor,opengm::Multiplier,int>)
   .def("productInplace", &pyacc::accSomeIFactorInplacePyList<PyIndependentFactor,opengm::Multiplier,int>)
   .def("product", &pyacc::accSomeInplacePyNumpy<PyIndependentFactor,opengm::Multiplier>)
   .def("product", &pyacc::accSomeCopyPyNumpy<PyIndependentFactor,opengm::Multiplier>)
   .def("product", &pyacc::accSomeInplacePyTuple<PyIndependentFactor,opengm::Multiplier,int>)
   .def("product", &pyacc::accSomeCopyPyTuple<PyIndependentFactor,opengm::Multiplier,int>)
   .def("product", &pyacc::accSomeInplacePyList<PyIndependentFactor,opengm::Multiplier,int>)
   .def("product", &pyacc::accSomeCopyPyList<PyIndependentFactor,opengm::Multiplier,int>)
   .def("product", &PyIndependentFactor::product)
   // interoperate with self
   .def(self + self)
   .def(self - self)
   .def(self * self)
   .def(self / self)
   .def(self += self)
   .def(self -= self)
   .def(self *= self)
   .def(self /= self)
   //interoperate with ValueType
   .def(self + ValueType())
   .def(self - ValueType())
   .def(self * ValueType())
   .def(self / ValueType())
   .def(ValueType() + self)
   .def(ValueType() - self)
   .def(ValueType() * self)
   .def(ValueType() / self)
   .def(self += ValueType())
   .def(self -= ValueType())
   .def(self *= ValueType())
   .def(self /= ValueType())
   //interoperate with FactorAdder
   .def(self + FactorGmAdder())
   .def(self - FactorGmAdder())
   .def(self * FactorGmAdder())
   .def(self / FactorGmAdder())
   .def(FactorGmAdder() + self)
   .def(FactorGmAdder() - self)
   .def(FactorGmAdder() * self)
   .def(FactorGmAdder() / self)
   .def(self += FactorGmAdder())
   .def(self -= FactorGmAdder())
   .def(self *= FactorGmAdder())
   .def(self /= FactorGmAdder())
      //interoperate with FactorAdder
   .def(self + FactorGmMultiplier())
   .def(self - FactorGmMultiplier())
   .def(self * FactorGmMultiplier())
   .def(self / FactorGmMultiplier())
   .def(FactorGmMultiplier() + self)
   .def(FactorGmMultiplier() - self)
   .def(FactorGmMultiplier() * self)
   .def(FactorGmMultiplier() / self)
   .def(self += FactorGmMultiplier())
   .def(self -= FactorGmMultiplier())
   .def(self *= FactorGmMultiplier())
   .def(self /= FactorGmMultiplier())
   ;
   
   

}


template void export_ifactor<GmValueType,GmIndexType>();
