#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include "copyhelper.hxx"
#include "nifty_iterator.hxx"
#include "utilities/shapeHolder.hxx"
#include "factorhelper.hxx"

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


using namespace boost::python;


   

template<class V,class I>
void export_ifactor(){
   typedef V ValueType;
   typedef I IndexType;
   typedef IndexType LabelType;
   typedef opengm::IndependentFactor<ValueType,IndexType,LabelType> PyIndependentFactor;
   
   import_array();
   docstring_options doc_options(true,true,false);
   typedef FactorShapeHolder<PyIndependentFactor> ShapeHolder;
   typedef FactorViHolder<PyIndependentFactor> ViHolder;
   //------------------------------------------------------------------------------------
   // shape-holder
   //------------------------------------------------------------------------------------   
   class_<ShapeHolder > ("IndependentFactorShape", 
   "Holds the shape of an independent factor.\n"
   "IndependentFactorShape is only a view to the factors shape,\n"
   "therefore only one pointer to the factor is stored",
   init<const PyIndependentFactor &>()[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const PyIndependentFactor& */>()])
   .def(init< >())
   .def("__iter__",boost::python::iterator<ShapeHolder>())
   .def("__len__", &ShapeHolder::size)
   .def("__str__",&ShapeHolder::asString,
	"Convert shape to a string\n"
	"Returns:\n"
	"  new allocated string"
   )
   .def("__array__", &ShapeHolder::toNumpy,
	"Convert shape to a 1d numpy ndarray\n"
	"Returns:\n"
	"  new allocated 1d numpy ndarray"
   )
   .def("__list__", &ShapeHolder::toList,
	"Convert shape to a list\n"
	"Returns:\n"
	"  new allocated list"
   )
   .def("__tuple__",&ShapeHolder::toTuple,
	"Convert shape to a tuple\n"
	"Returns:\n"
	"  new allocated tuple"
   )
   .def("__getitem__", &ShapeHolder::operator[], return_value_policy<return_by_value>(),(arg("variableIndex")),
   "Get the number of labels for a variable which is connected to this factor.\n\n"
   "Args:\n\n"
   "  variableIndex: variable index w.r.t. the factor"
   "Returns:\n"
   "  number of labels for the variable at ``variableIndex``\n\n"
   )
   .def("__copy__", &generic__copy__< ShapeHolder >)
   ;
   //------------------------------------------------------------------------------------
   // vi-holder
   //------------------------------------------------------------------------------------
   class_<ViHolder > ("IndependentFactorVariableIndices", 
   "Holds the variable indices of an factor.\n"
   "``FactorVariableIndices`` is only a view to the factors variable indices,\n"
   "therefore only one pointer to the factor is stored",
   init<const  PyIndependentFactor &>()[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const PyIndependentFactor& */>()] )
   .def(init< >())
   .def("__iter__",boost::python::iterator<ViHolder>())
   .def("__len__", &ViHolder::size)
   .def("__str__",&ViHolder::asString,
	"Convert shape to a string\n"
	"Returns:\n"
	"  new allocated string"
   )
   .def("__array__", &ViHolder::toNumpy,
	"Convert the variable indices  to a 1d numpy ndarray\n"
	"Returns:\n"
	"  new allocated 1d numpy ndarray"
   )
   .def("__list__", &ViHolder::toList,
	"Convert the variable indices  to a list\n"
	"Returns:\n"
	"  new allocated list"
   )
   .def("__tuple__",&ViHolder::toTuple,
	"Convert the variable indices  to a tuple\n"
	"Returns:\n"
	"  new allocated tuple"
   )
   .def("__getitem__", &ViHolder::operator[], return_value_policy<return_by_value>(),(arg("variableIndex")),
   "Get the number of variables for a variable which is connected to this factor.\n\n"
   "Args:\n\n"
   "  variableIndex: variable index w.r.t. the factor"
   "Returns:\n"
   "  number of Labels for the variable at ``variableIndex``\n\n"
   )
   .def("__copy__", &generic__copy__< ViHolder >)
   ;
   
   class_<PyIndependentFactor > ("IndependentFactor", init< >())
   .add_property("PyIndependentFactor", &PyIndependentFactor::size)
   .add_property("numberOfVariables", &PyIndependentFactor::numberOfVariables)
   .def("numberOfLabels", &PyIndependentFactor::numberOfLabels)
   .add_property("shape", &pyfactor::getShapeHolder<PyIndependentFactor>  )
   .add_property("shape", &pyfactor::getShapeHolder<PyIndependentFactor> ,
   "Get the shape of a independent factor, \n"
   "which is a sequence of the number of lables for all variables which are connected to this factor"
   )
   .def("_getitem__", &pyfactor::getValuePyTuple<PyIndependentFactor,int>, return_value_policy<return_by_value>())
   .def("_getitem",&pyfactor::getValuePyVector<PyIndependentFactor> , return_value_policy<return_by_value>())
   .def("_getitem", &pyfactor::getValuePyList<PyIndependentFactor,int>, return_value_policy<return_by_value>())
   .def("_getitem", &pyfactor::getValuePyNumpy<PyIndependentFactor>, return_value_policy<return_by_value>())
   .def("__str__", &pyfactor::printFactorPy<PyIndependentFactor>)
   // copy values to numpy order
   .def("copyValuesSwitchedOrder", &pyfactor::ifactorToNumpy<PyIndependentFactor>)
   // min 
   .def("minInplace", &pyacc::accSomeIFactorInplacePyNumpy<PyIndependentFactor,opengm::Minimizer>)
   .def("minInplace", &pyacc::accSomeIFactorInplacePyTuple<PyIndependentFactor,opengm::Minimizer,int>)
   .def("minInplace", &pyacc::accSomeIFactorInplacePyList<PyIndependentFactor,opengm::Minimizer,int>)
   .def("min", &pyacc::accSomeCopyPyNumpy<PyIndependentFactor,opengm::Minimizer>,return_value_policy<manage_new_object>())
   .def("min", &pyacc::accSomeCopyPyTuple<PyIndependentFactor,opengm::Minimizer,int>,return_value_policy<manage_new_object>())
   .def("min", &pyacc::accSomeCopyPyList<PyIndependentFactor,opengm::Minimizer,int>,return_value_policy<manage_new_object>())
   .def("min", &PyIndependentFactor::min)
    // max
   .def("maxInplace", &pyacc::accSomeIFactorInplacePyNumpy<PyIndependentFactor,opengm::Maximizer>)
   .def("maxInplace", &pyacc::accSomeIFactorInplacePyTuple<PyIndependentFactor,opengm::Maximizer,int>)
   .def("maxInplace", &pyacc::accSomeIFactorInplacePyList<PyIndependentFactor,opengm::Maximizer,int>)
   .def("max", &pyacc::accSomeCopyPyNumpy<PyIndependentFactor,opengm::Maximizer>,return_value_policy<manage_new_object>())
   .def("max", &pyacc::accSomeCopyPyTuple<PyIndependentFactor,opengm::Maximizer,int>,return_value_policy<manage_new_object>())
   .def("max", &pyacc::accSomeCopyPyList<PyIndependentFactor,opengm::Maximizer,int>,return_value_policy<manage_new_object>())
   .def("max", &PyIndependentFactor::max)
   //sum
   .def("sumInplace", &pyacc::accSomeIFactorInplacePyNumpy<PyIndependentFactor,opengm::Integrator>)
   .def("sumInplace", &pyacc::accSomeIFactorInplacePyTuple<PyIndependentFactor,opengm::Integrator,int>)
   .def("sumInplace", &pyacc::accSomeIFactorInplacePyList<PyIndependentFactor,opengm::Integrator,int>)
   .def("sum", &pyacc::accSomeCopyPyNumpy<PyIndependentFactor,opengm::Integrator>,return_value_policy<manage_new_object>())
   .def("sum", &pyacc::accSomeCopyPyTuple<PyIndependentFactor,opengm::Integrator,int>,return_value_policy<manage_new_object>())
   .def("sum", &pyacc::accSomeCopyPyList<PyIndependentFactor,opengm::Integrator,int>,return_value_policy<manage_new_object>())
   .def("sum", &PyIndependentFactor::sum)
   // product
   .def("productInplace", &pyacc::accSomeIFactorInplacePyNumpy<PyIndependentFactor,opengm::Multiplier>)
   .def("productInplace", &pyacc::accSomeIFactorInplacePyTuple<PyIndependentFactor,opengm::Multiplier,int>)
   .def("productInplace", &pyacc::accSomeIFactorInplacePyList<PyIndependentFactor,opengm::Multiplier,int>)
   .def("product", &pyacc::accSomeCopyPyNumpy<PyIndependentFactor,opengm::Multiplier>,return_value_policy<manage_new_object>())
   .def("product", &pyacc::accSomeCopyPyTuple<PyIndependentFactor,opengm::Multiplier,int>,return_value_policy<manage_new_object>())
   .def("product", &pyacc::accSomeCopyPyList<PyIndependentFactor,opengm::Multiplier,int>,return_value_policy<manage_new_object>())
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
   .def(self + opengm::python::FactorGmAdder())
   .def(self - opengm::python::FactorGmAdder())
   .def(self * opengm::python::FactorGmAdder())
   .def(self / opengm::python::FactorGmAdder())
   .def(opengm::python::FactorGmAdder() + self)
   .def(opengm::python::FactorGmAdder() - self)
   .def(opengm::python::FactorGmAdder() * self)
   .def(opengm::python::FactorGmAdder() / self)
   .def(self += opengm::python::FactorGmAdder())
   .def(self -= opengm::python::FactorGmAdder())
   .def(self *= opengm::python::FactorGmAdder())
   .def(self /= opengm::python::FactorGmAdder())
   //interoperate with FactorMultiplier
   .def(self + opengm::python::FactorGmMultiplier())
   .def(self - opengm::python::FactorGmMultiplier())
   .def(self * opengm::python::FactorGmMultiplier())
   .def(self / opengm::python::FactorGmMultiplier())
   .def(opengm::python::FactorGmMultiplier() + self)
   .def(opengm::python::FactorGmMultiplier() - self)
   .def(opengm::python::FactorGmMultiplier() * self)
   .def(opengm::python::FactorGmMultiplier() / self)
   .def(self += opengm::python::FactorGmMultiplier())
   .def(self -= opengm::python::FactorGmMultiplier())
   .def(self *= opengm::python::FactorGmMultiplier())
   .def(self /= opengm::python::FactorGmMultiplier())
   ;
   
}


template void export_ifactor<opengm::python::GmValueType,opengm::python::GmIndexType>();
