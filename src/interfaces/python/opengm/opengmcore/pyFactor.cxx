#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>

#include "copyhelper.hxx"
#include "nifty_iterator.hxx"
#include "utilities/shapeHolder.hxx"
#include "factorhelper.hxx"

using namespace boost::python;

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
   docstring_options doc_options(true,true,false);
   typedef FactorShapeHolder<PyFactor> ShapeHolder;
   typedef FactorViHolder<PyFactor> ViHolder;

   //------------------------------------------------------------------------------------
   // shape-holder
   //------------------------------------------------------------------------------------  
   class_<ShapeHolder > ("FactorShape", 
   "Holds the shape of a factor.\n"
   "``FactorShape`` is only a view to the factors shape,\n"
   "therefore only one pointer to the factor is stored",
   init<const  PyFactor &>()[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const PyFactor& */>()] )
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
   class_<ViHolder > ("FactorVariableIndices", 
   "Holds the variable indices of an factor.\n"
   "``FactorVariableIndices`` is only a view to the factors variable indices,\n"
   "therefore only one pointer to the factor is stored",
   init<const  PyFactor &>()[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const PyFactor& */>()] )
   .def(init< >())
   .def("__iter__",boost::python::iterator<ViHolder>())
   .def("__len__", &ViHolder::size)
   .def("__str__",&ViHolder::asString)
   .def("__array__", &ViHolder::toNumpy,
   "Convert the variable indices  to a 1d numpy ndarray\n"
   "Returns:\n"
   "  new allocated 1d numpy ndarray"
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
   //------------------------------------------------------------------------------------
   // factor
   //------------------------------------------------------------------------------------
   class_<PyFactor > ("Factor", init< >())
   .add_property("size", &PyFactor::size,
	"The number of entries in the factor's value\n"
	"table::\n\n"
	"    gm=opengm.graphicalModel([2,2,2,2])\n"
	"    fid=gm.addFunction(numpy.ones([2,2],dtype=numpy.uint64))\n"
	"    factorIndex=gm.addFactor(fid,[0,1])\n"
	"    assert( gm[factorIndex].size==4 )\n"
	"    fid=gm.addFunction(numpy.ones([2,2,2],dtype=numpy.uint64))\n"
	"    factorIndex=gm.addFactor(fid,[0,1,2])\n"
	"    assert( gm[factorIndex].size==8 )\n\n"  
   )
   .add_property("numberOfVariables", &PyFactor::numberOfVariables,
   "The number of variables which are connected to the\n"
   	"factor::\n\n"
   	"    #assuming gm,fid2 and fid3 exist:\n"
	"    factorIndex=gm.addFactor(fid2,[0,1])\n"
	"    assert( gm[factorIndex].numberOfVariables==2 )\n"
	"    factorIndex=gm.addFactor(fid3,[0,2,4])\n"
	"    assert( gm[factorIndex].numberOfVariables==3 )\n\n" 
   )
   .def("numberOfLabels", &PyFactor::numberOfLabels,(arg("variableIndex")),
	"Get the number of labels for a variable of the\n"
   	"factor::\n\n"
	"    gm=opengm.graphicalModel([2,3,4,5])\n"
	"    fid=gm.addFunction(numpy.ones([2,3],dtype=numpy.uint64))\n"
	"    factorIndex=gm.addFactor(fid,[0,1])\n"
	"    assert( gm[factorIndex].numberOfLabels(0)==2 )\n"
	"    assert( gm[factorIndex].numberOfLabels(1)==3 )\n"
	"    fid=gm.addFunction(numpy.ones([4,5],dtype=numpy.uint64))\n"
	"    factorIndex=gm.addFactor(fid,[2,4])\n"
	"    assert( gm[factorIndex].numberOfLabels(0)==4 )\n"
	"    assert( gm[factorIndex].numberOfLabels(1)==5 )\n"
   )
   .add_property("variableIndices", &pyfactor::getViHolder<PyFactor> ,
   "Get the variable indices of a factor (the indices of all variables which are connected to this factor)"
   )
   .add_property("shape", &pyfactor::getShapeHolder<PyFactor> ,
   "Get the shape of a factor, which is a sequence of the number of lables for all variables which are connected to this factor"
   )
   .add_property("functionType", &PyFactor::functionType,
   "Get the function type index of a factorm which indicated the type of the function this factor is connected to"
   )
   .add_property("functionIndex", &PyFactor::functionIndex,
   "Get the function index of a factor, which indicated the index of the function this factor is connected to"
   )
   .def("_getitem", &pyfactor::getValuePyNumpy<PyFactor>, return_value_policy<return_by_value>())
   .def("_getitem", &pyfactor::getValuePyTuple<PyFactor,int>, return_value_policy<return_by_value>())
   .def("_getitem", &pyfactor::getValuePyList<PyFactor,int>, return_value_policy<return_by_value>())
   .def("_getitem",&pyfactor::getValuePyVector<PyFactor> , return_value_policy<return_by_value>())
   .def("__str__", &pyfactor::printFactorPy<PyFactor>)
   .def("asIndependentFactor", &pyfactor::iFactorFromFactor<PyFactor,PyIndependentFactor> ,return_value_policy<manage_new_object>())
   .def("copyValues", &pyfactor::copyValuesCallByReturnPy<PyFactor>,
   "Copy the value table of a factor to a new allocated 1d-numpy array in last-coordinate-major-order"
   )
   .def("copyValuesSwitchedOrder", &pyfactor::copyValuesSwitchedOrderCallByReturnPy<PyFactor>,
   "Copy the value table of a factor to a new allocated 1d-numpy array in first-coordinate-major-order"
   )
   .def("isPotts", &PyFactor::isPotts,
   "Check if the factors value table can be written as Potts function"
   )
   .def("isGeneralizedPotts", &PyFactor::isGeneralizedPotts,
   "Check if the factors value table can be written as generalized Potts function"
   )
   .def("isSubmodular", &PyFactor::isSubmodular,
   "Check if the factor is submodular")
   .def("isSquaredDifference", &PyFactor::isSquaredDifference)
   .def("isTruncatedSquaredDifference", &PyFactor::isTruncatedSquaredDifference)
   .def("isAbsoluteDifference", &PyFactor::isAbsoluteDifference)
   .def("isTruncatedAbsoluteDifference", &PyFactor::isTruncatedAbsoluteDifference)
   // min
   .def("min", &pyacc::accSomeCopyPyNumpy<PyFactor,opengm::Minimizer>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Minimize / accumulate over some variables by of the factor.These variables are given by ``accVariables`` \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a 1d numpy.ndarray"
   )
   .def("min", &pyacc::accSomeCopyPyTuple<PyFactor,opengm::Minimizer,int>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Minimize / accumulate over some variables by of the factor.These variables are given by ``accVariables`` \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a tuple"
   )
   .def("min", &pyacc::accSomeCopyPyList<PyFactor,opengm::Minimizer,int>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Minimize / accumulate over some variables by of the factor.These variables are given by ``accVariables``. \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a list"
   )
   .def("min", &PyFactor::min,
   "Get the minimum value of the factor ( the minimum scalar in the factors value table)"
   )
   // max
   .def("max", &pyacc::accSomeCopyPyNumpy<PyFactor,opengm::Maximizer>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Minimize / accumulate over some variables by of the factor.These variables are given by ``accVariables`` \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a 1d numpy.ndarray"
   )
   .def("max", &pyacc::accSomeCopyPyTuple<PyFactor,opengm::Maximizer,int>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Minimize / accumulate over some variables by of the factor.These variables are given by ``accVariables`` \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a tuple"
   )
   .def("max", &pyacc::accSomeCopyPyList<PyFactor,opengm::Maximizer,int>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Minimize / accumulate over some variables by of the factor.These variables are given by ``accVariables``. \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a list"
   )
   .def("max", &PyFactor::max,
   "Get the maximum value of the factor ( the maximum scalar in the factors value table)"
   )
   //sum
   .def("sum", &pyacc::accSomeCopyPyNumpy<PyFactor,opengm::Integrator>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Integrate / accumulate over some variables by of the factor.These variables are given by ``accVariables`` \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a 1d numpy.ndarray"
   )
   .def("sum", &pyacc::accSomeCopyPyTuple<PyFactor,opengm::Integrator,int>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Integrate / accumulate over some variables by of the factor.These variables are given by ``accVariables`` \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a tuple"
   )
   .def("sum", &pyacc::accSomeCopyPyList<PyFactor,opengm::Integrator,int>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Integrate / accumulate over some variables by of the factor.These variables are given by ``accVariables``. \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a list"
   )
   .def("sum", &PyFactor::sum,
   "Get the sum of all values of the factor "
   )
   // product
   .def("product", &pyacc::accSomeCopyPyNumpy<PyFactor,opengm::Multiplier>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Multiply / accumulate over some variables by of the factor.These variables are given by ``accVariables`` \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a 1d numpy.ndarray"
   )
   .def("product", &pyacc::accSomeCopyPyTuple<PyFactor,opengm::Multiplier,int>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Multiply / accumulate over some variables by of the factor.These variables are given by ``accVariables`` \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a tuple"
   )
   .def("product", &pyacc::accSomeCopyPyList<PyFactor,opengm::Multiplier,int>,return_value_policy<manage_new_object>(),(arg("accVariables")),
   "Multiply / accumulate over some variables by of the factor.These variables are given by ``accVariables``. \n"
   "The result is an independentFactor. This independentFactor is only connected to the factors variables "
   "which where not in ``accVariables``.\n"
   "In this overloading the type of ``accVariables`` has to be a list"
   )
   .def("product", &PyFactor::product,
   "Get the product of all values of the factor "
   )
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
   .def(self + opengm::python::GmIndependentFactor())
   .def(self - opengm::python::GmIndependentFactor())
   .def(self * opengm::python::GmIndependentFactor())
   .def(self / opengm::python::GmIndependentFactor())
   .def(opengm::python::GmIndependentFactor() + self)
   .def(opengm::python::GmIndependentFactor() - self)
   .def(opengm::python::GmIndependentFactor() * self)
   .def(opengm::python::GmIndependentFactor() / self)
   ;
}


template void export_factor<opengm::python::GmAdder>();
template void export_factor<opengm::python::GmMultiplier>();   
