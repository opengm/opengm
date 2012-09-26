#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleCore

#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/noprefix.h>
#ifdef Bool
#undef Bool
#endif 

#include <stdexcept>
#include <string>
#include <sstream>
#include <ostream>
#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include "opengm_helpers.hxx"
#include "copyhelper.hxx"
#include "nifty_iterator.hxx"
#include "export_typedes.hxx"
#include "../converter.hxx"
#include "numpyview.hxx"


using namespace boost::python;

namespace pygm {

      //constructor from numpy array
      template<class GM,class VALUE_TYPE>
      GM *  gmConstructorPythonNumpy( NumpyView<VALUE_TYPE,1>  numberOfLabels) {        
         return new GM(typename GM::SpaceType(numberOfLabels.begin1d(), numberOfLabels.end1d()));
      }
      template<class GM,class VALUE_TYPE>
      GM *  gmConstructorPythonList(const boost::python::list & numberOfLabelsList) {
         typedef PythonIntListAccessor<VALUE_TYPE,true> Accessor;
         typedef opengm::AccessorIterator<Accessor,true> Iterator;
         Accessor accessor(numberOfLabelsList);
         Iterator begin(accessor,0);
         Iterator end(accessor,accessor.size());
         return new GM(typename GM::SpaceType(begin, end));
      }
      
      template<class GM,class VALUE_TYPE>
      void assignPythonList( GM & gm ,const boost::python::list & numberOfLabelsList) {
         typedef PythonIntListAccessor<VALUE_TYPE,true> Accessor;
         typedef opengm::AccessorIterator<Accessor,true> Iterator;
         Accessor accessor(numberOfLabelsList);
         Iterator begin(accessor,0);
         Iterator end(accessor,accessor.size());
         typename GM::SpaceType space(begin, end);
         gm.assign(space);
      }
      
      template<class GM,class VALUE_TYPE>
      void assignPythonNumpy( GM & gm ,NumpyView<VALUE_TYPE,1>  numberOfLabels) {
         typename GM::SpaceType space(numberOfLabels.begin1d(), numberOfLabels.end1d());
         gm.assign(space);
      }
      template<class GM>
      typename GM::IndexType addFactorPyNumpy
      (
         GM & gm,const typename GM::FunctionIdentifier & fid, NumpyView<typename  GM::IndexType,1>   vis
      ) {
         return gm.addFactor(fid, vis.begin1d(), vis.end1d());
      }
            
      template<class GM,class VALUE_TYPE>
      typename GM::IndexType addFactorPyList
      (
         GM & gm,const typename GM::FunctionIdentifier & fid, const boost::python::list &  vis
      ) {
         typedef PythonIntListAccessor<VALUE_TYPE,true> Accessor;
         typedef opengm::AccessorIterator<Accessor,true> Iterator;
         Accessor accessor(vis);
         Iterator begin(accessor,0);
         Iterator end(accessor,accessor.size());
         return gm.addFactor(fid, begin, end);
      }
      template<class GM,class VALUE_TYPE>
      typename GM::IndexType addFactorPyTuple
      (
         GM & gm,const typename GM::FunctionIdentifier & fid, const boost::python::tuple & vis
      ) {
         typedef PythonIntTupleAccessor<VALUE_TYPE,true> Accessor;
         typedef opengm::AccessorIterator<Accessor,true> Iterator;
         Accessor accessor(vis);
         Iterator begin(accessor,0);
         Iterator end(accessor,accessor.size());
         return gm.addFactor(fid, begin, end);
      }
      
      
      template<class GM>
      typename GM::IndexType numVarGm(const GM & gm) {
         return gm.numberOfVariables();
      }

      template<class GM>
      typename GM::IndexType numVarFactor(const GM & gm,const typename GM::IndexType factorIndex) {
         return gm.numberOfVariables(factorIndex);
      }

      template<class GM>
      typename GM::IndexType numFactorGm(const GM & gm) {
         return gm.numberOfFactors();
      }

      template<class GM>
      typename GM::IndexType numFactorVar(const GM & gm,const typename GM::IndexType variableIndex) {
         return gm.numberOfFactors(variableIndex);
      }
      

      
      template<class NUMPY_OBJECT,class VECTOR>
      size_t extractShape(const NUMPY_OBJECT & a,VECTOR & myshape){
         const boost::python::tuple &shape = boost::python::extract<boost::python::tuple > (a.attr("shape"));
         size_t dimension = boost::python::len(shape);
         myshape.resize(dimension);
         for (size_t d = 0; d < dimension; ++d)
            myshape[d] = boost::python::extract<typename VECTOR::value_type>(shape[d]);
         return dimension;
      }
      template<class NUMPY_OBJECT,class VECTOR>
      size_t extractStrides(const NUMPY_OBJECT & a,VECTOR & mystrides,const size_t dimension){
         intp* strides_ptr = PyArray_STRIDES(a.ptr());
         intp the_rank = dimension;//rank(a);
         for (intp i = 0; i < the_rank; i++) {
            mystrides.push_back(*(strides_ptr + i));
         }
         return dimension;
      }
      
      template<class NUMPY_OBJECT>
      void  * extractPtr(const NUMPY_OBJECT & a){
          return PyArray_DATA(a.ptr());
      }

      
      
      template<class GM>
      typename GM::FunctionIdentifier addFunctionNpPy( GM & gm,boost::python::numeric::array a) {
         //std::cout<<"add function c++\n";
         typedef opengm::ExplicitFunction<typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> ExplicitFunction;        
         ExplicitFunction fEmpty;
         std::pair< typename GM::FunctionIdentifier ,ExplicitFunction & > fidnRef=gm.addFunctionWithRefReturn(fEmpty);
         //std::cout<<"added  function c++\n";
         ExplicitFunction & f=fidnRef.second;
         //std::cout<<"get vie to numpy  function c++\n";
         NumpyView<typename GM::ValueType> numpyView(a);
         //std::cout<<"resize \n";
         
         //for(size_t i=0;i<numpyView.dimension();++i)
         //   std::cout<<numpyView.shapeBegin()[i]<<" ";
         //std::cout<<"\n";
         
         f.resize(numpyView.shapeBegin(), numpyView.shapeEnd());
         //std::cout<<"fill\n";
         if(numpyView.dimension()==1){
            size_t ind[1];
            size_t i = 0;
            for (ind[0] = 0; ind[0] < f.shape(0); ++ind[0]) {              
               f(i) = numpyView(ind[0]);
               ++i;
            }
         }
         else if(numpyView.dimension()==2){
            size_t ind[2];
            size_t i = 0;
            for (ind[1] = 0; ind[1] < f.shape(1); ++ind[1]){
               for (ind[0] = 0; ind[0] < f.shape(0); ++ind[0]) {
                  f(i) = numpyView(ind[0],ind[1]);
                  ++i;
               }
            }
         }
        
         else{
            opengm::ShapeWalker<typename ExplicitFunction::FunctionShapeIteratorType> walker(f.functionShapeBegin(),f.dimension());
            for (size_t i=0;i<f.size();++i) {
               typename GM::ValueType v=numpyView[walker.coordinateTuple().begin()];
               f(i) = v;
               ++walker;
            }
         }
         return fidnRef.first;
         
      }
  

      template<class GM>
      const typename GM::FactorType & getFactorPy(const GM & gm,const typename GM::IndexType factorIndex) {
         return gm.operator[](factorIndex);
      }

      template<class GM>
      const typename GM::FactorType  & getFactorStaticPy(const GM & gm, const int factorIndex) {
         return gm.operator[](factorIndex);
      }
      
      template<class GM>
      std::string printGmPy(const GM & gm) {
         std::stringstream ostr;
         ostr<<"-number of variables :"<<gm.numberOfVariables()<<std::endl;
         for(size_t i=0;i<GM::NrOfFunctionTypes;++i){
            ostr<<"-number of function(type-"<<i<<")"<<gm.numberOfFunctions(i)<<std::endl;
         }
         ostr<<"-number of factors :"<<gm.numberOfFactors()<<std::endl;
         ostr<<"-max. factor order :"<<gm.factorOrder();
         return ostr.str();
      }
      template<class GM>
      std::string operatorAsString(const GM & gm) {
         if(opengm::meta::Compare<typename GM::OperatorType,opengm::Adder>::value)
            return "adder";
         else
            return "multiplier";
      }
      
      template<class GM>
      typename GM::ValueType evaluatePyNumpy
      (
         const GM & gm,
         NumpyView<typename GM::IndexType,1> states
      ){
         return gm.evaluate(states.begin1d());
      }
      
      template<class GM,class INDEX_TYPE>
      typename GM::ValueType evaluatePyList
      (
         const GM & gm,
         boost::python::list states
      ){
         IteratorHolder< PythonIntListAccessor<INDEX_TYPE,true> > holder(states);
         return gm.evaluate(holder.begin());
      }
   }



template<class GM>
void export_gm() {


   typedef GM PyGm;
   typedef typename PyGm::SpaceType PySpace;
   typedef typename PyGm::ValueType ValueType;
   typedef typename PyGm::IndexType IndexType;
   typedef typename PyGm::LabelType LabelType;
   typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType> PyExplicitFunction;
   
   
   typedef typename PyGm::FunctionIdentifier PyFid;
   typedef typename PyGm::FactorType PyFactor;
   typedef typename PyFid::FunctionIndexType FunctionIndexType;
   typedef typename PyFid::FunctionTypeIndexType FunctionTypeIndexType;
	
   docstring_options doc_options(true, true, false);
  
	class_<PyGm > ("GraphicalModel", 
	"The central class of opengm which holds the factor graph and functions of the graphical model",
	init< >("Construct an empty graphical model with no variables ")
	)
	.def("__init__", make_constructor(&pygm::gmConstructorPythonNumpy<PyGm,IndexType> ,default_call_policies(),(arg("numberOfLabels"))),
	"Construct a gm from a numpy array which holds the number of labels for all variables.\n\n"
	"	The gm will have as many variables as the length of the numpy array\n\n"
	"Args:\n\n"
	"  numberOfLabels: holds the number of labels for each variable\n\n"
	)
	.def("__init__", make_constructor(&pygm::gmConstructorPythonList<PyGm,int> ,default_call_policies(),(arg("numberOfLabels"))),
	"Construct a gm from a python list which holds the number of labels for all variables.\n\n"
	"The gm will have as many variables as the length of the list\n"
	"Args:\n\n"
	"  numberOfLabels: holds the number of labels for each variable\n\n"
	)
	.def("assign", &pygm::assignPythonNumpy<PyGm,IndexType>,args("numberOfLabels"),
	"Assign a gm from a python list which holds the number of labels for all variables.\n\n"
	"	The gm will have as many variables as the length of the numpy array\n\n"
	"Args:\n\n"
	"  numberOfLabels: holds the number of labels for each variable\n\n"
	"Returns:\n"
   	"  None\n\n"
	)
	.def("assign", &pygm::assignPythonList<PyGm,int>,(arg("numberOfLabels")),
	"Assign a gm from a python list which holds the number of labels for all variables.\n\n"
	" The gm will have as many variables as the length of the list\n\n"
	"Args:\n\n"
	"  numberOfLabels: holds the number of labels for each variable\n\n"
	"Returns:\n"
   	"  None\n\n"
	)
	.def("__str__", &pygm::printGmPy<PyGm>,
	"Print a a gm as string"
	"Returns:\n"
   	"	A string which describes the graphical model \n\n"
	)
	.def("space", &PyGm::space , return_internal_reference<>(),
	"Get the variable space of the graphical model\n\n"
	"Returns:\n"
	"	A const reference to space of the gm."
	)
	.add_property("numberOfVariables", &pygm::numVarGm<PyGm>,
	"Get the number of variables of the graphical model"
	"Returns:\n"
	"	Number of variables."
	)
	.add_property("numberOfFactors", &pygm::numFactorGm<PyGm>,
	"The Number of factors of the graphical model\n\n"
	)
	.add_property("operator",&pygm::operatorAsString<PyGm>,
	"The operator of the graphical model as a string"
	)
	.def("numberOfVariablesOfFactor", &pygm::numVarFactor<PyGm>,args("factorIndex"),
	"Get the number of variables which are connected to a factor\n\n"
	"Args:\n\n"
	"  factorIndex: index to a factor in this gm\n\n"
	"Returns:\n"
   	"	The nubmer of variables which are connected \n\n"
   	"		to the factor at ``factorIndex``"
	)
	.def("numberOfFactorsOfVariable", &pygm::numFactorVar<PyGm>,args("variableIndex"),
	"Get the number of factors which are connected to a variable\n\n"
	"Args:\n\n"
	"  variableIndex: index of a variable w.r.t. the gm\n\n"
	"Returns:\n"
   	"	The nubmer of variables which are connected \n\n"
   	"		to the factor at ``factorIndex``"
	)
	.def("variableOfFactor",&PyGm::variableOfFactor,(arg("factorIndex"),arg("variableIndex")),
	"Get the variable index of a varible which is connected to a factor.\n\n"
	"Args:\n\n"
	"  factorIndex: index of a factor w.r.t the gm\n\n"
	"  variableIndex: index of a variable w.r.t the factor at ``factorIndex``\n\n"
	"Returns:\n"
   	"	The variableIndex w.r.t. the gm of the factor at ``factorIndex``"
	)
	.def("factorOfVariable",&PyGm::variableOfFactor,(arg("variableIndex"),arg("factorIndex")),
	"Get the variable index of a varible which is connected to a factor.\n\n"
	"Args:\n\n"
	"  factorIndex: index of a variable w.r.t the gm\n\n"
	"  variableIndex: index of a factor w.r.t the variable at ``variableInex``\n\n"
	"Returns:\n"
   	"	The variableIndex w.r.t. the gm of the factor at ``factorIndex``"
      
	)		
	.def("numberOfLabels", &PyGm::numberOfLabels,args("variableIndex"),
	"Get the number of labels for a variable\n\n"
	"Args:\n\n"
	"  variableIndex: index to a variable in this gm\n\n"
	"Returns:\n"
   	"	The nubmer of labels for the variable at ``variableIndex``"
	)
	.def("isAcyclic",&PyGm::isAcyclic,
	"check if the graphical is isAcyclic.\n\n"
	"Returns:\n"
	"	True if model has no loops / is acyclic\n\n"
	"	False if model has loops / is not acyclic\n\n"
	)
	.def("addFunction", &pygm::addFunctionNpPy<PyGm>,args("function"),
	"Adds a function to the graphical model."
	"Args:\n\n"
	"  function: a function/ value table\n\n"
	"		The type of \"function\"  has to be a numpy ndarray.\n\n"
	"Returns:\n"
   	"  	A function identifier (fid) .\n\n"
   	"		This fid is used to connect a factor to this function\n\n"
   	"Examples:\n"
	"	Adding 1th-order function with the shape [3]::\n\n"
	"		gm.graphicalModel([3,3,3])\n"
	"		f=numpy.array([0.8,1.4,0.1],dtype=numpy.float32)\n"
	"		fid=gm.addFunction(f)\n\n"
	"	Adding 2th-order function with  the shape [4,4]::\n\n"
	"		gm.graphicalModel([4,4,4])\n"
	"		f=numpy.ones([4,4],dtype=numpy.float32)\n"
	"		#fill the function with values\n"
	"		#..........\n"
	"		fid=gm.addFunction(f)\n\n"
	"	Adding 3th-order function with the shape [4,5,2]::\n\n"
	"		gm.graphicalModel([4,4,4,5,5,2])\n"
	"		f=numpy.ones([4,5,2],dtype=numpy.float32)\n"
	"		#fill the function with values\n"
	"		#..........\n"
	"		fid=gm.addFunction(f)\n\n"
	)
	.def("addFactor", &pygm::addFactorPyNumpy<PyGm>, (arg("fid"),arg("variableIndices")),
	"Adds a factor to the gm.\n\n"
	"	The factors will is connected to the function indicated with \"fid\".\n\n"
	"	The factors variables are given by ``variableIndices``. \"variableIndices\" has to be sorted.\n\n"
	"	In this overloading of \"addFactor\" the type of \"variableIndices\"  has to be a 1d numpy array\n\n"
	"Args:\n\n"
	"	variableIndices: the factors variables \n\n"
	"		``variableIndices`` has to be sorted.\n\n"
	"Returns:\n"
   	"  index of the added factor .\n\n"
	"Example\n"
    "	adding a factor to the graphical model::\n\n"
	"		# assuming there is a function \"f\"\n"
	"		fid=gm.addFunction(f)\n"
	"		vis=numpy.array([2,3,5],dtype=numpy.uint64)\n"
	"		#vis has to be sorted \n"	
	"		gm.addFactor(fid,vis)    \n\n"
	)
	.def("addFactor", &pygm::addFactorPyTuple<PyGm,int>, (arg("fid"),arg("variableIndices")),
	"Adds a factor to the gm.\n\n"
	"	The factors will is connected to the function indicated with \"fid\".\n\n"
	"	The factors variables are given by ``variableIndices``. \"variableIndices\" has to be sorted.\n\n"
	"	In this overloading of \"addFactor\" the type of \"variableIndices\"  has to be a tuple\n\n"
	"Args:\n\n"
	"	variableIndices: the factors variables \n\n"
	"		``variableIndices`` has to be sorted.\n\n"
	"Returns:\n"
   	"  index of the added factor .\n\n"
	"Example:\n"
    "	adding a factor to the graphical model::\n\n"
	"		# assuming there is a function \"f\"\n"
	"		fid=gm.addFunction(f)\n"
	"		vis=(2,3,5)\n"
	"		#vis has to be sorted \n"	
	"		gm.addFactor(fid,vis)    \n\n"
	)
	.def("addFactor", &pygm::addFactorPyList<PyGm,int>, (arg("fid"),arg("variableIndices")),
	"Adds a factor to the gm.\n\n"
	"	The factors will is connected to the function indicated with \"fid\".\n\n"
	"	The factors variables are given by ``variableIndices``. \"variableIndices\" has to be sorted.\n\n"
	"	In this overloading of \"addFactor\" the type of \"variableIndices\"  has to be a list\n\n"
	"Args:\n\n"
	"	variableIndices: the factors variables \n\n"
	"		``variableIndices`` has to be sorted.\n\n"
	"Returns:\n"
   	"  index of the added factor .\n\n"
	"Example:\n"
    "	adding a factor to the graphical model::\n\n"
	"		# assuming there is a function \"f\"\n"
	"		fid=gm.addFunction(f)\n"
	"		vis=[2,3,5]\n"
	"		#vis has to be sorted \n"	
	"		gm.addFactor(fid,vis)    \n\n"
	)
	.def("__getitem__", &pygm::getFactorStaticPy<PyGm>, return_internal_reference<>(),(arg("factorIndex")),
	"Get a factor of the graphical model\n\n"
	"Args:\n\n"
	"	factorIndex: index of a factor w.r.t. the gm \n\n"
	"		``factorIndex`` has to be a integral scalar::\n\n"
	"Returns:\n"
   	"  A const reference to the factor at ``factorIndex``.\n\n"
	"Example:\n"
	"    factor=gm[someFactorIndex]\n"
	)
	.def("evaluate",&pygm::evaluatePyNumpy<PyGm>,(arg("labels")),
	"Evaluates the factors of given a labelSequence.\n\n"
	"	In this overloading the type of  \"labelSequence\" has to be a 1d numpy array\n\n"
	"Args:\n\n"
	"	labelSequence: A labeling for all variables.\n\n"
	"		Has to as long as ``gm.numberOfVariables``.\n\n"
	"Returns:\n"
	"	The energy / probability for the given ``labelSequence``"
	)
	.def("evaluate",&pygm::evaluatePyList<PyGm,int>,(arg("labels")),
	"Evaluates the factors of given a labelSequence.\n\n"
	"	In this overloading the type of  \"labelSequence\" has to be a list\n\n"
	"Args:\n\n"
	"	labelSequence: A labeling for all variables.\n\n"
	"		Has to as long as ``gm.numberOfVariables``.\n\n"
	"Returns:\n"
	"	The energy / probability for the given ``labelSequence``"
	)
  ;
}


template void export_gm<GmAdder>();
template void export_gm<GmMultiplier>();
