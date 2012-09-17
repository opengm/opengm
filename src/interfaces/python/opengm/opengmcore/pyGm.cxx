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
         ostr<<"-number of variables :"<<gm.numberOfVariables()<<"\n";
         for(size_t i=0;i<GM::NrOfFunctionTypes;++i){
            ostr<<"-number of function(type-"<<i<<")"<<gm.numberOfFunctions(i)<<"\n";
         }
         ostr<<"-number of factors :"<<gm.numberOfFactors()<<"\n";
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

  
   class_<PyGm > ("GraphicalModel", init< >())
      .def("__init__", make_constructor(&pygm::gmConstructorPythonNumpy<PyGm,IndexType> ))
      .def("__init__", make_constructor(&pygm::gmConstructorPythonList<PyGm,int> ))
      .def("assign", &pygm::assignPythonNumpy<PyGm,IndexType>)
      .def("assign", &pygm::assignPythonList<PyGm,int>)
      .def("__str__", &pygm::printGmPy<PyGm>)
      .def("space", &PyGm::space , return_internal_reference<>())
      .add_property("numberOfVariables", &pygm::numVarGm<PyGm>)
      .add_property("numberOfFactors", &pygm::numFactorGm<PyGm>)
      .add_property("operator",&pygm::operatorAsString<PyGm>)
      .def("numberOfVariablesForFactor", &pygm::numVarFactor<PyGm>)
      .def("numberOfLabels", &PyGm::numberOfLabels)
      .def("numberOfFactorsForVariable", &pygm::numFactorVar<PyGm>)
      .def("isAcyclic",&PyGm::isAcyclic)
      //.def("addFunction", &PyGm::addFunction< PyExplicitFunction >)
      .def("addFunctionRaw", &pygm::addFunctionNpPy<PyGm>)
      .def("addFactor", &pygm::addFactorPyNumpy<PyGm>)
      .def("addFactor", &pygm::addFactorPyTuple<PyGm,int>)
      .def("addFactor", &pygm::addFactorPyList<PyGm,int>)
      .def("__getitem__", &pygm::getFactorStaticPy<PyGm>, return_internal_reference<>())
      .def("evaluate",&pygm::evaluatePyNumpy<PyGm>)
      .def("evaluate",&pygm::evaluatePyList<PyGm,int>)
      ;
}


template void export_gm<GmAdder>();
template void export_gm<GmMultiplier>();
