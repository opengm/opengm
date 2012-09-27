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
      
      //template<class GM>

      
      
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
      typename GM::IndexType addFactorsListPy
      (
         GM & gm,boost::python::list fids, boost::python::list vis
      ){
         typedef typename GM::FunctionIdentifier FidType;
         typedef typename GM::IndexType IndexType;
         size_t numFid=boost::python::len(fids);
         size_t numVis=boost::python::len(vis);
         if(numFid!=numVis || numFid!=1)
            throw opengm::RuntimeError("len(fids) must be 1 or len(vis)");
         FidType fid;
         if(numFid==1){
            // extract fid
            boost::python::extract<FidType> extractor(fids[0]);
            if(extractor.check())
               fid= static_cast<FidType >(extractor());
            else
               throw opengm::RuntimeError("wrong data type in fids list");
         }
         IndexType retFactorIndex=0;
         IndexType factorIndex=0;
         for(size_t i=0;i<numVis;++i){
            // extract fid
            if(numFid!=1){
               boost::python::extract<FidType> extractor(fids[i]);
               if(extractor.check())
                  fid= static_cast<FidType >(extractor());
               else
                  throw opengm::RuntimeError("wrong data type in fids list");
            }
            // extract vis
            bool extracted=false;
            // 1. try as list
            {
               boost::python::extract<boost::python::list> extractor(vis[i]);
               if(extractor.check()){
                  boost::python::list visI = static_cast<boost::python::list >(extractor());
                  factorIndex=addFactorPyList(fid,visI);
                  extracted=true;
               }
            }
            // 2. try as tuple
            if(!extracted){
               boost::python::extract<boost::python::tuple> extractor(vis[i]);
               if(extractor.check()){
                  boost::python::tuple visI = static_cast<boost::python::tuple >(extractor());
                  factorIndex=addFactorPyTuple(fid,visI);
                  extracted=true;
               }
            }
            // 3. try as numpy
            if(!extracted){
               boost::python::extract<boost::python::numeric::array> extractor(vis[i]);
               if(extractor.check()){
                  boost::python::numeric::array visI = static_cast<boost::python::numeric::array >(extractor());
                  factorIndex=addFactorPyTuple(fid,visI);
                  extracted=true;
               }
            }
            if(!extracted){
               throw opengm::RuntimeError("wrong data type in vis list");
            }
            else{
               if(i==0)
                  retFactorIndex=factorIndex;
            }
         }
         return retFactorIndex;
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
      boost::python::list addFunctionsListNpPy( GM & gm,boost::python::list functionList) {
         typedef typename GM::FunctionIdentifier FidType;
         size_t numF=boost::python::len(functionList);
         boost::python::list fidList;
         for(size_t i=0;i<numF;++i){
            boost::python::extract<boost::python::numeric::array> extractor(functionList[i]);
            if(extractor.check()){
               boost::python::numeric::array functionAsNumpy= static_cast<boost::python::numeric::array >(extractor());
               fidList.append(addFunctionNpPy(gm,functionAsNumpy));
            }
            else{
               throw opengm::RuntimeError("wrong data type in list");
            }
         }
         return fidList;
      }
      
      template<class GM>
      boost::python::list addFunctionsNpPy( GM & gm,NumpyView<typename GM::ValueType> view) {
         typedef typename GM::FunctionIdentifier FidType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename NumpyView<ValueType>::ShapeIteratorType ShapeIteratorType;
         typedef opengm::FastSequence<IndexType,1> FixedSeqType;
         typedef typename FixedSeqType::const_iterator FixedSeqIteratorType;
         typedef opengm::SubShapeWalker<ShapeIteratorType,FixedSeqIteratorType,FixedSeqIteratorType> SubWalkerType;
         typedef opengm::ExplicitFunction<typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> ExplicitFunction;        
          
         const size_t dim=view.dimension();
         const size_t numF=view.shape(0);
         if(dim<2){
            throw opengm::RuntimeError("functions dimension must be at least 2");
         }
         // allocate fixed coordinate and fixed coordinate values
         FixedSeqType fixedC(1);
         fixedC[0]=0;
         FixedSeqType fixedV(1);
         // fid return list
         boost::python::list fidList;
         // loop over 1 dimension/axis of the numpy ndarray view 
         for(size_t f=0;f<numF;++f){
            // add new function to gm (empty one and fill the ref.)
            ExplicitFunction fEmpty;
            std::pair<FidType ,ExplicitFunction & > fidnRef=gm.addFunctionWithRefReturn(fEmpty);
            // append "fid" to fid return list
            fidList.append(fidnRef.first);
            // reference to the function
            ExplicitFunction & function=fidnRef.second;
            // resizse
            function.resize(view.shapeBegin()+1,view.end());
            // subarray walker (walk over the subarray,first dimension is fixeed to the index "f")
            fixedV[0]=f;
            SubWalkerType subwalker(view.shapeBegin(),fixedC.begin(),fixedV.begin());
            const size_t subSize=subwalker.size();
            for(size_t i=0;i<subSize;++i,++subwalker){
               // fill gm function with values
               function(i)=view[subwalker.coordinateTuple().begin()];
            }
         }
         return fidList;
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



   namespace pygmgen{
      
      template<class GM>
      GM * grid2Order2d
      (
         NumpyView<typename GM::ValueType,3> unaryFunctions,
         boost::python::numeric::array binaryFunction,
         bool numpyOrder
      ){
         typedef typename GM::SpaceType Space;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename GM::FunctionIdentifier FunctionIdentifier;
         typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType> ExplicitFunctionType;
         typedef std::pair<FunctionIdentifier,ExplicitFunctionType &> FidRefPair;
         
         const size_t shape[]={unaryFunctions.shape(0),unaryFunctions.shape(1)};
         const size_t numVar=shape[0]*shape[1];
         const size_t numLabels=unaryFunctions.shape(2);
         GM * gm;
         { // scope to delete space
            Space space(numVar,numLabels);
            gm = new GM(space);
         }
         // add one (!) 2.-order-function to the gm
         FunctionIdentifier fid2=pygm::addFunctionNpPy(*gm,binaryFunction);
         IndexType c[2]={0,0};
         for(c[0]=0;c[0]<shape[0];++c[0]){
            for(c[1]=0;c[1]<shape[1];++c[1]){
               //unaries
               ExplicitFunctionType fempty;
               FidRefPair fidRef=gm->addFunctionWithRefReturn(fempty);
               FunctionIdentifier fid=fidRef.first;
               ExplicitFunctionType & f=fidRef.second;
               // resize f
               f.resize(&numLabels,&numLabels+1);
               // fill with data
               for(LabelType l=0;l<numLabels;++l)
                  f(l)=unaryFunctions(c[0],c[1],l);
               //connect 1.-order-function f to 1.-order-factor
               IndexType vi=numpyOrder? c[1]+c[0]*shape[1] :  c[0]+c[1]*shape[0];
               gm->addFactor(fid,&vi,&vi+1);
               // 2.-order-factors
               if(c[0]+1<shape[0]){
                  IndexType vi2=numpyOrder? c[1]+(c[0]+1)*shape[1] : (c[0]+1)+c[1]*shape[0];
                  const IndexType vis[]={vi<vi2?vi:vi2,vi<vi2?vi2:vi};
                  gm->addFactor(fid2,vis,vis+2);
               }
               if(c[1]+1<shape[1]){
                  IndexType vi2=numpyOrder? (c[1]+1)+c[0]*shape[1] : c[0]+(c[1]+1)*shape[0];
                  const IndexType vis[]={vi<vi2?vi:vi2,vi<vi2?vi2:vi};
                  gm->addFactor(fid2,vis,vis+2);
               }
            }  
         } 
         return gm;
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
   
   def("gridGm2dGenerator",&pygmgen::grid2Order2d<PyGm>,return_value_policy<manage_new_object>(),
   (arg("unaryFunctions"),arg("binaryFunction"),arg("numpyCoordinateOrder")=true),
   "Generate a 2th-order graphical model on a 2d-grid.\n\n"
	"	The 2th-order regularizer is the same on the complete grid\n\n"
	"Args:\n\n"
	"  unaryFunctions: 3d ndarray where the first dimension 2 dimension  \n\n"
   "     are the shape of the grid, the 3th-dimension is the label axis "
	"  binaryFunction: numberOfLabels x numberOfLabels 2d numpy ndarray which is  \n\n"
   "     the 2th-order regularizer ( which is the same on the complete grid )"
   "  numpyCoordinateOrder: Coordinate order which indicates which variable belongs to which coordinate"
	"Returns:\n"
   	"  The grid graphical model\n\n"
   );
   
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
   .def("addFunctions", &pygm::addFunctionsListNpPy<PyGm>,args("functions"),
	"Adds multiple functions to the graphical model."
	"Args:\n\n"
	"  functions: a list with function/ value table\n\n"
	"		The elemet type of the list \"functions\"  has to be a numpy ndarray.\n\n"
	"Returns:\n"
   "  	A list with function identifiers (fid) .\n\n"
   "		This fid's can be used to connect factors to this functions\n\n"
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
