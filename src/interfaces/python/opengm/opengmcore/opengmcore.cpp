#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleCoreOPENGM



#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
#include <stddef.h>
#include <deque>
#include <exception>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/utilities/tribool.hxx>
#include <opengm/inference/inference.hxx>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include "pyConfig.hxx"
#include "pyFactor.hxx"
#include "pyMovemaker.hxx"
#include "pyGmManipulator.hxx"
#include "pyIfactor.hxx" 
#include "pyGm.hxx"     
#include "pyFid.hxx"
#include "pyEnum.hxx"   
#include "pyFunctionTypes.hxx"
#include "pyFunctionGen.hxx"
#include "pySpace.hxx"
#include "pyVector.hxx"



//using namespace opengm::python;

void translateOpenGmRuntimeError(opengm::RuntimeError const& e){
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

void translateStdRuntimeError(std::runtime_error const& e){
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

using namespace boost::python;

template<class INDEX>
std::vector< std::vector < INDEX > > *
secondOrderGridVis(
   const size_t dx,
   const size_t dy,
   bool order
){
   typedef  std::vector<INDEX> InnerVec ;
   typedef  std::vector<InnerVec> VeVec;
   // calculate the number of factors...
   const size_t hFactors=(dx-1)*dy;
   const size_t vFactors=(dy-1)*dx;
   const size_t numFac=hFactors+vFactors;
   //
   VeVec *  vecVec=new VeVec(numFac,InnerVec(2));
   size_t fi=0;
   if(order){
      for(size_t x=0;x<dx;++x)
      for(size_t y=0;y<dy;++y){
         if(x+1<dx){
            (*vecVec)[fi][0]=(y+x*dy);
            (*vecVec)[fi][1]=(y+(x+1)*dy);
            ++fi;
         }
         if(y+1<dy){
            boost::python::list vis;
            (*vecVec)[fi][0]=(y+x*dy);
            (*vecVec)[fi][1]=((y+1)+x*dy);
            ++fi;
         }
      }
   }
   else{
      for(size_t x=0;x<dx;++x)
      for(size_t y=0;y<dy;++y){
         if(y+1<dy){
            (*vecVec)[fi][0]=(x+y*dx);
            (*vecVec)[fi][1]=(x+(y+1)*dx);
            ++fi;
         }
         if(x+1<dx){
            boost::python::list vis;
            (*vecVec)[fi][0]=(x+y*dx);
            (*vecVec)[fi][1]=((x+1)+y*dx);
            ++fi;
         }
      }
   }
   return vecVec;
}


// numpy extensions


template<class V>
boost::python::tuple findFirst(
   opengm::python::NumpyView<V,1> toFind,
   opengm::python::NumpyView<V,1> container
){
   typedef opengm::UInt64Type ResultTypePosition;
   // position
   boost::python::object position       = opengm::python::get1dArray<ResultTypePosition>(toFind.size());
   ResultTypePosition * castPtrPosition = opengm::python::getCastedPtr<ResultTypePosition>(position);
   // found
   boost::python::object found = opengm::python::get1dArray<bool>(toFind.size());
   bool * castPtrFound         = opengm::python::getCastedPtr<bool>(found);

   // fill map with positions of values to find 
   typedef std::map<V,size_t> MapType;
   typedef typename MapType::const_iterator MapIter;
   std::map<V,size_t> toFindPosition;
   for(size_t i=0;i<toFind.size();++i){
      toFindPosition.insert(std::pair<V,size_t>(toFind(i),i));
      castPtrFound[i]=false;
   }


   // find values
   size_t numFound=0;
   for(size_t i=0;i<container.size();++i){
      const V value = container(i);
      MapIter findVal=toFindPosition.find(value);

      if( findVal!=toFindPosition.end()){


         const size_t posInToFind = findVal->second;
         if(castPtrFound[posInToFind]==false){
            castPtrPosition[posInToFind]=static_cast<ResultTypePosition>(i);
            castPtrFound[posInToFind]=true;
            numFound+=1;
         }
         if(numFound==toFind.size()){
            break;
         }
      }
   }
   // return the positions and where if they have been found
   return boost::python::make_tuple(position,found);
}


template<class D>
typename D::value_type  dequeFront(const D & deque){return deque.front();}

template<class D>
typename D::value_type  dequeBack(const D & deque){return deque.back();}


template<class D>
typename D::value_type  dequePushBack(  
   D & deque,
   opengm::python::NumpyView<typename D::value_type,1> values
){
   for(size_t i=0;i<values.size();++i)
      deque.push_back(values(i));
}



BOOST_PYTHON_MODULE_INIT(_opengmcore) {
   Py_Initialize();
   PyEval_InitThreads();
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   boost::python::docstring_options docstringOptions(true,true,false);

   // specify that this module is actually a package
   object package = scope();
   package.attr("__path__") = "opengm";
   
   import_array();

   register_exception_translator<opengm::RuntimeError>(&translateOpenGmRuntimeError);
   register_exception_translator<std::runtime_error>(&translateStdRuntimeError);

   // converters 1d
   opengm::python::initializeNumpyViewConverters<bool,1>(); 
   opengm::python::initializeNumpyViewConverters<float,1>(); 
   opengm::python::initializeNumpyViewConverters<double,1>(); 
   opengm::python::initializeNumpyViewConverters<opengm::UInt32Type,1>();
   opengm::python::initializeNumpyViewConverters<opengm::UInt64Type,1>();
   opengm::python::initializeNumpyViewConverters<opengm::Int32Type,1>();
   opengm::python::initializeNumpyViewConverters<opengm::Int64Type,1>();
   // converters 2d
   opengm::python::initializeNumpyViewConverters<bool,2>(); 
   opengm::python::initializeNumpyViewConverters<float,2>(); 
   opengm::python::initializeNumpyViewConverters<double,2>(); 
   opengm::python::initializeNumpyViewConverters<opengm::UInt32Type,2>();
   opengm::python::initializeNumpyViewConverters<opengm::UInt64Type,2>();
   opengm::python::initializeNumpyViewConverters<opengm::Int32Type,2>();
   opengm::python::initializeNumpyViewConverters<opengm::Int64Type,2>();
   // converters 3d
   opengm::python::initializeNumpyViewConverters<bool,3>(); 
   opengm::python::initializeNumpyViewConverters<float,3>(); 
   opengm::python::initializeNumpyViewConverters<double,3>(); 
   opengm::python::initializeNumpyViewConverters<opengm::UInt32Type,3>();
   opengm::python::initializeNumpyViewConverters<opengm::UInt64Type,3>();
   opengm::python::initializeNumpyViewConverters<opengm::Int32Type,3>();
   opengm::python::initializeNumpyViewConverters<opengm::Int64Type,3>();
   // converters nd
   opengm::python::initializeNumpyViewConverters<bool,0>(); 
   opengm::python::initializeNumpyViewConverters<float,0>(); 
   opengm::python::initializeNumpyViewConverters<double,0>(); 
   opengm::python::initializeNumpyViewConverters<opengm::UInt32Type,0>();
   opengm::python::initializeNumpyViewConverters<opengm::UInt64Type,0>();
   opengm::python::initializeNumpyViewConverters<opengm::Int32Type,0>();
   opengm::python::initializeNumpyViewConverters<opengm::Int64Type,0>();
   

   
   std::string adderString="adder";
   std::string multiplierString="multiplier";
   
   scope current;
   std::string currentScopeName(extract<const char*>(current.attr("__name__")));
   
   currentScopeName="opengm";
   
   class_< opengm::meta::EmptyType > ("_EmptyType",init<>())
   ;

   def("secondOrderGridVis", &secondOrderGridVis<opengm::UInt64Type>,return_value_policy<manage_new_object>(),(arg("dimX"),arg("dimY"),arg("numpyOrder")=true),
	"Todo.."
	);
   

   // utilities
   {
      std::string substring="utilities";
      std::string submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      // Switch the scope to the submodule
      scope submoduleScope = submodule;

      //boost::python::def("findFirst",& findFirst<opengm::UInt32Type>);
      boost::python::def("findFirst",& findFirst<opengm::UInt64Type>);
      //boost::python::def("findFirst",& findFirst<opengm::Int32Type>);
      //boost::python::def("findFirst",& findFirst<opengm::Int64Type>);


      typedef std::deque<opengm::UInt64Type>  DequeUInt64;
      boost::python::class_<DequeUInt64>("DequeUInt64" ,init<>())
      .def("pop_front",&DequeUInt64::pop_front)
      .def("pop_back",&DequeUInt64::pop_back)
      .def("front",&dequeFront<DequeUInt64>)
      .def("back",&dequeBack<DequeUInt64>)
      .def("push_front",&DequeUInt64::push_front)
      .def("push_back",&DequeUInt64::push_back)
      .def("push_back",&dequePushBack<DequeUInt64>)
      .def("__len__",&DequeUInt64::size)
      .def("empty",&DequeUInt64::empty)
      .def("clear",&DequeUInt64::clear)
      ;
      
   }




   //export_rag();
   export_config();
   export_vectors<opengm::python::GmIndexType>();
   export_space<opengm::python::GmIndexType>();
   export_functiontypes<opengm::python::GmValueType,opengm::python::GmIndexType>();
   export_fid<opengm::python::GmIndexType>();
   export_ifactor<opengm::python::GmValueType,opengm::python::GmIndexType>();
   export_enum();
   export_function_generator<opengm::python::GmAdder,opengm::python::GmMultiplier>();
   //adder
   {
      std::string substring=adderString;
      std::string submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      // Switch the scope to the submodule, add methods and classes.
      scope submoduleScope = submodule;
      export_gm<opengm::python::GmAdder>();
      export_factor<opengm::python::GmAdder>();
      export_movemaker<opengm::python::GmAdder>();
      export_gm_manipulator<opengm::python::GmAdder>();
   }
   //multiplier
   {
      std::string substring=multiplierString;
      std::string submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      // Switch the scope to the submodule, add methods and classes.
      scope submoduleScope = submodule;
      export_gm<opengm::python::GmMultiplier>();
      export_factor<opengm::python::GmMultiplier>();
      export_movemaker<opengm::python::GmMultiplier>();
      export_gm_manipulator<opengm::python::GmMultiplier>();
   }
   
}
