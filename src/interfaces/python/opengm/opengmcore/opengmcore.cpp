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


#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/potts.hxx"


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
            (*vecVec)[fi][0]=(x+y*dx);
            (*vecVec)[fi][1]=((x+1)+y*dx);
            ++fi;
         }
      }
   }
   return vecVec;
}

template<class INDEX>
std::vector< std::vector < INDEX > > *
secondOrderGridVis3D(
   const size_t dx,
   const size_t dy,
   const size_t dz,
   bool order
){
   typedef  std::vector<INDEX> InnerVec ;
   typedef  std::vector<InnerVec> VeVec;
   
   VeVec* vecVec = new VeVec;
   if(order){
      for(size_t x=0;x<dx;++x)
      for(size_t y=0;y<dy;++y)
      for(size_t z=0;z<dz;++z){
	if(x+1<dx){
	  InnerVec vis(2);
	  vis[0] = (z+y*dz+x*dz*dy);
	  vis[1] = (z+y*dz+(x+1)*dz*dy);
	  vecVec->push_back(vis);
         }
	if(y+1<dy){
	  InnerVec vis(2);
	  vis[0] =(z+y*dz+x*dz*dy);
	  vis[1] =(z+(y+1)*dz+x*dz*dy);
	  vecVec->push_back(vis);
         }
	 if(z+1<dz){
	   InnerVec vis(2);
	   vis[0] = (z+y*dz+x*dz*dy);
	   vis[1] = ((z+1)+y*dz+x*dz*dy);
	   vecVec->push_back(vis);
         }
      }
   }
   else{
     for(size_t x=0;x<dx;++x)
     for(size_t y=0;y<dy;++y)
     for(size_t z=0;z<dz;++z){
       if(z+1<dx){
	 InnerVec vis(2);
	 vis[0] = (x+y*dx+z*dx*dy);
	 vis[1] = (x+y*dx+(z+1)*dx*dy);
	 vecVec->push_back(vis);
       }
       if(y+1<dy){
	 InnerVec vis(2);
	 vis[0] =(x+y*dx+z*dx*dy);
	 vis[1] =(x+(y+1)*dx+z*dx*dy);
	 vecVec->push_back(vis);
       }
       if(x+1<dz){
	 InnerVec vis(2);
	 vis[0] = (x+y*dx+z*dx*dy);
	 vis[1] = ((x+1)+y*dx+z*dx*dy);
	 vecVec->push_back(vis);
       }
     }
   }
   return vecVec;
}

struct CoordToVi{
    template<class ITER>
    CoordToVi(ITER shapeBegin, ITER shapeEnd, const bool numpyOrder)
    : shape_(shapeBegin, shapeEnd),
      strides_()
    {
        strides_.resize(shape_.size());
        size_t s=1;
        size_t d=shape_.size();
        if(numpyOrder){
            for(size_t ii=0; ii<d; ++ii){
                size_t i = d-1-ii;
                strides_[i] = s;
                s*=shape_[i];
            }
        }
        else{
            for(size_t i=0; i<d; ++i){
                strides_[i] = s;
                s*=shape_[i];
            }
        }
    }

    size_t operator()(const size_t x){
        return strides_[0]*x;
    }
    size_t operator()(const size_t x, const size_t y){
        return strides_[0]*x + strides_[1]*y;
    }
    size_t operator()(const size_t x, const size_t y, const size_t z){
        return strides_[0]*x + strides_[1]*y + strides_[2]*z;
    }
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

};



template< class GM >
GM *  pyPottsModel3d(
    opengm::python::NumpyView< typename GM::ValueType, 4> costVolume,
    opengm::python::NumpyView< typename GM::ValueType, 3> lambdaVolume,
    const bool numpyOrder 
){
    const size_t numLabels = costVolume.shape(3);
    const size_t numVar = lambdaVolume.size();
    typedef typename GM::SpaceType SpaceType;



    SpaceType space;
    space.reserve(numVar);
    for(size_t vi=0; vi<numVar; ++vi){
        space.addVariable(numLabels);
    }

    GM * gm = new GM(space);

    opengm::ExplicitFunction<
        typename GM::ValueType, typename GM::IndexType, typename GM::LabelType
    > ef(&numLabels, &numLabels+1);




    const size_t dz = costVolume.shape(2);
    const size_t dy = costVolume.shape(1);
    const size_t dx = costVolume.shape(0);


    CoordToVi toVi(lambdaVolume.shapeBegin(),lambdaVolume.shapeEnd() , numpyOrder);


    for(size_t z=0; z<dz; ++z)
    for(size_t y=0; y<dy; ++y)
    for(size_t x=0; x<dx; ++x){

        const size_t vi  = toVi(x, y, z);

        for(size_t l=0; l<numLabels; ++l){
            ef(&l)=costVolume(x, y, z, l);
        }
        gm->addFactor(gm->addFunction(ef), &vi, &vi+1);
    }

    size_t vis[2]={0,0};
    for(size_t z=0; z<dz; ++z)
    for(size_t y=0; y<dy; ++y)
    for(size_t x=0; x<dx; ++x){
        vis[0] = toVi(x, y, z);
        if(x+1<dx){
            vis[1] = toVi(x+1, y, z);
            const float l = (lambdaVolume(x,y,z) + lambdaVolume(x+1,y,z) ) /2.0;
            opengm::python::GmPottsFunction pf(numLabels, numLabels, 0.0, l);
            gm->addFactor(gm->addFunction(pf),vis,vis+2);
        }
        if(y+1<dy){
            vis[1] = toVi(x, y+1, z);
            const float l = (lambdaVolume(x,y,z) + lambdaVolume(x,y+1,z) ) /2.0;
            opengm::python::GmPottsFunction pf(numLabels, numLabels, 0.0, l);
            gm->addFactor(gm->addFunction(pf),vis,vis+2);
        }
        if(z+1<dz){
            vis[1] = toVi(x, y, z+1);
            const float l = (lambdaVolume(x,y,z) + lambdaVolume(x,y,z+1) ) /2.0;
            opengm::python::GmPottsFunction pf(numLabels, numLabels, 0.0, l);
            gm->addFactor(gm->addFunction(pf),vis,vis+2);
        }

    }
    return gm;

}

void  makeMaskedState(
    opengm::python::NumpyView< opengm::UInt32Type, 3> mask,
    opengm::python::NumpyView< opengm::UInt64Type, 1> arg,
    opengm::python::NumpyView< opengm::UInt32Type, 3> imgArg,
    opengm::UInt32Type noLabelIdx)  {
    const size_t dz = mask.shape(2);
    const size_t dy = mask.shape(1);
    const size_t dx = mask.shape(0);

    size_t numVar= 0;
    for(size_t z=0; z<dz; ++z)
    for(size_t y=0; y<dy; ++y)
    for(size_t x=0; x<dx; ++x){
      if ( mask(x,y,z) == 1) {
	imgArg(x,y,z) = arg(numVar);
	numVar++;
      }
      else {
	imgArg(x,y,z) = noLabelIdx;
      }
	
    }
}


void getStartingPointMasked(
    opengm::python::NumpyView< opengm::UInt32Type, 3> mask,
    opengm::python::NumpyView< opengm::UInt32Type, 3> imgArg,
    opengm::python::NumpyView< opengm::UInt32Type, 1> startingPoint) {
    const size_t dz = mask.shape(2);
    const size_t dy = mask.shape(1);
    const size_t dx = mask.shape(0);

    size_t numVar= 0;
    for(size_t z=0; z<dz; ++z)
    for(size_t y=0; y<dy; ++y)
    for(size_t x=0; x<dx; ++x){
      if ( mask(x,y,z) == 1) {
	startingPoint(numVar) = imgArg(x,y,z);
	numVar++;
      }
    }
}


template< class GM >
GM *  pyPottsModel3dMasked(
    opengm::python::NumpyView< typename GM::ValueType, 4> costVolume,
    opengm::python::NumpyView< typename GM::ValueType, 3> lambdaVolume,
    opengm::python::NumpyView< opengm::UInt32Type, 3> mask,
    opengm::python::NumpyView< opengm::UInt32Type, 1> idx2vi
		     ){
    const size_t numLabels = costVolume.shape(3);
    typedef typename GM::SpaceType SpaceType;


    const size_t dz = costVolume.shape(2);
    const size_t dy = costVolume.shape(1);
    const size_t dx = costVolume.shape(0);


    CoordToVi toVi(lambdaVolume.shapeBegin(),lambdaVolume.shapeEnd() , false);

    size_t cc= 0; 
    size_t numVar= 0;
    for(size_t z=0; z<dz; ++z)
    for(size_t y=0; y<dy; ++y)
    for(size_t x=0; x<dx; ++x){
      if ( mask(x,y,z) == 1) {
	idx2vi(cc) = numVar;
	numVar++;
      }
      cc++;
   
    
    }

    SpaceType space;
    space.reserve(numVar);
    for(size_t vi=0; vi<numVar; ++vi){
        space.addVariable(numLabels);
    }

    GM * gm = new GM(space);

    opengm::ExplicitFunction<
        typename GM::ValueType, typename GM::IndexType, typename GM::LabelType
    > ef(&numLabels, &numLabels+1);


    cc = 0;
    for(size_t z=0; z<dz; ++z)
    for(size_t y=0; y<dy; ++y)
    for(size_t x=0; x<dx; ++x){
      const size_t vi  = idx2vi(cc);
      cc++;
      if (mask(x,y,z) == 1) {
        for(size_t l=0; l<numLabels; ++l){
            ef(&l)=costVolume(x, y, z, l);
        }
        gm->addFactor(gm->addFunction(ef), &vi, &vi+1);
      }
    }

    size_t vis[2]={0,0};
    cc = 0;
    for(size_t z=0; z<dz; ++z)
    for(size_t y=0; y<dy; ++y)
    for(size_t x=0; x<dx; ++x){
      vis[0] = idx2vi(cc);
      cc++;
      if (mask(x,y,z) == 1) {
	if(x+1<dx && mask(x+1,y,z) == 1 ){
	  vis[1] = idx2vi(toVi(x+1, y, z));
	  const float l = (lambdaVolume(x,y,z) + lambdaVolume(x+1,y,z) ) /2.0;
	  opengm::python::GmPottsFunction pf(numLabels, numLabels, 0.0, l);
	  gm->addFactor(gm->addFunction(pf),vis,vis+2);
        }
        if(y+1<dy && mask(x,y+1,z) == 1 ){
	  vis[1] = idx2vi(toVi(x, y+1, z));
	  const float l = (lambdaVolume(x,y,z) + lambdaVolume(x,y+1,z) ) /2.0;
	  opengm::python::GmPottsFunction pf(numLabels, numLabels, 0.0, l);
	  gm->addFactor(gm->addFunction(pf),vis,vis+2);
        }
        if(z+1<dz && mask(x,y,z+1) == 1 ){
          vis[1] = idx2vi(toVi(x, y, z+1));
	  const float l = (lambdaVolume(x,y,z) + lambdaVolume(x,y,z+1) ) /2.0;
	  opengm::python::GmPottsFunction pf(numLabels, numLabels, 0.0, l);
	  gm->addFactor(gm->addFunction(pf),vis,vis+2);
      }
     }
    }
    return gm;

}



void export_makeMaskedState() {
    boost::python::def("_makeMaskedState",
        & makeMaskedState,
        (
            boost::python::args("mask"),
            boost::python::args("arg"),
            boost::python::args("imgArg"),
	    boost::python::args("labelIdx")
        )
    );
}


void export_getStartingPointMasked() {
    boost::python::def("_getStartingPointMasked",
        & getStartingPointMasked,
        (
            boost::python::args("mask"),
            boost::python::args("imgArg"),
	    boost::python::args("startingPoint")
        )
    );
}


template<class GM>
void export_potts_model_3d(){
    boost::python::def("_pottsModel3d",
        & pyPottsModel3d<GM>,
        (
            boost::python::args("costVolume"),
            boost::python::args("lambdaVolume"),
            boost::python::args("numpyOrder")=true
        ),
        boost::python::return_value_policy<boost::python::manage_new_object>()
    );
}

template<class GM>
void export_potts_model_3d_masked(){
    boost::python::def("_pottsModel3dMasked",
		       & pyPottsModel3dMasked<GM>,
        (
            boost::python::args("costVolume"),
            boost::python::args("lambdaVolume"),
            boost::python::args("maskVolume"),
            boost::python::args("idx2vi")
        ),
        boost::python::return_value_policy<boost::python::manage_new_object>()
    );
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

   // converters 4d
   opengm::python::initializeNumpyViewConverters<bool,4>(); 
   opengm::python::initializeNumpyViewConverters<float,4>(); 
   opengm::python::initializeNumpyViewConverters<double,4>(); 
   opengm::python::initializeNumpyViewConverters<opengm::UInt32Type,4>();
   opengm::python::initializeNumpyViewConverters<opengm::UInt64Type,4>();
   opengm::python::initializeNumpyViewConverters<opengm::Int32Type,4>();
   opengm::python::initializeNumpyViewConverters<opengm::Int64Type,4>();

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
   
   class_< opengm::meta::EmptyType > ("_EmptyType",init<>());

   def("secondOrderGridVis", &secondOrderGridVis<opengm::UInt64Type>,return_value_policy<manage_new_object>(),(arg("dimX"),arg("dimY"),arg("numpyOrder")=true),
	"Todo.."
	);

   def("secondOrderGridVis3D", 
       &secondOrderGridVis3D<opengm::UInt64Type>,
       return_value_policy<manage_new_object>(),(arg("dimX"),
						 arg("dimY"),
						 arg("dimZ"),
						 arg("numpyOrder")=true),
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
   export_makeMaskedState();
   export_getStartingPointMasked();
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

      export_potts_model_3d<opengm::python::GmAdder>();
      export_potts_model_3d_masked<opengm::python::GmAdder>();

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

      export_potts_model_3d<opengm::python::GmMultiplier>();
      export_potts_model_3d_masked<opengm::python::GmMultiplier>();
   }
   
}
