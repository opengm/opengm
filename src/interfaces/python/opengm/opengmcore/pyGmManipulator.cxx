
#include <boost/python.hpp>
#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <algorithm>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_manipulator.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include "opengm/utilities/functors.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/absolute_difference.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/pottsn.hxx"
#include "opengm/functions/pottsg.hxx"
#include "opengm/functions/squared_difference.hxx"
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "opengm/functions/sparsemarray.hxx"
#include "opengm/datastructures/partition.hxx"

#include "../gil.hxx"
#include "pyPythonFunction.hxx"




using namespace boost::python;

namespace pymanipulator {

   template<class GM>
   GM * getModifiedModel( opengm::GraphicalModelManipulator<GM> & gmManipulator){
      typedef opengm::GraphicalModelManipulator<GM> PyGmManipulator;

      typedef typename PyGmManipulator::MGM ModGmType;
      typedef typename GM::ValueType ValueType;
      typedef typename GM::IndexType IndexType;
      typedef typename GM::LabelType LabelType;
      typedef typename GM::SpaceType SpaceType;
      typedef typename GM::FunctionIdentifier Fid;
      typedef opengm::ExplicitFunction <ValueType,IndexType,LabelType> ExplicitFunction;
      // get mod model
      const ModGmType & modGm = gmManipulator.getModifiedModel();
      // space 
      std::vector<LabelType> nos(modGm.numberOfVariables());
      for(IndexType vi=0;vi<modGm.numberOfVariables();++vi){
         nos[vi]=modGm.numberOfLabels(vi);
      }
      GM * gm  = new GM(SpaceType(nos.begin(),nos.end()));


      // reservations
      IndexType facVisCount = 0 ;
      opengm::UInt64Type maxSize    = 0 ;
      for(IndexType fi=0;fi<modGm.numberOfFactors();++fi){
         facVisCount+=modGm[fi].numberOfVariables();
         maxSize = std::max(maxSize,opengm::UInt64Type(modGm[fi].size()));
      }
      gm->reserveFactorsVarialbeIndices(facVisCount);
      gm->reserveFactors(modGm.numberOfFactors());
      (*gm). template  reserveFunctions<ExplicitFunction>(modGm.numberOfFactors());


      // add functions and factors  
      ValueType * facVal = new ValueType[maxSize];
      for(IndexType fi=0;fi<modGm.numberOfFactors();++fi){
         modGm[fi].copyValues(facVal);
         ExplicitFunction fEmpty;
         Fid fid = gm->addFunction(fEmpty);
         ExplicitFunction & f = (*gm). template getFunction<ExplicitFunction>(fid);
         f.resize(modGm[fi].shapeBegin(),modGm[fi].shapeEnd());
         std::copy(facVal,facVal+modGm[fi].size(),f.begin());
         gm->addFactorNonFinalized(fid,modGm[fi].variableIndicesBegin(),modGm[fi].variableIndicesEnd()); 
      }
      gm->finalize();

      delete[] facVal;

      return gm;
   }


   template<class GM>
   boost::python::object getModifiedModelVariableIndices( opengm::GraphicalModelManipulator<GM> & gmManipulator){
      typedef opengm::GraphicalModelManipulator<GM> PyGmManipulator;

      typedef typename PyGmManipulator::MGM ModGmType;
      typedef typename GM::ValueType ValueType;
      typedef typename GM::IndexType IndexType;
      typedef typename GM::LabelType LabelType;
      typedef typename GM::SpaceType SpaceType;
      typedef typename GM::FunctionIdentifier Fid;
      typedef opengm::ExplicitFunction <ValueType,IndexType,LabelType> ExplicitFunction;

      const GM &        gm    = gmManipulator.getOriginalModel();
      const ModGmType & modGm = gmManipulator.getModifiedModel();


      boost::python::object obj = opengm::python::get1dArray<IndexType>(modGm.numberOfVariables());
      IndexType * castedPtr = opengm::python::getCastedPtr<IndexType>(obj);

      IndexType subVi=0;
      for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
         if(!gmManipulator.isFixed(vi)){
            castedPtr[subVi]=vi;
            ++subVi;
         }
      }
      return obj;
   }


   template<class GM>
   void fixVariables( 
      opengm::GraphicalModelManipulator<GM> & gmManipulator,
      opengm::python::NumpyView<typename GM::IndexType,1> vis,
      opengm::python::NumpyView<typename GM::LabelType,1> labels
   ){
      if(gmManipulator.isLocked())
         gmManipulator.unlock();
      OPENGM_CHECK_OP(gmManipulator.isLocked(),==,false,"must be onlocked");

      typedef typename GM::ValueType ValueType;
      typedef typename GM::IndexType IndexType;
      typedef typename GM::LabelType LabelType;

      OPENGM_CHECK_OP(vis.shape(0),==,labels.shape(0),"GraphicalModelManipulator.fixVariables error");

      for(IndexType v=0;v<vis.shape(0);++v){
         gmManipulator.fixVariable(vis(v),labels(v));
      }
      gmManipulator.lock();
   }

}

template<class GM>
void export_gm_manipulator() {
  
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   import_array();

   

   typedef GM PyGm;
   typedef typename PyGm::SpaceType PySpace;
   typedef typename PyGm::ValueType ValueType;
   typedef typename PyGm::IndexType IndexType;
   typedef typename PyGm::LabelType LabelType;
  
   typedef opengm::ExplicitFunction                      <ValueType,IndexType,LabelType> PyExplicitFunction;
   typedef opengm::PottsFunction                         <ValueType,IndexType,LabelType> PyPottsFunction;
   typedef opengm::PottsNFunction                        <ValueType,IndexType,LabelType> PyPottsNFunction;
   typedef opengm::PottsGFunction                        <ValueType,IndexType,LabelType> PyPottsGFunction;
   typedef opengm::AbsoluteDifferenceFunction            <ValueType,IndexType,LabelType> PyAbsoluteDifferenceFunction;
   typedef opengm::TruncatedAbsoluteDifferenceFunction   <ValueType,IndexType,LabelType> PyTruncatedAbsoluteDifferenceFunction;
   typedef opengm::SquaredDifferenceFunction             <ValueType,IndexType,LabelType> PySquaredDifferenceFunction;
   typedef opengm::TruncatedSquaredDifferenceFunction    <ValueType,IndexType,LabelType> PyTruncatedSquaredDifferenceFunction;
   typedef opengm::SparseFunction                        <ValueType,IndexType,LabelType> PySparseFunction; 
   typedef opengm::python::PythonFunction                <ValueType,IndexType,LabelType> PyPythonFunction; 

   typedef typename PyGm::FunctionIdentifier PyFid;
   typedef typename PyGm::FactorType PyFactor;
   typedef typename PyFid::FunctionIndexType FunctionIndexType;
   typedef typename PyFid::FunctionTypeIndexType FunctionTypeIndexType;


   //
   typedef opengm::GraphicalModelManipulator<PyGm> PyGmManipulator;


   
   class_<PyGmManipulator > ("GraphicalModelManipulator", 
   "Fix a subset of variables to a given state.",
   init<const  PyGm &>()[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const PyGm& */>()] )

   .def("buildModifiedModel",&PyGmManipulator::buildModifiedModel,"build the sub-model w.r.t. the fixedVariables")
   //.def("buildModifiedSubModels",&PyGmManipulator::buildModifiedSubModels,"build the sub-models w.r.t. the fixedVariables")
   .def("getModifiedModel", &pymanipulator::getModifiedModel<PyGm>,return_value_policy<manage_new_object>(),"get the modified gm" )
   .def("getModifiedModelVariableIndices", &pymanipulator::getModifiedModelVariableIndices<PyGm>,
      "get the variable indices of the modified model w.r.t. the unmodified model" )

   .def("fixVariables",&pymanipulator::fixVariables<PyGm>,( arg("variableIndices"),arg("labels") ),
   "fix a variable to a given label\n\n"
   "Args:\n\n"
    "  variableIndices: variables to fix\n\n"
   "  labels: labels of the variables to fix"
   )
   //.def("fixVariable",&PyGmManipulator::fixVariable,( arg("variableIndex"),arg("label") ),
   //"fix a variable to a given label\n\n"
   //"Args:\n\n"
   //"  variableIndex: variable to fix\n\n"
   //"  label: fix variable to this label"
   //)

   //.def("freeVariable",&PyGmManipulator::freeVariable,( arg("variableIndex") ),
   //"remove fixed label for a given variable\n\n"
   //"Args:\n\n"
   //"  variableIndex: remove fixed label for this variable"
   //)
   //.def("freeAllVariables",&PyGmManipulator::freeAllVariables,"remove fixed label for all variables")
   //.def("lock",&PyGmManipulator::lock,"lock the model")
   //.def("unlock",&PyGmManipulator::unlock,"unlock the model")

   //.def("isLocked", &PyGmManipulator::isLocked)

   #if 0
   

   
   .def("isFixed",&PyGmManipulator::isFixed,( arg("variableIndex") ),
   "check if a variable is fixed\n\n"
   "Args:\n\n"
   "  variableIndex: variable to check\n\n"
   )
   .def("numberOfSubmodels", &PyGmManipulator::numberOfSubmodels)








   //.def("__str__",&pyspace::asString<PySpace>)
   
   

   //.add_property("numberOfVariables", &PySpace::numberOfVariables,
   
   #endif
   ;
}


template void export_gm_manipulator<opengm::python::GmAdder>();
template void export_gm_manipulator<opengm::python::GmMultiplier>();
