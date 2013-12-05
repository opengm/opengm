#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleCore

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/stl_iterator.hpp>
#include <numpy/noprefix.h>
#ifdef Bool
#undef Bool
#endif 

#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <ostream>
#include <stddef.h>



#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
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

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


#include "nifty_iterator.hxx"
#include "../gil.hxx"
#include "../copyhelper.hxx"
#include <algorithm>
#include "utilities/shapeHolder.hxx"

#include "functionGenBase.hxx"



using namespace boost::python;

template<class Iter, class T>
Iter my_binary_find(Iter begin, Iter end, T val)
{
    // Finds the lower bound in at most log(last - first) + 1 comparisons
    Iter i = std::lower_bound(begin, end, val);

    if (i != end && *i == val)
        return i; // found
    else
        return end; // not found
}





namespace pygm {

   
      template<class GM>
      std::vector<typename  GM::FunctionIdentifier>  * 
      addFunctionsFromGenerator(GM & gm,
         FunctionGeneratorBase<opengm::python::GmAdder,opengm::python::GmMultiplier> * generatorPtr
      ){
         std::vector<typename  GM::FunctionIdentifier>  * vec=NULL;
         {
            releaseGIL rgil;
            vec=generatorPtr->addFunctions(gm);
         }
         return vec;
      }
      

      //constructor from numpy array
      template<class GM,class VALUE_TYPE>
      inline GM *  gmConstructorPythonNumpy( opengm::python::NumpyView<VALUE_TYPE,1>  numberOfLabels,const size_t resNumVarsFac) {        
         return new GM(typename GM::SpaceType(numberOfLabels.begin(), numberOfLabels.end()),resNumVarsFac);
      }
      template<class GM,class VALUE_TYPE>
      inline GM *  gmConstructorPythonAny(const boost::python::object & obj,const size_t resNumVarsFac) {
            stl_input_iterator<VALUE_TYPE> begin(obj), end;
            return new GM(typename GM::SpaceType(begin, end),resNumVarsFac);
      }
      template<class GM>
      inline GM *  gmConstructorVector(const std::vector<typename GM::LabelType> & numLabels,const size_t resNumVarsFac) {
            return new GM(typename GM::SpaceType(numLabels.begin(), numLabels.end()),resNumVarsFac);
      }
      template<class GM>
      inline GM *  gmConstructorSimple(const typename GM::IndexType numVar,const typename GM::LabelType numLabels,const size_t resNumVarsFac ) {
            typename GM::SpaceType space;
            space.reserve(numVar);
            for(typename GM::IndexType i=0;i<numVar;++i)
               space.addVariable(numLabels);
            return new GM(space,resNumVarsFac);
      }

      template<class GM>
      inline void assign_Vector( GM & gm ,const std::vector<typename GM::LabelType> & numberOfLabels) {
         gm.assign(typename GM::SpaceType(numberOfLabels.begin(), numberOfLabels.end()));
      }

      template<class GM,class VALUE_TYPE>
      inline void assign_Any( GM & gm ,const boost::python::object & obj) {
         stl_input_iterator<VALUE_TYPE> begin(obj), end;
         gm.assign(typename GM::SpaceType(begin, end));
      }
      
      template<class GM,class VALUE_TYPE>
      inline void assign_Numpy( GM & gm ,opengm::python::NumpyView<VALUE_TYPE,1>  numberOfLabels) {
         gm.assign( typename GM::SpaceType(numberOfLabels.begin(), numberOfLabels.end()));
      }


      
      template<class GM>
      inline typename GM::IndexType addFactor_Numpy
      (
         GM & gm,const typename GM::FunctionIdentifier & fid, opengm::python::NumpyView<typename  GM::IndexType,1>   vis, const bool finalize
      ) {
         if(finalize)
            return gm.addFactor(fid, vis.begin(), vis.end());
         else
            return gm.addFactorNonFinalized(fid, vis.begin(), vis.end());
      }
      
      template<class GM>
      inline typename GM::IndexType addFactor_Vector
      (
         GM & gm,const typename GM::FunctionIdentifier & fid, const std::vector<typename GM::IndexType> & vis, const bool finalize
      ) {
         if(finalize)
            return gm.addFactor(fid, vis.begin(), vis.end());
         else
            return gm.addFactorNonFinalized(fid, vis.begin(), vis.end());
      }

      template<class GM,class VALUE_TYPE>
      inline typename GM::IndexType addFactor_Any
      (
         GM & gm,const typename GM::FunctionIdentifier & fid, const boost::python::object &  vis, const bool finalize
      ) {
         stl_input_iterator<VALUE_TYPE> begin(vis), end;
         if(finalize)
            return gm.addFactor(fid, begin, end);
         else
            return gm.addFactorNonFinalized(fid, begin,end);
      }

      
      template<class GM>
      typename GM::IndexType addFactors_Vector_VectorVector
      (
         GM & gm,const std::vector<typename GM::FunctionIdentifier> & fids, std::vector< std::vector< typename GM::IndexType > > vis,const bool finalize
      ){
         typedef typename GM::FunctionIdentifier FidType;
         typedef typename GM::IndexType IndexType;
         size_t numFid=fids.size();
         size_t numVis=vis.size();
         IndexType retFactorIndex=0;
         if(numFid!=numVis && numFid!=1)
            throw opengm::RuntimeError("len(fids) must be 1 or len(vis)");
         {
            releaseGIL rgil;

            FidType fid;
            if(numFid==1)
               fid=fids[0];
            for(size_t i=0;i<numVis;++i){
               // extract fid
               if(numFid!=1)
                  fid=fids[i];

               if(finalize)
                  retFactorIndex=gm.addFactor(fid,vis[i].begin(),vis[i].end());
               else
                  retFactorIndex=gm.addFactorNonFinalized(fid,vis[i].begin(),vis[i].end());
            }
         }
         return retFactorIndex;
      }
      

      template<class GM>
      typename GM::IndexType addUnaryFactors_Vector_Numpy
      (
         GM & gm,const std::vector<typename GM::FunctionIdentifier> & fids, opengm::python::NumpyView<typename GM::IndexType,1> vis,const bool finalize
      ){
         typedef typename GM::FunctionIdentifier FidType;
         typedef typename GM::IndexType IndexType;
         size_t numFid=fids.size();
         size_t numVis=vis.shape(0);
         IndexType retFactorIndex=0;
         if(numFid!=numVis && numFid!=1)
            throw opengm::RuntimeError("len(fids) must be 1 or len(vis)");
         {
            releaseGIL rgil;
 
            FidType fid;
            if(numFid==1)
               fid=fids[0];
            for(size_t i=0;i<numVis;++i){
               // extract fid
               if(numFid!=1)
                  fid=fids[i];
               const IndexType vi=vis[i];
               if(finalize)
                  retFactorIndex=gm.addFactor(fid,&vi,&vi+1);
               else
                  retFactorIndex=gm.addFactorNonFinalized(fid,&vi,&vi+1);
            }
         }
         return retFactorIndex;
      }


      template<class GM>
      typename GM::IndexType addFactors_Vector_Numpy
      (
         GM & gm, const std::vector<typename GM::FunctionIdentifier> & fids, opengm::python::NumpyView<typename GM::IndexType,2> vis,const bool finalize
      ){
         //NumpyView<typename GM::IndexType,2> vis=NumpyView<typename GM::IndexType,2>(visn);
         typedef typename GM::FunctionIdentifier FidType;
         typedef typename GM::IndexType IndexType;
         size_t numFid=fids.size();
         size_t numVis=vis.shape(0);
         size_t factorOrder=vis.shape(1);
         if(numFid!=numVis && numFid!=1)
            throw opengm::RuntimeError("len(fids) must be 1 or len(vis)");
         FidType fid;
         IndexType retFactorIndex=0;
         if(numFid==1)
            fid=fids[0];
         
         {
            releaseGIL rgil;
            opengm::FastSequence<IndexType,5> visI(factorOrder);
            for(size_t i=0;i<numVis;++i){
               if(numFid!=1)
                  fid=fids[i];
               for(size_t j=0;j<factorOrder;++j){
                  visI[j]=vis(i,j);
               }
               if(finalize)
                  retFactorIndex=gm.addFactor(fid,visI.begin(),visI.end()); 
               else
                  retFactorIndex=gm.addFactorNonFinalized(fid,visI.begin(),visI.end()); 
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
      
      template<class GM,class FUNCTION>
      inline typename GM::FunctionIdentifier addFunctionGenericPy( 
         GM & gm,
         const FUNCTION & f
      ){
         return gm.addFunction(f);
      }


      template<class GM,class FUNCTION>
      inline std::vector<typename GM::FunctionIdentifier> * addFunctionsGenericVectorPy( 
         GM & gm,
         const std::vector<FUNCTION> & f
      ){
         typedef std::vector<typename GM::FunctionIdentifier> FidVector;
         FidVector * fidVec = NULL;
         {
            releaseGIL rgil;
            fidVec = new FidVector(f.size());
            for(size_t i=0;i<f.size();++i){
               (*fidVec)[i]=gm.addFunction(f[i]);
            }
         }
         return fidVec;
      }


      template<class GM>
      typename GM::FunctionIdentifier addFunctionNpPy( 
         GM & gm,
         opengm::python::NumpyView<typename GM::ValueType> view
         //boost::python::numeric::array a
      ){
         typedef opengm::ExplicitFunction<typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> PyExplicitFunction;        
         PyExplicitFunction fEmpty;
         typename GM::FunctionIdentifier  fid;
         {
            releaseGIL rgil;

            fid=gm.addFunction(fEmpty);
            PyExplicitFunction & f=gm.template getFunction<PyExplicitFunction>(fid);
            f.resize(view.shapeBegin(), view.shapeEnd());
            //std::cout<<"fill\n";
            if(view.dimension()==1){
               size_t ind[1];
               size_t i = 0;
               for (ind[0] = 0; ind[0] < f.shape(0); ++ind[0]) {              
                  f(i) = view(ind[0]);
                  ++i;
               }
            }
            else if(view.dimension()==2){
               size_t ind[2];
               size_t i = 0;
               for (ind[1] = 0; ind[1] < f.shape(1); ++ind[1]){
                  for (ind[0] = 0; ind[0] < f.shape(0); ++ind[0]) {
                     f(i) = view(ind[0],ind[1]);
                     ++i;
                  }
               }
            }  
            else{
               opengm::ShapeWalker<typename PyExplicitFunction::FunctionShapeIteratorType> walker(f.functionShapeBegin(),f.dimension());
               for (size_t i=0;i<f.size();++i) {
                  typename GM::ValueType v=view[walker.coordinateTuple().begin()];
                  f(i) = v;
                  ++walker;
               }
            }
         }
         return fid;
      }
  
      
      template<class GM>
      std::vector<typename GM::FunctionIdentifier> * addFunctionsListNpPy( GM & gm,boost::python::list functionList) {

         typedef typename GM::FunctionIdentifier FidType;
         size_t numF=boost::python::len(functionList);
         std::vector<typename GM::FunctionIdentifier> * fidVec= new std::vector<typename GM::FunctionIdentifier>(numF);

         for(size_t i=0;i<numF;++i){
            boost::python::extract<boost::python::numeric::array> extractor(functionList[i]);
            if(extractor.check()){
              //boost::python::numeric::array functionAsNumpy= static_cast<boost::python::numeric::array >(extractor());
               typedef opengm::python::NumpyView<typename GM::ValueType> NView;
               NView nview= static_cast<NView >(extractor());
               
               (*fidVec)[i]= pygm::addFunctionNpPy<GM>(gm,nview);
            }
            else{
               throw opengm::RuntimeError("wrong data type in list");
            }
         }
         return fidVec;
      }
      
      template<class GM>
      std::vector<typename GM::FunctionIdentifier> * addFunctionsNpPy( GM & gm,opengm::python::NumpyView<typename GM::ValueType> view) {
         typedef typename GM::FunctionIdentifier FidType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename opengm::python::NumpyView<ValueType>::ShapeIteratorType ShapeIteratorType;
         typedef opengm::FastSequence<IndexType,1> FixedSeqType;
         typedef opengm::SubShapeWalker<ShapeIteratorType,FixedSeqType,FixedSeqType> SubWalkerType;
         typedef opengm::ExplicitFunction<typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> ExplicitFunction;   


         std::vector<typename GM::FunctionIdentifier> * fidVec=NULL;
         {
            releaseGIL rgil;
            const size_t dim=view.dimension();
            const size_t fDim=dim-1;
            const size_t numF=view.shape(0);
            fidVec= new std::vector<typename GM::FunctionIdentifier>(numF);
            if(dim<2){
               throw opengm::RuntimeError("functions dimension must be at least 2");
            }


            if(fDim==1){
               for(size_t f=0;f<numF;++f){
                  ExplicitFunction functionEmpty;
                  FidType fid=gm.addFunction(functionEmpty);
                  (*fidVec)[f]=fid;
                  ExplicitFunction & function=gm. template getFunction<ExplicitFunction>(fid);
                  function.resize(view.shapeBegin()+1,view.shapeEnd());
                  for(LabelType l0=0;l0<function.shape(0);++l0){
                     function(l0)=view(f,l0);
                  }
               }
            }
            else if (fDim==2){
               for(size_t f=0;f<numF;++f){
                  ExplicitFunction functionEmpty;
                  FidType fid=gm.addFunction(functionEmpty);
                  (*fidVec)[f]=fid;
                  ExplicitFunction & function=gm. template getFunction<ExplicitFunction>(fid);
                  function.resize(view.shapeBegin()+1,view.shapeEnd());
                  for(LabelType l1=0;l1<function.shape(1);++l1)
                  for(LabelType l0=0;l0<function.shape(0);++l0){
                     function(l0,l1)=view(f,l0,l1);
                  }
               }
            }
            else if (fDim==3){
               for(size_t f=0;f<numF;++f){
                  ExplicitFunction functionEmpty;
                  FidType fid=gm.addFunction(functionEmpty);
                  (*fidVec)[f]=fid;
                  ExplicitFunction & function=gm. template getFunction<ExplicitFunction>(fid);
                  function.resize(view.shapeBegin()+1,view.shapeEnd());
                  for(LabelType l2=0;l2<function.shape(2);++l2)
                  for(LabelType l1=0;l1<function.shape(1);++l1)
                  for(LabelType l0=0;l0<function.shape(0);++l0){
                     function(l0,l1,l2)=view(f,l0,l1,l2);
                  }
               }
            }
            else if (fDim==4){
               for(size_t f=0;f<numF;++f){
                  ExplicitFunction functionEmpty;
                  FidType fid=gm.addFunction(functionEmpty);
                  (*fidVec)[f]=fid;
                  ExplicitFunction & function=gm. template getFunction<ExplicitFunction>(fid);
                  function.resize(view.shapeBegin()+1,view.shapeEnd());
                  for(LabelType l3=0;l3<function.shape(3);++l3)
                  for(LabelType l2=0;l2<function.shape(2);++l2)
                  for(LabelType l1=0;l1<function.shape(1);++l1)
                  for(LabelType l0=0;l0<function.shape(0);++l0){
                     function(l0,l1,l2,l3)=view(f,l0,l1,l2,l3);
                  }
               }
            }
            else if (fDim==5){
               for(size_t f=0;f<numF;++f){
                  ExplicitFunction functionEmpty;
                  FidType fid=gm.addFunction(functionEmpty);
                  (*fidVec)[f]=fid;
                  ExplicitFunction & function=gm. template getFunction<ExplicitFunction>(fid);
                  function.resize(view.shapeBegin()+1,view.shapeEnd());
                  LabelType c[6];
                  c[0]=f;
                  for(c[5]=0;c[5]<function.shape(4);++c[5])
                  for(c[4]=0;c[4]<function.shape(3);++c[4])
                  for(c[3]=0;c[3]<function.shape(2);++c[3])
                  for(c[2]=0;c[2]<function.shape(1);++c[2])
                  for(c[1]=0;c[1]<function.shape(0);++c[1]){
                     function(c+1)=view[c];
                  }
               }
            }
            else{

               // allocate fixed coordinate and fixed coordinate values
               FixedSeqType fixedC(1);
               fixedC[0]=0;
               FixedSeqType fixedV(1);
               FixedSeqType subCoordinate(dim);
               // fid return list
               // loop over 1 dimension/axis of the numpy ndarray view 
               for(size_t f=0;f<numF;++f){
                  // add new function to gm (empty one and fill the ref.)
                  ExplicitFunction functionEmpty;
                  FidType fid=gm.addFunction(functionEmpty);
                  (*fidVec)[f]=fid;
                  ExplicitFunction & function=gm. template getFunction<ExplicitFunction>(fid);
                  function.resize(view.shapeBegin()+1,view.shapeEnd());

                  OPENGM_CHECK_OP(function.dimension(),==,dim-1,"");
                  // append "fid" to fid return list
                  
                  // subarray walker (walk over the subarray,first dimension is fixed to the index "f")
                  fixedV[0]=f;
                  SubWalkerType subwalker(view.shapeBegin(),dim,fixedC,fixedV);
                  const size_t subSize=subwalker.subSize();
                  for(size_t i=0;i<subSize;++i,++subwalker){
                     // fill gm function with values
                     for(size_t j=0;j<dim-1;++j)
                        subCoordinate[j]=subwalker.coordinateTuple()[j+1];
                     function(subCoordinate.begin())=view[subwalker.coordinateTuple().begin()];
                  }
               }
            }
         }
         return fidVec;
      }
      
      template<class GM>
      std::vector<typename GM::FunctionIdentifier> * addUnaryFunctionsNpPy( GM & gm,opengm::python::NumpyView<typename GM::ValueType,2> view) {
         typedef typename GM::FunctionIdentifier FidType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename opengm::python::NumpyView<ValueType>::ShapeIteratorType ShapeIteratorType;
         typedef opengm::FastSequence<IndexType,1> FixedSeqType;
         //typedef typename FixedSeqType::const_iterator FixedSeqIteratorType;
         typedef opengm::SubShapeWalker<ShapeIteratorType,FixedSeqType,FixedSeqType> SubWalkerType;
         typedef opengm::ExplicitFunction<typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> ExplicitFunction;        
         std::vector<typename GM::FunctionIdentifier> * fidVec= NULL;

         const size_t numF=view.shape(0);
         const size_t numLabels=view.shape(1);
         fidVec= new std::vector<typename GM::FunctionIdentifier>(numF);


         releaseGIL rgil;
         for(size_t f=0;f<numF;++f){
            // add new function to gm (empty one and fill the ref.)
            ExplicitFunction functionEmpty;
            FidType fid=gm.addFunction(functionEmpty);
            ExplicitFunction & function=gm. template getFunction<ExplicitFunction>(fid);
            function.resize(view.shapeBegin()+1,view.shapeEnd());
            (*fidVec)[f]=fid;
            for(size_t i=0;i<numLabels;++i){
               // fill gm function with values
               function(i)=view(f,i);
            }
         }
         
         return fidVec;
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
         else if(opengm::meta::Compare<typename GM::OperatorType,opengm::Multiplier>::value)
            return "multiplier";
         else
            throw opengm::RuntimeError("internal error,wrong operator type");
      }
      
      template<class GM>
      typename GM::ValueType evaluatePyNumpy
      (
         const GM & gm,
         opengm::python::NumpyView<typename GM::IndexType,1> states
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

      template<class GM,class LABEL_TYPE>
      typename GM::ValueType evaluatePyVector
      (
         const GM & gm,
         const std::vector<LABEL_TYPE> labels
      ){
         typename GM::ValueType val;
         {
            releaseGIL rgil;
            val = gm.evaluate(labels.begin());
         }
         return val;
      }



      template<class GM>
      boost::python::list factorRange
      (
         const GM & gm,
         int start,
         int stop,
         int increment
      ){
         boost::python::list factorList;
         for(int i=start;i<stop;i+=increment){
            factorList.append(gm[i]);
         }
         return factorList;
      }

      template<class GM>
      FactorsOfVariableHolder<GM> getFactosOfVariableHolder
      (
         const GM & gm,
         const size_t variableIndex
      ) {
         return FactorsOfVariableHolder< GM >(gm,variableIndex);
      }


      template<class GM>
      boost::python::list
      variablesAdjacency(
         const GM & gm
      ){
         boost::python::list variablesAdjacencyList;
         typedef typename GM::IndexType IndexType;
         for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
            std::set<IndexType> visSet;
            for(IndexType f=0;f<gm.numberOfFactors(vi);++f){
               const IndexType fi=gm.factorOfVariable(vi,f);
               const IndexType numVarF=gm[fi].numberOfVariables();
               if(numVarF>=2){
                  for(IndexType v=0;v<numVarF;++v){
                     const IndexType otherVi=gm[fi].variableIndex(v);
                     if(otherVi!=vi){
                        visSet.insert(otherVi);
                     }
                  }
               }
            }
            //Vis set is complete for this variable
            boost::python::list visList;
            for(typename std::set<IndexType>::const_iterator iter=visSet.begin();iter!=visSet.end();++iter){
               visList.append(*iter);
            }
            variablesAdjacencyList.append(visList);
         }
         return variablesAdjacencyList;
      }
      

      
      template<class GM>
      inline void reserveFunctions(GM & gm ,const size_t size,const std::string & fname){
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

         if(fname==std::string("explicit")){
            return gm. template  reserveFunctions<PyExplicitFunction>(size);
         }
         else if(fname==std::string("potts")){
            return gm. template  reserveFunctions<PyPottsFunction>(size);
         }
         else if(fname==std::string("potts-n")){
            return gm. template  reserveFunctions<PyPottsNFunction>(size);
         }
         else if(fname==std::string("potts-g")){
            return gm. template  reserveFunctions<PyPottsGFunction>(size);
         }
         else if(fname==std::string("truncated-absolute-difference")){
            return gm. template  reserveFunctions<PyTruncatedAbsoluteDifferenceFunction>(size);
         }
         else if(fname==std::string("truncated-squared-difference")){
            return gm. template  reserveFunctions<PyTruncatedSquaredDifferenceFunction>(size);
         }
         else if(fname==std::string("sparse")){
            return gm. template  reserveFunctions<PySparseFunction>(size);
         }
         else if(fname==std::string("python")){
            return gm. template  reserveFunctions<PyPythonFunction>(size);
         }
         else{
            throw opengm::RuntimeError(fname + std::string(" is an unknown function type name"));
         }
      }

      template<class GM>
      boost::python::object factorIndicesFromVariableIndices(
         const GM & gm,
         opengm::python::NumpyView<typename GM::IndexType,1> vis
      ){
         //releaseGIL * rgil= new releaseGIL;

         typedef typename GM::IndexType IndexType;
         typedef typename GM::ValueType ValueType;
         typedef std::set<IndexType> SetType; 
         typedef typename SetType::const_iterator SetIter;

         SetType factorSet;
         for(size_t i=0;i<vis.size();++i){
            const IndexType vi=vis(i);
            for(size_t f=0;f<gm.numberOfFactors(vi);++f){
               factorSet.insert(gm.factorOfVariable(vi,f));
            }
         }

         boost::python::object obj = opengm::python::get1dArray<ValueType>(factorSet.size());
         ValueType * castedPtr = opengm::python::getCastedPtr<ValueType>(obj);
         size_t c=0;
         for(SetIter iter=factorSet.begin();iter!=factorSet.end();++iter){
            castedPtr[c]=*iter;
            ++c;
         }
         
         //delete rgil;
         return obj;
      }


      template<class GM>
      boost::python::object moveLocalOpt(
         const GM & gm,
         const std::string  & acc 
      ){
         //releaseGIL * rgil= new releaseGIL;

         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::OperatorType OperatorType;

         boost::python::object obj = opengm::python::get1dArray<LabelType>(gm.numberOfVariables());
         LabelType * castedPtr = opengm::python::getCastedPtr<LabelType>(obj);

         LabelType maxLabel = 0 ;
         for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
            castedPtr[vi]=0;
            maxLabel = std::max(maxLabel,gm.numberOfLabels(vi));
         }

         ValueType * unaries = new ValueType[maxLabel];
         ValueType * buffer  = new ValueType[maxLabel];

         if(acc==std::string("minimizer")){

            for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
               const LabelType nLabels = gm.numberOfLabels(vi);

               size_t nUnaries = 0 ;
               size_t nFac    = gm.numberOfFactors(vi);

               for(size_t f=0;f<nFac;++f){
                  const IndexType fi = gm.factorOfVariable(vi,f);
                  if(gm[fi].numberOfVariables()==1){

                     if(nUnaries==0){
                        gm[fi].copyValues(unaries);
                        ++nUnaries;
                     }
                     else{
                        gm[fi].copyValues(buffer);
                        for(LabelType l=0;l<nLabels;++l){
                           OperatorType::op(unaries[l],buffer[l]);
                        }
                     }
                  }
               }
               // find the minimum label
               LabelType minLabel = 0;
               ValueType minValue = unaries[0];
               if(nUnaries!=0){
                  for(LabelType l=1;l<nLabels;++l){
                     if(unaries[l]<minValue){
                        minValue=unaries[l];
                        minLabel=l;
                     }
                  }
               }
               castedPtr[minLabel];
            }
         }
         
         //delete rgil;
         return obj;
      }
      

      template<class GM>
      boost::python::object variableIndicesFromFactorIndices(
         const GM & gm,
         opengm::python::NumpyView<typename GM::IndexType,1> factorIndices
      ){
         //releaseGIL * rgil= new releaseGIL;

         typedef typename GM::IndexType IndexType;
         typedef typename GM::ValueType ValueType;
         typedef std::set<IndexType> SetType; 
         typedef typename SetType::const_iterator SetIter;

         SetType variableSet;
         for(size_t i=0;i<factorIndices.size();++i){
            const IndexType fi=factorIndices(i);
            for(size_t v=0;v<gm.numberOfVariables(fi);++v){
               variableSet.insert(gm.variableOfFactor(fi,v));
            }
         }

         boost::python::object obj = opengm::python::get1dArray<ValueType>(variableSet.size());
         ValueType * castedPtr = opengm::python::getCastedPtr<ValueType>(obj);
         size_t c=0; 
         for(SetIter iter=variableSet.begin();iter!=variableSet.end();++iter){
            castedPtr[c]=*iter;
            ++c;
         }
         
         //delete rgil;
         return boost::python::extract<boost::python::numeric::array>(obj);
      }


      ////////////////////////////////
      ///  VECTORIZED FACTOR API HELPERS
      /////////////////////////////////
      /*

      template<class VALUE_TYPE>
      inline boost::python::object opengm::python::get1dArray(const size_t size){
         npy_intp dims[1]={static_cast<int>(size)};
         boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(int(1),  dims, typeEnumFromType<VALUE_TYPE>() )));
         return obj;
      }

      template<class VALUE_TYPE>
      inline boost::python::object get2dArray(const size_t size1,const size_t size2){
         npy_intp dims[2]={static_cast<int>(size1),static_cast<int>(size2)};
         boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(int(2),  dims, typeEnumFromType<VALUE_TYPE>() )));
         return obj;
      }

      template<class VALUE_TYPE>
      inline VALUE_TYPE * opengm::python::getCastedPtr(boost::python::object obj){
         void *array_data = PyArray_DATA((PyArrayObject*) obj.ptr());
         return  static_cast< VALUE_TYPE *>(array_data);
      }

      inline boost::python::numeric::array opengm::python::objToArray(boost::python::object obj){
         return boost::python::extract<boost::python::numeric::array > (obj);
      }
      */

      template<class GM>
      boost::python::tuple getCCFromLabes(
         const GM & gm,
         opengm::python::NumpyView<typename GM::LabelType,1> labels
      ){
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;

         // merge with UFD
         opengm::Partition<IndexType> ufd(gm.numberOfVariables());
         for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
            const LabelType label=labels(vi);
            const IndexType numFacVar = static_cast<IndexType>(gm.numberOfFactors(vi));
            for(IndexType f=0;f<numFacVar;++f){
               const IndexType fi        = gm.factorOfVariable(vi,f);
               const IndexType numVarFac = gm[fi].numberOfVariables();
               for(size_t v=0;v<numVarFac;++v){
                  const IndexType vi2=gm[fi].variableIndex(v);
                  const LabelType label2=labels(vi2);
                  if(vi!=vi2 && label==label2){
                     ufd.merge(vi,vi2);
                  }
               }
            }
         }
         std::map<IndexType,IndexType> repLabeling;
         ufd.representativeLabeling(repLabeling);
         const size_t numberOfCCs=ufd.numberOfSets();

         // get array
         boost::python::object obj = opengm::python::get1dArray<IndexType>(gm.numberOfVariables());
         IndexType * castPtr       = opengm::python::getCastedPtr<IndexType>(obj);
         for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
            IndexType findVi=ufd.find(vi);
            IndexType denseRelabling=repLabeling[findVi];
            castPtr[vi]=denseRelabling;
         }
         return boost::python::make_tuple(opengm::python::objToArray(obj),numberOfCCs);
      }



      ////////////////////////////////
      ///  VECTORIZED FACTOR API
      ///
      ///
      /////////////////////////////////

      template<class GM>
      boost::python::numeric::array factor_withOrder(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices,const size_t factorOrder){
         typedef typename GM::IndexType ResultType;
         size_t size=0;
         for(size_t i=0;i<factorIndices.size();++i){
            if(gm[factorIndices(i)].numberOfVariables()==factorOrder)
               ++size;
         }
         // get array
         boost::python::object obj = opengm::python::get1dArray<ResultType>(size);
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         // fill array
         size_t counter=0;
         for(size_t i=0;i<factorIndices.size();++i){
            if(gm[factorIndices(i)].numberOfVariables()==factorOrder){
               castPtr[counter]=factorIndices(i);
               ++counter;
            }
         }          
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_numberOfVariables(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef typename GM::IndexType ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].numberOfVariables();
         return opengm::python::objToArray(obj);
      }




      template<class GM>
      boost::python::numeric::array factor_size(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef typename GM::IndexType ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].size();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_functionIndex(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef typename GM::IndexType ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].functionIndex();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_functionType(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef typename GM::IndexType ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].functionType();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_isSubmodular(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef bool ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].isSubmodular();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_isAbsoluteDifference(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef bool ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].isAbsoluteDifference();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_isGeneralizedPotts(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef bool ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].isGeneralizedPotts();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_isTruncatedSquaredDifference(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef bool ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].isTruncatedSquaredDifference();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_isTruncatedAbsoluteDifference(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef bool ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].isTruncatedAbsoluteDifference();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_isSquaredDifference(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef bool ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].isSquaredDifference();
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_isPotts(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef bool ResultType;
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<ResultType>(obj);
         for(size_t i=0;i<factorIndices.size();++i)
            castPtr[i]=gm[factorIndices(i)].isPotts();
         return opengm::python::objToArray(obj);
      }


      template<class GM,class SCALAR_TYPE>
      boost::python::numeric::array factor_scalarRetFunction(
         const GM & gm,
         boost::python::object function,
         opengm::python::NumpyView<typename GM::IndexType,1> factorIndices
      ){
         typedef SCALAR_TYPE ResultType;
         boost::python::object obj = opengm::python::get1dArray<SCALAR_TYPE>(factorIndices.size());
         ResultType * castPtr      = opengm::python::getCastedPtr<SCALAR_TYPE>(obj);
         for(size_t i=0;i<factorIndices.size();++i){
            castPtr[i]=boost::python::extract<SCALAR_TYPE>(function(  gm[factorIndices(i)] )) ;
         }
         return opengm::python::objToArray(obj);
      }





      template<class GM>
      boost::python::numeric::array factor_variableIndices(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef typename GM::IndexType ResultType;
         // get order from first factor in factorIndices
         const size_t numberOfVariables = gm[factorIndices(0)].numberOfVariables();
         const size_t numFactors        = factorIndices.size();       
         // allocate numpy array
         boost::python::object obj =opengm::python::get2dArray<ResultType>(numFactors,numberOfVariables);
         opengm::python::NumpyView<ResultType,2> numpyArray(obj);
         for(size_t i=0;i<numFactors;++i){
            const size_t fi     = factorIndices(i);
            const size_t numVar = gm[fi].numberOfVariables();
            if(numVar!=numberOfVariables){
               throw opengm::RuntimeError("within this function all factors must have the same order");
            }
            for(size_t v=0;v<numVar;++v){
               numpyArray(i,v)=gm[fi].variableIndex(v);
            }
         }
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_numberOfLabels(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices){
         typedef typename GM::IndexType ResultType;
         // get order from first factor in factorIndices
         const size_t numberOfVariables = gm[factorIndices(0)].numberOfVariables();
         const size_t numFactors        = factorIndices.size();       
         // allocate numpy array
         boost::python::object obj =opengm::python::get2dArray<ResultType>(numFactors,numberOfVariables);
         opengm::python::NumpyView<ResultType,2> numpyArray(obj);
         for(size_t i=0;i<numFactors;++i){
            const size_t fi     = factorIndices(i);
            const size_t numVar = gm[fi].numberOfVariables();
            if(numVar!=numberOfVariables){
               throw opengm::RuntimeError("within this function all factors must have the same order");
            }
            for(size_t v=0;v<numVar;++v){
               numpyArray(i,v)=gm[fi].numberOfLabels(v);
            }
         }
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_gmLablingToFactorLabeling(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices, opengm::python::NumpyView<typename GM::LabelType,1> labels){
         typedef typename GM::LabelType ResultType;
         // get order from first factor in factorIndices
         const size_t numberOfVariables = gm[factorIndices(0)].numberOfVariables();
         const size_t numFactors        = factorIndices.size();       
         // allocate numpy array
         boost::python::object obj =opengm::python::get2dArray<ResultType>(numFactors,numberOfVariables);
         opengm::python::NumpyView<ResultType,2> numpyArray(obj);
         for(size_t i=0;i<numFactors;++i){
            const size_t fi     = factorIndices(i);
            const size_t numVar = gm[fi].numberOfVariables();
            if(numVar!=numberOfVariables){
               throw opengm::RuntimeError("within this function all factors must have the same order");
            }
            for(size_t v=0;v<numVar;++v){
               numpyArray(i,v)=labels(gm[fi].variableIndex(v));
            }
         }
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_evaluateGmLabeling(const GM & gm,opengm::python::NumpyView<typename GM::IndexType,1> factorIndices, opengm::python::NumpyView<typename GM::LabelType,1> labels){
         typedef typename GM::ValueType ResultType;
         // get order from first factor in factorIndices
         const size_t numberOfVariables = gm[factorIndices(0)].numberOfVariables();
         const size_t numFactors        = factorIndices.size();       
         // allocate numpy array
         boost::python::object obj = opengm::python::get1dArray<ResultType>(numFactors);
         opengm::python::NumpyView<ResultType,2> numpyArray(obj);

         std::vector<typename GM::LabelType> factorLabels(numberOfVariables);

         for(size_t i=0;i<numFactors;++i){
            const size_t fi     = factorIndices(i);
            const typename GM::FactorType factor=gm[fi];
            const size_t numVar = factor.numberOfVariables();
            if(numVar!=numberOfVariables){
               throw opengm::RuntimeError("within this function all factors must have the same order");
            }
            for(size_t v=0;v<numVar;++v){
               factorLabels[v]=labels(gm[fi].variableIndex(v));
            }
            numpyArray(i)=factor(factorLabels.begin());
         }
         return opengm::python::objToArray(obj);
      }

      template<class GM>
      boost::python::numeric::array factor_evaluateFactorLabeling(
         const GM & gm,
         opengm::python::NumpyView<typename GM::IndexType,1> factorIndices, 
         opengm::python::NumpyView<typename GM::LabelType,2> labels
      ){
         typedef typename GM::ValueType ResultType;
         // get order from first factor in factorIndices
         const size_t numberOfVariables = gm[factorIndices(0)].numberOfVariables();
         const size_t numFactors        = factorIndices.size();     
         const size_t numGivenLabels    = labels.shape(0);
         const size_t givenOrder        = labels.shape(1);   

         OPENGM_CHECK_OP(numberOfVariables , == ,givenOrder, "labels have wrong shape");
         OPENGM_CHECK(numGivenLabels==1 || numGivenLabels==numFactors,"labels have wrong shape");


         // allocate numpy array
         boost::python::object obj = opengm::python::get1dArray<ResultType>(numFactors);
         opengm::python::NumpyView<ResultType,1> numpyArray(obj);

         std::vector<typename GM::LabelType> factorLabels(numberOfVariables);

         for(size_t i=0;i<numFactors;++i){
            const size_t fi     = factorIndices(i);
            const typename GM::FactorType factor=gm[fi];
            const size_t numVar = factor.numberOfVariables();
            if(numVar!=numberOfVariables){
               throw opengm::RuntimeError("within this function all factors must have the same order");
            }

            size_t labelIndex=i;
            if(i>=numGivenLabels)
               labelIndex=numGivenLabels-1;

            for(size_t v=0;v<numVar;++v){
               factorLabels[v]=static_cast<typename GM::LabelType>(labels(labelIndex,v));
            }
            numpyArray(i)=factor(factorLabels.begin());
         }
         return opengm::python::objToArray(obj);
      }
      template<class GM>
      boost::python::numeric::array factor_fullIncluedFactors(
         const GM & gm,
         opengm::python::NumpyView<typename GM::IndexType,1> factorIndices, 
         opengm::python::NumpyView<typename GM::IndexType,1> vis
      ){
         typedef typename GM::IndexType IndexType;
         typedef typename GM::IndexType ResultType;
         // get order from first factor in factorIndices
         const size_t numberOfVariables = gm[factorIndices(0)].numberOfVariables();
         const size_t numFactors        = factorIndices.size();       


         std::set<IndexType> visSet;
         std::set<IndexType> fisSet;
         std::set<IndexType> factorCandidates;

         typedef typename std::set<IndexType>::const_iterator SetIter;


         const bool fullSubset=factorIndices.size()==gm.numberOfFactors();

         // fill fis set   
         if(fullSubset==false){
            for(IndexType f=0;f<factorIndices.size();++f){
               const IndexType fi=factorIndices[f];
               fisSet.insert(fi);
            }
         }

         // fill vis set and candiate factors
         for(IndexType v=0;v<vis.size();++v){
            const IndexType vi=vis[v];
            visSet.insert(vi);
         }

         // fill vis set and candiate factors
         for(IndexType v=0;v<vis.size();++v){
            const IndexType vi=vis[v];
            const IndexType numFacVar=static_cast<IndexType>(gm.numberOfFactors(vi));
            for(IndexType f=0;f<numFacVar;++f){
               const IndexType fi = gm.factorOfVariable(vi,f);
               bool includeFactor=true;
               if(fullSubset){

               }
               else if( fisSet.find(fi)==fisSet.end()){
                  includeFactor=false;
               }
               if( factorCandidates.find(fi)==factorCandidates.end()){
                  const IndexType numVarF=gm[fi].numberOfVariables();
                  for(IndexType v=0;v<numVarF;++v){
                     const IndexType vi=gm[fi].variableIndex(v);
                     if(visSet.find(vi)==visSet.end()){
                        includeFactor=false;
                        break;
                     }
                  }
               }
               if(includeFactor){
                  factorCandidates.insert(fi);
               }
            }
         }

         // allocate numpy array
         boost::python::object obj = opengm::python::get1dArray<ResultType>(factorCandidates.size());
         opengm::python::NumpyView<ResultType,2> numpyArray(obj);

         IndexType counter=0;
         for(SetIter fiter=factorCandidates.begin();fiter!=factorCandidates.end();++fiter){
            const IndexType fi = *fiter;
            numpyArray[counter]=fi;
            counter+=1;
         }
         return opengm::python::objToArray(obj);
      }



      template<class GM>
      boost::python::tuple factor_check(
         const GM & gm,
         opengm::python::NumpyView<typename GM::IndexType,1> factorIndices
      ){
         // allocate numpy array
         boost::python::object obj1 = opengm::python::get1dArray<float>(3);
         boost::python::object obj2 = opengm::python::get1dArray<float>(2);

         opengm::python::NumpyView<float,1> a1(obj1);
         opengm::python::NumpyView<float,1> a2(obj2);

         a1(0)=0;a1(1)=1;a1(2)=2;
         a2(0)=3;a2(1)=4;

         return boost::python::make_tuple(obj1,obj2);
      }



      /*
      template<class GM>
      boost::python::tuple factor_subfactors_alpha_expansion(
         const GM & gm,
         opengm::python::NumpyView<typename GM::IndexType,1> factorIndices,
         opengm::python::NumpyView<typename GM::LabelType,1> gmLabels,
         const typename GM::LabelType alpha
      ){
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndependentFactorType IndependentFactorType;
         ///////////
         // Count factors:
         //  - factors which have only alphas as labeling are excluded
         //  - factors which have no alpha as labeling are full included
         //  - factors which have at least an an alpha and some other labels are partial included
         size_t nOnlyAlpha    = 0;
         size_t nNoAlpha      = 0;
         size_t nPartialAlpha = 0;
         {

         }

         // allocate numpy arrays for indices of factors
         boost::python::object objOnlyAlpha    = opengm::python::get1dArray<float>(nOnlyAlpha);
         boost::python::object objNoAlpha      = opengm::python::get1dArray<float>(nNoAlpha);
         boost::python::object objPartialAlpha = opengm::python::get1dArray<float>(nPartialAlpha);

         opengm::python::NumpyView<float,1> aOnlyAlpha(objOnlyAlpha);
         opengm::python::NumpyView<float,1> aNoAlpha(objNoAlpha);
         opengm::python::NumpyView<float,1> objPartialAlphaNoAlpha(objPartialAlpha);

         // allocate std::vector< IndependentFactors>
         std::vector<IndependentFactorType> partialAlphaFactors(objPartialAlpha);

         ///////////
         // Fill indices of counted factors :
         //  - factors which have only alphas as labeling are excluded
         //  - factors which have no alpha as labeling are full included
         //  - factors which have at least an an alpha and some other labels are partial included
         // Fill partialAlphaFactors
         //  - fix values and set up indices w.r.t this graphical model
         //  => implement relabel in std::vector<IndependentFactorType> 
         {

         }
      }
      */

   }


   namespace pygmgen{
      
      template<class GM>
      GM * grid2Order2d
      (
         opengm::python::NumpyView<typename GM::ValueType,3> unaryFunctions,
         opengm::python::NumpyView<typename GM::ValueType> binaryFunction,
         bool numpyOrder
      ){
         typedef typename GM::SpaceType Space;
         typedef typename GM::ValueType ValueType;
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         typedef typename GM::FunctionIdentifier FunctionIdentifier;
         typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType> ExplicitFunctionType;
         typedef std::pair<FunctionIdentifier,ExplicitFunctionType &> FidRefPair;
         

         GM * gm=NULL;
         {
            releaseGIL rgil;
            const size_t shape[]={unaryFunctions.shape(0),unaryFunctions.shape(1)};
            const size_t numVar=shape[0]*shape[1];
            const size_t numLabels=unaryFunctions.shape(2);
            { // scope to delete space
               Space space(numVar,numLabels);
               gm = new GM(space);
            }
            // add one (!) 2.-order-function to the gm
            
            if(binaryFunction.dimension()!=2){
               throw opengm::RuntimeError("binaryFunction dimension must be 2");
            }
            FunctionIdentifier fid2=pygm::addFunctionNpPy(*gm,binaryFunction);
            IndexType c[2]={0,0};
            ExplicitFunctionType f(&numLabels,&numLabels+1);
            for(c[0]=0;c[0]<shape[0];++c[0]){
               for(c[1]=0;c[1]<shape[1];++c[1]){
                  //unaries
                  // fill with data
                  for(LabelType l=0;l<numLabels;++l)
                     f(l)=unaryFunctions(c[0],c[1],l);
                  FunctionIdentifier fid=gm->addFunction(f);
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
         }
         return gm;
      }
}




template<class GM>
void export_gm() {

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
	
   docstring_options doc_options(true, true, false);
   




   typedef FactorsOfVariableHolder<PyGm>  FactorOfVarHolder;
   //------------------------------------------------------------------------------------
   // factor-holder
   //------------------------------------------------------------------------------------  
   class_<FactorOfVarHolder > ("FactorsOfVariable", 
   "Holds the factor indices of all factors conencted to a variable.\n"
   "``FactorsOfVariable`` is only a view to real data,\n"
   "therefore only one pointer  is stored",
   init<const PyGm &,const size_t >()[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const PyGm& */>()])
   .def(init< >())
   .def("__len__", &FactorOfVarHolder::size)
   .def("__str__",&FactorOfVarHolder::asString,
   "Convert shape to a string\n"
   "Returns:\n"
   "  new allocated string"
   )
   .def("__array__", &FactorOfVarHolder::toNumpy,
   "Convert FactorsOfVariable to a 1d numpy ndarray\n"
   "Returns:\n"
   "  new allocated 1d numpy ndarray"
   )
   .def("__list__", &FactorOfVarHolder::toList,
   "Convert FactorsOfVariable to a list\n"
   "Returns:\n"
   "  new allocated list"
   )
   .def("__tuple__",&FactorOfVarHolder::toTuple,
   "Convert FactorsOfVariable to a tuple\n"
   "Returns:\n"
   "  new allocated tuple"
   )
   .def("__getitem__", &FactorOfVarHolder::operator[], return_value_policy<return_by_value>(),(arg("factorIndex")),
   "Get the factor index w.r.t. the graphical model .\n\n"
   "Args:\n\n"
   "  factorIndex: factor index w.r.t. the number of factors connected with the variable"
   "Returns:\n"
   "  factor index w.r.t. the graphical model\n\n"
   )
   .def("__copy__", &generic__copy__< FactorOfVarHolder >)
   ;

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
	"The central class of ``opengm`` which holds the factor graph and functions of the graphical model",
	init< >(
      "Construct an empty graphical model with no variables "
      "Example:\n\n"
      "     Construct an empty ::\n\n"
      "        >>> gm=opengm.addder.GraphicalModel()\n"
      "        >>> int(gm.numberOfVariables)\n"
      "        0\n"
      "        \n\n"
   )
	)
   .def("__init__", make_constructor(&pygm::gmConstructorPythonAny<PyGm,LabelType> ,default_call_policies(),
         (
            arg("numberOfLabels"),
            arg("reserveNumFactorsPerVariable")=1
         )
      ),
      "Construct a gm from any iterable python object where the iterabe object holds the number of labels for each variable.\n\n"
      "The gm will have as many variables as the length of the iterable sequence\n\n"
      "Args:\n\n"
      "  numberOfLabels: holds the number of labels for each variable\n\n"
      "  reserveNumFactorsPerVariable: reserve a certain number of factors for each varialbe (default\n\n"
      "     This can speedup adding factors.\n\n"
      "Example:\n\n"
      "     Construct a gm from generator expression ::\n\n"
      "        >>> gm=opengm.addder.GraphicalModel(2 for x in xrange(100))\n"
      "        >>> int(gm.numberOfVariables)\n"
      "        100\n"
      "        >>> int(gm.numberOfLabels(0))\n"
      "        2\n"
      "        \n\n"
      "     Construct a gm from list and tuples ::\n\n"
      "        >>> gm=opengm.adder.GraphicalModel( [3]*10 )\n"
      "        >>> int(gm.numberOfVariables)\n"
      "        10\n"
      "        >>> int(gm.numberOfLabels(0))\n"
      "        3\n"
      "        >>> gm=opengm.adder.GraphicalModel( (2,4,6) )\n"
      "        >>> int(gm.numberOfVariables)\n"
      "        3\n"
      "        >>> int(gm.numberOfLabels(0))\n"
      "        2\n"
      "        >>> int(gm.numberOfLabels(1))\n"
      "        4\n"
      "        >>> int(gm.numberOfLabels(2))\n"
      "        6\n"
      "        \n\n"
      "     And factors can be reserved for the varialbes ::\n\n"
      "        >>> gm=opengm.adder.GraphicalModel(numberOfLabels=[2]*10,reserveNumFactorsPerVariable=5)\n"
      "        >>> gm=opengm.adder.GraphicalModel(numberOfLabels=(2,2,2),reserveNumFactorsPerVariable=3)\n"
      "        \n\n"
      "Note:\n\n"
      ".. seealso::"
      "     :func:`opengm.gm` :func:`opengm.graphicalModel`"
   )
	.def("__init__", make_constructor(&pygm::gmConstructorPythonNumpy<PyGm,LabelType> ,default_call_policies(),
         (
            arg("numberOfLabels"),
            arg("reserveNumFactorsPerVariable")=1
         )
      ),
      "Construct a gm from a 1d numpy ndarray which holds the number of labels for each variable.\n\n"
      "The gm will have as many variables as the length of the ndarray\n\n"
      "Args:\n\n"
      "  numberOfLabels: holds the number of labels for each variable\n\n"
      "  reserveNumFactorsPerVariable: reserve a certain number of factors for each varialbe (default\n\n"
      "     This can speedup adding factors.\n\n"
      "Example:\n\n"
      "     Construct a gm from generator expression ::\n\n"
      "        >>> gm=opengm.addder.GraphicalModel(numpy.ones(100,dtype=numpy.uint64)*4)\n"
      "        >>> int(gm.numberOfVariables)\n"
      "        \n\n"
      "        100\n"
      "        >>> int(gm.numberOfLabels(0))\n"
      "        4\n"
      "     And factors can be reserved for the varialbes ::\n\n"
      "        >>> gm=opengm.addder.GraphicalModel(numpy.ones(100,dtype=numpy.uint64,reserveNumFactorsPerVariable=3) )\n"
      "        \n\n"
      "Note:\n\n"
      ".. seealso::"
      "     :func:`opengm.gm` :func:`opengm.graphicalModel`"
	)
   .def("__init__", make_constructor(&pygm::gmConstructorSimple<PyGm>,default_call_policies(),
      (
         arg("numberOfVariables"),
         arg("numberOfLabels"),
         arg("reserveNumFactorsPerVariable")=1
      )
   ),
   "Construct a gm where each variable will have the same number of labels\n\n"
   "The gm will have as many variables as given by ``numberOfVariables`` \n"
   "Args:\n\n"
   "  numberOfVariables: is the number of varables for the gm\n\n"
   "  numberOfLabels: is the number of labels for each variable\n\n"
   "  reserveNumFactorsPerVariable: reserve a certain number of factors for each varialbe.\n\n"
   "     This can speedup adding factors.\n\n"
   "Example:\n\n"
   "     Construct a gm with 10 variables each having 2 possible labels::\n\n"
   "        >>> gm=opengm.addder.GraphicalModel(numberOfVariables=10,numberOfLabels=2)\n"
   "        >>> gm.numberOfVariables\n"
   "        10\n"
   "        \n\n"
   )
   .def("__init__",make_constructor(&pygm::gmConstructorVector<PyGm> ,default_call_policies(),(arg("numberOfLabels"))),
   "Construct a gm from a gm from a ``opengm.IndexVector``\n\n"
   "Args:\n\n"
   "  numberOfLabels: holds the number of labels for each variable\n\n"
   )
   .def("assign", &pygm::assign_Any<PyGm,LabelType>,(arg("numberOfLabels")),
   "Assign a graphical model from any number of labels sequence\n\n"
   "Args:\n\n"
   "  numberOfLabels: holds the number of labels for each variable\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.adder.GraphicalModel.__init__`"
   )
   .def("moveLocalOpt",&pygm::moveLocalOpt<PyGm>)
   .def("_getCCFromLabes",&pygm::getCCFromLabes<PyGm>)
   .def("_factor_check",&pygm::factor_check<PyGm>)
   .def("_factor_evaluateFactorLabeling",&pygm::factor_evaluateFactorLabeling<PyGm>)
   .def("_factor_evaluateGmLabeling",&pygm::factor_evaluateGmLabeling<PyGm>)
   .def("_factor_gmLablingToFactorLabeling",&pygm::factor_gmLablingToFactorLabeling<PyGm>)
   .def("_factor_withOrder",&pygm::factor_withOrder<PyGm>)
   .def("_factor_variableIndices",&pygm::factor_variableIndices<PyGm>)
   .def("_factor_numberOfLabels",&pygm::factor_numberOfLabels<PyGm>)
   .def("_factor_numberOfVariables",&pygm::factor_numberOfVariables<PyGm>)
   .def("_factor_isSubmodular",&pygm::factor_isSubmodular<PyGm>)
    // TODO -> change this to default api
   .def("_variableIndices",&pygm::variableIndicesFromFactorIndices<PyGm>)
   .def("_factorIndices",&pygm::factorIndicesFromVariableIndices<PyGm>)
   .def("_factor_fullIncluedFactors",&pygm::factor_fullIncluedFactors<PyGm>)
   .def("_factor_scalarRetFunction_bool",&pygm::factor_scalarRetFunction<PyGm,bool>)
   .def("_factor_scalarRetFunction_uint64",&pygm::factor_scalarRetFunction<PyGm,opengm::UInt64Type>)
   .def("_factor_scalarRetFunction_int64",&pygm::factor_scalarRetFunction<PyGm,opengm::Int64Type>)
   .def("_factor_scalarRetFunction_float32",&pygm::factor_scalarRetFunction<PyGm,opengm::Float32Type>)
   .def("_factor_scalarRetFunction_float64",&pygm::factor_scalarRetFunction<PyGm,opengm::Float64Type>)


	.def("assign", &pygm::assign_Numpy<PyGm,LabelType>,args("numberOfLabels"),
   "Assign a graphical model from  number of labels sequence which is a 1d ``numpy.ndarray`` \n\n"
   "Args:\n\n"
   "  numberOfLabels: holds the number of labels for each variable\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.adder.GraphicalModel.__init__`"
	)
   .def("assign", &pygm::assign_Vector<PyGm>,args("numberOfLabels"),
   "Assign a graphical model from  number of labels sequence which is a ``opengm.IndexVector`` \n\n"
   "Args:\n\n"
   "  numberOfLabels: holds the number of labels for each variable\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.adder.GraphicalModel.__init__`"
   )
   .def("reserveFactors",&PyGm::reserveFactors,(arg("numberOfFactors")),
   "reserve space for factors. \n\n"
   "This can speedup adding factors\n\n" 
   "Args:\n\n"
   "  numberOfFactors: the number of factor to reserve\n\n"
   "Example:\n\n"
   "    Reserve some factors\n\n"
   "        >>> gm=gm([2]*10)\n"
   "        >>> gm.reserveFactors(10)\n"
   "        \n\n"
   )
   .def("reserveFactorsVarialbeIndices",&PyGm::reserveFactorsVarialbeIndices,(arg("size")),
   "reserve space for factors varialbe indices (stored in one std::vector for all factors). \n\n"
   "This can speedup adding factors\n\n" 
   "Args:\n\n"
   "  size: total size of variable indices\n\n"
   "Example:\n\n"
   "    Reserve space for varaiable indices of  9 second order factors\n\n"
   "        >>> gm=gm([2]*10)\n"
   "        >>> gm.reserveFactorsVarialbeIndices(9*2)\n"
   "        \n\n"
   )
   .def("reserveFunctions",&pygm::reserveFunctions<PyGm>,(arg("numberOfFunctions"),arg("functionTypeName")),"reserve space for functions of a certain type")
	.def("__str__", &pygm::printGmPy<PyGm>,
	"Print a a gm as string"
	"Returns:\n"
   "	A string which describes the graphical model \n\n"
	)
	.def("space", &PyGm::space , return_internal_reference<>(),
	"Get the variable space of the graphical model\n\n"
	"Returns:\n"
	"	A const reference to space of the gm."
   "Example:\n\n"
   "    Get variable space\n\n"
   "        >>> gm=gm([2]*10)\n"
   "        >>> space=gm.space()\n"
   "        >>> len(space)\n"
   "        10\n"
   "        \n\n"
	)
	.add_property("numberOfVariables", &pygm::numVarGm<PyGm>,
	"Number of variables of the graphical model"
   "Example:\n\n"
   "    Get the number of variables of a gm\n\n"
   "        >>> gm=gm([2]*5)\n"
   "        >>> int(gm.numberOfVariables)\n"
   "        5\n"
   "        \n\n"
	)
	.add_property("numberOfFactors", &pygm::numFactorGm<PyGm>,
	"Number of factors of the graphical model\n\n"
   "Example:\n\n"
   "    Get the number of factors of a gm\n\n"
   "        >>> import opengm\n"
   "        >>> gm=gm([2]*5)\n"
   "        >>> int(gm.numberOfFactors)\n"
   "        0\n"
   "        >>> fid=gm.addFunction(opengm.PottsFunction([2,2],1.0,0.0))\n"
   "        >>> int(gm.addFactor(fid,[0,1]))\n"
   "        0\n"
   "        >>> int(gm.numberOfFactors)\n"
   "        1\n"
   "        >>> int(gm.addFactor(fid,[1,2]))\n"
   "        1\n"
   "        >>> int(gm.numberOfFactors)\n"
   "        2\n"
   "        \n\n"
	)
	.add_property("operator",&pygm::operatorAsString<PyGm>,
	"The operator of the graphical model as a string"
   "Example:\n\n"
   "    Get the operator of a gm as string\n\n"
   "        >>> import opengm\n"
   "        >>> gm=opengm.gm([2]*5)\n"
   "        >>> gm.operator\n"
   "        'adder'\n"
   "        >>> gm=opengm.gm([2]*5,operator='adder')\n"
   "        >>> gm.operator\n"
   "        'adder'\n"
   "        >>> gm=opengm.gm([2]*5,operator='multiplier')\n"
   "        >>> gm.operator\n"
   "        'multiplier'\n"
   "        \n\n"
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
	.def("factorOfVariable",&PyGm::factorOfVariable,(arg("variableIndex"),arg("factorIndex")),
	"Get the factor index of a factor which is connected to the variable at variable index.\n\n"
	"Args:\n\n"
	"  variableIndex: index of a variable w.r.t the gm\n\n"
	"  factorIndex: index of a factor w.r.t the number of factors which are connected to this variable``\n\n"
	"Returns:\n"
   	"	The factor index w.r.t. the gm "
      
	)
   .def("factorsOfVariable",&pygm::getFactosOfVariableHolder<PyGm>,(arg("variableIndex")),
   "Get the sequence of factor indices (w.r.t. the graphical model) of all factors connected"
   " to the variable at ``variableIndex`` "
   "Args:\n\n"
   "  variableIndex: index of a variable w.r.t the gm\n\n"
   "Returns:\n"
   "A sequence of factor indices (w.r.t. the graphical model) of all factors connected"
   " to the variable at ``variableIndex`` "
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

   .def("_addFunctions_list", &pygm::addFunctionsListNpPy<PyGm>,return_value_policy<manage_new_object>(),args("functions"))
   .def("_addFunctions_numpy", &pygm::addFunctionsNpPy<PyGm>,return_value_policy<manage_new_object>(),args("functions"))
   .def("_addUnaryFunctions_numpy", &pygm::addUnaryFunctionsNpPy<PyGm>,return_value_policy<manage_new_object>(),args("functions"))
   .def("_addFunctions_generator", &pygm::addFunctionsFromGenerator<PyGm>,return_value_policy<manage_new_object>(),args("functions"))
    // WARNING,THIS IS UNTESTED....TEST ME!!!!!!
   .def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PyPottsFunction>,return_value_policy<manage_new_object>(),args("functions"))
   .def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PyPottsNFunction>,return_value_policy<manage_new_object>(),args("functions"),"todo")
   .def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PyPottsGFunction>,return_value_policy<manage_new_object>(),args("functions"),"todo")
   //.def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PyAbsoluteDifferenceFunction>,return_value_policy<manage_new_object>(),args("functions"),"todo")
   .def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PyTruncatedAbsoluteDifferenceFunction>,return_value_policy<manage_new_object>(),args("functions"),"todo")
   //.def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PySquaredDifferenceFunction>,return_value_policy<manage_new_object>(),args("functions"),"todo")
   .def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PyTruncatedSquaredDifferenceFunction>,return_value_policy<manage_new_object>(),args("functions"),"todo")
   .def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PySparseFunction>,return_value_policy<manage_new_object>(),args("functions"),"todo")
   .def("_addFunctions_vector",&pygm::addFunctionsGenericVectorPy<PyGm,PyPythonFunction>,return_value_policy<manage_new_object>(),args("functions"),"todo")


   .def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PyPottsFunction>,args("function"))
   .def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PyPottsFunction>,args("function"))
   .def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PyPottsNFunction>,args("function"))
   .def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PyPottsGFunction>,args("function"))
   //.def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PyAbsoluteDifferenceFunction>,args("function"))
   .def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PyTruncatedAbsoluteDifferenceFunction>,args("function"))
   //.def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PySquaredDifferenceFunction>,args("function"))
   .def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PyTruncatedSquaredDifferenceFunction>,args("function"))
   .def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PySparseFunction>,args("function"))
   .def("_addFunction",&pygm::addFunctionGenericPy<PyGm,PyPythonFunction>,args("function"))
	.def("_addFunction", &pygm::addFunctionNpPy<PyGm>,args("function"))
   .def("_addFactor", &pygm::addFactor_Any<PyGm,int>, (arg("fid"),arg("variableIndices"),arg("finalize")))
	.def("_addFactor", &pygm::addFactor_Numpy<PyGm>, (arg("fid"),arg("variableIndices"),arg("finalize")))
   .def("_addFactor", &pygm::addFactor_Vector<PyGm>, (arg("fid"),arg("variableIndices"),arg("finalize")))
   .def("_addUnaryFactors_vector_numpy", &pygm::addUnaryFactors_Vector_Numpy<PyGm>, (arg("fid"),arg("variableIndices"),arg("finalize")))
   .def("_addFactors_vector_numpy", &pygm::addFactors_Vector_Numpy<PyGm>, (arg("fid"),arg("variableIndices"),arg("finalize")))
   .def("_addFactors_vector_vectorvector", &pygm::addFactors_Vector_VectorVector<PyGm>, (arg("fid"),arg("variableIndices"),arg("finalize")))
   .def("finalize",&PyGm::finalize,
      "finalize the graphical model after adding all factors \n\n"
      "this method must be called if any non finalized factor has been added (addFactor / addFactors with finalize=False)"
   )
	.def("__getitem__", &pygm::getFactorStaticPy<PyGm>, return_internal_reference<>(),(arg("factorIndex")),
	"Get a factor of the graphical model\n\n"
	"Args:\n\n"
	"	factorIndex: index of a factor w.r.t. the gm \n\n"
	"		``factorIndex`` has to be a integral scalar::\n\n"
	"Returns:\n"
   	"  A const reference to the factor at ``factorIndex``.\n\n"
	)
	.def("_evaluate_numpy",&pygm::evaluatePyNumpy<PyGm>,(arg("labels")),
	"Evaluates the factors of given a labelSequence.\n\n"
	"	In this overloading the type of  \"labelSequence\" has to be a 1d numpy array\n\n"
	"Args:\n\n"
	"	labelSequence: A labeling for all variables.\n\n"
	"		Has to as long as ``gm.numberOfVariables``.\n\n"
	"Returns:\n"
	"	The energy / probability for the given ``labelSequence``"
	)
	.def("_evaluate_list",&pygm::evaluatePyList<PyGm,int>,(arg("labels")),
	"Evaluates the factors of given a labelSequence.\n\n"
	"	In this overloading the type of  \"labelSequence\" has to be a list\n\n"
	"Args:\n\n"
	"	labelSequence: A labeling for all variables.\n\n"
	"		Has to as long as ``gm.numberOfVariables``.\n\n"
	"Returns:\n"
	"	The energy / probability for the given ``labelSequence``"
	)
   .def("_evaluate_vector",&pygm::evaluatePyVector<PyGm,LabelType>,(arg("labels")),
   "Evaluates the factors of given a labelSequence.\n\n"
   "  In this overloading the type of  \"labelSequence\" has to be a std::vector<LabelType> \n\n"
   "Args:\n\n"
   "  labelSequence: A labeling for all variables.\n\n"
   "     Has to as long as ``gm.numberOfVariables``.\n\n"
   "Returns:\n"
   "  The energy / probability for the given ``labelSequence``"
   )
   .def("variablesAdjacency",&pygm::variablesAdjacency<PyGm>,"generate variable adjacency")
  ;
}


template void export_gm<opengm::python::GmAdder>();
template void export_gm<opengm::python::GmMultiplier>();
