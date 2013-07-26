#pragma once
#ifndef OPENGM_GRAPHICALMODEL_EXPLICIT_STORAGE_HXX
#define OPENGM_GRAPHICALMODEL_EXPLICIT_STORAGE_HXX

#include <vector>
#include <algorithm>
#include <numeric>
#include <map>

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/opengm.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"

namespace opengm{

template<
   class T, 
   class OPERATOR, 
   class FUNCTION_TYPE_LIST , 
   class SPACE 
>
class GraphicalModel;
    
    
template<unsigned int I,unsigned int D,bool END>
class FunctionIteratation;


/// \brief ExplicitStorage (continous storage) of a graphical model function data
///
/// ExplicitStorage stores all instances of all functions into a single continous
/// stride of memory.
/// Given a factor one gets the begin pointer of the corresponding function.
/// The storage of the function is in last coordinate major order.
///
/// Usage:
/// \code
/// opengm::ExplicitStorage<GmType> storage(gm);
/// typedef GmType::FactorType::ShapeIteratorType ShapeIteratorType;
/// // loop over all factors
/// for(size_t factor=0;factor<gmA.numberOfFactors();++factor){
///    // "usual" way to iterate over the factors values with coordinates
///    // is with the shapewalker class
///    opengm::ShapeWalker< ShapeIteratorType > walker(gm[factor].shapeBegin(),gm[factor].numberOfVariables());
///    // get the begin pointer to the storage of the factors function
///    GmType::ValueType const *  ptr=storageA[gmA[factor]];
///    // toy / test loop over all factors values
///    // and compare (expensive) direct access via the factors
///    // and cheap access via pointer
///    for (size_t i = 0; i < gmA[factor].size(); ++i) {
///        OPENGM_TEST_EQUAL(gmA[factor](walker.coordinateTuple().begin()),ptr[i]);
///         ++walker;
///    }
/// }
/// 
/// \endcode

template<class GM>
class ExplicitStorage {
    
   template<unsigned int I,unsigned int D,bool END>
   friend class  FunctionIteratation;
   typedef GM GraphicalModelType;
   typedef typename GraphicalModelType::LabelType LabelType;        
   typedef typename GraphicalModelType::IndexType IndexType;       
   typedef typename GraphicalModelType::ValueType ValueType;        
   typedef typename GraphicalModelType::OperatorType OperatorType;               
   typedef typename GraphicalModelType::FactorType FactorType;                     
   typedef typename GraphicalModelType::IndependentFactorType IndependentFactorType; 
   typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;

public:
    ExplicitStorage(const GraphicalModelType & gm )
    :gm_(gm),
    functionSize_(GraphicalModelType::NrOfFunctionTypes),
    functionTypeStart_(GraphicalModelType::NrOfFunctionTypes,0){
        //some offset calculation to get a function index
        // from the fid (=type + index)
        size_t numFTotal=0; 
        for(size_t i=0;i<GraphicalModelType::NrOfFunctionTypes;++i){
            functionSize_[i]=gm_.numberOfFunctions(i);
            numFTotal+=functionSize_[i];
        }
        //
        functionIndexToStart_.resize(numFTotal);
        functionTypeStart_[0]=0;
        for(size_t i=1;i<GraphicalModelType::NrOfFunctionTypes;++i){
            for(size_t ii=0;ii<i;++ii){
                functionTypeStart_[i]+=functionSize_[ii];
            }
        }
        
        // calculate how much storage is needed
        size_t storageSize=0;
        FunctionIteratation<0,GraphicalModelType::NrOfFunctionTypes,false>::size(*this,storageSize);
        
        
        // allocate memory
        data_ = new ValueType[storageSize];
        dataSize_=storageSize;
        
        // write function into allocated memory 
        size_t currentOffset=0;
        FunctionIteratation<0,GraphicalModelType::NrOfFunctionTypes,false>::store(*this,currentOffset);
        OPENGM_ASSERT(currentOffset==storageSize);
    }
    ~ExplicitStorage(){
        delete[] data_;
    }
    
    ValueType const * operator[](const FactorType & factor)const{
        const size_t scalarIndex=fidToIndex(factor.functionType(),factor.functionIndex());
        return data_+functionIndexToStart_[scalarIndex];
    }
    
private:
    size_t dataSize_;
    const GraphicalModelType & gm_;
    std::vector<size_t> functionSize_;
    std::vector<size_t> functionTypeStart_;
    std::vector<size_t> functionIndexToStart_;
    ValueType * data_;
    
    
    size_t fidToIndex(const size_t functionType,const size_t functionIndex)const{
        return functionTypeStart_[functionType]+functionIndex;
    }
    
    ValueType const * getFunction(size_t functionType,size_t functionIndex){
        
    }
    
};

/// \brief Convert any graphical model into an explicit graphical model
/// 
/// An explicit graphical model (gm) is a gm using only the explicit function for
/// storage.
///
/// Usage:
/// \code
/// typedef opengm::ConvertToExplicit<MaybeNonExplicitGmType>::ExplicitGraphicalModelType GmTypeExplicit;
/// GmTypeExplicit explicitGm;
/// opengm::ConvertToExplicit<MaybeNonExplicitGmType>::convert(maybeNonExplicit,explicitGm);
/// \endcode
template<class GM>
class ConvertToExplicit{
private:
    typedef typename GM::ValueType GmValueType;
    typedef typename GM::OperatorType GmOperatorTye;
    typedef typename GM::FunctionIdentifier GmFunctionIdentifier;
public:
    typedef GraphicalModel<GmValueType,GmOperatorTye,ExplicitFunction<GmValueType>,DiscreteSpace< > > ExplicitGraphicalModelType;
private:
    typedef typename ExplicitGraphicalModelType::FunctionIdentifier FunctionIdentifier;
public:
    static void convert(const GM & gm,ExplicitGraphicalModelType & explicitGm){
        DiscreteSpace< > space;
        space.reserve(gm.numberOfVariables());
        for(size_t v=0;v<gm.numberOfVariables();++v){
            space.addVariable(gm.numberOfLabels(v));
        }
        explicitGm.assign(space);
        std::map<GmFunctionIdentifier,FunctionIdentifier> fidMap;
        //convert adds all the explicit functions
        FunctionIteratation<0,GM::NrOfFunctionTypes,false>::convert(gm,explicitGm,fidMap);
        for(size_t f=0;f<gm.numberOfFactors();++f){
            const typename GM::FactorType & factor=gm[f];
            FunctionIdentifier explicitFid=fidMap[GmFunctionIdentifier(factor.functionIndex(),factor.functionType())];
            explicitGm.addFactor(explicitFid,factor.variableIndicesBegin(),factor.variableIndicesEnd());
        }
    }
};



template<unsigned int IX,unsigned int DX>
class FunctionIteratation<IX,DX,false>{
    public:
    template<class STORAGE>
    static void size( STORAGE & storage ,size_t  & neededStorage){
        
        const size_t numF=storage.functionSize_[IX];
        for(size_t f=0;f<numF;++f){
            neededStorage+=storage.gm_. template functions<IX>()[f].size();
        }
        FunctionIteratation<IX+1,DX,IX+1==DX >::size(storage,neededStorage);
    }
    
    template<class STORAGE>
    static void store( STORAGE & storage ,size_t  & currentOffset){
        // get function type
        typedef typename STORAGE::GraphicalModelType::FunctionTypeList FTypeList;
        typedef typename meta::TypeAtTypeList<FTypeList,IX>::type FunctionType;
        typedef typename FunctionType::FunctionShapeIteratorType FunctionShapeIteratorType;
        const size_t numF=storage.functionSize_[IX];
        for(size_t f=0;f<numF;++f){
            const FunctionType & function = storage.gm_. template functions<IX>()[f];
            const size_t functionSize=function.size();
            // compute function index (1. scalar index)
            const size_t fIndex=storage.fidToIndex(IX,f);
            
            OPENGM_ASSERT(fIndex<storage.functionIndexToStart_.size());
            // remember offset
            storage.functionIndexToStart_[fIndex]=currentOffset;
            // write function into memory
            ShapeWalker< FunctionShapeIteratorType > walker(function.functionShapeBegin(),function.dimension());
            for (size_t i = 0; i < functionSize; ++i) {
                OPENGM_ASSERT(currentOffset+i<storage.dataSize_);
               storage.data_[currentOffset+i]=function(walker.coordinateTuple().begin());
               ++walker;
            }
            currentOffset+=functionSize;
        }
        FunctionIteratation<IX+1,DX,IX+1==DX >::store(storage,currentOffset);
    }
    
    template<class GM,class GM_EXPLICIT,class FID_MAP>
    static void convert(const GM & gm ,GM_EXPLICIT & gmExplicit , FID_MAP & fidMap){
        // get function type
        typedef typename GM::FunctionTypeList FTypeList;
        typedef typename meta::TypeAtTypeList<FTypeList,IX>::type FunctionType;
        typedef typename FunctionType::FunctionShapeIteratorType FunctionShapeIteratorType;
        const size_t numF=gm. template functions<IX>().size();
        // loop  over all function of the type "FunctionType"
        for(size_t f=0;f<numF;++f){
            const FunctionType & function = gm. template functions<IX>()[f];
            const size_t functionSize=function.size();
            
            
            ExplicitFunction<typename GM_EXPLICIT::ValueType> explicitFunction(function.functionShapeBegin(),function.functionShapeEnd());
            ShapeWalker< FunctionShapeIteratorType > walker(function.functionShapeBegin(),function.dimension());
            for (size_t i = 0; i < functionSize; ++i) {
               explicitFunction(i)=function(walker.coordinateTuple().begin());
               ++walker;
            }
            // add function and "fid"-mapping
            typename GM::FunctionIdentifier gmFid(f,IX);
            fidMap[gmFid]=gmExplicit.addFunction(explicitFunction);
        }
        FunctionIteratation<IX+1,DX,IX+1==DX >::convert(gm,gmExplicit,fidMap);
    }
    
    
};


template<unsigned int IX,unsigned int DX>
class FunctionIteratation<IX,DX,true>{
    public:
    template<class STORAGE>
    static void size( STORAGE & storage ,size_t  & neededStorage){
        // do nothing
    }
    
    template<class STORAGE>
    static void store( STORAGE & storage ,size_t  & currentOffset){
         // do nothing
    }
    
    template<class GM,class GM_EXPLICIT,class FID_MAP>
    static void convert(const GM & gm ,GM_EXPLICIT & gmExplicit , FID_MAP & fidMap){
         // do nothing
    }
};

}

#endif //OPENGM_GRAPHICALMODEL_EXPLICIT_STORAGE_HXX
