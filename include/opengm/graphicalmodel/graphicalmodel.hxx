#pragma once
#ifndef OPENGM_GRAPHICALMODEL_HXX
#define OPENGM_GRAPHICALMODEL_HXX

#include <exception>
#include <set>
#include <vector>
#include <queue>

#include "opengm/opengm.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/datastructures/randomaccessset.hxx"
#include "opengm/graphicalmodel/graphicalmodel_function_wrapper.hxx"
#include "opengm/graphicalmodel/graphicalmodel_explicit_storage.hxx"
#include "opengm/graphicalmodel/graphicalmodel_factor.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/graphicalmodel/graphviews/factorgraph.hxx"
#include "opengm/utilities/accessor_iterator.hxx"
#include "opengm/utilities/shape_accessor.hxx"
#include "opengm/utilities/metaprogramming.hxx"

namespace opengm {

namespace hdf5 {
   template<class GM>
      void save(const GM&, const std::string&, const std::string&);
   template<class GM_>
      void load(GM_& gm, const std::string&, const std::string&);
   template<class, size_t, size_t, bool>
      struct SaveAndLoadFunctions;
}

   template<unsigned int I,unsigned int D,bool END>
      class  FunctionIteratation;
/// \cond HIDDEN_SYMBOLS
namespace detail_graphical_model {
   template<class FUNCTION_TYPE>
      struct FunctionData;
   template<class T, class INDEX_TYPE>
      struct FunctionAdjacencyData;
   template<class FUNCTION_TYPE>
      struct FunctionDataUnit;
   template<class FUNCTION_TYPE, class INDEX_TYPE>
      struct FunctionAdjacencyDataUnit;
}
/// \endcond 

template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
   struct FunctionIdentification;

/// \cond HIDDEN_SYMBOLS
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
   class GraphicalModelEdit;
/// \endcond 

/// \brief GraphicalModel
///
/// \ingroup graphical_models
template<
   class T, 
   class OPERATOR, 
   class FUNCTION_TYPE_LIST = meta::TypeList<ExplicitFunction<T>, meta::ListEnd>, 
   class SPACE = opengm::DiscreteSpace<size_t, size_t>, 
   bool EDITABLE = false
>
class GraphicalModel
:  public GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>, 
   public FactorGraph<
      GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>,
      typename SPACE::IndexType
   > 
{
public:
   typedef GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE> GraphicalModelType;
   typedef SPACE SpaceType;
   typedef typename SpaceType::IndexType IndexType;
   typedef typename SpaceType::LabelType LabelType;
   typedef T ValueType;
      
   typedef typename meta::GenerateFunctionTypeList<
      FUNCTION_TYPE_LIST, 
      ExplicitFunction<T,IndexType,LabelType>, 
      EDITABLE
   >::type FunctionTypeList;
      
   enum FunctionInformation{
      NrOfFunctionTypes = meta::LengthOfTypeList<FunctionTypeList>::value
   };
      
   typedef FunctionIdentification<IndexType, UInt8Type> FunctionIdentifier;
   typedef IndependentFactor<ValueType, IndexType, LabelType> IndependentFactorType; 
   typedef Factor<GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE> > FactorType;
   typedef OPERATOR OperatorType; 
      
   /// \cond HIDDEN_SYMBOLS
   template<bool M>
   struct Rebind {
      typedef GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, M> RebindType;
   };
   /// \endcond

   GraphicalModel();
   GraphicalModel(const GraphicalModel&);
   template<class FUNCTION_TYPE_LIST_OTHER, bool IS_EDITABLE>
   GraphicalModel(const GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST_OTHER, SPACE, IS_EDITABLE>&);
   GraphicalModel(const SpaceType& ,const size_t reserveFactorsPerVariable=0);
   GraphicalModel& operator=(const GraphicalModel&);
   template<class FUNCTION_TYPE_LIST_OTHER, bool IS_EDITABLE>
   GraphicalModel& operator=(const GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST_OTHER, SPACE, IS_EDITABLE>&);

   const SpaceType& space() const;
   IndexType numberOfVariables() const;
   IndexType numberOfVariables(const IndexType) const;
   IndexType numberOfLabels(const IndexType) const;
   IndexType numberOfFunctions(const size_t) const;
   IndexType numberOfFactors() const;
   IndexType numberOfFactors(const IndexType) const;
   IndexType variableOfFactor(const IndexType, const IndexType) const;
   IndexType factorOfVariable(const IndexType, const IndexType) const;
   const FactorType& operator[](const IndexType) const;
   template<class ITERATOR>
      ValueType evaluate(ITERATOR) const;
   /// \cond HIDDEN_SYMBOLS
   template<class ITERATOR>
      bool isValidIndexSequence(ITERATOR, ITERATOR) const;
   /// \endcond
   size_t factorOrder() const;

   void assign(const SpaceType& );
   IndexType addVariable(const IndexType); 
   template<class FUNCTION_TYPE>
      FunctionIdentifier addFunction(const FUNCTION_TYPE&);
   template<class FUNCTION_TYPE>
      std::pair<FunctionIdentifier,FUNCTION_TYPE &> addFunctionWithRefReturn(const FUNCTION_TYPE&);
   template<class FUNCTION_TYPE>
      FunctionIdentifier addSharedFunction(const FUNCTION_TYPE&);
   template<class FUNCTION_TYPE>
      FUNCTION_TYPE& getFunction(const FunctionIdentifier&);
   template<class ITERATOR>
      IndexType addFactor(const FunctionIdentifier&, ITERATOR, ITERATOR);

   // reserve stuff
   template <class FUNCTION_TYPE>
   void reserveFunctions(const size_t numF){
         typedef meta::SizeT<
            meta::GetIndexInTypeList<
               FunctionTypeList, 
               FUNCTION_TYPE
            >::value
         > TLIndex;
         this-> template functions<TLIndex::value>().reserve(numF);
   }
   
   void reserveFactors(const size_t numF){
      factors_.reserve(numF);
   }
   
protected:
   template<size_t FUNCTION_INDEX>
      const std::vector<typename meta::TypeAtTypeList<FunctionTypeList, FUNCTION_INDEX>::type>& functions() const;
   template<size_t FUNCTION_INDEX>
      std::vector<typename meta::TypeAtTypeList<FunctionTypeList, FUNCTION_INDEX>::type>& functions();

private:
   SpaceType space_;
   meta::Field<FunctionTypeList, detail_graphical_model::FunctionDataUnit> functionDataField_;
   std::vector<RandomAccessSet<IndexType> > variableFactorAdjaceny_;
   std::vector<FactorType> factors_;

template<typename, typename, typename , typename , bool>  
   friend class GraphicalModelEdit;
template<size_t>
   friend struct detail_graphical_model::FunctionWrapper;
template<size_t, size_t , bool>
   friend struct detail_graphical_model::FunctionWrapperExecutor;
template<typename GM>
   friend void opengm::hdf5::save(const GM&, const std::string&, const std::string&);
template<typename GM>
   friend void opengm::hdf5::load(GM&, const std::string&, const std::string&);

template<class , size_t , size_t , bool>
   friend struct opengm::hdf5::SaveAndLoadFunctions;
template<typename, typename>
   friend struct GraphicalModelEqualityTest;
template<typename, typename, typename >
   friend class IndependentFactor;
template<typename>
   friend class Factor;
template<typename, typename, typename , typename , bool>
   friend class GraphicalModel;
template <size_t , size_t, bool >
   friend struct opengm::functionwrapper::executor::FactorInvariant;
template<unsigned int I,unsigned int D,bool END>
    friend class  FunctionIteratation;
template<class GM>
    friend class ExplicitStorage;
};

/// \cond HIDDEN_SYMBOLS
/// Interface to change a graphical model
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
class GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>
{
public:
   typedef GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true> HostGmType;

   GraphicalModelEdit();
   template<class IteratorVi>
      void replaceFactor(const size_t, size_t, IteratorVi, IteratorVi);
   void isolateFactor(const size_t);
   template<class INDEX_ITERATOR, class VALUE_ITERATOR>
      void introduceEvidence(INDEX_ITERATOR, INDEX_ITERATOR, VALUE_ITERATOR);
   template<class INDEX_ITERATOR, class STATE_ITERATOR>
      void fixVariables(size_t, INDEX_ITERATOR, INDEX_ITERATOR, STATE_ITERATOR); 

protected:
   template<size_t FUNCTION_INDEX>
      void addFunctionToAdjacency();
   void addFactorToAdjacency(const size_t , const size_t, const size_t);
   void assignGm(HostGmType*);
   void initializeFactorFunctionAdjacency(); 
      
   HostGmType* gm_;
      
   template<size_t FUNCTION_INDEX>
      std::vector<RandomAccessSet<typename SPACE::IndexType> >& factorFunctionAdjacencies();
   template<size_t FUNCTION_INDEX>
      const std::vector<RandomAccessSet<typename SPACE::IndexType> >& factorFunctionAdjacencies() const;

private:     
   typedef typename meta::GenerateFunctionTypeList<
      FUNCTION_TYPE_LIST, 
      opengm::ExplicitFunction<T,typename SPACE::IndexType,typename SPACE::LabelType>, 
      true
   >::type FTL;

   meta::Field2<
      FTL, 
      typename SPACE::IndexType, 
      detail_graphical_model::FunctionAdjacencyDataUnit
   > functionAdjacencyDataField_;
      
template<size_t>
   friend struct detail_graphical_model::FunctionWrapper;
template<size_t, size_t , bool>
   friend struct detail_graphical_model::FunctionWrapperExecutor;
};
   
/// Interface to change a graphical model
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
class GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, false> {
public:
   typedef GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, false> HostGmType;

   GraphicalModelEdit();     

protected:
   template<size_t FUNCTION_INDEX>
      void addFunctionToAdjacency();
   void addFactorToAdjacency(const size_t, const size_t, const size_t);
   void assignGm(HostGmType * );
   void initializeFactorFunctionAdjacency(); 
      
template<size_t>
   friend struct detail_graphical_model::FunctionWrapper;
template<size_t, size_t , bool>
   friend struct detail_graphical_model::FunctionWrapperExecutor;
};
/// \endcond

/// \cond HIDDEN_SYMBOLS
template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
struct FunctionIdentification {
   typedef FUNCTION_INDEX_TYPE FunctionIndexType;
   typedef FunctionIndexType IndexType;
   typedef FUNCTION_TYPE_INDEX_TYPE FunctionTypeIndexType;

   FunctionIdentification(const FunctionIndexType=FunctionIndexType(0), const FunctionTypeIndexType=FunctionTypeIndexType(0));
   bool operator <  (const FunctionIdentification& ) const;
   bool operator >  (const FunctionIdentification& ) const;
   bool operator <= (const FunctionIdentification& ) const;
   bool operator >= (const FunctionIdentification& ) const;
   bool operator == (const FunctionIdentification& ) const;

   FunctionTypeIndexType getFunctionType()const{return functionType;};
   FunctionIndexType getFunctionIndex()const{return functionIndex;};

   FunctionIndexType functionIndex;
   FunctionTypeIndexType functionType;
};
/// \endcond

/// \brief return the order (number of factors) connected to a specific variable
/// \sa FactorGraph
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::numberOfFactors
(
   const IndexType variableIndex
) const {
   OPENGM_ASSERT(variableIndex < numberOfVariables());
   return variableFactorAdjaceny_[variableIndex].size();
}
   
/// \brief return the order (number of variables) of a specific factor
/// \sa FactorGraph
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::numberOfVariables
(
   const IndexType factorIndex
) const 
{
   OPENGM_ASSERT(factorIndex < numberOfFactors());
   return factors_[factorIndex].numberOfVariables();
}
   
/// \brief return the number of functions of a specific type
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::numberOfFunctions
(
   const size_t functionTypeIndex
) const 
{
   typedef meta::SizeT<GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::NrOfFunctionTypes> NoFt;
   return detail_graphical_model::FunctionWrapper<NoFt::value>::numberOfFunctions(this, functionTypeIndex);
}
   
/// \brief return the k-th variable of the j-th factor
/// \sa FactorGraph
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::variableOfFactor
(
   const IndexType factorIndex, 
   const IndexType variableNumber
) const 
{
   OPENGM_ASSERT(factorIndex < numberOfFactors());
   OPENGM_ASSERT(variableNumber < numberOfVariables(factorIndex));
   return factors_[factorIndex].variableIndex(variableNumber);
}
   
/// \brief return the k-th factor connected to the j-th variable
/// \sa FactorGraph
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::factorOfVariable
(
   const IndexType variableIndex, 
   const IndexType factorNumber
) const 
{
   OPENGM_ASSERT(variableIndex < numberOfVariables());
   OPENGM_ASSERT(factorNumber < numberOfFactors(variableIndex));
   return variableFactorAdjaceny_[variableIndex][factorNumber];
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::GraphicalModel()
:  GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>(), 
   space_(), 
   functionDataField_(), 
   variableFactorAdjaceny_(), 
   factors_(0, FactorType(this)) 
{
   this->assignGm(this);    
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::GraphicalModel
(
   const GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>& gm
)
:  GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>(), 
   space_(gm.space_), 
   functionDataField_(gm.functionDataField_), 
   variableFactorAdjaceny_(gm.variableFactorAdjaceny_), 
   factors_(gm.numberOfFactors()) 
{
   for(size_t i = 0; i<this->factors_.size(); ++i) {
      factors_[i].gm_=this;
      factors_[i].functionIndex_=gm.factors_[i].functionIndex_;
      factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_;
      factors_[i].variableIndices_=gm.factors_[i].variableIndices_;
   }
   this->assignGm(this);
   this->initializeFactorFunctionAdjacency();
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class FUNCTION_TYPE_LIST_OTHER, bool IS_EDITABLE>
inline GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::GraphicalModel
(
   const GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST_OTHER, SPACE, IS_EDITABLE>& gm
)
:  GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>(), 
   space_(gm.space_), 
   //functionDataField_(gm.functionDataField_), 
   variableFactorAdjaceny_(gm.variableFactorAdjaceny_), 
   factors_(gm.numberOfFactors()) 
{
   typedef GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST_OTHER, SPACE, IS_EDITABLE> OtherGmType;
   if(meta::HasTypeInTypeList<typename OtherGmType::FunctionTypeList, opengm::ExplicitFunction<T,IndexType,LabelType> >::value==false) {
      for(size_t i = 0; i<this->factors_.size(); ++i) {  
         factors_[i].gm_=this;
         factors_[i].functionIndex_=gm.factors_[i].functionIndex_;
         factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_;
         factors_[i].variableIndices_=gm.factors_[i].variableIndices_;
      }
   }
   else{
      typedef typename meta::SizeT<
         meta::GetIndexInTypeListSafely<
            typename OtherGmType::FunctionTypeList, 
            opengm::ExplicitFunction<T,IndexType,LabelType>, 
            OtherGmType::NrOfFunctionTypes
            >::value
      > ExplicitFunctionPosition;
      OPENGM_ASSERT(static_cast<size_t>(ExplicitFunctionPosition::value)<static_cast<size_t>(OtherGmType::NrOfFunctionTypes));
      for(size_t i = 0; i<this->factors_.size(); ++i) {  
         factors_[i].gm_=this;
         const size_t typeId=gm.factors_[i].functionTypeId_;
         if(typeId<ExplicitFunctionPosition::value) {
            factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_;
         }
         else if(typeId==ExplicitFunctionPosition::value) {
            factors_[i].functionTypeId_=NrOfFunctionTypes-1;
         }
         else{
            factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_-1;
         }         
         factors_[i].functionIndex_=gm.factors_[i].functionIndex_;
         factors_[i].variableIndices_=gm.factors_[i].variableIndices_;
      }
   }
   detail_graphical_model::FunctionWrapper<OtherGmType::NrOfFunctionTypes>::assignFunctions(gm, *this);
   this->assignGm(this);
   this->initializeFactorFunctionAdjacency();
   if(!NO_DEBUG) {
      try{
         for(size_t i = 0; i<this->factors_.size(); ++i) {  
            this->factors_[i].testInvariant();
         }
      }
      catch(...) {
         throw RuntimeError("Construction Failed");
      }
   }
}

/// \brief construct a graphical model based on a label space
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::GraphicalModel
(
   const SpaceType& space,
   const size_t reserveFactorsPerVariable
)
:  GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>(), 
   space_(space), 
   functionDataField_(), 
   variableFactorAdjaceny_(space.numberOfVariables()), 
   factors_(0, FactorType(this)) 
{  
   if(reserveFactorsPerVariable==0){
      variableFactorAdjaceny_.resize(space.numberOfVariables());
   }
   else{
      RandomAccessSet<IndexType> reservedSet;
      reservedSet.reserve(reserveFactorsPerVariable);
      variableFactorAdjaceny_.resize(space.numberOfVariables(),reservedSet);
   }
   this->assignGm(this);
}
/// \brief add a new variable to the graphical model and underlying label space
/// \return index of the newly added variable
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::addVariable
(
   const IndexType nLabels
) 
{
   space_.addVariable(nLabels);
   variableFactorAdjaceny_.push_back(RandomAccessSet<size_t>());
   return space_.numberOfVariables() - 1;    
}

/// \brief clear the graphical model and construct a new one based on a label space
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline void
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::assign
(
   const SPACE& space
) 
{
   (*this) = GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>(space);
   this->assignGm(this);
}

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::numberOfVariables() const 
{
   return space_.numberOfVariables();
}

/// \brief return the number of labels of an indicated variable
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::numberOfLabels
(
   const IndexType index
) const 
{
   OPENGM_ASSERT(index < this->numberOfVariables());
   return space_.numberOfLabels(index);
}

/// \brief access a factor of the graphical model
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline const typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::FactorType&
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::operator[]
(
   const IndexType index
) const 
{
   OPENGM_ASSERT(index < this->numberOfFactors());
   return factors_[index];
}

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::numberOfFactors() const 
{
   return this->factors_.size();
}

/// \brief return the label space underlying the graphical model
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline const SPACE&
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::space() const 
{
   return this->space_;
}

/// \brief evaluate the modeled function for a given labeling
/// \param labelIndices iterator to the beginning of a sequence of label indices
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class ITERATOR>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::ValueType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::evaluate
(
   ITERATOR labelIndices
) const 
{
   ValueType v;
   OperatorType::neutral(v);
   for(size_t j = 0; j < factors_.size(); ++j) {
      size_t nvar = factors_[j].numberOfVariables();
      if(factors_[j].numberOfVariables() == 0) {
         nvar = 1;
      };
      std::vector<size_t> factor_state(nvar, static_cast<size_t> (0));
      for(size_t i = 0; i < factors_[j].numberOfVariables(); ++i) {
         OPENGM_ASSERT( static_cast<LabelType>(labelIndices[factors_[j].variableIndex(i)]) 
            < static_cast<LabelType>(factors_[j].numberOfLabels(i)));
         factor_state[i] = labelIndices[factors_[j].variableIndex(i)];
      }
      OperatorType::op(factors_[j](factor_state.begin()), v);
   }
   return v;
}

/// \param begin iterator to the beginning of a sequence of label indices
/// \param begin iterator to the end of a sequence of label indices
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class ITERATOR>
inline bool
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::isValidIndexSequence
(
   ITERATOR begin, 
   ITERATOR end
) const 
{
   ITERATOR previousIt = begin;
   while(begin != end) {
      if(*begin >= this->numberOfVariables()) {
         return false;
      }
      if(previousIt != begin && *previousIt >= *begin) {
         return false;
      }
      previousIt = begin;
      ++begin;
   }
   return true;
}

/// \brief return the maximum of the orders of all factors
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline size_t
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::factorOrder() const 
{
   size_t factorOrder = 0;
   for(size_t i = 0; i < numberOfFactors(); i++) {
      if(factors_[i].numberOfVariables() > factorOrder)
         factorOrder = factors_[i].numberOfVariables();
   }
   return factorOrder;
}

/// \brief add a function to the graphical model
/// \param function a copy of function is stored in the model
/// \return the identifier of the new function that can be used e.g. with the function addFactor
/// \sa addFactor
/// \sa getFunction
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class FUNCTION_TYPE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::FunctionIdentifier
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::addFunction
(
   const FUNCTION_TYPE& function
) 
{
   // find index of FUNCTION_TYPE in Typelist
   typedef meta::SizeT<
      meta::GetIndexInTypeList<
         FunctionTypeList, 
         FUNCTION_TYPE
      >::value
   > TLIndex;
   typedef typename meta::SmallerNumber<TLIndex::value, GraphicalModelType::NrOfFunctionTypes>::type MetaBoolAssertType;
   OPENGM_META_ASSERT(MetaBoolAssertType::value, WRONG_FUNCTION_TYPE_INDEX);
   FunctionIdentifier functionIdentifier;
   functionIdentifier.functionType = TLIndex::value;
   const size_t functionIndex=this-> template functions<TLIndex::value>().size();
   functionIdentifier.functionIndex = functionIndex;
   this-> template functions<TLIndex::value>().push_back(function);
   OPENGM_ASSERT(functionIndex==this-> template functions<TLIndex::value>().size()-1);
   this-> template addFunctionToAdjacency < TLIndex::value > ();
   return functionIdentifier;
}
   

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class FUNCTION_TYPE>
inline std::pair<typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::FunctionIdentifier,FUNCTION_TYPE &> 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::addFunctionWithRefReturn
(
   const FUNCTION_TYPE& function
){
   // find index of FUNCTION_TYPE in Typelist
   typedef meta::SizeT<
      meta::GetIndexInTypeList<
         FunctionTypeList, 
         FUNCTION_TYPE
      >::value
   > TLIndex;
   typedef typename meta::SmallerNumber<TLIndex::value, GraphicalModelType::NrOfFunctionTypes>::type MetaBoolAssertType;
   OPENGM_META_ASSERT(MetaBoolAssertType::value, WRONG_FUNCTION_TYPE_INDEX);
   FunctionIdentifier functionIdentifier;
   functionIdentifier.functionType = TLIndex::value;
   const size_t functionIndex=this-> template functions<TLIndex::value>().size();
   functionIdentifier.functionIndex = functionIndex;
   this-> template functions<TLIndex::value>().push_back(function);
   OPENGM_ASSERT(functionIndex==this-> template functions<TLIndex::value>().size()-1);
   this-> template addFunctionToAdjacency < TLIndex::value > ();
   std::pair<FunctionIdentifier,FUNCTION_TYPE &> fidFunction(functionIdentifier,this-> template functions<TLIndex::value>().back());
   return fidFunction;
}


/// \brief add a function to the graphical model avoiding duplicates (requires search)
/// \return the identifier of the function that can be used e.g. with the function addFactor
/// \sa addFactor
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class FUNCTION_TYPE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::FunctionIdentifier
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::addSharedFunction
(
   const FUNCTION_TYPE& function
) 
{
   //const size_t dim=function.dimension();
   // find index of FUNCTION_TYPE in Typelist
   typedef meta::SizeT<
      meta::GetIndexInTypeList<
         FunctionTypeList, 
         FUNCTION_TYPE
      >::value
   > TLIndex;
   typedef typename meta::SmallerNumber<TLIndex::value, GraphicalModelType::NrOfFunctionTypes>::type MetaBoolAssertType;
   OPENGM_META_ASSERT(MetaBoolAssertType::value, WRONG_FUNCTION_TYPE_INDEX);
   FunctionIdentifier functionIdentifier;
   functionIdentifier.functionType = TLIndex::value;
   // search if function is already in the gm
   for(size_t i=0;i<this-> template functions<TLIndex::value>().size();++i) {
      if(function == this-> template functions<TLIndex::value>()[i]) {
         functionIdentifier.functionIndex = static_cast<IndexType>(i);
         OPENGM_ASSERT(function==this-> template functions<TLIndex::value>()[functionIdentifier.functionIndex]);
         return functionIdentifier;
      }
   } 
   functionIdentifier.functionIndex = this-> template functions<TLIndex::value>().size();
   this-> template functions<TLIndex::value>().push_back(function);
   OPENGM_ASSERT(functionIdentifier.functionIndex==this-> template functions<TLIndex::value>().size()-1);
   this-> template addFunctionToAdjacency < TLIndex::value > ();
   return functionIdentifier;
}



/// \brief access functions
///
/// For example:
/// \code
/// opengm::ExplicitFunction<double> f = gm.getFunction<opengm::ExplicitFunction<double> >(fid);
/// \endcode
/// If your function and graphical model type both depend on one or more common template parameters,
/// you may have to add the .template keyword for some compilers:
/// \code
/// opengm::ExplicitFunction<double> f = gm.template getFunction< FunctionType >(fid);
/// \endcode
/// \param functionIdentifier identifier of the underlying function, cf. addFunction
/// \sa addFunction
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class FUNCTION_TYPE>
FUNCTION_TYPE& 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::getFunction
(
   const FunctionIdentifier& fid
) 
{
   typedef meta::SizeT<
      meta::GetIndexInTypeList<
         FunctionTypeList, 
         FUNCTION_TYPE
      >::value
   > TLIndex;
   return this-> template functions<TLIndex::value>()[fid.getFunctionIndex()];
}


   
/// \brief add a factor to the graphical model
/// \param functionIdentifier identifier of the underlying function, cf. addFunction
/// \param begin iterator to the beginning of a sequence of variable indices
/// \param end iterator to the end of a sequence of variable indices
/// \sa addFunction
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class ITERATOR>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::addFactor
(
   const FunctionIdentifier& functionIdentifier, 
   ITERATOR begin, 
   ITERATOR end
) 
{
   // create factor
   //FactorType factor();
   const IndexType factorIndex = this->factors_.size();
   this->factors_.push_back(FactorType(this, functionIdentifier.functionIndex, functionIdentifier.functionType , begin, end));
   for(size_t i=0;i<factors_.back().numberOfVariables();++i) {
      const FactorType factor =factors_.back();
      if(i!=0){
         OPENGM_CHECK_OP(factor.variableIndex(i-1),<,factor.variableIndex(i),
            "variable indices of a factor must be sorted");
      }
      OPENGM_CHECK_OP(factor.variableIndex(i),<,this->numberOfVariables(),
         "variable indices of a factor must smaller than gm.numberOfVariables()");
      this->variableFactorAdjaceny_[factor.variableIndex(i)].insert(factorIndex);
      //++begin;
   }
   this->addFactorToAdjacency(functionIdentifier.functionIndex, factorIndex, functionIdentifier.functionType);
   this->factors_[factorIndex].testInvariant();
   return factorIndex;
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
inline GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>&
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::operator=
(
   const GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>& gm
) {
   if(this!=&gm) {
      this->space_ = gm.space_;
      this->functionDataField_=gm.functionDataField_;
      this->factors_.resize(gm.factors_.size());
      this->variableFactorAdjaceny_=gm.variableFactorAdjaceny_;     
      for(size_t i = 0; i<this->factors_.size(); ++i) {  
         factors_[i].gm_=this;
         factors_[i].functionIndex_=gm.factors_[i].functionIndex_;
         factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_;
         factors_[i].variableIndices_=gm.factors_[i].variableIndices_;
      }
      this->assignGm(this);
      this->initializeFactorFunctionAdjacency();
   }
   return *this;
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<class FUNCTION_TYPE_LIST_OTHER, bool IS_EDITABLE>
inline GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>&
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::operator=
(
   const GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST_OTHER, SPACE, IS_EDITABLE>& gm
) {
   if(this!=&gm) {
      this->space_ = gm.space_;
      this->factors_.resize(gm.factors_.size());
      this->variableFactorAdjaceny_=gm.variableFactorAdjaceny_;     
      typedef GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST_OTHER, SPACE, IS_EDITABLE> OtherGmType;
      if(meta::HasTypeInTypeList<typename OtherGmType::FunctionTypeList, opengm::ExplicitFunction<T,IndexType,LabelType> > ::value==false) {
         for(size_t i = 0; i<this->factors_.size(); ++i) {  
            factors_[i].gm_=this;
            factors_[i].functionIndex_=gm.factors_[i].functionIndex_;
            factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_;
            factors_[i].variableIndices_=gm.factors_[i].variableIndices_;
         }
      }
      else{
         typedef typename meta::SizeT<
            meta::GetIndexInTypeListSafely<
               typename OtherGmType::FunctionTypeList, 
               opengm::ExplicitFunction<T,IndexType,LabelType>, 
               OtherGmType::NrOfFunctionTypes
               >::value
         > ExplicitFunctionPosition;
         OPENGM_ASSERT(static_cast<size_t>(ExplicitFunctionPosition::value)<static_cast<size_t>(OtherGmType::NrOfFunctionTypes));
         for(size_t i = 0; i<this->factors_.size(); ++i) {  
            factors_[i].gm_=this;
            const size_t typeId=gm.factors_[i].functionTypeId_;
            if(typeId<ExplicitFunctionPosition::value) {
               factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_;
            }
            else if(typeId==ExplicitFunctionPosition::value) {
               factors_[i].functionTypeId_=NrOfFunctionTypes-1;
            }
            else{
               factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_-1;
            }
            factors_[i].functionIndex_=gm.factors_[i].functionIndex_;
            factors_[i].variableIndices_=gm.factors_[i].variableIndices_;
         }
      }
      detail_graphical_model::FunctionWrapper<OtherGmType::NrOfFunctionTypes>::assignFunctions(gm, *this);
      this->assignGm(this);
      this->initializeFactorFunctionAdjacency();
   }
   if(!NO_DEBUG) {
      try{
         for(size_t i = 0; i<this->factors_.size(); ++i) {  
            this->factors_[i].testInvariant();
         }
      }
      catch(...) {
         throw RuntimeError("Construction Failed");
      }
   }
   return *this;
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<size_t FUNCTION_INDEX>
const std::vector<  
   typename meta::TypeAtTypeList<
      typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::FunctionTypeList, FUNCTION_INDEX
   >::type 
>& 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::functions() const 
{
   return meta::FieldAccess::template byIndex<FUNCTION_INDEX>
      (this->functionDataField_).functionData_.functions_;
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE, bool EDITABLE>
template<size_t FUNCTION_INDEX>
std::vector<  
   typename meta::TypeAtTypeList<
      typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::FunctionTypeList, 
      FUNCTION_INDEX
   >::type 
>& 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, EDITABLE>::functions() 
{
   return meta::FieldAccess::template byIndex<FUNCTION_INDEX>
      (this->functionDataField_).functionData_.functions_;
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<size_t FUNCTION_INDEX>
inline std::vector<RandomAccessSet<typename SPACE::IndexType> >& 
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::factorFunctionAdjacencies() 
{
   return meta::FieldAccess::template byIndex<FUNCTION_INDEX>
      (this->functionAdjacencyDataField_).functionAdjacencyData_.functionFactorAdjacencies_;
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<size_t FUNCTION_INDEX>
inline const std::vector<RandomAccessSet<typename SPACE::IndexType> >& 
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::factorFunctionAdjacencies() const 
{
   return meta::FieldAccess::template byIndex<FUNCTION_INDEX>
      (this->functionAdjacencyDataField_).functionAdjacencyData_.functionFactorAdjacencies_;
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::GraphicalModelEdit()
:  functionAdjacencyDataField_() 
{
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, false>::GraphicalModelEdit() 
{}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<size_t FUNCTION_INDEX> 
inline void
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, false>::addFunctionToAdjacency() 
{}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline void
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, false>::addFactorToAdjacency
(
   const size_t i , 
   const size_t j , 
   const size_t k
) 
{}
      
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline void
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, false>::assignGm
(
   typename GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, false>::HostGmType* gm
) 
{}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline void
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, false>::initializeFactorFunctionAdjacency() 
{}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class ITERATOR>
void
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::replaceFactor
(
   const size_t factorIndex, 
   size_t explicitFunctionIndex, 
   ITERATOR begin, 
   ITERATOR end
) 
{
   typedef meta::SizeT<
      meta::Decrement<
         HostGmType::NrOfFunctionTypes
      >::value
   > ExplicitFunctionPosition;
   OPENGM_ASSERT(explicitFunctionIndex<gm_->numberOfFunctions(ExplicitFunctionPosition::value));
   OPENGM_ASSERT( size_t(std::distance(begin, end))==size_t(gm_->template functions<ExplicitFunctionPosition::value>()[explicitFunctionIndex].dimension()));
   //this->gm_->factors_[factorIndex].testInvariant();
   OPENGM_ASSERT(factorIndex<this->gm_->numberOfFactors());
   //OPENGM_ASSERT(opengm::isSorted(begin, end));
   // update the ajdacency between factors and variables
   const size_t newNumVar=std::distance(begin, end);
   const size_t oldNumVar=this->gm_->factors_[factorIndex].numberOfVariables();
   bool MustUpdateAdj=false;
   if(newNumVar==oldNumVar) {
      for(size_t i=0;i<newNumVar;++i) {
         if(begin[i]!=gm_->factors_[factorIndex].variableIndex(i)) {
            MustUpdateAdj=true;
            break;
         }
      }
   }
   else {
      MustUpdateAdj=true;
   }
   if(MustUpdateAdj==true) {
      for(size_t i=0;i<oldNumVar;++i) {
         const size_t vi=gm_->factors_[factorIndex].variableIndex(i);
         gm_->variableFactorAdjaceny_[vi].erase(factorIndex);
      }
      this->gm_->factors_[factorIndex].variableIndices_.assign(begin, end);
      for(size_t i=0;i<newNumVar;++i) {
         gm_->variableFactorAdjaceny_[begin[i]].insert(factorIndex);
      }
   }
   const size_t currentFunctionIndex = this->gm_->factors_[factorIndex].functionIndex_;
   const size_t currentFunctionType = this->gm_->factors_[factorIndex].functionTypeId_;
   size_t ei = explicitFunctionIndex;
   this->template factorFunctionAdjacencies<ExplicitFunctionPosition::value>()[ei].insert(factorIndex);
   typedef detail_graphical_model::FunctionWrapper<HostGmType::NrOfFunctionTypes> WrapperType;
   WrapperType::swapAndDeleteFunction(this->gm_, factorIndex, currentFunctionIndex, currentFunctionType, ei);
   // set the factors functionIndex and FunctionType to the new one
   gm_->factors_[factorIndex].functionIndex_ = ei;
   gm_->factors_[factorIndex].functionTypeId_ = ExplicitFunctionPosition::value;
   this-> template factorFunctionAdjacencies<ExplicitFunctionPosition::value>()[ei].insert(factorIndex);
}

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
void 
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::isolateFactor
(
   const size_t factorIndex
) {
   typedef meta::SizeT<
      meta::Decrement<
         HostGmType::NrOfFunctionTypes
      >::value
   > ExplicitFunctionPosition;
   //this->gm_->factors_[factorIndex].testInvariant();
   const size_t currentFunctionIndex = this->gm_->factors_[factorIndex].functionIndex_;
   switch (this->gm_->factors_[factorIndex].functionTypeId_) {
      case static_cast<size_t>(ExplicitFunctionPosition::value) :{
         const size_t sizeAdj = this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[currentFunctionIndex].size();
         if (sizeAdj > 1) {
            // push back the new function / a copy of the function we want to isolate
            gm_->template functions < ExplicitFunctionPosition::value > ().push_back
               (gm_->template functions < ExplicitFunctionPosition::value > ()[currentFunctionIndex]);
            this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ().push_back(RandomAccessSet<typename SPACE::IndexType > ());
            this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ().back().insert(factorIndex);
            typename HostGmType::FunctionIdentifier id;
            id.functionIndex=gm_-> template functions < ExplicitFunctionPosition::value > ().size()-1;
            id.functionType =ExplicitFunctionPosition::value;
            this->replaceFactor(
               factorIndex, id.functionIndex, 
               this->gm_->factors_[factorIndex].variableIndices_.begin(), 
               this->gm_->factors_[factorIndex].variableIndices_.end()
            );
            OPENGM_ASSERT(this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[id.functionIndex].size() == 1);
            OPENGM_ASSERT(this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[id.functionIndex].begin()[0] == factorIndex);
            OPENGM_ASSERT(gm_->factors_[factorIndex].functionIndex() == id.functionIndex);
            OPENGM_ASSERT(gm_->factors_[factorIndex].functionType() == ExplicitFunctionPosition::value);
         } else {
            OPENGM_ASSERT(this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[currentFunctionIndex].size() == 1);
            OPENGM_ASSERT(this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[currentFunctionIndex].begin()[0] == factorIndex);
         }
      }
      break;
      default:{
         // copy function
         const size_t factorDimension = gm_->factors_[factorIndex].numberOfVariables();
         if (factorDimension != 0) {
            typedef typename HostGmType::FactorType::ShapeIteratorType FactorShapeIteratorType;
            FactorShapeIteratorType factorShapeBegin = gm_->factors_[factorIndex].shapeBegin();
            FactorShapeIteratorType factorShapeEnd = gm_->factors_[factorIndex].shapeEnd();
            // push back new explicit function
            // get the function index
            const size_t newFunctionIndex = gm_-> template functions < ExplicitFunctionPosition::value > ().size();
            // push back new explicit function
            gm_-> template functions < ExplicitFunctionPosition::value > ().push_back(
            ExplicitFunction<T,typename HostGmType::IndexType,typename HostGmType::LabelType>(factorShapeBegin, factorShapeEnd));
            // push back empty adjacency
            this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ().push_back(RandomAccessSet<typename SPACE::IndexType > ());
            ExplicitFunction<T,typename HostGmType::IndexType,typename HostGmType::LabelType>& newFunction = gm_-> template functions < ExplicitFunctionPosition::value > ()[newFunctionIndex];
            // fill new function with data
            ShapeWalker< FactorShapeIteratorType > walker(factorShapeBegin, factorDimension);
            for (size_t i = 0; i < newFunction.size(); ++i) {
               newFunction(walker.coordinateTuple().begin()) =(this->gm_->factors_[factorIndex]).operator()(walker.coordinateTuple().begin());
               ++walker;
            }
            typename HostGmType::FunctionIdentifier id;
            id.functionIndex=newFunctionIndex;
            id.functionType=ExplicitFunctionPosition::value;
            this->replaceFactor
               (
               factorIndex, id.functionIndex, 
               this->gm_->factors_[factorIndex].variableIndices_.begin(), 
               this->gm_->factors_[factorIndex].variableIndices_.end()
               );
            OPENGM_ASSERT(this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[newFunctionIndex].size() == 1);
            OPENGM_ASSERT(this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[id.functionIndex].begin()[0] == factorIndex);
            OPENGM_ASSERT(this->gm_->factors_[factorIndex].functionIndex() == id.functionIndex);
            OPENGM_ASSERT(this->gm_->factors_[factorIndex].functionType() == ExplicitFunctionPosition::value);
         } 
         else {
            // push back new explicit function
            // get the function index
            const size_t newFunctionIndex = this->gm_->template functions < ExplicitFunctionPosition::value > ().size();
            // push back new explicit function
            size_t scalarIndex[] = {0};
            this->gm_->template functions < ExplicitFunctionPosition::value > ().push_back(ExplicitFunction<T,typename HostGmType::IndexType,typename HostGmType::LabelType>(gm_->factors_[factorIndex](scalarIndex)));
            // push back empty adjacency
            this-> template factorFunctionAdjacencies<ExplicitFunctionPosition::value>().push_back(RandomAccessSet<typename SPACE::IndexType > ());         
            typename HostGmType::FunctionIdentifier id;
            id.functionIndex=this->gm_->template functions < ExplicitFunctionPosition::value > ().size() - 1;
            id.functionType=ExplicitFunctionPosition::value;
            this->replaceFactor(
               factorIndex, id.functionIndex, 
               this->gm_->factors_[factorIndex].variableIndices_.begin(), 
               this->gm_->factors_[factorIndex].variableIndices_.end()
            );
            OPENGM_ASSERT(this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[newFunctionIndex].size() == 1);
            OPENGM_ASSERT(this->template factorFunctionAdjacencies < ExplicitFunctionPosition::value > ()[id.functionIndex].begin()[0] == factorIndex);
            OPENGM_ASSERT(gm_->factors_[factorIndex].functionIndex() == id.functionIndex);
            OPENGM_ASSERT(gm_->factors_[factorIndex].functionType() == ExplicitFunctionPosition::value);
         }
      }
   }
}

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class INDEX_ITERATOR, class VALUE_ITERATOR>
inline void
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::introduceEvidence
(
   INDEX_ITERATOR begin, 
   INDEX_ITERATOR end, 
   VALUE_ITERATOR value
) {
   if(opengm::isSorted(begin, end) == false) {
      std::vector<typename std::iterator_traits<INDEX_ITERATOR>::value_type > tmpIndexContainer(begin, end);
      std::vector<typename std::iterator_traits<VALUE_ITERATOR>::value_type > tmpValueContainer(tmpIndexContainer.size());
      for(size_t i = 0; i < tmpIndexContainer.size(); ++i) {
         tmpValueContainer[i] = *value;
         ++value;
      }
      opengm::doubleSort(tmpIndexContainer.begin(), tmpIndexContainer.end(), tmpValueContainer.begin());
      //OPENGM_ASSERT(opengm::isSorted(tmpIndexContainer.begin(), tmpIndexContainer.end()));
      for(size_t j = 0; j<this->gm_->numberOfFactors(); ++j) {
         this->isolateFactor(j);
         this->fixVariables(j, tmpIndexContainer.begin(), tmpIndexContainer.end(), tmpValueContainer.begin());
      }
   }
   else {
      for(size_t j = 0; j<this->gm_->numberOfFactors(); ++j) {
         this->isolateFactor(j);
         this->fixVariables(j, begin, end, value);
      }
   }
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class INDEX_ITERATOR, class STATE_ITERATOR>
void
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::fixVariables
(
   size_t factorIndex, 
   INDEX_ITERATOR beginIndex, 
   INDEX_ITERATOR endIndex, 
   STATE_ITERATOR beginStates
) 
{
   typedef meta::SizeT<
      meta::Decrement<
         HostGmType::NrOfFunctionTypes
      >::value
   > ExplicitFunctionPosition;
   //gm_->factors_[factorIndex].testInvariant();
   //this->testInvariant();
   if(gm_->factors_[factorIndex].variableIndices_.size() != 0) {         
      OPENGM_ASSERT(factorIndex < gm_->factors_.size());
      OPENGM_ASSERT(opengm::isSorted(beginIndex, endIndex));
      opengm::FastSequence<typename HostGmType::IndexType> variablesToFix;
      opengm::FastSequence<typename HostGmType::IndexType> variablesNotToFix;
      opengm::FastSequence<typename HostGmType::IndexType> positionOfVariablesToFix;
      opengm::FastSequence<typename HostGmType::LabelType> newStates;
      opengm::FastSequence<typename HostGmType::LabelType> newShape;
      // find the variables to fix
      while(beginIndex != endIndex) {
         size_t counter = 0;
         OPENGM_ASSERT(*beginIndex < this->gm_->numberOfVariables());
         if(*beginIndex>gm_->factors_[factorIndex].variableIndices_.back()) {
            break;
         }
         for(size_t i = counter; i<gm_->factors_[factorIndex].variableIndices_.size(); ++i) {
            if(*beginIndex<gm_->factors_[factorIndex].variableIndices_[i])break;
            else if(*beginIndex == gm_->factors_[factorIndex].variableIndices_[i]) {
               ++counter;
               variablesToFix.push_back(*beginIndex);
               newStates.push_back(*beginStates);
               positionOfVariablesToFix.push_back(i);
            }
         }
         ++beginIndex;
         ++beginStates;
      }
      for(size_t i = 0; i<gm_->factors_[factorIndex].variableIndices_.size(); ++i) {
         bool found = false;
         for(size_t j = 0; j < variablesToFix.size(); ++j) {
            if(variablesToFix[j] == gm_->factors_[factorIndex].variableIndices_[i]) {
               found = true;
               break;
            }
         }
         if(found == false) {
            variablesNotToFix.push_back(gm_->factors_[factorIndex].variableIndices_[i]);
            newShape.push_back(gm_->factors_[factorIndex].numberOfLabels(i));
         }
      }
      if(variablesToFix.size() != 0) {
         this->isolateFactor(factorIndex);
         OPENGM_ASSERT(this->gm_->operator[](factorIndex).functionType() == ExplicitFunctionPosition::value);
         ExplicitFunction<T,typename HostGmType::IndexType,typename HostGmType::LabelType>& factorFunction =gm_->template functions<ExplicitFunctionPosition::value>()[gm_->factors_[factorIndex].functionIndex_];
         //std::vector<LabelType> fullCoordinate(this->numberOfVariables());
         if(variablesToFix.size() == gm_->factors_[factorIndex].variableIndices_.size()) {
               ExplicitFunction<T,typename HostGmType::IndexType,typename HostGmType::LabelType> tmp(factorFunction(newStates.begin()));
            factorFunction = tmp;
            gm_->factors_[factorIndex].variableIndices_.clear();
         }
         else {
            SubShapeWalker<
               typename HostGmType::FactorType::ShapeIteratorType , 
                  opengm::FastSequence<typename HostGmType::IndexType>, 
                  opengm::FastSequence<typename HostGmType::LabelType>
               >
               subWalker(gm_->factors_[factorIndex].shapeBegin(), factorFunction.dimension(), positionOfVariablesToFix, newStates);
            ExplicitFunction<T,typename HostGmType::IndexType,typename HostGmType::LabelType> tmp(newShape.begin(), newShape.end());
            const size_t subSize = subWalker.subSize();
            subWalker.resetCoordinate();
            for(size_t i = 0; i < subSize; ++i) {
               tmp(i) = factorFunction( subWalker.coordinateTuple().begin());
               ++subWalker;
            }
            factorFunction = tmp;
            gm_->factors_[factorIndex].variableIndices_.assign(variablesNotToFix.begin(), variablesNotToFix.end());         
         }
         OPENGM_ASSERT(factorFunction.dimension()==variablesNotToFix.size());
         OPENGM_ASSERT(newShape.size()==variablesNotToFix.size());
         OPENGM_ASSERT(factorFunction.dimension()==newShape.size());
      }
   }
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<size_t FUNCTION_INDEX>
inline void 
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::addFunctionToAdjacency() 
{
   OPENGM_ASSERT(gm_!=NULL);
   this-> template factorFunctionAdjacencies<FUNCTION_INDEX>().push_back(RandomAccessSet<typename HostGmType::IndexType > ());
   OPENGM_ASSERT(this-> template factorFunctionAdjacencies<FUNCTION_INDEX>().size() ==this->gm_-> template functions<FUNCTION_INDEX>().size());
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline void 
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::addFactorToAdjacency
(
   const size_t functionIndex, 
   const size_t factorIndex, 
   const size_t functionType
) 
{
   typedef detail_graphical_model::FunctionWrapper<HostGmType::NrOfFunctionTypes> WrapperType;
   WrapperType::addFactorFunctionAdjacency(this->gm_, functionIndex, factorIndex, functionType);
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline void 
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::assignGm(HostGmType * gm) 
{
   this->gm_ = gm;
}

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline void 
GraphicalModelEdit<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE, true>::initializeFactorFunctionAdjacency() 
{
   detail_graphical_model::FunctionWrapper<HostGmType::NrOfFunctionTypes >::initializeFactorFunctionAdjacencies(gm_);
} 
   
template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
inline 
FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE>::FunctionIdentification
( 
   const FUNCTION_INDEX_TYPE functionIndex,
   const FUNCTION_TYPE_INDEX_TYPE functionType
)
:  functionIndex(functionIndex), 
   functionType(functionType) 
{}
   
template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
inline bool 
FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE>::operator < 
(
   const FunctionIdentification& rhs
) const 
{
   if(functionType < rhs.functionType)
       return true;
   else 
       return functionIndex < rhs.functionIndex;
}
   
template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
inline bool 
FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE>::operator > 
(
   const FunctionIdentification& rhs
) const 
{
   if(functionType >rhs.functionType)
       return true;
   else 
       return functionIndex > rhs.functionIndex;
}
   
template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
inline bool 
FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE>::operator <= 
(
   const FunctionIdentification& rhs
) const 
{
   if(functionType <= rhs.functionType)
       return true;
   else 
       return functionIndex <= rhs.functionIndex;
}
   
template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
inline bool 
FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE>::operator >= 
(
   const FunctionIdentification& rhs
) const 
{
   if(functionType >=rhs.functionType)
       return true;
   else 
       return functionIndex >= rhs.functionIndex;
}
   
template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
inline bool 
FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE>::operator == 
(
   const FunctionIdentification& rhs
) const
{
   return  (functionType == rhs.functionType) &&  (functionIndex == rhs.functionIndex);
}

/// \cond HIDDEN_SYMBOLS
namespace detail_graphical_model {
   template<class FUNCTION_TYPE>
   struct FunctionData {
      std::vector<FUNCTION_TYPE> functions_;
   };

   template<class T, class INDEX_TYPE>
   struct FunctionAdjacencyData {
      std::vector<RandomAccessSet<INDEX_TYPE> > functionFactorAdjacencies_;
   };

   template<class FUNCTION_TYPE>
   struct FunctionDataUnit{
      FunctionData<FUNCTION_TYPE> functionData_;
   };

   template<class FUNCTION_TYPE, class INDEX_TYPE>
   struct FunctionAdjacencyDataUnit{
      FunctionAdjacencyData<FUNCTION_TYPE, INDEX_TYPE> functionAdjacencyData_;
   };
} // namespace detail_graphical_model
/// \endcond

} //namespace opengm

#endif // #ifndef OPENGM_GRAPHICALMODEL_HXX
