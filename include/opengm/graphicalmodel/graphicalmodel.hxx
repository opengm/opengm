#pragma once
#ifndef OPENGM_GRAPHICALMODEL_HXX
#define OPENGM_GRAPHICALMODEL_HXX

#include <exception>
#include <set>
#include <vector>
#include <queue>
#include <string>

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

   template<class FUNCTION_TYPE>
   struct FunctionDataUnit;
}
/// \endcond 

template<class FUNCTION_INDEX_TYPE, class FUNCTION_TYPE_INDEX_TYPE>
   struct FunctionIdentification;


/// \brief GraphicalModel
///
/// Implements the graphical model interface
/// see also for factorgraph_view
///
/// \ingroup graphical_models
template<
   class T, 
   class OPERATOR, 
   class FUNCTION_TYPE_LIST = meta::TypeList<ExplicitFunction<T>, meta::ListEnd>, 
   class SPACE = opengm::DiscreteSpace<size_t, size_t>
>
class GraphicalModel
:  public FactorGraph<
      GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>,
      typename SPACE::IndexType
   > 
{
public:
   typedef GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE> GraphicalModelType;
   typedef SPACE SpaceType;
   typedef typename SpaceType::IndexType IndexType;
   typedef typename SpaceType::LabelType LabelType;
   typedef T ValueType;
      
   typedef typename meta::GenerateFunctionTypeList<
      FUNCTION_TYPE_LIST, 
      ExplicitFunction<T,IndexType,LabelType>,false // refactor me
   >::type FunctionTypeList;
      
   enum FunctionInformation{
      NrOfFunctionTypes = meta::LengthOfTypeList<FunctionTypeList>::value
   };
      
   typedef FunctionIdentification<IndexType, UInt8Type> FunctionIdentifier;
   typedef IndependentFactor<ValueType, IndexType, LabelType> IndependentFactorType; 
   typedef Factor<GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE> > FactorType;
   typedef OPERATOR OperatorType; 


   GraphicalModel();
   GraphicalModel(const GraphicalModel&);
   GraphicalModel(const SpaceType& ,const size_t reserveFactorsPerVariable=0);
   GraphicalModel& operator=(const GraphicalModel&);

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

   template<class ITERATOR>
      IndexType addFactorNonFinalized(const FunctionIdentifier&, ITERATOR, ITERATOR);

   void finalize();

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

   void reserveFactorsVarialbeIndices(const size_t size){
      factorsVis_.reserve(size);
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
   std::vector<IndexType>  factorsVis_;
   IndexType order_;


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
template<typename, typename, typename , typename >
   friend class GraphicalModel;
template <size_t , size_t, bool >
   friend struct opengm::functionwrapper::executor::FactorInvariant;
template<unsigned int I,unsigned int D,bool END>
    friend class  FunctionIteratation;
template<class GM>
    friend class ExplicitStorage;
};


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
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::numberOfFactors
(
   const IndexType variableIndex
) const {
   OPENGM_ASSERT(variableIndex < numberOfVariables());
   return variableFactorAdjaceny_[variableIndex].size();
}
   
/// \brief return the order (number of variables) of a specific factor
/// \sa FactorGraph
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::numberOfVariables
(
   const IndexType factorIndex
) const 
{
   OPENGM_ASSERT(factorIndex < numberOfFactors());
   return factors_[factorIndex].numberOfVariables();
}
   
/// \brief return the number of functions of a specific type
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::numberOfFunctions
(
   const size_t functionTypeIndex
) const 
{
   typedef meta::SizeT<GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::NrOfFunctionTypes> NoFt;
   return detail_graphical_model::FunctionWrapper<NoFt::value>::numberOfFunctions(this, functionTypeIndex);
}
   
/// \brief return the k-th variable of the j-th factor
/// \sa FactorGraph
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::variableOfFactor
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
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::factorOfVariable
(
   const IndexType variableIndex, 
   const IndexType factorNumber
) const 
{
   OPENGM_ASSERT(variableIndex < numberOfVariables());
   OPENGM_ASSERT(factorNumber < numberOfFactors(variableIndex));
   return variableFactorAdjaceny_[variableIndex][factorNumber];
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::GraphicalModel()
:  space_(), 
   functionDataField_(), 
   variableFactorAdjaceny_(), 
   factors_(0, FactorType(this)),
   factorsVis_(),
   order_(0)
{
   //this->assignGm(this);    
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::GraphicalModel
(
   const GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>& gm
)
:  space_(gm.space_), 
   functionDataField_(gm.functionDataField_), 
   variableFactorAdjaceny_(gm.variableFactorAdjaceny_), 
   factors_(gm.numberOfFactors()),
   factorsVis_(gm.factorsVis_),
   order_(gm.factorOrder())
{
   for(size_t i = 0; i<this->factors_.size(); ++i) {
      factors_[i].gm_=this;
      factors_[i].functionIndex_=gm.factors_[i].functionIndex_;
      factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_;
      //factors_[i].order_=gm.factors_[i].order_;
      //factors_[i].indexInVisVector_=gm.factors_[i].indexInVisVector_;
      factors_[i].vis_=gm.factors_[i].vis_;
      factors_[i].vis_.assignPtr(this->factorsVis_);
   }
   //this->assignGm(this);
   //this->initializeFactorFunctionAdjacency();
}
   


/// \brief construct a graphical model based on a label space
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::GraphicalModel
(
   const SpaceType& space,
   const size_t reserveFactorsPerVariable
)
:  space_(space), 
   functionDataField_(), 
   variableFactorAdjaceny_(space.numberOfVariables()), 
   factors_(0, FactorType(this)),
   order_(0)
{  
   if(reserveFactorsPerVariable==0){
      variableFactorAdjaceny_.resize(space.numberOfVariables());
   }
   else{
      RandomAccessSet<IndexType> reservedSet;
      reservedSet.reserve(reserveFactorsPerVariable);
      variableFactorAdjaceny_.resize(space.numberOfVariables(),reservedSet);
   }
   //this->assignGm(this);
}
/// \brief add a new variable to the graphical model and underlying label space
/// \return index of the newly added variable
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::addVariable
(
   const IndexType nLabels
) 
{
   space_.addVariable(nLabels);
   variableFactorAdjaceny_.push_back(RandomAccessSet<size_t>());
   return space_.numberOfVariables() - 1;    
}

/// \brief clear the graphical model and construct a new one based on a label space
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline void
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::assign
(
   const SPACE& space
) 
{
   (*this) = GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>(space);
   //this->assignGm(this);
}

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::numberOfVariables() const 
{
   return space_.numberOfVariables();
}

/// \brief return the number of labels of an indicated variable
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::numberOfLabels
(
   const IndexType index
) const 
{
   OPENGM_ASSERT(index < this->numberOfVariables());
   return space_.numberOfLabels(index);
}

/// \brief access a factor of the graphical model
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline const typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::FactorType&
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::operator[]
(
   const IndexType index
) const 
{
   OPENGM_ASSERT(index < this->numberOfFactors());
   return factors_[index];
}

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::numberOfFactors() const 
{
   return this->factors_.size();
}

/// \brief return the label space underlying the graphical model
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline const SPACE&
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::space() const 
{
   return this->space_;
}

/// \brief evaluate the modeled function for a given labeling
/// \param labels iterator to the beginning of a sequence of label indices
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class ITERATOR>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::ValueType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::evaluate
(
   ITERATOR labels
) const 
{
   ValueType v;
   //std::vector<LabelType> factor_state(numberOfVariables()+1);
   std::vector<LabelType> factor_state(factorOrder()+1);
   OperatorType::neutral(v);
   for(size_t j = 0; j < factors_.size(); ++j) {
      //size_t nvar = factors_[j].numberOfVariables();
      //if(factors_[j].numberOfVariables() == 0) {
      //   nvar = 1;
      //};
      //factor_state.resize(nvar, static_cast<LabelType> (0));
      factor_state[0]=0;
      for(size_t i = 0; i < factors_[j].numberOfVariables(); ++i) {
         // OPENGM_ASSERT_OP( static_cast<LabelType>(labels[factors_[j].variableIndex(i)]) 
         //   ,< ,static_cast<LabelType>(factors_[j].numberOfLabels(i)));
         factor_state[i] = labels[factors_[j].variableIndex(i)];
      }
      OperatorType::op(factors_[j](factor_state.begin()), v);
   }
   return v;
}

/// \param begin iterator to the beginning of a sequence of label indices
/// \param begin iterator to the end of a sequence of label indices
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class ITERATOR>
inline bool
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::isValidIndexSequence
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
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline size_t
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::factorOrder() const 
{
   for(size_t i = 0; i < numberOfFactors(); i++) {
      OPENGM_ASSERT(factors_[i].numberOfVariables()<=order_);
   } 
   return order_;
/*
   size_t factorOrder = 0;
   for(size_t i = 0; i < numberOfFactors(); i++) {
      if(factors_[i].numberOfVariables() > factorOrder)
         factorOrder = factors_[i].numberOfVariables();
   }
   return factorOrder;
*/
}

/// \brief add a function to the graphical model
/// \param function a copy of function is stored in the model
/// \return the identifier of the new function that can be used e.g. with the function addFactor
/// \sa addFactor
/// \sa getFunction
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class FUNCTION_TYPE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::FunctionIdentifier
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::addFunction
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
   //this-> template addFunctionToAdjacency < TLIndex::value > ();
   return functionIdentifier;
}
   

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class FUNCTION_TYPE>
inline std::pair<typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::FunctionIdentifier,FUNCTION_TYPE &> 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::addFunctionWithRefReturn
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
   //this-> template addFunctionToAdjacency < TLIndex::value > ();
   std::pair<FunctionIdentifier,FUNCTION_TYPE &> fidFunction(functionIdentifier,this-> template functions<TLIndex::value>().back());
   return fidFunction;
}


/// \brief add a function to the graphical model avoiding duplicates (requires search)
/// \return the identifier of the function that can be used e.g. with the function addFactor
/// \sa addFactor
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class FUNCTION_TYPE>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::FunctionIdentifier
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::addSharedFunction
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
   //this-> template addFunctionToAdjacency < TLIndex::value > ();
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
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class FUNCTION_TYPE>
FUNCTION_TYPE& 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::getFunction
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
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class ITERATOR>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::addFactor
(
   const FunctionIdentifier& functionIdentifier, 
   ITERATOR begin, 
   ITERATOR end
) 
{
  
   const IndexType indexInVisVector = factorsVis_.size();
   IndexType factorOrder = 0;
   while(begin!=end){
      factorsVis_.push_back(*begin);
      ++begin;
      ++factorOrder;
   }
   order_ = std::max(order_,factorOrder);

   // create factor
   //FactorType factor();
   const IndexType factorIndex = this->factors_.size();
   this->factors_.push_back(FactorType(this, functionIdentifier.functionIndex, functionIdentifier.functionType , factorOrder, indexInVisVector));
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
   //this->addFactorToAdjacency(functionIdentifier.functionIndex, factorIndex, functionIdentifier.functionType);
   //this->factors_[factorIndex].testInvariant();
   return factorIndex;
}
   



/// \brief add a factor to the graphical model
/// \param functionIdentifier identifier of the underlying function, cf. addFunction
/// \param begin iterator to the beginning of a sequence of variable indices
/// \param end iterator to the end of a sequence of variable indices
/// \sa addFunction
/// 
///  IF FACTORS ARE ADDED WITH THIS FUNCTION , gm.finalize() needs to be called
///
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<class ITERATOR>
inline typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::IndexType
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::addFactorNonFinalized
(
   const FunctionIdentifier& functionIdentifier, 
   ITERATOR begin, 
   ITERATOR end
) 
{


   const IndexType indexInVisVector = factorsVis_.size();
   IndexType factorOrder = 0;
   while(begin!=end){
      factorsVis_.push_back(*begin);
      ++begin;
      ++factorOrder;
   }
   order_ = std::max(order_,factorOrder);


   // create factor
   //FactorType factor();
   const IndexType factorIndex = this->factors_.size();
   this->factors_.push_back(FactorType(this, functionIdentifier.functionIndex, functionIdentifier.functionType , factorOrder, indexInVisVector));

   for(size_t i=0;i<factors_.back().numberOfVariables();++i) {
      const FactorType factor =factors_.back();
      if(i!=0){
         OPENGM_CHECK_OP(factor.variableIndex(i-1),<,factor.variableIndex(i),
            "variable indices of a factor must be sorted");
      }
      OPENGM_CHECK_OP(factor.variableIndex(i),<,this->numberOfVariables(),
         "variable indices of a factor must smaller than gm.numberOfVariables()");
      //this->variableFactorAdjaceny_[factor.variableIndex(i)].insert(factorIndex);
      //++begin;
   }
   //this->addFactorToAdjacency(functionIdentifier.functionIndex, factorIndex, functionIdentifier.functionType);
   //this->factors_[factorIndex].testInvariant();
   return factorIndex;
}
   

template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
void 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::finalize(){

   std::vector<std::set<IndexType> >  variableFactorAdjaceny(this->numberOfVariables());
   for(IndexType fi=0; fi < this->numberOfFactors();++fi){

      const FactorType & factor = factors_[fi];
      const IndexType numVar =  factor.numberOfVariables();
      for(IndexType v=0;v<numVar;++v){
         const IndexType vi=factor.variableIndex(v);
         variableFactorAdjaceny[vi].insert(fi);
      }
   }

   for(IndexType vi=0;vi<this->numberOfVariables();++vi){
      this->variableFactorAdjaceny_[vi].assignFromSet(variableFactorAdjaceny[vi]);
   }
}


template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
inline GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>&
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::operator=
(
   const GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>& gm
) {
   if(this!=&gm) {
      this->space_ = gm.space_;
      this->functionDataField_=gm.functionDataField_;
      this->factors_.resize(gm.factors_.size());
      this->variableFactorAdjaceny_=gm.variableFactorAdjaceny_;    
      this->factorsVis_ = gm.factorsVis_; 
      this->order_ = gm.order_;
      for(size_t i = 0; i<this->factors_.size(); ++i) {  
         factors_[i].gm_=this;
         factors_[i].functionIndex_=gm.factors_[i].functionIndex_;
         factors_[i].functionTypeId_=gm.factors_[i].functionTypeId_;
         factors_[i].vis_=gm.factors_[i].vis_;
         factors_[i].vis_.assignPtr(this->factorsVis_);
      }
      //this->assignGm(this);
      //this->initializeFactorFunctionAdjacency();
   }
   return *this;
}
   
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<size_t FUNCTION_INDEX>
const std::vector<  
   typename meta::TypeAtTypeList<
      typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::FunctionTypeList, FUNCTION_INDEX
   >::type 
>& 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::functions() const 
{
   return meta::FieldAccess::template byIndex<FUNCTION_INDEX>
      (this->functionDataField_).functionData_.functions_;
}
   
template<class T, class OPERATOR, class FUNCTION_TYPE_LIST, class SPACE>
template<size_t FUNCTION_INDEX>
std::vector<  
   typename meta::TypeAtTypeList<
      typename GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::FunctionTypeList, 
      FUNCTION_INDEX
   >::type 
>& 
GraphicalModel<T, OPERATOR, FUNCTION_TYPE_LIST, SPACE>::functions() 
{
   return meta::FieldAccess::template byIndex<FUNCTION_INDEX>
      (this->functionDataField_).functionData_.functions_;
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
   const FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE> & rhs
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
   const FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE> & rhs
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
   const FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE> & rhs
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
   const FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE> & rhs
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
   const FunctionIdentification<FUNCTION_INDEX_TYPE, FUNCTION_TYPE_INDEX_TYPE> & rhs
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

   // template<class T, class INDEX_TYPE>
   //struct FunctionAdjacencyData {
   //   std::vector<RandomAccessSet<INDEX_TYPE> > functionFactorAdjacencies_;
   //};

   template<class FUNCTION_TYPE>
   struct FunctionDataUnit{
      FunctionData<FUNCTION_TYPE> functionData_;
   };

   //template<class FUNCTION_TYPE, class INDEX_TYPE>
   //struct FunctionAdjacencyDataUnit{
   //   FunctionAdjacencyData<FUNCTION_TYPE, INDEX_TYPE> functionAdjacencyData_;
   //};
} // namespace detail_graphical_model
/// \endcond

} //namespace opengm

#endif // #ifndef OPENGM_GRAPHICALMODEL_HXX
