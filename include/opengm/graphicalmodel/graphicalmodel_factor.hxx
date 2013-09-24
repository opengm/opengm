#pragma once
#ifndef OPENGM_GRAPHICALMODEL_FACTOR_HXX
#define OPENGM_GRAPHICALMODEL_FACTOR_HXX

#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include <numeric>
#include <map>
#include <list>
#include <set>
#include <functional>

#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/datastructures/sparsemarray/sparsemarray.hxx"
#include "opengm/opengm.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/sorting.hxx"
#include "opengm/utilities/functors.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/graphicalmodel/graphicalmodel_factor_operator.hxx"
#include "opengm/graphicalmodel/graphicalmodel_factor_accumulator.hxx"

#include "opengm/utilities/vector_view.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

template<
   class T, 
   class OPERATOR, 
   class FUNCTION_TYPE_LIST, 
   class SPACE
> class GraphicalModel;

template<class GRAPHICAL_MODEL> class Factor;

namespace hdf5 {
   template<class GM>
      void save(const GM&, const std::string&, const std::string&);
   template<class GM_>
      void load(GM_& gm, const std::string&, const std::string&);
}

namespace functionwrapper {
   namespace executor {
      template<size_t IX, size_t DX, bool END>
         struct FactorInvariant;
      template<size_t IX, size_t DX>
         struct FactorInvariant<IX, DX, false> {
            template<class GM, class FACTOR>
            void static op(const GM &, const FACTOR &);
         };
      template<size_t IX, size_t DX>
         struct FactorInvariant<IX, DX, true> {
            template<class GM, class FACTOR>
            void static op(const GM &, const FACTOR &);
         };
   } // namespace executor
} // namespace functionwrapper

namespace detail_graphical_model {
   template<size_t DX>
      struct FunctionWrapper;
   template<size_t DX, class VALUE_TYPE>
      struct FunctionValueWrapper;
   template<size_t IX, size_t DX, bool end>
      struct FunctionWrapperExecutor;
   template<size_t IX, size_t DX, bool end, class VALUE_TYPE>
      struct FunctionValueWrapperExecutor;
}

/// \endcond

/// \brief Abstraction (wrapper class) of factors, independent of the function used to implement the factor
template<class GRAPHICAL_MODEL>
class Factor {
public:
   typedef GRAPHICAL_MODEL* GraphicalModelPointerType;
   typedef GRAPHICAL_MODEL GraphicalModelType;
   enum FunctionInformation {
      NrOfFunctionTypes = GraphicalModelType::NrOfFunctionTypes
   };

   typedef typename GraphicalModelType::FunctionTypeList FunctionTypeList;
   typedef typename GraphicalModelType::ValueType ValueType;
   typedef typename GraphicalModelType::LabelType LabelType;
   typedef typename GraphicalModelType::IndexType IndexType;

   /// \cond HIDDEN_SYMBOLS
   typedef FactorShapeAccessor<Factor<GRAPHICAL_MODEL> > ShapeAccessorType;
   /// \endcond
   typedef VectorView<  std::vector<IndexType> , IndexType > VisContainerType;
   typedef typename opengm::AccessorIterator<ShapeAccessorType, true> ShapeIteratorType;
   typedef typename VisContainerType::const_iterator VariablesIteratorType;





   // construction and assignment
   Factor();
   Factor(GraphicalModelPointerType);
   Factor(const Factor&);
   Factor(GraphicalModelPointerType, const IndexType, const UInt8Type, const IndexType,const IndexType);
   Factor& operator=(const Factor&);

   /*
   template<int FUNCTION_TYPE>
      IndexType size() const;
   */
   IndexType size() const;
   IndexType numberOfVariables() const;
   IndexType numberOfLabels(const IndexType) const;
   IndexType shape(const IndexType) const;
   IndexType variableIndex(const IndexType) const;
   ShapeIteratorType shapeBegin() const;
   ShapeIteratorType shapeEnd() const;
   template<size_t FUNCTION_TYPE_INDEX>
      const typename meta::TypeAtTypeList<FunctionTypeList, FUNCTION_TYPE_INDEX>::type& function() const;
   VariablesIteratorType variableIndicesBegin() const;
   VariablesIteratorType variableIndicesEnd() const;
   template<class ITERATOR>
      ValueType operator()(ITERATOR) const;
   
   template<class ITERATOR>
      void copyValues(ITERATOR iterator) const;
  template<class ITERATOR>
      void copyValuesSwitchedOrder(ITERATOR iterator) const;
   template<int FunctionType, class ITERATOR>
      ValueType operator()(ITERATOR) const;
   UInt8Type functionType() const;
   IndexType functionIndex() const;
   template<class ITERATOR>
      void variableIndices(ITERATOR out) const;
   bool isPotts() const;
   bool isGeneralizedPotts() const;
   bool isSubmodular() const;
   bool isSquaredDifference() const;
   bool isTruncatedSquaredDifference() const;
   bool isAbsoluteDifference() const;
   bool isTruncatedAbsoluteDifference() const;
   template<int PROPERTY>
   bool binaryProperty()const;
   template<int PROPERTY>
   ValueType valueProperty()const;
   
   template<class FUNCTOR>
      void forAllValuesInAnyOrder(FUNCTOR & functor)const;
   template<class FUNCTOR>
      void forAtLeastAllUniqueValues(FUNCTOR & functor)const;
   template<class FUNCTOR>
       void forAllValuesInOrder(FUNCTOR & functor)const;
   template<class FUNCTOR>
       void forAllValuesInSwitchedOrder(FUNCTOR & functor)const;

   ValueType sum() const;
   ValueType product() const;
   ValueType min() const;
   ValueType max() const;
   IndexType dimension()const{return this->numberOfVariables();}
private:
   void testInvariant() const;
   //std::vector<IndexType> & variableIndexSequence();
   const VisContainerType & variableIndexSequence() const;
   template<size_t FUNCTION_TYPE_INDEX>
   typename meta::TypeAtTypeList<FunctionTypeList, FUNCTION_TYPE_INDEX>::type& function();

   GraphicalModelPointerType gm_;
   IndexType functionIndex_;
   opengm::UInt8Type functionTypeId_;
   //std::vector<IndexType> variableIndices_;
   //IndexType order_;
   //IndexType indexInVisVector_;
   VisContainerType vis_;
template<typename, typename, typename, typename>
   friend class GraphicalModel;
template<size_t>
   friend struct opengm::detail_graphical_model::FunctionWrapper;
template<size_t, size_t, bool>
   friend struct opengm::detail_graphical_model::FunctionWrapperExecutor;
template<typename>
   friend class Factor;
template<typename GM_>
   friend void opengm::hdf5::save(const GM_&, const std::string&, const std::string&);
template<typename GM_>
   friend void opengm::hdf5::load(GM_&, const std::string&, const std::string&);
template<typename, typename, typename >
   friend class IndependentFactor;

// friends for unary
template<class, class, class, class>
   friend class opengm::functionwrapper::binary::OperationWrapperSelector;
template<class , class, class>
   friend class opengm::functionwrapper::unary::OperationWrapperSelector;

template<class, class, class, class, class, class, class>
   friend class opengm::functionwrapper::binary::OperationWrapper;

template <class, size_t>
   friend class opengm::meta::GetFunction;
template<class, class, class>
   friend class opengm::functionwrapper::AccumulateSomeWrapper;

template<class, class, class, size_t, size_t, bool >
   friend class opengm::functionwrapper::executor::AccumulateSomeExecutor;

template<class, class, class, size_t, size_t, bool>
   friend class opengm::functionwrapper::executor::binary::InplaceOperationExecutor;

template<class A, class B, class OP, size_t IX, size_t DX, bool>
  friend class opengm::functionwrapper::executor::unary::OperationExecutor;
};

/// Factor (with corresponding function and variable indices), independent of a GraphicalModel
template<class T, class I, class L>
class IndependentFactor {
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;
   typedef ExplicitFunction<ValueType,IndexType,LabelType> FunctionType;
   typedef typename meta::TypeListGenerator<FunctionType>::type FunctionTypeList;
   enum FunctionInformation {
      NrOfFunctionTypes = 1
   };
   typedef const size_t* ShapeIteratorType;
   typedef typename std::vector<IndexType>::const_iterator VariablesIteratorType;

   typedef std::vector<IndexType> VisContainerType;

   IndependentFactor();
   IndependentFactor(const ValueType);

   template<class VARIABLE_INDEX_ITERATOR, class SHAPE_ITERATOR>
      IndependentFactor(VARIABLE_INDEX_ITERATOR, VARIABLE_INDEX_ITERATOR, SHAPE_ITERATOR, SHAPE_ITERATOR);
   template<class VARIABLE_INDEX_ITERATOR, class SHAPE_ITERATOR>
      IndependentFactor(VARIABLE_INDEX_ITERATOR, VARIABLE_INDEX_ITERATOR, SHAPE_ITERATOR, SHAPE_ITERATOR, const ValueType);
   template<class GRAPHICAL_MODEL, class VARIABLE_INDEX_ITERATOR>
      IndependentFactor(const GRAPHICAL_MODEL&, VARIABLE_INDEX_ITERATOR, VARIABLE_INDEX_ITERATOR, const ValueType = ValueType());
   IndependentFactor(const IndependentFactor&);
   template<class GRAPHICAL_MODEL>
      IndependentFactor(const Factor<GRAPHICAL_MODEL>&);
   IndependentFactor& operator=(const IndependentFactor&);
   template<class GRAPHICAL_MODEL>
      IndependentFactor& operator=(const Factor<GRAPHICAL_MODEL>&);
   template<class VARIABLE_INDEX_ITERATOR, class SHAPE_ITERATOR>
      void assign(VARIABLE_INDEX_ITERATOR, VARIABLE_INDEX_ITERATOR, SHAPE_ITERATOR, SHAPE_ITERATOR);
   template<class VARIABLE_INDEX_ITERATOR, class SHAPE_ITERATOR>
      void assign(VARIABLE_INDEX_ITERATOR, VARIABLE_INDEX_ITERATOR, SHAPE_ITERATOR, SHAPE_ITERATOR, const ValueType);
   template<class GRAPHICAL_MODEL, class VARIABLE_INDEX_ITERATOR>
      void assign(const GRAPHICAL_MODEL&, VARIABLE_INDEX_ITERATOR, VARIABLE_INDEX_ITERATOR);
   template<class GRAPHICAL_MODEL, class VARIABLE_INDEX_ITERATOR>
      void assign(const GRAPHICAL_MODEL&, VARIABLE_INDEX_ITERATOR, VARIABLE_INDEX_ITERATOR, const ValueType);
   void assign(const ValueType);

   ShapeIteratorType shapeBegin() const;
   ShapeIteratorType shapeEnd() const;
   VariablesIteratorType variableIndicesBegin()const;
   VariablesIteratorType variableIndicesEnd()const;
   const std::vector<IndexType>& variableIndexSequence() const;
   template<size_t FUNCTION_TYPE_INDEX>
      const FunctionType& function() const;
   size_t numberOfVariables() const;
   IndexType numberOfLabels(const IndexType) const;
   IndexType shape(const size_t dimIndex) const;
   size_t size() const;
   IndexType variableIndex(const size_t) const;
   template<class ITERATOR>
      void variableIndices(ITERATOR) const;
   template<class ITERATOR>
      T operator()(ITERATOR) const;
   T operator()(const IndexType) const;
   T operator()(const IndexType, const IndexType) const;
   T operator()(const IndexType, const IndexType, const IndexType) const;
   T operator()(const IndexType, const IndexType, const IndexType, const IndexType) const;

   template<class INDEX_ITERATOR, class STATE_ITERATOR>
      void fixVariables(INDEX_ITERATOR, INDEX_ITERATOR, STATE_ITERATOR);
   template<class ITERATOR>
      T& operator()(ITERATOR);
   T& operator()(const IndexType);
   T& operator()(const IndexType, const IndexType);
   T& operator()(const IndexType, const IndexType, const IndexType);
   T& operator()(const IndexType, const IndexType, const IndexType, const IndexType);
   template<class UNARY_OPERATOR_TYPE>
      void operateUnary(UNARY_OPERATOR_TYPE unaryOperator);
   template<class BINARY_OPERATOR_TYPE>
      void operateBinary(const T value, BINARY_OPERATOR_TYPE binaryOperator);
   template<class GRAPHICAL_MODEL, class BINARY_OPERATOR_TYPE>
      void operateBinary(const Factor<GRAPHICAL_MODEL>&, BINARY_OPERATOR_TYPE binaryOperator);
   template<class BINARY_OPERATOR_TYPE>
      void operateBinary(const IndependentFactor<T, I, L>&, BINARY_OPERATOR_TYPE binaryOperator);
   template<class BINARY_OPERATOR_TYPE>
      void operateBinary(const IndependentFactor<T, I, L>&, const IndependentFactor<T, I, L>&, BINARY_OPERATOR_TYPE binaryOperator);
   void subtractOffset();

   template<class ACCUMULATOR>
      void accumulate(ValueType&, std::vector<LabelType>&) const;
   template<class ACCUMULATOR>
      void accumulate(ValueType&) const;
   template<class ACCUMULATOR, class VariablesIterator>
      void accumulate(VariablesIterator, VariablesIterator, IndependentFactor<T, I, L> &) const;
   template<class ACCUMULATOR, class VariablesIterator>
      void accumulate(VariablesIterator, VariablesIterator) ;
   const FunctionType& function() const;

   bool isPotts()
      { return function_.isPotts(); }
   bool isGeneralizedPotts()
      { return function_.isGeneralizedPotts(); }
   bool isSubmodular()
      { return function_.isSubmodular(); }
   bool isSquaredDifference()
      { return function_.isSquaredDifference(); }
   bool isTruncatedSquaredDifference()
      { return function_.isTruncatedSquaredDifference(); }
   bool isAbsoluteDifference()
      { return function_.isAbsoluteDifference(); }
   bool isTruncatedAbsoluteDifference()
      { return function_.isTruncatedAbsoluteDifference(); }

   T min() {return function_.min();}
   T max() {return function_.max();}
   T sum() {return function_.sum();}
   T product() {return function_.product();}

private:
   template<size_t FUNCTION_TYPE_INDEX>
      FunctionType& function();
   std::vector<IndexType>& variableIndexSequence();

   std::vector<IndexType> variableIndices_;
   FunctionType function_;

template<typename>
   friend class Factor;
template<typename, typename, typename, typename>
   friend class GraphicalModel;
//friends for unary
template<class, class, class, class>
   friend class opengm::functionwrapper::binary::OperationWrapperSelector;
template<class , class, class>
   friend class opengm::functionwrapper::unary::OperationWrapperSelector;
template<class, class, class, class, class, class, class>
   friend class opengm::functionwrapper::binary::OperationWrapper;
template <class, size_t>
   friend class opengm::meta::GetFunction;
template<class, class, class>
   friend class opengm::functionwrapper::AccumulateSomeWrapper;
template<class, class, class, size_t, size_t, bool>
   friend class opengm::functionwrapper::executor::AccumulateSomeExecutor;
template<class, class, class, size_t, size_t, bool>
   friend class opengm::functionwrapper::executor::binary::InplaceOperationExecutor;
template<class A, class B, class OP, size_t IX, size_t DX, bool>
  friend class opengm::functionwrapper::executor::unary::OperationExecutor;
template<class ACC, class A, class ViAccIterator>
   friend void accumulate(A &, ViAccIterator, ViAccIterator );
};


template<class GRAPHICAL_MODEL>
inline Factor<GRAPHICAL_MODEL>::Factor()
:  gm_(NULL), 
   functionIndex_(), 
   vis_()
{}

/// \brief factors are usually not constructed directly but obtained from operator[] of GraphicalModel
template<class GRAPHICAL_MODEL>
inline Factor<GRAPHICAL_MODEL>::Factor
(
   GraphicalModelPointerType gm, 
   const typename  Factor<GRAPHICAL_MODEL>::IndexType functionIndex, 
   const UInt8Type functionTypeId, 
   const typename  Factor<GRAPHICAL_MODEL>::IndexType  order,
   const typename  Factor<GRAPHICAL_MODEL>::IndexType  indexInVisVector
)
:  gm_(gm), 
   functionIndex_(functionIndex), 
   functionTypeId_(functionTypeId), 
   vis_(gm->factorsVis_, indexInVisVector,order)
{
   /*
   if(!opengm::NO_DEBUG) {
      if(variableIndices_.size() != 0) {
         OPENGM_ASSERT(variableIndices_[0] < gm->numberOfVariables());
         for(size_t i = 1; i < variableIndices_.size(); ++i) {
            OPENGM_ASSERT(variableIndices_[i] < gm->numberOfVariables());
         }
      }
   }
   */
}

/// \brief factors are usually not constructed directly but obtained from operator[] of GraphicalModel
template<class GRAPHICAL_MODEL>
inline Factor<GRAPHICAL_MODEL>::Factor
(
   GraphicalModelPointerType gm
)
:  gm_(gm), 
   functionIndex_(0), 
   functionTypeId_(0), 
   vis_(gm_->factorsVis_)
{}

template<class GRAPHICAL_MODEL>
inline Factor<GRAPHICAL_MODEL>::Factor
(
   const Factor& src
)
:  gm_(src.gm_), 
   functionIndex_(src.functionIndex_), 
   functionTypeId_(src.functionTypeId_), 
   vis_(src.vis_)
{}

template<class GRAPHICAL_MODEL>
inline Factor<GRAPHICAL_MODEL>&
Factor<GRAPHICAL_MODEL>::operator=
(
   const Factor& src
)
{
   if(&src != this) {
      functionTypeId_ = src.functionTypeId_;
      functionIndex_ = src.functionIndex_;
      //variableIndices_ = src.variableIndices_;
      vis_=src.vis_;
   }
   return *this;
}

template<class GRAPHICAL_MODEL>
typename Factor<GRAPHICAL_MODEL>::ShapeIteratorType
Factor<GRAPHICAL_MODEL>::shapeBegin() const 
{
   return ShapeIteratorType(ShapeAccessorType(this), 0);
}

template<class GRAPHICAL_MODEL>
typename Factor<GRAPHICAL_MODEL>::ShapeIteratorType
Factor<GRAPHICAL_MODEL>::shapeEnd() const
{
   return ShapeIteratorType(ShapeAccessorType(this), vis_.size());
}



template<class GRAPHICAL_MODEL>
inline const typename Factor<GRAPHICAL_MODEL>::VisContainerType &
Factor<GRAPHICAL_MODEL>::variableIndexSequence() const 
{
   return this->vis_;
}


/// \brief return the number of labels of the j-th variable
template<class GRAPHICAL_MODEL>
inline typename Factor<GRAPHICAL_MODEL>::IndexType 
Factor<GRAPHICAL_MODEL>::numberOfLabels
(
   const IndexType j
) const {
   return gm_->numberOfLabels(vis_[j]);
}

template<class GRAPHICAL_MODEL>
inline typename  Factor<GRAPHICAL_MODEL>::IndexType
Factor<GRAPHICAL_MODEL>::numberOfVariables() const 
{
   return vis_.size();
}

/// \brief return the index of the j-th variable
template<class GRAPHICAL_MODEL>
inline typename  Factor<GRAPHICAL_MODEL>::IndexType
Factor<GRAPHICAL_MODEL>::variableIndex(
   const IndexType j
) const 
{
   return vis_[j];
}

/// \brief return the extension a value table encoding this factor would have in the dimension of the j-th variable
template<class GRAPHICAL_MODEL>
inline typename Factor<GRAPHICAL_MODEL>::IndexType
Factor<GRAPHICAL_MODEL>::shape(
   const IndexType j
) const 
{
   OPENGM_ASSERT(j < vis_.size());
   return gm_->numberOfLabels(vis_[j]);
}

/// \brief evaluate the factor for a sequence of labels
/// \param begin iterator to the beginning of a sequence of labels
template<class GRAPHICAL_MODEL>
template<class ITERATOR>
inline typename Factor<GRAPHICAL_MODEL>::ValueType
Factor<GRAPHICAL_MODEL>::operator()(
   ITERATOR begin
) const
{
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::getValue (this->gm_, begin, functionIndex_, functionTypeId_);
}


/// \brief copies the values of a factors into an iterator
/// \param begin output iterator to store the factors values in last coordinate major order
template<class GRAPHICAL_MODEL>
template<class ITERATOR>
inline void
Factor<GRAPHICAL_MODEL>::copyValues(
   ITERATOR begin
) const
{
   opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::getValues (this->gm_, begin, functionIndex_, functionTypeId_);
}

template<class GRAPHICAL_MODEL>
template<class ITERATOR>
inline void
Factor<GRAPHICAL_MODEL>::copyValuesSwitchedOrder(
   ITERATOR begin
) const
{
   opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::getValuesSwitchedOrder (this->gm_, begin, functionIndex_, functionTypeId_);
}

/// \brief evaluate the factor for a sequence of labels
/// \param begin iterator to the beginning of a sequence of labels
template<class GRAPHICAL_MODEL>
template<int FunctionType, class ITERATOR>
inline typename Factor<GRAPHICAL_MODEL>::ValueType
Factor<GRAPHICAL_MODEL>::operator()
(
   ITERATOR begin
) const {
   return gm_-> template functions<FunctionType>()[functionIndex_].operator()(begin);
}

/// \brief compute a  binary property of a factor 
///
/// The property must be one of the properties defined in
/// opengm::BinartyProperties::Values 
///
/// Usage:
/// \code
/// const size_t factorIndex=0;
/// bool isPotts=gm[factorIndex]. template binaryProperty<opengm::BinaryProperties::IsPotts>();
/// OPENGM_ASSERT(gm[factorIndex].isPotts() == isPotts);
/// \endcode
template<class GRAPHICAL_MODEL>
template<int PROPERTY>
inline bool
Factor<GRAPHICAL_MODEL>::binaryProperty() const 
{
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >:: template binaryProperty<GRAPHICAL_MODEL,PROPERTY> (this->gm_, functionIndex_, functionTypeId_);
}


/// \brief compute a property of a factor 
///
/// The property must be one of the properties defined in
/// opengm::ValueProperties::Values 
///
/// Usage:
/// \code
/// const size_t factorIndex=0;
/// GmType::ValueType sum=gm[factorIndex]. template valueProperty<opengm::ValueProperties::Sum>();
/// OPENGM_ASSERT(gm[factorIndex].sum() == sum);
/// \endcode
template<class GRAPHICAL_MODEL>
template<int PROPERTY>
inline typename GRAPHICAL_MODEL::ValueType
Factor<GRAPHICAL_MODEL>::valueProperty() const 
{
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >:: template valueProperty<GRAPHICAL_MODEL,PROPERTY> (this->gm_, functionIndex_, functionTypeId_);
}

/// \brief call a functor for all values in no defined order
///
/// Usage:
/// \code
/// opengm::AccumulationFunctor<opengm::Adder,ValueType> functor;
/// const size_t factorIndex=0;
/// gm[factorIndex].forAllValuesInAnyOrder(functor);
/// OPENGM_ASSERT(gm[factorIndex].sum()==functor.value());
/// \endcode
template<class GRAPHICAL_MODEL>
template<class FUNCTOR>
inline void 
Factor<GRAPHICAL_MODEL>::forAllValuesInAnyOrder
(
   FUNCTOR & functor
)const{
   opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >:: template forAllValuesInAnyOrder<GRAPHICAL_MODEL,FUNCTOR> (this->gm_,functor, functionIndex_, functionTypeId_);
}


/// \brief call a functor for at least all unique values in no defined order
///
/// Usage:
/// \code
/// opengm::AccumulationFunctor<opengm::Maximizer,ValueType> functor;
/// const size_t factorIndex=0;
/// gm[factorIndex].forAtLeastAllUniqueValues(functor);
/// OPENGM_ASSERT(gm[factorIndex].max()==functor.value());
/// \endcode
template<class GRAPHICAL_MODEL>
template<class FUNCTOR>
inline void 
Factor<GRAPHICAL_MODEL>::forAtLeastAllUniqueValues
(
   FUNCTOR & functor
)const{
   opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >:: template forAtLeastAllUniqueValues<GRAPHICAL_MODEL,FUNCTOR> (this->gm_,functor, functionIndex_, functionTypeId_);
}

/// \brief call a functor for all values in last coordinate major order
///
/// Usage:
/// \code
/// opengm::AccumulationFunctor<opengm::Multipler,ValueType> functor;
/// const size_t factorIndex=0;
/// gm[factorIndex].forAllValuesInOrder(functor);
/// OPENGM_ASSERT(gm[factorIndex].product()==functor.value());
/// \endcode
template<class GRAPHICAL_MODEL>
template<class FUNCTOR>
inline void 
Factor<GRAPHICAL_MODEL>::forAllValuesInOrder
(
   FUNCTOR & functor
)const{
  opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >:: template forAllValuesInOrder<GRAPHICAL_MODEL,FUNCTOR> (this->gm_,functor, functionIndex_, functionTypeId_); 
}

template<class GRAPHICAL_MODEL>
template<class FUNCTOR>
inline void 
Factor<GRAPHICAL_MODEL>::forAllValuesInSwitchedOrder
(
   FUNCTOR & functor
)const{
  opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >:: template forAllValuesInSwitchedOrder<GRAPHICAL_MODEL,FUNCTOR> (this->gm_,functor, functionIndex_, functionTypeId_); 
}

template<class GRAPHICAL_MODEL>
inline bool
Factor<GRAPHICAL_MODEL>::isPotts() const 
{
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::isPotts (this->gm_, functionIndex_, functionTypeId_);
}

template<class GRAPHICAL_MODEL>
inline bool
Factor<GRAPHICAL_MODEL>::isGeneralizedPotts() const 
{
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::isGeneralizedPotts (this->gm_, functionIndex_, functionTypeId_);
}

template<class GRAPHICAL_MODEL>
inline bool
Factor<GRAPHICAL_MODEL>::isSubmodular() const 
{
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::isSubmodular (this->gm_, functionIndex_, functionTypeId_);
}

template<class GRAPHICAL_MODEL>
inline bool
Factor<GRAPHICAL_MODEL>::isSquaredDifference() const 
{
   if(this->numberOfVariables()==2) {
      return opengm::detail_graphical_model::FunctionWrapper<
            Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
         >::isSquaredDifference(this->gm_, functionIndex_, functionTypeId_);
   }
   else {
      return false;
   }
}

template<class GRAPHICAL_MODEL>
inline bool
Factor<GRAPHICAL_MODEL>::isTruncatedSquaredDifference() const 
{
   if(this->numberOfVariables()==2) {
      return opengm::detail_graphical_model::FunctionWrapper<
            Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
         >::isTruncatedSquaredDifference(this->gm_, functionIndex_, functionTypeId_);
   }
   else {
      return false;
   }
}

template<class GRAPHICAL_MODEL>
inline bool
Factor<GRAPHICAL_MODEL>::isAbsoluteDifference() const 
{
   if(this->numberOfVariables() == 2) {
      return opengm::detail_graphical_model::FunctionWrapper<
            Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
         >::isAbsoluteDifference(this->gm_, functionIndex_, functionTypeId_);
   }
   else {
      return false;
   }
}

template<class GRAPHICAL_MODEL>
inline bool
Factor<GRAPHICAL_MODEL>::isTruncatedAbsoluteDifference() const 
{
   if(this->numberOfVariables()==2) {
      return opengm::detail_graphical_model::FunctionWrapper<
         Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
      >::isTruncatedAbsoluteDifference (this->gm_, functionIndex_, functionTypeId_);
   }
   else{
      return false;
   }
}

template<class GRAPHICAL_MODEL>
inline typename Factor<GRAPHICAL_MODEL>::ValueType
Factor<GRAPHICAL_MODEL>::sum() const {
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::sum (this->gm_, functionIndex_, functionTypeId_);
}

template<class GRAPHICAL_MODEL>
inline typename Factor<GRAPHICAL_MODEL>::ValueType
Factor<GRAPHICAL_MODEL>::product() const {
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::product (this->gm_, functionIndex_, functionTypeId_);
}

template<class GRAPHICAL_MODEL>
inline typename Factor<GRAPHICAL_MODEL>::ValueType
Factor<GRAPHICAL_MODEL>::min() const {
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::min (this->gm_, functionIndex_, functionTypeId_);
}

template<class GRAPHICAL_MODEL>
inline typename Factor<GRAPHICAL_MODEL>::ValueType
Factor<GRAPHICAL_MODEL>::max() const {
   return opengm::detail_graphical_model::FunctionWrapper<
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes
   >::max (this->gm_, functionIndex_, functionTypeId_);
}

template<class GRAPHICAL_MODEL>
template<class ITERATOR>
inline void Factor<GRAPHICAL_MODEL>::variableIndices
(
   ITERATOR out
) const {
   for(IndexType j = 0; j < numberOfVariables(); ++j) {
      *out = this->variableIndex(j);
      ++out;
   }
}

template<class GRAPHICAL_MODEL>
template<size_t FUNCTION_TYPE_INDEX>
inline typename meta::TypeAtTypeList< typename Factor<GRAPHICAL_MODEL>::FunctionTypeList, FUNCTION_TYPE_INDEX>::type&
Factor<GRAPHICAL_MODEL>::function() {
   typedef typename meta::SmallerNumber<FUNCTION_TYPE_INDEX, Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes>::type MetaBoolAssertType;
   OPENGM_META_ASSERT(MetaBoolAssertType::value, WRONG_FUNCTION_TYPE_INDEX);
   return meta::FieldAccess::template byIndex<FUNCTION_TYPE_INDEX>(gm_->functionDataField_).
      functionData_.functions_[functionIndex_];
}

template<class GRAPHICAL_MODEL>
template<size_t FUNCTION_TYPE_INDEX>
inline const typename meta::TypeAtTypeList< typename Factor<GRAPHICAL_MODEL>::FunctionTypeList, FUNCTION_TYPE_INDEX>::type&
Factor<GRAPHICAL_MODEL>::function() const {
   typedef typename meta::SmallerNumber<FUNCTION_TYPE_INDEX, Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes>::type MetaBoolAssertType;
   OPENGM_META_ASSERT(MetaBoolAssertType::value, WRONG_FUNCTION_TYPE_INDEX);
   return meta::FieldAccess::template byIndex<FUNCTION_TYPE_INDEX>(gm_->functionDataField_).
      functionData_.functions_[functionIndex_];
}

template<class GRAPHICAL_MODEL>
inline typename  Factor<GRAPHICAL_MODEL>::VisContainerType::const_iterator
Factor<GRAPHICAL_MODEL>::variableIndicesBegin() const {
   return vis_.begin();
}

template<class GRAPHICAL_MODEL>
inline typename  Factor<GRAPHICAL_MODEL>::VisContainerType::const_iterator
Factor<GRAPHICAL_MODEL>::variableIndicesEnd() const {
   return  vis_.end();
}

template<class GRAPHICAL_MODEL>
inline typename Factor<GRAPHICAL_MODEL>::IndexType
Factor<GRAPHICAL_MODEL>::size() const 
{
   if(vis_.size() != 0) {
      size_t val = this->shape(0);
      for(size_t i = 1; i<this->numberOfVariables(); ++i) {
         val *= this->shape(i);
      }
      return val;
   }
   return 1;
}

template<class GRAPHICAL_MODEL>
inline void Factor<GRAPHICAL_MODEL>::testInvariant() const {
   opengm::functionwrapper::executor::FactorInvariant
   <
      0, 
      Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes, 
      meta::EqualNumber<Factor<GRAPHICAL_MODEL>::NrOfFunctionTypes, 0>::value
   >::op( *gm_, *this);
}

template<class GRAPHICAL_MODEL>
inline opengm::UInt8Type
Factor<GRAPHICAL_MODEL>::functionType() const {
   return static_cast<UInt8Type> (functionTypeId_);
}

template<class GRAPHICAL_MODEL>
inline typename Factor<GRAPHICAL_MODEL>::IndexType
Factor<GRAPHICAL_MODEL>::functionIndex()const {
   return functionIndex_;
}

template<class T, class I, class L>
inline IndependentFactor<T, I, L>::IndependentFactor()
:  variableIndices_(), 
   function_(1.0)
{}

/// \brief construct a constant (order 0) independent factor
template<class T, class I, class L>
inline IndependentFactor<T, I, L>::IndependentFactor
(
   const ValueType constant
)
:  variableIndices_(), 
   function_(constant)
{}

/// \brief construct an independent factor using variable indices and the extension of a value table
/// \param beginVi iterator to the beginning of a sequence of variable indices
/// \param endVi iterator to the end of a sequence of variable indices
/// \param shapeBegin iterator to the beginning of a sequence of extensions
/// \param shapeBegin iterator to the end of a sequence of extensions
template<class T, class I, class L>
template<class VARIABLE_INDEX_ITERATOR, class SHAPE_ITERATOR>
inline IndependentFactor<T, I, L>::IndependentFactor
(
   VARIABLE_INDEX_ITERATOR beginVi, 
   VARIABLE_INDEX_ITERATOR endVi, 
   SHAPE_ITERATOR beginShape, 
   SHAPE_ITERATOR endShape
)
:  variableIndices_(beginVi, endVi), 
   function_(beginShape, endShape, 1)
{
   OPENGM_ASSERT(std::distance(beginVi, endVi) == std::distance(beginShape, endShape));
   OPENGM_ASSERT(opengm::isSorted(beginVi, endVi));
}

template<class T, class I, class L>
template<class VARIABLE_INDEX_ITERATOR, class SHAPE_ITERATOR>
inline IndependentFactor<T, I, L>::IndependentFactor
(
   VARIABLE_INDEX_ITERATOR beginVi, 
   VARIABLE_INDEX_ITERATOR endVi, 
   SHAPE_ITERATOR beginShape, 
   SHAPE_ITERATOR endShape,
   const ValueType constant
)
:  variableIndices_(beginVi, endVi), 
   function_(beginShape, endShape, constant)
{
   OPENGM_ASSERT(std::distance(beginVi, endVi) == std::distance(beginShape, endShape));
   OPENGM_ASSERT(opengm::isSorted(beginVi, endVi));
}

template<class T, class I, class L>
template<class VARIABLE_INDEX_ITERATOR, class SHAPE_ITERATOR>
inline void IndependentFactor<T, I, L>::assign
(
   VARIABLE_INDEX_ITERATOR beginVi, 
   VARIABLE_INDEX_ITERATOR endVi, 
   SHAPE_ITERATOR beginShape, 
   SHAPE_ITERATOR endShape
) {
   OPENGM_ASSERT(std::distance(beginVi, endVi) == std::distance(beginShape, endShape));
   OPENGM_ASSERT(opengm::isSorted(beginVi, endVi));
   function_.assign();
   function_.resize(beginShape, endShape, 1);
   variableIndices_.assign(beginVi, endVi);
}

template<class T, class I, class L>
template<class VARIABLE_INDEX_ITERATOR, class SHAPE_ITERATOR>
inline void IndependentFactor<T, I, L>::assign
(
   VARIABLE_INDEX_ITERATOR beginVi, 
   VARIABLE_INDEX_ITERATOR endVi, 
   SHAPE_ITERATOR beginShape, 
   SHAPE_ITERATOR endShape,
   const ValueType constant
) {
   OPENGM_ASSERT(std::distance(beginVi, endVi) == std::distance(beginShape, endShape));
   OPENGM_ASSERT(opengm::isSorted(beginVi, endVi));
   function_.assign();
   function_.resize(beginShape, endShape, constant);
   variableIndices_.assign(beginVi, endVi);
}

/// \brief return the function of the independent factor
template<class T, class I, class L>
template<size_t FUNCTION_TYPE_INDEX>
inline typename IndependentFactor<T, I, L>::FunctionType&
IndependentFactor<T, I, L>::function() 
{
   return function_;
}

template<class T, class I, class L>
inline const typename IndependentFactor<T, I, L>::FunctionType&
IndependentFactor<T, I, L>::function() const 
{
   return function_;
}

template<class T, class I, class L>
template<size_t FUNCTION_TYPE_INDEX>
inline const typename IndependentFactor<T, I, L>::FunctionType&
IndependentFactor<T, I, L>::function() const
{
   return function_;
}

template<class T, class I, class L>
inline void
IndependentFactor<T, I, L>::assign
(
   const ValueType constant
) 
{
   typename IndependentFactor<T, I, L>::FunctionType c(constant);
   function_ = c;
   variableIndices_.clear();
}

template<class T, class I, class L>
template<class GRAPHICAL_MODEL, class VARIABLE_INDEX_ITERATOR>
inline void IndependentFactor<T, I, L>::assign
(
   const GRAPHICAL_MODEL& gm, 
   VARIABLE_INDEX_ITERATOR begin, 
   VARIABLE_INDEX_ITERATOR end
)
{
   OPENGM_ASSERT(opengm::isSorted(begin, end));
   this->variableIndices_.assign(begin, end);
   std::vector<size_t> factorShape(variableIndices_.size());
   for(size_t i = 0; i < factorShape.size(); ++i) {
      factorShape[i] = gm.numberOfLabels(variableIndices_[i]);
   }
   this->function_.assign();
   this->function_.resize(factorShape.begin(), factorShape.end());
}

template<class T, class I, class L>
template<class GRAPHICAL_MODEL, class VARIABLE_INDEX_ITERATOR>
void inline IndependentFactor<T, I, L>::assign
(
   const GRAPHICAL_MODEL& gm, 
   VARIABLE_INDEX_ITERATOR begin, 
   VARIABLE_INDEX_ITERATOR end, 
   const ValueType value
) {
   OPENGM_ASSERT(opengm::isSorted(begin, end));
   this->variableIndices_.assign(begin, end);
   std::vector<size_t> factorShape(variableIndices_.size());
   for(size_t i = 0; i < factorShape.size(); ++i) {
      factorShape[i] = static_cast<size_t> (gm.numberOfLabels(this->variableIndices_[i]));
      //factorShape[i]=gm.numbersOfStates_[ this->variableIndices_[i] ];
      //  (gm.numberOfLabels(  1));
   }
   this->function_.assign();
   this->function_.resize(factorShape.begin(), factorShape.end(), value);
}

template<class T, class I, class L>
template<class GRAPHICAL_MODEL, class VARIABLE_INDEX_ITERATOR>
inline IndependentFactor<T, I, L>::IndependentFactor
(
   const GRAPHICAL_MODEL& gm, 
   VARIABLE_INDEX_ITERATOR begin, 
   VARIABLE_INDEX_ITERATOR end, 
   const ValueType value
)
:  variableIndices_(begin, end)
{
   OPENGM_ASSERT(opengm::isSorted(begin, end));
   std::vector<size_t> shape(variableIndices_.size());
   for(size_t i = 0; i < shape.size(); ++i) {
      shape[i] = gm.numberOfLabels(variableIndices_[i]);
   }
   this->function_.assign();
   this->function_.resize(shape.begin(), shape.end(), value);
}

template<class T, class I, class L>
inline IndependentFactor<T, I, L>::IndependentFactor
(
   const IndependentFactor<T, I, L>& src
)
:  variableIndices_(src.variableIndices_)
{
   if(src.variableIndices_.size() == 0) {
      FunctionType tmp(src.function_(0));
      function_ = tmp;
   }
   else {
      function_ = src.function_;
   }
}

template<class T, class I, class L>
template<class GRAPHICAL_MODEL>
inline IndependentFactor<T, I, L>::IndependentFactor
(
   const Factor<GRAPHICAL_MODEL>& src
)
:  variableIndices_(src.variableIndicesBegin(), src.variableIndicesEnd()) {
   //resize dst function
   const size_t dimension = src.numberOfVariables();
   if(dimension!=0) {
      function_.assign();
      function_.resize(src.shapeBegin(), src.shapeEnd());
      //iterators and walkersbeginbegin
      ShapeWalker< typename Factor<GRAPHICAL_MODEL>::ShapeIteratorType> walker(src.shapeBegin(), dimension);
      const opengm::FastSequence<size_t> & coordinate = walker.coordinateTuple();
      for(size_t scalarIndex = 0; scalarIndex < function_.size(); ++scalarIndex) {
         function_(coordinate.begin()) = src(coordinate.begin());
         ++walker;
      }
   }
   else{
      function_.assign();
      size_t indexToScalar[]={0};
      //function_(indexToScalar)=(src.operator()(indexToScalar));

      ExplicitFunction<T,I,L> tmp(src.operator()(indexToScalar));
      function_ = tmp;
   }
}

template<class T, class I, class L>
inline IndependentFactor<T, I, L>&
IndependentFactor<T, I, L>::operator=
(
   const IndependentFactor& src
)
{
   if(this != &src) {
      function_ = src.function_;
      variableIndices_ = src.variableIndices_;
   }
   return *this;
}

template<class T, class I, class L>
template<class GRAPHICAL_MODEL>
IndependentFactor<T, I, L>&
IndependentFactor<T, I, L>::operator=
(
   const Factor<GRAPHICAL_MODEL>& src
)
{
   this->variableIndices_.resize(src.numberOfVariables());
   for(size_t i=0;i<src.numberOfVariables();++i)
      variableIndices_[i] = src.variableIndex(i);
   //resize dst function
   const size_t dimension = src.numberOfVariables();
   if(dimension!=0) {
      function_.assign();
      function_.resize(src.shapeBegin(), src.shapeEnd());
      //iterators and walkers
      ShapeWalker< typename Factor<GRAPHICAL_MODEL>::ShapeIteratorType> walker(src.shapeBegin(), dimension);
      const opengm::FastSequence<size_t> & coordinate = walker.coordinateTuple();
      for(size_t scalarIndex = 0; scalarIndex < function_.size(); ++scalarIndex) {
         function_(scalarIndex) = src(coordinate.begin());
         ++walker;
      }
   }
   else {
      size_t indexToScalar[]={0};
      function_=ExplicitFunction<T>(src(indexToScalar));
   }
  return * this;
}

template<class T, class I, class L>
inline size_t
IndependentFactor<T, I, L>::numberOfVariables() const 
{
   return variableIndices_.size();
}

/// \brief return the number of labels of a specific variable 
template<class T, class I, class L>
inline typename IndependentFactor<T, I, L>::IndexType
IndependentFactor<T, I, L>::numberOfLabels
(
   const IndexType index
) const 
{
   OPENGM_ASSERT(index < variableIndices_.size());
   return function_.shape(index);
}

/// \brief return the number of entries of the value table of the function
template<class T, class I, class L>
inline size_t
IndependentFactor<T, I, L>::size() const 
{
   return function_.size();
}

/// \brief return the extension of the value table of the of the function in a specific dimension
template<class T, class I, class L>
inline typename IndependentFactor<T, I, L>::IndexType
IndependentFactor<T, I, L>::shape
(
   const size_t index
) const {
   if(variableIndices_.size() == 0) {
      return 0;
   }
   OPENGM_ASSERT(index < variableIndices_.size());
   return function_.shape(index);
}

template<class T, class I, class L>
inline typename IndependentFactor<T, I, L>::ShapeIteratorType
IndependentFactor<T, I, L>::shapeBegin() const 
{
   return function_.shapeBegin();
}

template<class T, class I, class L>
inline typename IndependentFactor<T, I, L>::ShapeIteratorType
IndependentFactor<T, I, L>::shapeEnd() const 
{
   return function_.shapeEnd();
}

template<class T, class I, class L>
inline typename IndependentFactor<T, I, L>::VariablesIteratorType
IndependentFactor<T, I, L>::variableIndicesBegin() const 
{
   return variableIndices_.begin();
}

template<class T, class I, class L>
inline typename IndependentFactor<T, I, L>::VariablesIteratorType
IndependentFactor<T, I, L>::variableIndicesEnd() const 
{
   return variableIndices_.end();
}


/// \brief return the index of the j-th variable
template<class T, class I, class L>
inline I
IndependentFactor<T, I, L>::variableIndex
(
   const size_t index
) const {
   OPENGM_ASSERT(index < variableIndices_.size());
   return variableIndices_[index];
}

template<class T, class I, class L>
inline void
IndependentFactor<T, I, L> ::subtractOffset() {
   if(variableIndices_.size() == 0) {
      function_(0) = static_cast<ValueType> (0);
   }
   else {
      T v;
      std::vector<size_t> states;
      opengm::accumulate<Minimizer>(*this, v, states);
      (*this) -= v;
   }
}

/// \brief evaluate the function underlying the factor, given labels to be assigned the variables
/// \param begin iterator to the beginning of a sequence of labels
template<class T, class I, class L>
template<class ITERATOR>
inline T
IndependentFactor<T, I, L>::operator()
(
   ITERATOR begin
) const {
   return function_(begin);
}

template<class T, class I, class L>
inline std::vector<I>&
IndependentFactor<T, I, L>::variableIndexSequence() 
{
   return this->variableIndices_;
}

template<class T, class I, class L>
inline const std::vector<I>&
IndependentFactor<T, I, L>::variableIndexSequence() const 
{
   return this->variableIndices_;
}

/// \brief evaluate an independent factor with 1 variable
template<class T, class I, class L>
inline T
IndependentFactor<T, I, L>::operator()
(
   const IndexType x0
) const 
{
   return function_(x0);
}

/// \brief evaluate an independent factor with 2 variables
template<class T, class I, class L>
inline T
IndependentFactor<T, I, L>::operator()
(
   const IndexType x0, 
   const IndexType x1
) const 
{
   OPENGM_ASSERT(2 == variableIndices_.size());
   return function_(x0, x1);
}

/// \brief evaluate an independent factor with 3 variables
template<class T, class I, class L>
inline T
IndependentFactor<T, I, L>::operator()
(
   const IndexType x0, 
   const IndexType x1, 
   const IndexType x2
) const 
{
   OPENGM_ASSERT(3 == variableIndices_.size());
   return function_(x0, x1, x2);
}

/// \brief evaluate an independent factor with 4 variables
template<class T, class I, class L>
inline T
IndependentFactor<T, I, L>::operator()
(
   const IndexType x0, 
   const IndexType x1, 
   const IndexType x2, 
   const IndexType x3
) const 
{
   OPENGM_ASSERT(4 == variableIndices_.size());
   return function_(x0, x1, x2, x3);
}

template<class T, class I, class L>
template<class UNARY_OPERATOR_TYPE>
inline void
IndependentFactor<T, I, L>::operateUnary
(
   UNARY_OPERATOR_TYPE unaryOperator
) {
   if(this->variableIndices_.size() != 0) {
      for(size_t i = 0; i < function_.size(); ++i) {
         function_(i) = static_cast<T> (unaryOperator(function_(i)));
      }
   }
   else {
      function_(0) = static_cast<T> (unaryOperator(function_(0)));
   }
}

template<class T, class I, class L>
template<class BINARY_OPERATOR_TYPE>
inline void
IndependentFactor<T, I, L>::operateBinary
(
   const T value, 
   BINARY_OPERATOR_TYPE binaryOperator
) {
   if(this->variableIndices_.size() != 0) {
      for(size_t i = 0; i < function_.size(); ++i) {
         function_(i) = static_cast<T> (binaryOperator(function_(i), value));
      }
   }
   else {
      function_(0) = static_cast<T> (binaryOperator(function_(0), value));
   }
}

template<class T, class I, class L>
template<class GRAPHICAL_MODEL, class BINARY_OPERATOR_TYPE>
inline void
IndependentFactor<T, I, L>::operateBinary
(
   const Factor<GRAPHICAL_MODEL>& srcB, 
   BINARY_OPERATOR_TYPE binaryOperator
) {
   opengm::operateBinary(*this, srcB, binaryOperator);
}

template<class T, class I, class L>
template<class BINARY_OPERATOR_TYPE>
inline void
IndependentFactor<T, I, L>::operateBinary
(
   const IndependentFactor<T, I, L>& srcB, 
   BINARY_OPERATOR_TYPE binaryOperator
) {
   opengm::operateBinary(*this, srcB, binaryOperator);
}

template<class T, class I, class L>
template<class BINARY_OPERATOR_TYPE>
inline void
IndependentFactor<T, I, L>::operateBinary
(
   const IndependentFactor<T, I, L>& srcA, 
   const IndependentFactor<T, I, L>& srcB, 
   BINARY_OPERATOR_TYPE binaryOperator
) {
   opengm::operateBinary(srcA, srcB, *this, binaryOperator);
}

template<class T, class I, class L>
template<class ACCUMULATOR>
inline void
IndependentFactor<T, I, L>::accumulate
(
   T& result, 
   std::vector<LabelType>& resultState
) const {
   opengm::accumulate<ACCUMULATOR> (*this, result, resultState);
}

template<class T, class I, class L>
template<class ACCUMULATOR>
inline void
IndependentFactor<T, I, L>::accumulate
(
   T& result
) const {
   opengm::accumulate<ACCUMULATOR> (*this, result);
}

template<class T, class I, class L>
template<class ACCUMULATOR, class VariablesIterator>
inline void
IndependentFactor<T, I, L>::accumulate
(
   VariablesIterator begin, 
   VariablesIterator end, 
   IndependentFactor<T, I, L>& dstFactor
) const {
   opengm::accumulate<ACCUMULATOR> (*this, begin, end, dstFactor);
}

template<class T, class I, class L>
template<class ACCUMULATOR, class VariablesIterator>
inline void
IndependentFactor<T, I, L>::accumulate(
   VariablesIterator begin, 
   VariablesIterator end
) {
   opengm::accumulate<ACCUMULATOR> (*this, begin, end);
}

/// \brief evaluate the independent factor 
/// \param begin iterator to the beginning of a sequence of labels
template<class T, class I, class L>
template<class ITERATOR>
inline T&
IndependentFactor<T, I, L>::operator()(
   ITERATOR begin
) {
   return function_(begin);
}

/// \brief evaluate an independent factor with 1 variables
template<class T, class I, class L>
inline T&
IndependentFactor<T, I, L>::operator()(
   const IndexType x0
) {
   return function_(x0);
}

/// \brief evaluate an independent factor with 2 variables
template<class T, class I, class L>
inline T&
IndependentFactor<T, I, L>::operator()(
   const IndexType x0, 
   const IndexType x1
) {
   OPENGM_ASSERT(2 == variableIndices_.size());
   return function_(x0, x1);
}

/// \brief evaluate an independent factor with 3 variables
template<class T, class I, class L>
inline T& IndependentFactor<T, I, L>::operator()(
   const IndexType x0, 
   const IndexType x1, 
   const IndexType x2
) {
   OPENGM_ASSERT(3 == variableIndices_.size());
   return function_(x0, x1, x2);
}

/// \brief evaluate an independent factor with 4 variables
template<class T, class I, class L>
inline T& IndependentFactor<T, I, L>::operator()(
   const IndexType x0, 
   const IndexType x1, 
   const IndexType x2, 
   const IndexType x3
) {
      OPENGM_ASSERT(4 == variableIndices_.size());
      return function_(x0, x1, x2, x3);
}

template<class T, class I, class L>
template<class ITERATOR>
inline void
IndependentFactor<T, I, L>::variableIndices
(
   ITERATOR out
) const {
   for(size_t j=0; j<variableIndices_.size(); ++j) {
      *out = variableIndices_[j];
      ++out;
   }
}

/// \brief assign specific labels to a specific subset of variables (reduces the order)
/// \param beginIndex iterator to the beginning of a sequence of variable indices
/// \param endIndex iterator to the end of a sequence of variable indices
/// \param beginLabels iterator to the beginning of a sequence of labels
template<class T, class I, class L>
template<class INDEX_ITERATOR, class STATE_ITERATOR>
void
IndependentFactor<T, I, L>::fixVariables
(
   INDEX_ITERATOR beginIndex, 
   INDEX_ITERATOR endIndex, 
   STATE_ITERATOR beginLabels
) {
   if(this->variableIndices_.size() != 0) {
      OPENGM_ASSERT(opengm::isSorted(beginIndex, endIndex));
      opengm::FastSequence<IndexType> variablesToFix;
      opengm::FastSequence<IndexType> variablesNotToFix;
      opengm::FastSequence<IndexType> positionOfVariablesToFix;
      opengm::FastSequence<LabelType> newStates;
      opengm::FastSequence<LabelType> newShape;
      // find the variables to fix:
      while(beginIndex != endIndex) {
         size_t counter = 0;
         if(*beginIndex>this->variableIndices_.back()) {
            break;
         }
         for(size_t i = counter; i<this->variableIndices_.size(); ++i) {
            if(*beginIndex<this->variableIndices_[i])break;
            else if(*beginIndex == this->variableIndices_[i]) {
               ++counter;
               variablesToFix.push_back(*beginIndex);
               newStates.push_back(*beginLabels);
               positionOfVariablesToFix.push_back(i);
            }
         }
         ++beginIndex;
         ++beginLabels;
      }
      for(size_t i = 0; i<this->variableIndices_.size(); ++i) {
         bool found = false;
         for(size_t j = 0; j < variablesToFix.size(); ++j) {
            if(variablesToFix[j] == this->variableIndices_[i]) {
               found = true;
               break;
            }
         }
         if(found == false) {
            variablesNotToFix.push_back(this->variableIndices_[i]);
            newShape.push_back(this->numberOfLabels(i));
         }
      }
      if(variablesToFix.size() != 0) {
         FunctionType& factorFunction = this->function_;
         std::vector<LabelType> fullCoordinate(this->numberOfVariables());
         if(variablesToFix.size() == this->variableIndices_.size()) {
            FunctionType tmp(factorFunction(newStates.begin()));
            factorFunction = tmp;
            this->variableIndices_.clear();
         }
         else {
            SubShapeWalker< 
                ShapeIteratorType, 
                opengm::FastSequence<IndexType>, 
                opengm::FastSequence<LabelType> 
            > subWalker
               (shapeBegin(), factorFunction.dimension(), positionOfVariablesToFix, newStates);
            FunctionType tmp(newShape.begin(), newShape.end());
            const size_t subSize = subWalker.subSize();
            subWalker.resetCoordinate();
            for(size_t i = 0; i < subSize; ++i) {
               tmp(i) = factorFunction(subWalker.coordinateTuple().begin());
               ++subWalker;
            }
            factorFunction = tmp;
            this->variableIndices_.assign(variablesNotToFix.begin(), variablesNotToFix.end());
         }
         OPENGM_ASSERT(factorFunction.dimension()==variablesNotToFix.size());
         OPENGM_ASSERT(newShape.size()==variablesNotToFix.size());
         OPENGM_ASSERT(factorFunction.dimension()==newShape.size());
      }
   }
}

#define OPENGM_INDEPENDENT_FACTOR_OPERATION_GENERATION(BINARY_OPERATOR_SYMBOL, BINARY_INPLACE_OPERATOR_SYMBOL, BINARY_FUNCTOR_NAME) \
template<class T, class I, class L> \
inline IndependentFactor<T, I, L> & \
operator BINARY_INPLACE_OPERATOR_SYMBOL \
( \
   IndependentFactor<T, I, L>& op1, \
   const T& op2 \
) { \
   op1.operateBinary(op2, BINARY_FUNCTOR_NAME<T>()); \
   return op1; \
} \
template<class GRAPHICAL_MODEL> \
inline IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > & \
operator BINARY_INPLACE_OPERATOR_SYMBOL \
( \
   IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType >& op1, \
   const Factor<GRAPHICAL_MODEL>& op2 \
) { \
   op1.operateBinary(op2, BINARY_FUNCTOR_NAME<typename GRAPHICAL_MODEL::ValueType> ()); \
   return op1; \
} \
template<class T, class I, class L> \
inline IndependentFactor<T, I, L> & \
operator BINARY_INPLACE_OPERATOR_SYMBOL \
( \
   IndependentFactor<T, I, L>& op1, \
   const IndependentFactor<T, I, L>& op2 \
) { \
   op1.operateBinary(op2, BINARY_FUNCTOR_NAME<T> ()); \
   return op1; \
} \
template<class T, class I, class L> \
inline IndependentFactor<T, I, L> \
operator BINARY_OPERATOR_SYMBOL \
( \
   const IndependentFactor<T, I, L>& op1, \
   const IndependentFactor<T, I, L>& op2 \
) { \
   IndependentFactor<T, I, L> tmp; \
   opengm::operateBinary(op1, op2, tmp, BINARY_FUNCTOR_NAME <T> ()); \
   return tmp; \
} \
template<class T, class I, class L> \
inline IndependentFactor<T, I, L> \
operator BINARY_OPERATOR_SYMBOL \
( \
   const T & op1, \
   const IndependentFactor<T, I, L>& op2 \
) { \
   IndependentFactor<T, I, L> tmp; \
   opengm::operateBinary(op1, op2, tmp, BINARY_FUNCTOR_NAME <T> ()); \
   return tmp; \
} \
template<class T, class I, class L> \
inline IndependentFactor<T, I, L> \
operator BINARY_OPERATOR_SYMBOL \
( \
   const IndependentFactor<T, I, L>& op1, \
   const T & op2  \
) { \
   IndependentFactor<T, I, L> tmp; \
   opengm::operateBinary(op1, op2, tmp, BINARY_FUNCTOR_NAME <T> ()); \
   return tmp; \
} \
template<class GRAPHICAL_MODEL> \
inline IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > \
operator BINARY_OPERATOR_SYMBOL \
( \
   const Factor<GRAPHICAL_MODEL> & op1, \
   const IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType >& op2 \
) { \
   IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > tmp; \
   opengm::operateBinary(op1, op2, tmp, BINARY_FUNCTOR_NAME <typename GRAPHICAL_MODEL::ValueType> ()); \
   return tmp; \
} \
template<class GRAPHICAL_MODEL> \
inline IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > \
operator BINARY_OPERATOR_SYMBOL \
( \
   const IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType >& op1, \
   const Factor<GRAPHICAL_MODEL> & op2  \
) { \
   IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > tmp; \
   opengm::operateBinary(op1, op2, tmp, BINARY_FUNCTOR_NAME <typename GRAPHICAL_MODEL::ValueType> ()); \
   return tmp; \
} \
template<class GRAPHICAL_MODEL> \
inline IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > \
operator BINARY_OPERATOR_SYMBOL \
( \
   const Factor<GRAPHICAL_MODEL>& op1, \
   const Factor<GRAPHICAL_MODEL>& op2 \
) { \
   IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > tmp; \
   opengm::operateBinary(op1, op2, tmp, BINARY_FUNCTOR_NAME <typename GRAPHICAL_MODEL::ValueType> ()); \
   return tmp; \
} \
template<class GRAPHICAL_MODEL> \
inline IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > \
operator BINARY_OPERATOR_SYMBOL \
( \
   const Factor<GRAPHICAL_MODEL>& op1, \
   const typename GRAPHICAL_MODEL::ValueType & op2 \
) { \
   IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > tmp; \
   opengm::operateBinary(op1, op2, tmp, BINARY_FUNCTOR_NAME <typename GRAPHICAL_MODEL::ValueType> ()); \
   return tmp; \
} \
template<class GRAPHICAL_MODEL> \
inline IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > \
operator BINARY_OPERATOR_SYMBOL \
( \
   const typename GRAPHICAL_MODEL::ValueType & op1, \
   const Factor<GRAPHICAL_MODEL>& op2 \
) { \
   IndependentFactor<typename GRAPHICAL_MODEL::ValueType, typename GRAPHICAL_MODEL::IndexType, typename GRAPHICAL_MODEL::LabelType > tmp; \
   opengm::operateBinary(op1, op2, tmp, BINARY_FUNCTOR_NAME <typename GRAPHICAL_MODEL::ValueType> ()); \
   return tmp; \
} \

OPENGM_INDEPENDENT_FACTOR_OPERATION_GENERATION( ||, |=, std::logical_or)
OPENGM_INDEPENDENT_FACTOR_OPERATION_GENERATION( &&, &=, std::logical_and)
OPENGM_INDEPENDENT_FACTOR_OPERATION_GENERATION( + , +=, std::plus)
OPENGM_INDEPENDENT_FACTOR_OPERATION_GENERATION( - , -=, std::minus)
OPENGM_INDEPENDENT_FACTOR_OPERATION_GENERATION( * , *=, std::multiplies)
OPENGM_INDEPENDENT_FACTOR_OPERATION_GENERATION( / , /=, std::divides)

/// \cond HIDDEN_SYMBOLS

namespace functionwrapper {

   namespace executor {

      template<size_t IX, size_t DX>
      template<class GM, class FACTOR>
      void FactorInvariant<IX, DX, false>::op
      (
         const GM & gm, 
         const FACTOR & factor
      ) {
         typedef typename GM::IndexType IndexType;
         typedef typename GM::LabelType LabelType;
         if(factor.functionType() == IX) {
            const IndexType functionIndex     = static_cast<IndexType>(factor.functionIndex());
            const size_t numVar               = static_cast<size_t>(factor.numberOfVariables());
            const size_t dimFunction          = static_cast<size_t>(meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[functionIndex].dimension());
            const IndexType numberOfFunctions = static_cast<IndexType>(meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_.size());

            OPENGM_CHECK_OP(functionIndex , < ,numberOfFunctions,
               "function index must be smaller than numberOfFunctions for that given function type")
            OPENGM_CHECK_OP(numVar , ==  ,dimFunction,
               "number of variable indices of the factor must match the functions dimension")
            for(size_t i = 0; i < numVar; ++i) {
               const LabelType numberOfLabelsOfFunction = meta::FieldAccess::template byIndex<IX> (gm.functionDataField_).functionData_.functions_[functionIndex].shape(i);
               OPENGM_CHECK_OP(factor.numberOfLabels(i) , == , numberOfLabelsOfFunction,
                  "number of labels of the variables in a factor must match the functions shape")
            }
         }
         else {
            FactorInvariant
            <
               meta::Increment<IX>::value, 
               DX, 
               meta::EqualNumber< meta::Increment<IX>::value, DX>::value
            >::op(gm, factor);
         }
      }

      template<size_t IX, size_t DX>
      template<class GM, class FACTOR>
      void FactorInvariant<IX, DX, true>::op
      (
         const GM & gm, 
         const FACTOR & factor
      ) {
         throw RuntimeError("Incorrect function type id.");
      }

   } // namespace executor
} // namespace functionwrapper

/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_GRAPHICALMODEL_FACTOR_HXX
