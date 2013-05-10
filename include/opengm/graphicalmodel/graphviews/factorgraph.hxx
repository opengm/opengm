#pragma once
#ifndef OPENGM_FACTORGRAPH_HXX
#define OPENGM_FACTORGRAPH_HXX

#include <algorithm>
#include <limits>

#include "opengm/utilities/accessor_iterator.hxx"
#include "opengm/datastructures/randomaccessset.hxx"
#include "opengm/datastructures/partition.hxx"

#include <typeinfo>
namespace opengm {

/// Interface that makes an object of type S (the template parameter) 
/// look like a (non-editable) factor graph.
template<class S,class I>
class FactorGraph {
private:
   class VariableAccessor;
   class FactorAccessor;
   typedef I IndexType;
public:
   typedef S SpecialType;
   typedef AccessorIterator<VariableAccessor, true> ConstVariableIterator;
   typedef AccessorIterator<FactorAccessor, true> ConstFactorIterator;

   // required interface of S (the template parameter)
   size_t numberOfVariables() const;
   size_t numberOfVariables(const size_t) const;
   size_t numberOfFactors() const;
   size_t numberOfFactors(const size_t) const;
   size_t variableOfFactor(const size_t, const size_t) const;
   size_t factorOfVariable(const size_t, const size_t) const;

   // functions that need not be member functions of S (the template parameter)
   ConstVariableIterator variablesOfFactorBegin(const size_t) const;
   ConstVariableIterator variablesOfFactorEnd(const size_t) const;
   ConstFactorIterator factorsOfVariableBegin(const size_t) const;
   ConstFactorIterator factorsOfVariableEnd(const size_t) const;
   bool variableFactorConnection(const size_t, const size_t) const;
   bool factorVariableConnection(const size_t, const size_t) const;
   bool variableVariableConnection(const size_t, const size_t) const;
   bool factorFactorConnection(const size_t, const size_t) const;
   bool isAcyclic() const;
   bool isConnected(marray::Vector<size_t>& representatives) const;
   bool isChain(marray::Vector<size_t>&) const;
   bool isGrid(marray::Matrix<size_t>&) const;

   size_t maxFactorOrder() const;
   bool maxFactorOrder(const size_t maxOrder) const;
   size_t numberOfNthOrderFactorsOfVariable(const size_t, const size_t) const;
   size_t numberOfNthOrderFactorsOfVariable(const size_t, const size_t, marray::Vector<size_t>&) const;
   size_t secondVariableOfSecondOrderFactor(const size_t, const size_t) const;

   // export functions
   void variableAdjacencyMatrix(marray::Matrix<bool>&) const;
   void variableAdjacencyList(std::vector<std::set<IndexType> >&) const;
   void variableAdjacencyList(std::vector<RandomAccessSet<IndexType> >&) const;
   void factorAdjacencyList(std::vector<std::set<IndexType>  >&) const;
   void factorAdjacencyList(std::vector<RandomAccessSet<IndexType> >&) const;

protected:
   // cast operators
   operator S&() 
      { return static_cast<S&>(*this); }
   operator S const&() const 
      { return static_cast<const S&>(*this); }

   template <class LIST>
   bool shortestPath(const size_t, const size_t, LIST&, const LIST& = LIST()) const;
   template <class LIST>
   bool twoHopConnected(const size_t, const size_t, LIST&) const;

private:
   class VariableAccessor {
   public:
      typedef size_t value_type;

      VariableAccessor(const FactorGraph<S,I>* factorGraph = NULL, const size_t factor = 0)
         : factorGraph_(factorGraph), factor_(factor) 
         {}
      VariableAccessor(const FactorGraph<S,I>& factorGraph, const size_t factor = 0)
         : factorGraph_(&factorGraph), factor_(factor) 
         {}
      size_t size() const
         { OPENGM_ASSERT(factorGraph_ != NULL);
           return factorGraph_->numberOfVariables(factor_); }
      const size_t operator[](const size_t number) const
         { OPENGM_ASSERT(factorGraph_ != NULL);
           return factorGraph_->variableOfFactor(factor_, number); }
      bool operator==(const VariableAccessor& a) const
         { OPENGM_ASSERT(factorGraph_ != NULL);
           return factor_ == a.factor_ && factorGraph_ == a.factorGraph_; }

   private:
      const FactorGraph<S,I>* factorGraph_;
      size_t factor_;
   };

   class FactorAccessor {
   public:
      typedef I value_type;

      FactorAccessor(const FactorGraph<S,I>* factorGraph = NULL, const size_t variable = 0)
         : factorGraph_(factorGraph), variable_(variable) 
         {}
      FactorAccessor(const FactorGraph<S,I>& factorGraph, const size_t variable = 0)
         : factorGraph_(&factorGraph), variable_(variable) 
         {}
      size_t size() const
         { OPENGM_ASSERT(factorGraph_ != NULL);
           return factorGraph_->numberOfFactors(variable_); }
      const size_t operator[](const size_t number) const
         { OPENGM_ASSERT(factorGraph_ != NULL);
           return factorGraph_->factorOfVariable(variable_, number); }
      bool operator==(const FactorAccessor& a) const
         { OPENGM_ASSERT(factorGraph_ != NULL);
           return variable_ == a.variable_ && factorGraph_ == a.factorGraph_; }

   private:
      const FactorGraph<S,I>* factorGraph_;
      size_t variable_;
   };

   template<class LIST>
      void templatedVariableAdjacencyList(LIST&) const;
   template<class LIST>
      void templatedFactorAdjacencyList(LIST&) const;
};

/// \brief total number of variable nodes in the factor graph
/// \return number of variable nodes
template<class S,class I>
inline size_t 
FactorGraph<S,I>::numberOfVariables() const
{
   return static_cast<const SpecialType&>(*this).numberOfVariables();
}

/// \brief number of variable nodes connected to a factor node
/// \param factor factor index
/// \return number of variable nodes 
template<class S,class I>
inline size_t 
FactorGraph<S,I>::numberOfVariables
(
   const size_t factor
) const
{
   return static_cast<const SpecialType&>(*this).numberOfVariables(factor);
}

/// \brief total number of factor nodes in the factor graph
/// \return number of factor nodes 
template<class S,class I>
inline size_t 
FactorGraph<S,I>::numberOfFactors() const
{
   return static_cast<const SpecialType&>(*this).numberOfFactors();
}

/// \brief number of factor nodes connected to a variable node
/// \param variable variable index
/// \return number of factor nodes 
template<class S,class I>
inline size_t 
FactorGraph<S,I>::numberOfFactors
(
   const size_t variable
) const
{
   return static_cast<const SpecialType&>(*this).numberOfFactors(variable);
}

/// \brief j-th variable node connected to a factor node
/// \param factor factor index
/// \param j number of the variable w.r.t. the factor
/// \return variable index
template<class S,class I>
inline size_t 
FactorGraph<S,I>::variableOfFactor
(
   const size_t factor, 
   const size_t j
) const
{
   return static_cast<const SpecialType&>(*this).variableOfFactor(factor, j);
}

/// \brief j-th factor node connected to a variable node
/// \param variable variable index
/// \param j number of the factor w.r.t. the variable
/// \return factor index
template<class S,class I>
inline size_t 
FactorGraph<S,I>::factorOfVariable
(
   const size_t variable, 
   const size_t j
) const
{
   return static_cast<const SpecialType&>(*this).factorOfVariable(variable, j);
}

/// \brief constant iterator to the beginning of the squence of variables connected to a factor
/// \param factor factor index
/// \return iterator
template<class S,class I>
inline typename FactorGraph<S,I>::ConstVariableIterator 
FactorGraph<S,I>::variablesOfFactorBegin
(
   const size_t factor
) const
{
   VariableAccessor accessor(this, factor);
   return ConstVariableIterator(accessor);
}

/// \brief constant iterator to the end of the squence of variables connected to a factor
/// \param factor factor index
/// \return iterator
template<class S,class I>
inline typename FactorGraph<S,I>::ConstVariableIterator 
FactorGraph<S,I>::variablesOfFactorEnd
(
   const size_t factor
) const
{
   VariableAccessor accessor(this, factor);
   return ConstVariableIterator(accessor, numberOfVariables(factor));
}

/// \brief constant iterator to the beginning of the squence of factors connected to a variable
/// \param variable variable index
/// \return iterator
template<class S,class I>
inline typename FactorGraph<S,I>::ConstFactorIterator 
FactorGraph<S,I>::factorsOfVariableBegin
(
   const size_t variable
) const
{
   FactorAccessor accessor(this, variable);
   return ConstFactorIterator(accessor);
}

/// \brief constant iterator to the end of the squence of factors connected to a variable
/// \param variable variable index
/// \return iterator
template<class S,class I>
inline typename FactorGraph<S,I>::ConstFactorIterator 
FactorGraph<S,I>::factorsOfVariableEnd
(
   const size_t variable
) const
{
   FactorAccessor accessor(this, variable);
   return ConstFactorIterator(accessor, numberOfFactors(variable));
}

/// \brief return true if a factor is connected to a variable
/// \param variable variable index
/// \param factor factor index
/// \return result
template<class S,class I>
inline bool 
FactorGraph<S,I>::variableFactorConnection
(
   const size_t variable,
   const size_t factor
) const
{
   OPENGM_ASSERT(factor < numberOfFactors());
   OPENGM_ASSERT(variable < numberOfVariables());
   if(!NO_DEBUG) {
      for(size_t j=1; j<numberOfVariables(factor); ++j) {
         OPENGM_ASSERT(variableOfFactor(factor, j-1) < variableOfFactor(factor, j));
      }
   }
   return std::binary_search(variablesOfFactorBegin(factor), 
      variablesOfFactorEnd(factor), variable);
   return false;
}

/// \brief return true if a variable is connected to a factor
/// \param factor factor index
/// \param variable variable index
/// \return result
template<class S,class I>
inline bool 
FactorGraph<S,I>::factorVariableConnection
(
   const size_t factor, 
   const size_t variable
) const
{
   OPENGM_ASSERT(factor < numberOfFactors());
   OPENGM_ASSERT(variable < numberOfVariables());
   return variableFactorConnection(variable, factor);
}

/// \brief return true if a variable is connected to a variable
/// \param variable1 variable index
/// \param variable2 variable index
/// \return result
template<class S,class I>
inline bool 
FactorGraph<S,I>::variableVariableConnection
(
   const size_t variable1, 
   const size_t variable2
) const
{
   OPENGM_ASSERT(variable1 < numberOfVariables());
   OPENGM_ASSERT(variable2 < numberOfVariables());
   if(variable1 != variable2) {
      ConstFactorIterator it1 = factorsOfVariableBegin(variable1);
      ConstFactorIterator it2 = factorsOfVariableBegin(variable2);
      while(it1 != factorsOfVariableEnd(variable1) && it2 != factorsOfVariableEnd(variable2)) {
         if(*it1 < *it2) {
            ++it1;
         }
         else if(*it1 == *it2) {
            return true;
         }
         else {
            ++it2;
         }
      }
   }
   return false;
}

/// \brief return true if the factor graph (!) is acyclic
/// \return result
template<class S,class I>
bool
FactorGraph<S,I>::isAcyclic() const
{
   const size_t NO_FACTOR = numberOfFactors();
   const size_t NO_VARIABLE = numberOfVariables();
   const size_t ROOT_FACTOR = numberOfVariables() + 1;
   std::vector<size_t> factorFathers(numberOfFactors(), NO_VARIABLE);
   std::vector<size_t> variableFathers(numberOfVariables(), NO_FACTOR);
   std::queue<size_t> factorQueue;
   std::queue<size_t> variableQueue;
   for(size_t F = 0; F < numberOfFactors(); ++F) {
      if(factorFathers[F] == NO_VARIABLE) {
         factorFathers[F] = ROOT_FACTOR;
         factorQueue.push(F);
         while(!factorQueue.empty()) {
            while(!factorQueue.empty()) {
               const size_t f = factorQueue.front();
               factorQueue.pop();
               for(size_t j = 0; j < numberOfVariables(f); ++j) {
                  const size_t v = variableOfFactor(f, j);
                  if(variableFathers[v] == NO_FACTOR) {
                     variableFathers[v] = f;
                     variableQueue.push(v);
                  }
                  else if(factorFathers[f] != v) {
                     return false;
                  }
               }
            }
            while(!variableQueue.empty()) {
               const size_t v = variableQueue.front();
               variableQueue.pop();
               for(size_t j = 0; j < numberOfFactors(v); ++j) {
                  const size_t f = factorOfVariable(v, j);
                  if(factorFathers[f] == NO_VARIABLE) {
                     factorFathers[f] = v;
                     factorQueue.push(f);
                  }
                  else if(variableFathers[v] != f) {
                     return false;
                  }
               }
            }
         }
      }
   }
   return true;
}

/// \brief return true if the factor graph (!) is connected
/// \param[out] representatives A vector of variable id's where each id is a representative of a connected component.
/// \return result
template<class S,class I>
bool
FactorGraph<S,I>::isConnected(marray::Vector<size_t>& representatives) const
{
   // check if factor graph has zero variables
   if(numberOfVariables() == 0) {
      return false;
   }

   // create a partition of all connected components
   Partition<size_t> connectedComponents(numberOfVariables());
   // iterate over all factors
   for(size_t i = 0; i < numberOfFactors(); i++) {
      // iterate over all connected variables of factor and merge them to one partition
      const ConstVariableIterator variablesBegin = variablesOfFactorBegin(i);
      const ConstVariableIterator variablesEnd = variablesOfFactorEnd(i);
      OPENGM_ASSERT(variablesBegin != variablesEnd);

      for(ConstVariableIterator iter = variablesBegin + 1; iter != variablesEnd; iter++) {
         connectedComponents.merge(*(iter - 1), *iter);
      }
   }

   // check number of connected components
   OPENGM_ASSERT(connectedComponents.numberOfSets() > 0);

   representatives = marray::Vector<size_t>(connectedComponents.numberOfSets());
   connectedComponents.representatives(representatives.begin());

   if(connectedComponents.numberOfSets() == 1) {
      return true;
   } else {
      return false;
   }
}

/// \brief return true if the factor graph (!) is a chain
/// \param[out] chainIDs A vector representing the chain, where chain(i) contains the corresponding variable ID.
/// \return result
template<class S,class I>
inline bool
FactorGraph<S,I>::isChain(marray::Vector<size_t>& chainIDs) const {
   const size_t numVariables = numberOfVariables();

   // check if factor graph has zero variables
   if(numVariables == 0) {
      return false;
   }

   // check Factor Order
   if(!maxFactorOrder(2)) {
      return false;
   }

   // special case: graph contains only one variable
   if(numVariables == 1) {
      chainIDs = marray::Vector<size_t>(numVariables);
      chainIDs[0] = 0;
      return true;
   }

   // find ends
   marray::Vector<size_t> ends(2);
   size_t detectedEnds = 0;
   for(size_t i = 0; i < numVariables; i++) {
      size_t countSecondOrderFactors = numberOfNthOrderFactorsOfVariable(i, 2);
      if(countSecondOrderFactors == 1) {
         if(detectedEnds > 1) {
            return false;
         }
         ends[detectedEnds] = i;
         detectedEnds++;
      }
   }

   // two ends found?
   if(detectedEnds != 2) {
      return false;
   }

   chainIDs = marray::Vector<size_t>(numVariables);
   // set ends
   chainIDs[0] = ends[0];
   chainIDs[numVariables - 1] = ends[1];

   // try to traverse from first end to second end
   // set predecessor and successor of ends[0]
   size_t predecessor = ends[0];
   OPENGM_ASSERT(numberOfVariables() < std::numeric_limits<size_t>::max());
   size_t successor = std::numeric_limits<size_t>::max();
   for(ConstFactorIterator iter = factorsOfVariableBegin(ends[0]); iter != factorsOfVariableEnd(ends[0]); iter++) {
      if(numberOfVariables(*iter) == 2) {
         successor = secondVariableOfSecondOrderFactor(ends[0], *iter);
      }
   }

   OPENGM_ASSERT(successor != std::numeric_limits<size_t>::max());

   // traverse chain while successor != ends[1]
   size_t countTraversedVariables = 1;
   while(successor != ends[1]) {
      marray::Vector<size_t> secondOrderFactorIds;
      size_t countSecondOrderFactors = numberOfNthOrderFactorsOfVariable(successor, 2, secondOrderFactorIds);
      if(countSecondOrderFactors > 2) {
         return false;
      }
      // add successor to chainIDs
      chainIDs[countTraversedVariables] = successor;
      countTraversedVariables++;

      // update predecessor and successor
      OPENGM_ASSERT(secondOrderFactorIds.size() == 2);
      for(size_t j = 0; j < 2; j++) {
         size_t possibleSuccesor = secondVariableOfSecondOrderFactor(successor, secondOrderFactorIds[j]);
         if(possibleSuccesor != predecessor) {
            predecessor = successor;
            successor = possibleSuccesor;
            break;
         }
      }
   }

   if(countTraversedVariables != numVariables - 1) {
      // end of chain reached too soon
      return false;
   }
   OPENGM_ASSERT(countTraversedVariables == numVariables - 1);

   // check if last variable is really the expected second end
   OPENGM_ASSERT(chainIDs[numVariables - 1] == ends[1]);

   return true;
}

/// \brief return true if the factor graph (!) is a grid
/// \param[out] gridIDs A matrix representing the grid, where grid(i,j) contains the corresponding variable ID.
/// \return result
template<class S,class I>
inline bool
FactorGraph<S,I>::isGrid(marray::Matrix<size_t>& gridIDs) const {

   // check if factor graph has zero variables
   if(numberOfVariables() == 0) {
      return false;
   }

   // check Factor Order
   if(!maxFactorOrder(2)) {
      return false;
   }

   // special case: graph contains only one variable
   if(numberOfVariables() == 1) {
      gridIDs = marray::Matrix<size_t>(1,1);
      gridIDs(0, 0) = 0;
      return true;
   }

   // check one dimensional case (e.g. graph is a chain)
   marray::Vector<size_t> chainIDs;
   bool graphIsChain = isChain(chainIDs);
   if(graphIsChain) {
      gridIDs = marray::Matrix<size_t>(1, chainIDs.size());
      for(size_t i = 0; i < chainIDs.size(); i++) {
         gridIDs(0, i) = chainIDs[i];
      }
      return true;
   }

   // find corner variables (variables connected to two second order factors)
   // and outer hull variables
   marray::Vector<size_t> cornerIDs(4);
   size_t numCornersFound = 0;
   std::list<size_t> outerHullVariableIDs;

   for(size_t i = 0; i < numberOfVariables(); i++) {
      size_t countSecondOrderFactors = numberOfNthOrderFactorsOfVariable(i, 2);
      if(countSecondOrderFactors == 2) {
         // corner found
         if(numCornersFound > 3) {
            return false;
         }
         cornerIDs(numCornersFound) = i;
         numCornersFound++;
         // corner is also an outer hull variable
         outerHullVariableIDs.push_back(i);
      } else if(countSecondOrderFactors == 3) {
         outerHullVariableIDs.push_back(i);
      } else if(countSecondOrderFactors > 4) {
         // variable is connected to too many other variables
         return false;
      }
   }

   if(numCornersFound < 4) {
      return false;
   }

   OPENGM_ASSERT(numCornersFound == 4);

   // find shortest path from one corner to all other corners
   std::vector<std::list<size_t> > shortestPaths(3);
   if(!shortestPath(cornerIDs(0), cornerIDs(1), shortestPaths[0], outerHullVariableIDs)) {
      return false;
   }

   if(!shortestPath(cornerIDs(0), cornerIDs(2), shortestPaths[1], outerHullVariableIDs)) {
      return false;
   }

   if(!shortestPath(cornerIDs(0), cornerIDs(3), shortestPaths[2], outerHullVariableIDs)) {
      return false;
   }

   // find diagonal corner
   size_t diagonalCorner = 1;
   for(size_t i = 1; i < 4; i++) {
      if(shortestPaths[i - 1].size() > shortestPaths[diagonalCorner].size()) {
         diagonalCorner = i;
      }
   }

   // compute shortest paths from adjacent corners to diagonal corner
   std::vector<std::list<size_t> > shortestAdjacentCornerPaths(2);
   size_t shortestAdjacentCornerPathsComputed = 0;
   for(size_t i = 1; i < 4; i++) {
      if(i != diagonalCorner) {
         if(!shortestPath(cornerIDs(i), cornerIDs(diagonalCorner), shortestAdjacentCornerPaths[shortestAdjacentCornerPathsComputed], outerHullVariableIDs)) {
            return false;
         }
         shortestAdjacentCornerPathsComputed++;
      }
   }
   OPENGM_ASSERT(shortestAdjacentCornerPathsComputed == 2);

   // compute grid dimension
   std::vector<size_t> dimension(2);
   size_t dimensionIndex = 0;
   for(size_t i = 1; i < 4; i++) {
      if(i != diagonalCorner) {
         dimension[dimensionIndex] = shortestPaths[i - 1].size();
         dimensionIndex++;
      }
   }
   OPENGM_ASSERT(dimensionIndex == 2);

   //check dimensions
   if(dimension[0] != shortestAdjacentCornerPaths[1].size()) {
      return false;
   }
   if(dimension[1] != shortestAdjacentCornerPaths[0].size()) {
      return false;
   }

   // create storage
   gridIDs = marray::Matrix<size_t>(dimension[0], dimension[1]);

   // fill outer values
   // from first corner to adjacent corners
   bool transpose = false;
   for(size_t i = 1; i < 4; i++) {
      if(i != diagonalCorner) {
         size_t indexHelper = 0;
         if(transpose == false) {
            for(typename std::list<size_t>::iterator iter = shortestPaths[i - 1].begin(); iter != shortestPaths[i - 1].end(); iter++) {
               gridIDs(indexHelper, 0) = *iter;
               indexHelper++;
            }
            transpose = true;
         } else {
            for(typename std::list<size_t>::iterator iter = shortestPaths[i - 1].begin(); iter != shortestPaths[i - 1].end(); iter++) {
               gridIDs(0, indexHelper) = *iter;
               indexHelper++;
            }
         }

      }
   }

   // from diagonal corner to adjacent corners
   transpose = false;
   for(size_t i = 0; i <= 1; i++) {
      size_t indexHelper = 0;
      if(transpose == false) {
         for(typename std::list<size_t>::iterator iter = shortestAdjacentCornerPaths[i].begin(); iter != shortestAdjacentCornerPaths[i].end(); iter++) {
            gridIDs(dimension[0] - 1, indexHelper) = *iter;
            indexHelper++;
         }
         transpose = true;
      } else {
         for(typename std::list<size_t>::iterator iter = shortestAdjacentCornerPaths[i].begin(); iter != shortestAdjacentCornerPaths[i].end(); iter++) {
            gridIDs(indexHelper, dimension[1] - 1) = *iter;
            indexHelper++;
         }
      }
   }

   // fill inner values
   for(size_t i = 1; i < dimension[0] - 1; i++) {
      for(size_t j = 1; j < dimension[1] - 1; j++) {
         std::vector<size_t> oneHopVariables;
         if(twoHopConnected(gridIDs(i - 1, j), gridIDs(i, j - 1), oneHopVariables)) {
            if(oneHopVariables.size() < 2) {
               return false;
            }
            OPENGM_ASSERT(oneHopVariables.size() == 2);
            if(oneHopVariables[0] != gridIDs(i - 1, j - 1)) {
               gridIDs(i, j) = oneHopVariables[0];
            } else {
               gridIDs(i, j) = oneHopVariables[1];
            }
         } else {
            return false;
         }
      }
   }

   return true;
}

/// \brief return maximum factor order
/// \return maximum factor order
template<class S,class I>
inline size_t
FactorGraph<S,I>::maxFactorOrder() const {
   size_t maxFactorOrder = 0;
   for(size_t i = 0; i < numberOfFactors(); i++) {
      if(numberOfVariables(i) > maxFactorOrder) {
         maxFactorOrder = numberOfVariables(i);
      }
   }
   return maxFactorOrder;
}

/// \brief return true if the maximum factor order is less or equal to maxOrder
/// \param maxOrder maximum allowed factor order
/// \return result
template<class S,class I>
inline bool
FactorGraph<S,I>::maxFactorOrder(const size_t maxOrder) const {
   for(size_t i = 0; i < numberOfFactors(); i++) {
      if(numberOfVariables(i) > maxOrder) {
         return false;
      }
   }
   return true;
}

/// \brief return number of factors with order n which are connected to variable
/// \param variable variable index
/// \param n desired order of factors
/// \return result
template<class S,class I>
inline size_t
FactorGraph<S,I>::numberOfNthOrderFactorsOfVariable(const size_t variable, const size_t n) const {
   OPENGM_ASSERT(variable < numberOfVariables());
   size_t countNthOrderFactors = 0;
   for(ConstFactorIterator iter = factorsOfVariableBegin(variable); iter != factorsOfVariableEnd(variable); iter++) {
      if(numberOfVariables(*iter) == n) {
         countNthOrderFactors++;
      }
   }
   return countNthOrderFactors;
}

/// \brief return number of factors with order n which are connected to variable and stores the corresponding factorIDs
/// \param variable variable index
/// \param n desired order of factors
/// \param[out] factorIDs factorIDs of all n'th order factors connected to a given variable
/// \return result
template<class S,class I>
inline size_t
FactorGraph<S,I>::numberOfNthOrderFactorsOfVariable(const size_t variable, const size_t n, marray::Vector<size_t>& factorIDs) const {
   OPENGM_ASSERT(variable < numberOfVariables());
   // FIXME this might be done more efficiently without numberOfNthOrderFactorsOfVariable(variable, n) if marray::Vector<size_t> would support something like push_back()
   size_t countNthOrderFactors = numberOfNthOrderFactorsOfVariable(variable, n);
   factorIDs = marray::Vector<size_t>(countNthOrderFactors);
   for(ConstFactorIterator iter = factorsOfVariableBegin(variable); iter != factorsOfVariableEnd(variable); iter++) {
      if(numberOfVariables(*iter) == n) {
         countNthOrderFactors--;
         factorIDs[countNthOrderFactors] = *iter;
      }
   }
   OPENGM_ASSERT(countNthOrderFactors == 0);
   return factorIDs.size();
}

/// \brief return returns the id of the second variable which is connected to a given variable via a second order factor
/// \param variable variable index
/// \param factor factor index
/// \return result
template<class S,class I>
inline size_t
FactorGraph<S,I>::secondVariableOfSecondOrderFactor(const size_t variable, const size_t factor) const {
   OPENGM_ASSERT(variable < numberOfVariables());
   OPENGM_ASSERT(factor < numberOfFactors());
   OPENGM_ASSERT(numberOfVariables(factor) == 2);
   OPENGM_ASSERT(variableFactorConnection(variable, factor));
   for(ConstVariableIterator iter = variablesOfFactorBegin(factor); iter != variablesOfFactorEnd(factor); iter++) {
      if(*iter != variable) {
         return *iter;
      }
   }
   return variable;
}

/// \brief return true if a factor is connected to a factor
/// \param factor1 variable index
/// \param factor2 variable index
/// \return result
template<class S,class I>
inline bool 
FactorGraph<S,I>::factorFactorConnection
(
   const size_t factor1, 
   const size_t factor2
) const
{
   OPENGM_ASSERT(factor1 < numberOfFactors());
   OPENGM_ASSERT(factor2 < numberOfFactors());
   if(factor1 != factor2) {
      ConstVariableIterator it1 = variablesOfFactorBegin(factor1);
      ConstVariableIterator it2 = variablesOfFactorBegin(factor2);
      while(it1 != variablesOfFactorEnd(factor1) && it2 != variablesOfFactorEnd(factor2)) {
         if(*it1 < *it2) {
            ++it1;
         }
         else if(*it1 == *it2) {
            return true;
         }
         else {
            ++it2;
         }
      }
   }
   return false;
}

/// \brief outputs the factor graph as a variable adjacency matrix
/// \param out matrix
template<class S,class I>
inline void 
FactorGraph<S,I>::variableAdjacencyMatrix
(
   marray::Matrix<bool>& out
) const
{
   out = marray::Matrix<bool>(numberOfVariables(), numberOfVariables(), false);
   for(size_t factor=0; factor<numberOfFactors(); ++factor) {
      for(size_t j=0; j<numberOfVariables(factor); ++j) {
         for(size_t k=j+1; k<numberOfVariables(factor); ++k) {
            const size_t variable1 = variableOfFactor(factor, j);
            const size_t variable2 = variableOfFactor(factor, k);
            out(variable1, variable2) = true;
            out(variable2, variable1) = true;
         }
      }
   }
}

/// \brief outputs the factor graph as variable adjacency lists
/// \param out  variable adjacency lists (as a vector of RandomAccessSets)
template<class S,class I>
inline void 
FactorGraph<S,I>::variableAdjacencyList
(
   std::vector<RandomAccessSet<IndexType> >& out
) const
{
   templatedVariableAdjacencyList(out);
}

/// \brief outputs the factor graph as variable adjacency lists
/// \param out  variable adjacency lists (as a vector of sets)
template<class S,class I>
inline void 
FactorGraph<S,I>::variableAdjacencyList
(
   std::vector<std::set<IndexType> >& out
) const
{
   templatedVariableAdjacencyList(out);
}

/// \brief outputs the factor graph as variable adjacency lists
/// \param out variable adjacency lists (e.g. std::vector<std::set<size_t> >)
template<class S,class I>
template<class LIST>
inline void 
FactorGraph<S,I>::templatedVariableAdjacencyList
(
   LIST& out
) const
{
   out.clear();
   out.resize(numberOfVariables());
   for(size_t factor=0; factor<numberOfFactors(); ++factor) {
      for(size_t j=0; j<numberOfVariables(factor); ++j) {
         for(size_t k=j+1; k<numberOfVariables(factor); ++k) {
            const size_t variable1 = variableOfFactor(factor, j);
            const size_t variable2 = variableOfFactor(factor, k);
            out[variable1].insert(variable2);
            out[variable2].insert(variable1);
         }
      }
   }
}

template<class S,class I>
inline void 
FactorGraph<S,I>::factorAdjacencyList
(
   std::vector<std::set<IndexType> >& out
) const
{
   templatedFactorAdjacencyList(out);
}

template<class S,class I>
inline void 
FactorGraph<S,I>::factorAdjacencyList
(
   std::vector< RandomAccessSet<IndexType> >& out
) const
{
   templatedFactorAdjacencyList(out);
}

template<class S,class I>
template<class LIST>
inline void 
FactorGraph<S,I>::templatedFactorAdjacencyList
(
   LIST& out
) const
{
   out.clear();
   out.resize(numberOfFactors());
   for(size_t f=0; f<numberOfFactors(); ++f) {
      for(size_t v=0 ;v<numberOfVariables(f); ++v) {
         for(size_t ff=0;ff<numberOfFactors(v);++ff) {
            const size_t fOfVar=factorOfVariable(v,ff);
            if(f!=fOfVar) {
               out[f].insert(fOfVar);
            }
         }
      }
   }
}

/// \brief computes the shortest path from s to t using Dijkstra's algorithm with uniform distances
/// \param s ID of the start variable
/// \param t ID of the target variable
/// \param[out] path returns computed path from s to t
/// \param allowedVariables path is only allowed to contain variables which are given here (if empty, all variables are allowed)
template<class S,class I>
template <class LIST>
inline bool
FactorGraph<S,I>::shortestPath(const size_t s, const size_t t, LIST& path, const LIST& allowedVariables) const {
   OPENGM_ASSERT(s < numberOfVariables());
   OPENGM_ASSERT(t < numberOfVariables());
   OPENGM_ASSERT(allowedVariables.size() <= numberOfVariables());
   OPENGM_ASSERT(numberOfVariables() != std::numeric_limits<size_t>::max());
   const size_t infinity = std::numeric_limits<size_t>::max();

   bool useAllVariables = (allowedVariables.size() == 0) || (allowedVariables.size() == numberOfVariables());

   if(useAllVariables) {
      std::vector<size_t> distances(numberOfVariables(), infinity);
      std::vector<size_t> previous(numberOfVariables(), infinity);
      distances[s] = 0;
      LIST Q;
      for(size_t i = 0; i < numberOfVariables(); i++) {
         Q.push_back(i);
      }
      while(Q.size() !=0) {
         typename LIST::iterator currentIter = Q.begin();
         for(typename LIST::iterator iter = Q.begin(); iter != Q.end(); iter++) {
            if(distances[*iter] < distances[*currentIter]) {
               currentIter = iter;
            }
         }
         if(distances[*currentIter] == infinity) {
            // all remaining variables are inaccessible from s
            return false;
         }
         if(*currentIter == t) {
            // target found
            break;
         }
         size_t currentID = *currentIter;
         Q.erase(currentIter);
         // iterate over all neighbor variables of *current which are still in Q
         for(ConstFactorIterator factorIter = factorsOfVariableBegin(currentID); factorIter != factorsOfVariableEnd(currentID); factorIter++) {
            for(ConstVariableIterator variableIter = variablesOfFactorBegin(*factorIter); variableIter != variablesOfFactorEnd(*factorIter); variableIter++) {
               if(std::find(Q.begin(), Q.end(), *variableIter) != Q.end()) {
                  size_t newDistance = distances[currentID] + 1;
                  if(newDistance < distances[*variableIter]) {
                     distances[*variableIter] = newDistance;
                     previous[*variableIter] = currentID;
                  }
               }
            }
         }
      }
      OPENGM_ASSERT(previous[t] != infinity);
      // create path
      size_t u = t;
      while(previous[u] != infinity) {
         path.push_front(u);
         u = previous[u];
      }
      path.push_front(s);
      return true;
   } else {
      OPENGM_ASSERT(std::find(allowedVariables.begin(), allowedVariables.end(), s) != allowedVariables.end());
      OPENGM_ASSERT(std::find(allowedVariables.begin(), allowedVariables.end(), t) != allowedVariables.end());
      std::vector<size_t> distances(allowedVariables.size(), infinity);
      std::vector<size_t> previous(allowedVariables.size(), infinity);
      std::map<size_t, size_t> local2actualIDs;
      std::map<size_t, size_t> actual2localIDs;
      LIST Q;
      size_t counter = 0;
      for(typename LIST::const_iterator iter = allowedVariables.begin(); iter != allowedVariables.end(); iter++) {
         Q.push_back(counter);
         local2actualIDs.insert(std::pair<size_t, size_t>(counter, *iter));
         actual2localIDs.insert(std::pair<size_t, size_t>(*iter, counter));
         counter++;
      }
      distances[actual2localIDs.find(s)->second] = 0;
      while(Q.size() !=0) {
         typename LIST::iterator currentIter = Q.begin();
         for(typename LIST::iterator iter = Q.begin(); iter != Q.end(); iter++) {
            if(distances[*iter] < distances[*currentIter]) {
               currentIter = iter;
            }
         }
         if(distances[*currentIter] == infinity) {
            // all remaining variables are inaccessible from s
            return false;
         }
         // get actual ID
         size_t actualID = local2actualIDs.find(*currentIter)->second;
         if(actualID == t) {
            // target found
            break;
         }
         size_t currentLocalID = *currentIter;
         Q.erase(currentIter);
         // iterate over all neighbor variables of *current which are in allowedVariables and are still in Q
         for(ConstFactorIterator factorIter = factorsOfVariableBegin(actualID); factorIter != factorsOfVariableEnd(actualID); factorIter++) {
            for(ConstVariableIterator variableIter = variablesOfFactorBegin(*factorIter); variableIter != variablesOfFactorEnd(*factorIter); variableIter++) {
               const std::map<size_t, size_t>::const_iterator actual2localIDsCurrentPosition = actual2localIDs.find(*variableIter);
               if(actual2localIDsCurrentPosition != actual2localIDs.end()) {
                  size_t localID = actual2localIDsCurrentPosition->second;
                  if(std::find(Q.begin(), Q.end(), localID) != Q.end()) {
                     size_t newDistance = distances[currentLocalID] + 1;
                     if(newDistance < distances[localID]) {
                        distances[localID] = newDistance;
                        previous[localID] = currentLocalID;
                     }
                  }
               }
            }
         }
      }
      OPENGM_ASSERT(actual2localIDs.find(t)->second != infinity);
      // create path
      size_t u = actual2localIDs.find(t)->second;
      while(previous[u] != infinity) {

         path.push_front(local2actualIDs.find(u)->second);
         u = previous[u];
      }
      path.push_front(s);
      return true;
   }
   return false;
}

/// \brief checks if variabel1 is connected to variable2 via two hops
/// \param variable1 ID of the first variable
/// \param variable2 ID of the second variable
/// \param[out] oneHopVariables a List of all possible one hop variables in the two hop path from variable1 to variable2
template<class S,class I>
template <class LIST>
inline bool
FactorGraph<S,I>::twoHopConnected(const size_t variable1, const size_t variable2, LIST& oneHopVariables) const {
   OPENGM_ASSERT(variable1 < numberOfVariables());
   OPENGM_ASSERT(variable2 < numberOfVariables());
   oneHopVariables.clear();
   if(variable1 != variable2) {
      for(ConstFactorIterator factorIter = factorsOfVariableBegin(variable1); factorIter != factorsOfVariableEnd(variable1); factorIter++) {
         for(ConstVariableIterator variableIter = variablesOfFactorBegin(*factorIter); variableIter != variablesOfFactorEnd(*factorIter); variableIter++) {
            if((*variableIter != variable1) || (*variableIter != variable2)) {
               if(variableVariableConnection(*variableIter, variable2)) {
                  oneHopVariables.push_back(*variableIter);
               }
            }
         }
      }
   }
   if(oneHopVariables.size() == 0) {
      return false;
   } else {
      return true;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_FACTORGRAPH_HXX
