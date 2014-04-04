#pragma once
#ifndef OPENGM_GRAPHICALMODELDECOMPOSER_HXX
#define OPENGM_GRAPHICALMODELDECOMPOSER_HXX

#include <exception>
#include <set>
#include <vector>
#include <list>
#include <map>
#include <queue>
#include <limits>

#include "opengm/datastructures/partition.hxx"
#include "opengm/graphicalmodel/decomposition/graphicalmodeldecomposition.hxx"
#include "opengm/functions/scaled_view.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

template <class GM>
class GraphicalModelDecomposer
{
public:
   typedef GM                                              GraphicalModelType;
   typedef GraphicalModelDecomposition                     DecompositionType;
   typedef typename GraphicalModelType::FactorType                     FactorType;
   typedef typename GraphicalModelType::FunctionIdentifier             FunctionIdentifierType;
   typedef typename DecompositionType::SubFactor           SubFactorType;
   typedef typename DecompositionType::SubVariable         SubVariableType;   
   typedef typename DecompositionType::SubFactorListType   SubFactorListType;
   typedef typename DecompositionType::SubVariableListType SubVariableListType;

   typedef typename GraphicalModelType::ValueType                      ValueType;
   typedef typename GraphicalModelType::OperatorType                   OperatorType;
   //typedef ScaledViewFunction<GraphicalModelType>                      ViewFunctionType;
   //typedef GraphicalModel<ValueType, OperatorType, ViewFunctionType> SubGmType;

   // Constructor
   GraphicalModelDecomposer();

   // Decompose Methods 
   DecompositionType decomposeManual(const GraphicalModelType&, const std::vector<std::vector<size_t> >& subModelFactors) const;
   DecompositionType decomposeIntoTree(const GraphicalModelType&) const; 
   DecompositionType decomposeIntoSpanningTrees(const GraphicalModelType&) const; 
   DecompositionType decomposeIntoKFans(const GraphicalModelType&, const std::vector<std::set<size_t> >&) const;
   DecompositionType decomposeIntoKFans(const GraphicalModelType&, const size_t k) const;
   DecompositionType decomposeIntoClosedBlocks(const GraphicalModelType&, const std::vector<std::set<size_t> >&) const;
   DecompositionType decomposeIntoOpenBlocks(const GraphicalModelType&, const std::vector<std::set<size_t> >&) const;
   DecompositionType decomposeIntoClosedBlocks(const GraphicalModelType&, const size_t) const;
};

template <class GM> 
inline GraphicalModelDecomposer<GM>::
GraphicalModelDecomposer()
{}
 
template <class GM> 
inline typename GraphicalModelDecomposer<GM>::DecompositionType GraphicalModelDecomposer<GM>::
decomposeManual
(
   const GraphicalModelType& gm, 
   const std::vector<std::vector<size_t> >& subModelFactors
) const
{      
   DecompositionType decomposition = GraphicalModelDecomposition(gm.numberOfVariables(),gm.numberOfFactors(),0);
   for(size_t subModel=0; subModel<subModelFactors.size(); ++subModel) {
      decomposition.addSubModel();  
   }
   for(size_t subModel=0; subModel<subModelFactors.size(); ++subModel) {
      std::vector<size_t> subVariableIds(gm.numberOfVariables(),gm.numberOfVariables());
      //for(size_t factorId=0; factorId<subModelFactors[subModel].size(); ++factorId) {
      for(size_t nn=0; nn<subModelFactors[subModel].size(); ++nn) {
         size_t factorId = subModelFactors[subModel][nn];
         std::vector<size_t> subVariableIndices(gm[factorId].numberOfVariables());
         for(size_t j=0; j<gm[factorId].numberOfVariables(); ++j) {
            const size_t variableIndex = gm[factorId].variableIndex(j);
            if(subVariableIds[variableIndex] == gm.numberOfVariables()) {
               subVariableIds[variableIndex] = decomposition.addSubVariable(subModel,variableIndex);
            }
            subVariableIndices[j] = subVariableIds[variableIndex]; 
         }
         decomposition.addSubFactor( subModel, factorId, subVariableIndices );   
      }
   }
   decomposition.reorder();
   return decomposition;
}

template <class GM>
inline typename GraphicalModelDecomposer<GM>::DecompositionType GraphicalModelDecomposer<GM>::
decomposeIntoTree
(
   const GraphicalModelType& gm
) const
{ 
   DecompositionType decomposition = GraphicalModelDecomposition(gm.numberOfVariables(),gm.numberOfFactors(),0);

   decomposition.addSubModel(); 
   Partition<size_t> partition(gm.numberOfVariables()); 
   for(size_t variableId=0; variableId<gm.numberOfVariables();++variableId) {
      decomposition.addSubVariable(0,variableId);
   }
   for(size_t factorId=0; factorId<gm.numberOfFactors(); ++factorId) {
      std::map<size_t, size_t> counters;
      std::vector<size_t> subVariableIndices(gm[factorId].numberOfVariables());
      for(size_t j=0; j<gm[factorId].numberOfVariables(); ++j) { 
         const size_t variableIndex = gm[factorId].variableIndex(j);
         if( ++counters[partition.find(variableIndex)] > 1) {
            subVariableIndices[j] = decomposition.addSubVariable(0,variableIndex);
         }
         else{
            subVariableIndices[j] = variableIndex;
         }
      }
      decomposition.addSubFactor(0,factorId,subVariableIndices);
      for(size_t j=1; j<gm[factorId].numberOfVariables(); ++j) {
         partition.merge(gm[factorId].variableIndex(j-1), gm[factorId].variableIndex(j));
      }
   } 
   decomposition.reorder();
   return decomposition;
}
   
template <class GM>
inline typename GraphicalModelDecomposer<GM>::DecompositionType GraphicalModelDecomposer<GM>::
decomposeIntoSpanningTrees
(
   const GraphicalModelType& gm
) const
{   
   DecompositionType decomposition = GraphicalModelDecomposition(gm.numberOfVariables(),gm.numberOfFactors(),0);
    
   std::list<size_t> blackList;
   std::list<size_t> grayList;
   std::list<size_t> whiteList;
   for(size_t j=0; j<gm.numberOfFactors(); ++j) {
      whiteList.push_back(j);
   }
     
   while(!whiteList.empty()) {
      size_t subModelId  = decomposition.addSubModel();
      for(size_t variableId=0; variableId<gm.numberOfVariables();++variableId) {
         decomposition.addSubVariable(subModelId,variableId);
      }
        
      Partition<size_t> partition(gm.numberOfVariables());
      std::list<size_t>* currentList;

      for(size_t listSwap=0; listSwap<2; ++listSwap) {
         if(listSwap == 0) {
            currentList = &whiteList; // add white factors in the first round
         } 
         else {
            currentList = &blackList; // add black factors in the second round
         } 

         std::list<size_t>::iterator it = currentList->begin();
         while(it != currentList->end()) {
            // check if *it can be inserted into the submodel
            bool insert = true;
            const FactorType& factor = gm[*it];
            std::map<size_t, size_t> counters; 
            for(size_t j=0; j<factor.numberOfVariables(); ++j) { 
               if( ++counters[partition.find(factor.variableIndex(j))] > 1) { 
                  //factor has 2 variabels of the same set
                  insert = false;
                  break;
               }
            }
            if(insert) {
               std::vector<size_t> subVariableIndices(factor.numberOfVariables());
               for(size_t j=0; j<factor.numberOfVariables(); ++j) {
                  const size_t variableId = factor.variableIndex(j);
                  subVariableIndices[j] = variableId;	    
               }
               decomposition.addSubFactor(subModelId,(*it),subVariableIndices);
                 
               if(currentList == &whiteList) {
                  grayList.push_back(*it);
                  it = currentList->erase(it);
               }
               else {
                  ++it;
               }
               for(size_t j=1; j<factor.numberOfVariables(); ++j) {
                  partition.merge(factor.variableIndex(j-1), factor.variableIndex(j));
               }
            }
            else {
               ++it;
            }
         }
      }
      blackList.insert(blackList.end(), grayList.begin(), grayList.end());
      grayList.clear();
   }
   return decomposition;
} 
 
template <class GM>
inline typename GraphicalModelDecomposer<GM>::DecompositionType GraphicalModelDecomposer<GM>::
decomposeIntoKFans
(
   const GraphicalModelType& gm, 
   const size_t k
) const
{ 
   DecompositionType decomposition = GraphicalModelDecomposition(gm.numberOfVariables(),gm.numberOfFactors(),0);
    
   const size_t numberOfVariables   = gm.numberOfVariables();
   const size_t numberOfSubproblems = (size_t)(ceil((double)(numberOfVariables)/(double)(k)));
   std::vector<std::set<size_t> > innerFanVariables(numberOfSubproblems);
   size_t counter = 0;
   for(size_t subModelId=0; subModelId<numberOfSubproblems; ++subModelId) {
      for(size_t i=0; i<k; ++i) {
         innerFanVariables[subModelId].insert(counter);
         counter = (counter+1) % numberOfVariables;
      }
   }
   return decomposeIntoKFans(gm, innerFanVariables);
}
   
template <class GM>
inline typename GraphicalModelDecomposer<GM>::DecompositionType GraphicalModelDecomposer<GM>::
decomposeIntoKFans
(
   const GraphicalModelType& gm, 
   const std::vector<std::set<size_t> >& innerFanVariables
) const
{ 
   DecompositionType decomposition = GraphicalModelDecomposition(gm.numberOfVariables(),gm.numberOfFactors(),0);
    
   //const size_t numberOfVariables   = gm.numberOfVariables();
   const size_t numberOfSubproblems = innerFanVariables.size();
     
   for(size_t subModelId=0;subModelId<numberOfSubproblems;++subModelId) {
      decomposition.addSubModel();
      for(size_t variableId=0; variableId<gm.numberOfVariables();++variableId) {
         decomposition.addSubVariable(subModelId,variableId);
      }

      // find factors of subproblems
      for(size_t factorId=0; factorId<gm.numberOfFactors(); ++factorId) {
         if(gm[factorId].numberOfVariables()==0) {
            std::vector<size_t> subVariableIndices(0);
            decomposition.addSubFactor(subModelId,factorId,subVariableIndices);
         }
         else if(gm[factorId].numberOfVariables()==1) {
            std::vector<size_t> subVariableIndices(1,gm[factorId].variableIndex(0));
            decomposition.addSubFactor(subModelId,factorId,subVariableIndices);
         }
         else if(gm[factorId].numberOfVariables()==2) {
            if(  (innerFanVariables[subModelId].count(gm[factorId].variableIndex(0)) > 0 )  ||
                  (innerFanVariables[subModelId].count(gm[factorId].variableIndex(1)) > 0 )) {
               std::vector<size_t> subVariableIndices(2);
               subVariableIndices[0] = gm[factorId].variableIndex(0);  
               subVariableIndices[1] = gm[factorId].variableIndex(1); 
               decomposition.addSubFactor(subModelId,factorId,subVariableIndices);
            }
         }
         else{
            throw RuntimeError("The k-fan decomposition currently supports only models of order <= 2.");
         }
      }
   }
   return decomposition;
} 
 
template <class GM>
inline typename GraphicalModelDecomposer<GM>::DecompositionType GraphicalModelDecomposer<GM>::
decomposeIntoOpenBlocks
(
   const GraphicalModelType& gm, 
   const std::vector<std::set<size_t> >& innerVariables
) const
{ 
   DecompositionType decomposition = GraphicalModelDecomposition(gm.numberOfVariables(),gm.numberOfFactors(),0);
    
   const size_t numberOfVariables   = gm.numberOfVariables();
   const size_t numberOfSubproblems = innerVariables.size();
   std::vector<size_t> subVariableMap(numberOfVariables);
     
   for(size_t subModelId=0;subModelId<numberOfSubproblems;++subModelId) {
      decomposition.addSubModel();
      for(typename std::vector<size_t>::iterator it = subVariableMap.begin(); it !=subVariableMap.end(); ++it)
         *it = std::numeric_limits<std::size_t>::max();
      for(typename std::set<size_t>::const_iterator it=innerVariables[subModelId].begin();it!=innerVariables[subModelId].end(); ++it)
         subVariableMap[*it]=decomposition.addSubVariable(subModelId,*it);
       
      // find factors of subproblems
      for(size_t factorId=0; factorId<gm.numberOfFactors(); ++factorId) {
         if(gm[factorId].numberOfVariables()==0) {
            std::vector<size_t> subVariableIndices(0);
            decomposition.addSubFactor(subModelId,factorId,subVariableIndices);
         }
         else{
            bool test = true; 
            for(size_t i=0; i<gm[factorId].numberOfVariables(); ++i) {
               test = test && (subVariableMap[gm[factorId].variableIndex(i)] != std::numeric_limits<std::size_t>::max());
            }
            if(test) { 
               std::vector<size_t> subVariableIndices(gm[factorId].numberOfVariables()); 
               for(size_t i=0; i<gm[factorId].numberOfVariables(); ++i) {
                  subVariableIndices[i] = subVariableMap[gm[factorId].variableIndex(i)];
               }
               decomposition.addSubFactor(subModelId,factorId,subVariableIndices);
            }
         }
      }         
   }
   return decomposition;
}
 
template <class GM>
inline typename GraphicalModelDecomposer<GM>::DecompositionType GraphicalModelDecomposer<GM>::
decomposeIntoClosedBlocks
(
   const GraphicalModelType& gm, 
   const size_t numBlocks
) const
{
   std::vector<std::set<size_t> > innerVariables(numBlocks);
   double fractalBlocksize = (1.0*gm.numberOfVariables())/numBlocks;
   size_t var = 0;
   for(size_t i=0; i<numBlocks;++i) {
      while(var < fractalBlocksize*(i+1)+0.0000001 && var < gm.numberOfVariables()) {
         innerVariables[i].insert(var++);  
      }
      if( var != gm.numberOfVariables()) {
         --var;
      }
   } 
   return decomposeIntoClosedBlocks(gm,innerVariables);  
}

template <class GM>
inline typename GraphicalModelDecomposer<GM>::DecompositionType GraphicalModelDecomposer<GM>::
decomposeIntoClosedBlocks
(
   const GraphicalModelType& gm, 
   const std::vector<std::set<size_t> >& innerVariables
) const
{ 
   DecompositionType decomposition = GraphicalModelDecomposition(gm.numberOfVariables(),gm.numberOfFactors(),0);
    
   const size_t numberOfVariables   = gm.numberOfVariables();
   const size_t numberOfSubproblems = innerVariables.size();
   std::vector<size_t> subVariableMap(numberOfVariables);
     
   for(size_t subModelId=0;subModelId<numberOfSubproblems;++subModelId) {
      decomposition.addSubModel();
      for(typename std::vector<size_t>::iterator it = subVariableMap.begin(); it !=subVariableMap.end(); ++it)
         *it = std::numeric_limits<std::size_t>::max();
      for(typename std::set<size_t>::const_iterator it=innerVariables[subModelId].begin();it!=innerVariables[subModelId].end(); ++it)
         subVariableMap[*it]=decomposition.addSubVariable(subModelId,*it);
       
      // find factors of subproblems
      for(size_t factorId=0; factorId<gm.numberOfFactors(); ++factorId) {
         if(gm[factorId].numberOfVariables()==0) {
            std::vector<size_t> subVariableIndices(0);
            decomposition.addSubFactor(subModelId,factorId,subVariableIndices);
         }
         else{
            bool test = false; 
            for(size_t i=0; i<gm[factorId].numberOfVariables(); ++i) {
               test = test || (innerVariables[subModelId].count(gm[factorId].variableIndex(i))>0);
            }
            if(test) { 
               std::vector<size_t> subVariableIndices(gm[factorId].numberOfVariables()); 
               for(size_t i=0; i<gm[factorId].numberOfVariables(); ++i) {
                  const size_t varId = gm[factorId].variableIndex(i);
                  if(subVariableMap[varId] == std::numeric_limits<std::size_t>::max()) {
                     subVariableMap[varId] = decomposition.addSubVariable(subModelId,varId);
                  }
                  subVariableIndices[i] = subVariableMap[gm[factorId].variableIndex(i)];
               }
               decomposition.addSubFactor(subModelId,factorId,subVariableIndices);
            }
         }
      }         
   }
   decomposition.reorder();
   return decomposition;
} 

/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_GRAPHICALMODELDECOMPOSER_HXX
