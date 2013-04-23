#pragma once
#ifndef OPENGM_GRAPHICALMODELDECOMPOSITION_HXX
#define OPENGM_GRAPHICALMODELDECOMPOSITION_HXX

#include <vector>
#include <list>
#include <set>
#include <map>
#include <limits>

namespace opengm {

/// \cond HIDDEN_SYMBOLS

class GraphicalModelDecomposition
{
public:
   class SubFactor {
   public:
      size_t subModelId_;
      size_t subFactorId_;
      std::vector<size_t> subIndices_;
      SubFactor(const size_t&, const size_t&, const std::vector<size_t>&); 
      SubFactor(const size_t&, const std::vector<size_t>&);
   };  
   typedef SubFactor EmptySubFactor;
/*
   class EmptySubFactor {
   public:
      size_t subModelId_;
      size_t subFactorId_;
      std::vector<size_t> subIndices_;
      EmptySubFactor(const size_t&, const std::vector<size_t>&);
   }; 
*/
   class SubVariable {
   public:
      size_t subModelId_;
      size_t subVariableId_;
      SubVariable(const size_t&, const size_t&);
   };

   typedef std::list<SubFactor>      SubFactorListType;
   typedef std::list<SubFactor>      EmptySubFactorListType;
   typedef std::list<SubVariable>    SubVariableListType;
   
   GraphicalModelDecomposition();
   GraphicalModelDecomposition(const size_t numVariables, const size_t numFactors, const size_t numSubModels=0);
   size_t addSubModel();
   size_t addSubFactor(const size_t& subModel, const size_t& factorId, const std::vector<size_t>& subIndices);
   size_t addSubFactor(const size_t& subModel, const std::vector<size_t>& indices, const std::vector<size_t>& subIndices);
   size_t addSubVariable(const size_t& subModel, const size_t& variableId);

   void reorder(); 
   void complete();
   template <class GM> bool isValid(const GM&) const;
   const std::vector<SubFactorListType>& getFactorLists() const                             
      { return subFactorLists_; }
   const std::map<std::vector<size_t>,EmptySubFactorListType>& getEmptyFactorLists() const  
      { return emptySubFactorLists_; }
   const std::vector<SubVariableListType>& getVariableLists() const                         
      {return subVariableLists_; }
   size_t numberOfSubModels() const                   
      { return numberOfSubModels_; }
   size_t numberOfSubVariables(size_t subModel) const 
      { return numberOfSubVariables_[subModel]; }
   size_t numberOfSubFactors(size_t subModel) const   
      { return numberOfSubFactors_[subModel]; }

private:
   size_t                                               numberOfVariables_;
   size_t                                               numberOfFactors_;
   size_t                                               numberOfSubModels_;
   std::vector<size_t>                                  numberOfSubFactors_;    //vectorsize = numberOfModels
   std::vector<size_t>                                  numberOfSubVariables_;  //vectorsize = numberOfModels
   std::vector<SubFactorListType>                       subFactorLists_;        //vectorsize = numberOfFactors
   std::vector<SubVariableListType>                     subVariableLists_;      //vectorsize = numberOfVariabels
   std::map<std::vector<size_t>,EmptySubFactorListType> emptySubFactorLists_;   //mapindex   = realFactorIndices
};

inline   
GraphicalModelDecomposition::SubFactor::
SubFactor
(
   const size_t& sM, 
   const size_t& sF, 
   const std::vector<size_t>& sI
)
:  subModelId_(sM), 
   subFactorId_(sF), 
   subIndices_(sI)
{}
   
inline
GraphicalModelDecomposition::SubFactor::
SubFactor
(
   const size_t& sM, 
   const std::vector<size_t>& sI
)
:  subModelId_(sM), 
   subIndices_(sI)
{}
    
inline
GraphicalModelDecomposition::SubVariable::
SubVariable
(
   const size_t& sM, 
   const size_t& sV
)
:  subModelId_(sM), 
   subVariableId_(sV)
{}
 
inline
GraphicalModelDecomposition::
GraphicalModelDecomposition()
{
   numberOfVariables_ = 0;
   numberOfFactors_   = 0;
   numberOfSubModels_ = 0;
}
      
inline   
GraphicalModelDecomposition::
GraphicalModelDecomposition
(
   const size_t numNodes, 
   const size_t numFactors, 
   const size_t numSubModels
)
:  numberOfVariables_(numNodes), 
   numberOfFactors_(numFactors), 
   numberOfSubModels_(numSubModels)
{
   numberOfSubFactors_.resize(numberOfSubModels_,0);
   numberOfSubVariables_.resize(numberOfSubModels_,0);
   subFactorLists_.resize(numberOfFactors_); 
   subVariableLists_.resize(numberOfVariables_);
}
   
inline size_t 
GraphicalModelDecomposition::addSubModel()
{  
   numberOfSubFactors_.push_back(0);
   numberOfSubVariables_.push_back(0);
   return numberOfSubModels_++;
}

inline size_t 
GraphicalModelDecomposition::addSubFactor
(
   const size_t& subModel, 
   const size_t& factorId, 
   const std::vector<size_t>& subIndices
)
{
   OPENGM_ASSERT(subModel < numberOfSubModels_);
   OPENGM_ASSERT(factorId < numberOfFactors_);
   if(!NO_DEBUG) {
      for(size_t i=0; i<subIndices.size(); ++i) {
         OPENGM_ASSERT(subIndices[i] < numberOfSubVariables_[subModel]); 
      }
   }
   subFactorLists_[factorId].push_back(SubFactor(subModel,numberOfSubFactors_[subModel],subIndices));
   return numberOfSubFactors_[subModel]++;
}  

inline size_t 
GraphicalModelDecomposition::addSubFactor
(
   const size_t& subModel, 
   const std::vector<size_t>& indices, 
   const std::vector<size_t>& subIndices
)
{
   OPENGM_ASSERT(subModel < numberOfSubModels_);
   if(!NO_DEBUG) {
      for(size_t i=0; i<subIndices.size(); ++i) {
         OPENGM_ASSERT(subIndices[i] < numberOfSubVariables_[subModel]); 
      }
   }
   emptySubFactorLists_[indices].push_back(SubFactor(subModel,subIndices));
   return numberOfSubFactors_[subModel]++;
} 

inline size_t 
GraphicalModelDecomposition::addSubVariable
(
   const size_t& subModel, 
   const size_t& variableId
) {
   OPENGM_ASSERT(subModel < numberOfSubModels_);
   OPENGM_ASSERT(variableId < numberOfVariables_);
   subVariableLists_[variableId].push_back(SubVariable(subModel,numberOfSubVariables_[subModel]));
   return  numberOfSubVariables_[subModel]++;
}

inline void GraphicalModelDecomposition::complete()
{ 
   SubVariableListType::iterator    it;
   SubFactorListType::iterator      it2; 
   EmptySubFactorListType::iterator it3;

   // build mapping: (subModel, subVariable) -> (realVariable)
   std::vector<std::vector<size_t> > subVariable2realVariable(numberOfSubModels_);
   for(size_t subModelId=0; subModelId<numberOfSubModels_; ++subModelId) {
      subVariable2realVariable[subModelId].resize(numberOfSubVariables_[subModelId],0);
   }
   for(size_t varId=0; varId<numberOfVariables_; ++varId) {
      for(it = subVariableLists_[varId].begin(); it!=subVariableLists_[varId].end();++it) {
         const size_t& subModelId    = (*it).subModelId_;
         const size_t& subVariableId = (*it).subVariableId_;
         subVariable2realVariable[subModelId][subVariableId] = varId; 
      }
   }
      
   // build mapping: (realVariable) -> (realUnaryFactor)
   std::vector<size_t> realVariable2realUnaryFactors(numberOfVariables_,std::numeric_limits<std::size_t>::max());
   for(size_t factorId=0; factorId<numberOfFactors_; ++factorId) {
      if(!subFactorLists_[factorId].empty() && subFactorLists_[factorId].front().subIndices_.size()==1) {
         const size_t& subModelId     = subFactorLists_[factorId].front().subModelId_;
         const size_t& subVariableId  = subFactorLists_[factorId].front().subIndices_[0];
         const size_t& realVariableId = subVariable2realVariable[subModelId][subVariableId];
         realVariable2realUnaryFactors[realVariableId] = factorId;
      }
   }

   // add missing unary Factors
   for(size_t varId=0; varId<numberOfVariables_; ++varId) {
      if(subVariableLists_[varId].size()>1) {
         std::vector<std::set<size_t> > required(numberOfSubModels_);
         // find Missing SubFactors 
         for(it = subVariableLists_[varId].begin(); it!=subVariableLists_[varId].end();++it) {
            const size_t& subModelId = (*it).subModelId_; 
            const size_t& subVariableId = (*it).subVariableId_;
            required[subModelId].insert(subVariableId);
         }
         if(realVariable2realUnaryFactors[varId] < std::numeric_limits<size_t>::max()) {
            const size_t& factorId = realVariable2realUnaryFactors[varId];
            // find included SubFactors
            for(it2 = subFactorLists_[factorId].begin(); it2!=subFactorLists_[factorId].end();++it2) {
               const size_t& subModelId = (*it2).subModelId_;
               const size_t& subVariableId = (*it2).subIndices_[0];
               required[subModelId].erase(subVariableId);
            }
            // add SubFactor that are still missing
            for(size_t subModelId=0; subModelId<numberOfSubModels_; ++subModelId) {
               for(std::set<size_t>::const_iterator setit=required[subModelId].begin(); setit!=required[subModelId].end(); setit++) {
                  const std::vector<size_t> subIndices(1,(*setit));
                  addSubFactor(subModelId, factorId, subIndices);
               }
            }     
         }
         else{
            // find included SubFactors
            std::vector<size_t> subIndices(1,varId);
            for(it3 = emptySubFactorLists_[subIndices].begin(); it3!=emptySubFactorLists_[subIndices].end();++it3) {
               const size_t& subModelId    = (*it3).subModelId_;
               const size_t& subVariableId = (*it3).subIndices_[0];
               required[subModelId].erase(subVariableId);
            }
            // add SubFactor that are still missing
            for(size_t subModelId=0; subModelId<numberOfSubModels_; ++subModelId) {
               for(std::set<size_t>::const_iterator setit=required[subModelId].begin(); setit!=required[subModelId].end(); setit++) {
                  const std::vector<size_t> indices(1,varId);
                  const std::vector<size_t> subIndices(1,(*setit));
                  addSubFactor(subModelId,indices, subIndices);
               }
            }       
         }
      }
   }
}

inline void GraphicalModelDecomposition::reorder()
{ 
   SubVariableListType::iterator it;
   SubFactorListType::iterator it2;
   EmptySubFactorListType::iterator it3;      
   std::map<std::vector<size_t>, EmptySubFactorListType>::iterator it4;
   
   std::vector<size_t> varCount(numberOfSubModels_,0);
   std::vector<std::vector<size_t> > subVarMap(numberOfSubModels_);
   for(size_t subModel=0; subModel<numberOfSubModels_; ++subModel) {
      subVarMap[subModel].resize(numberOfSubVariables_[subModel],0);
   }

   // re-order SubVariableIds
   for(size_t varId=0; varId<numberOfVariables_; ++varId) {
      for(it = subVariableLists_[varId].begin(); it!=subVariableLists_[varId].end();++it) {
         const size_t& subModelId    = (*it).subModelId_;
         const size_t& subVariableId = (*it).subVariableId_;
         subVarMap[subModelId][subVariableId] = varCount[subModelId];
         (*it).subVariableId_ =  varCount[subModelId]++;
      }
   }

   // reset FactorSubIndices
   for(size_t factorId=0; factorId<numberOfFactors_; ++factorId) {
      for(it2 = subFactorLists_[factorId].begin(); it2!=subFactorLists_[factorId].end();++it2) {
         const size_t& subModelId = (*it2).subModelId_;
         for(size_t i=0; i<(*it2).subIndices_.size();++i) {
            (*it2).subIndices_[i] = subVarMap[subModelId][(*it2).subIndices_[i]];
         }
         for(size_t i=1; i<(*it2).subIndices_.size();++i) {
            OPENGM_ASSERT((*it2).subIndices_[i-1] < (*it2).subIndices_[i]);
         }
      }
   } 
   for(it4=emptySubFactorLists_.begin() ; it4 != emptySubFactorLists_.end(); it4++ ) {
      for(it3 = (*it4).second.begin(); it3!=(*it4).second.end();++it3) {
         const size_t& subModelId = (*it3).subModelId_;
         for(size_t i=0; i<(*it3).subIndices_.size();++i) {
            (*it3).subIndices_[i] = subVarMap[subModelId][(*it3).subIndices_[i]];
         }
         for(size_t i=1; i<(*it3).subIndices_.size();++i) {
            OPENGM_ASSERT((*it3).subIndices_[i-1] < (*it3).subIndices_[i]);
         }
      }
   }
}

template <class GM>
bool GraphicalModelDecomposition::isValid(const GM& gm) const
{
   OPENGM_ASSERT(subVariableLists_.size() == gm.numberOfVariables());
   OPENGM_ASSERT(subFactorLists_.size() == gm.numberOfFactors());
   if(!NO_DEBUG) {
      for(size_t i=0; i<gm.numberOfVariables(); ++i) {
         OPENGM_ASSERT(subVariableLists_[i].size() > 0);
      }
   }
   for(size_t i=0; i<gm.numberOfFactors(); ++i) {
      OPENGM_ASSERT(subFactorLists_[i].size() > 0);
      for(SubFactorListType::const_iterator it=subFactorLists_[i].begin() ; it != subFactorLists_[i].end(); it++ ) {
         OPENGM_ASSERT((*it).subIndices_.size() == gm[i].numberOfVariables());
         // check consistency of SubIndices of SubFactors
         for(size_t j=0; j<gm[i].numberOfVariables(); ++j) {
            const SubVariableListType &list = subVariableLists_[gm[i].variableIndex(j)];
            bool check = false;
            for(SubVariableListType::const_iterator it2=list.begin() ; it2 != list.end(); it2++ ) {
               if(((*it2).subModelId_ == (*it).subModelId_) && ((*it2).subVariableId_ == (*it).subIndices_[j])) {
                  check = true;
               }
            }
            OPENGM_ASSERT(check);
         }
      }
   }

   bool ret = true; 
   if(subVariableLists_.size() != gm.numberOfVariables()) { 
      ret = false; 
   }
   if(subFactorLists_.size() != gm.numberOfFactors()) { 
      ret = false; 
   }
   for(size_t i=0; i<gm.numberOfVariables(); ++i) {
      if(subVariableLists_[i].size()==0) {
         ret = false;
      }
   } 
   for(size_t i=0; i<gm.numberOfFactors(); ++i) {
      if(subFactorLists_[i].size()==0) {
         ret = false;
      } 
      for(SubFactorListType::const_iterator it=subFactorLists_[i].begin() ; it != subFactorLists_[i].end(); it++ ) {
         if((*it).subIndices_.size() != gm[i].numberOfVariables()) {
            ret = false;
         } 
         //Check Consistency of SubIndices of SubFactors
         //This might be very timeconsuming
         /*
         for(size_t j=0; j<gm[i].numberOfVariables(); ++j) {
            const SubVariableListType &list = subVariableLists_[gm[i].variableIndex(j)];
            bool check = false;
            for(SubVariableListType::const_iterator it2=list.begin() ; it2 != list.end(); it2++ ) {
               if( (*it2).subModelId == (*it).subModelId && (*it2).subVariableId == (*it).subIndices_[j]) {
                  check = true;
               }
            }
            if(!check) {ret=false;}
         }
         */
      }
   }
   return ret;
}

/// \endcond

} // namespace opengm

#endif // #ifndef OPENGM_GRAPHICALMODELDECOMPOSITION_HXX

