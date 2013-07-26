#pragma once
#ifndef OPENGM_GRAPHICALMODEL_MANIPULATOR_HXX
#define OPENGM_GRAPHICALMODEL_MANIPULATOR_HXX

#include <exception>
#include <set>
#include <vector>
#include <queue>

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/functions/view.hxx"
#include "opengm/functions/view_fix_variables_function.hxx"
#include "opengm/functions/constant.hxx"
#include <opengm/utilities/metaprogramming.hxx>

namespace opengm {

/// \brief GraphicalModelManipulator
///
/// Invariant: Order of the variables in the modified subgraphs is the same as in the original graph
///
/// \ingroup graphical_models
   template<class GM>
   class GraphicalModelManipulator
   {
   public:
      typedef GM                             OGM;
      typedef typename GM::SpaceType         OSpaceType;
      typedef typename GM::IndexType         IndexType;
      typedef typename GM::LabelType         LabelType;
      typedef typename GM::ValueType         ValueType;

      typedef typename opengm::DiscreteSpace<IndexType, LabelType> MSpaceType;
      typedef typename meta::TypeListGenerator< ViewFixVariablesFunction<GM>, ViewFunction<GM>, ConstantFunction<ValueType, IndexType, LabelType> >::type MFunctionTypeList;
      typedef GraphicalModel<ValueType, typename GM::OperatorType, MFunctionTypeList, MSpaceType> MGM;

      GraphicalModelManipulator(GM& gm);

      //BuildModels
      void buildModifiedModel();
      void buildModifiedSubModels();

      //Get Models
      const OGM& getOriginalModel() const;
      const MGM& getModifiedModel() const; 
      const MGM& getModifiedSubModel(size_t) const;
      
      //GetInfo
      size_t numberOfSubmodels() const;
      void modifiedState2OriginalState(const std::vector<LabelType>&, std::vector<LabelType>&) const;
      void modifiedSubStates2OriginalState(const std::vector<std::vector<LabelType> >&, std::vector<LabelType>&) const;
      bool isLocked() const;

      //Manipulation
      void fixVariable(const typename GM::IndexType, const typename GM::LabelType);
      void freeVariable(const  typename GM::IndexType);
      void freeAllVariables();
      void unlock();  
      void lock(); 

   private:
      void expand(IndexType, IndexType,  std::vector<bool>&);

      //General Members
      const OGM& gm_;                            // original model
      bool locked_;                              // if true no more manipulation is allowed 
      std::vector<bool> fixVariable_;            // flag if variables are fixed
      std::vector<LabelType> fixVariableLabel_;  // label of fixed variables (otherwise undefined)


      //Modified Model
      bool validModel_;                          // true if modified model is valid
      MGM mgm_;                                  // modified model

      //Modified SubModels
      bool validSubModels_;                      // true if themodified submodels are valid            
      std::vector<MGM> submodels_;               // modified submodels           
      std::vector<IndexType> var2subProblem_;    // subproblem of variable (for fixed variables undefined)
   };
  
   template<class GM>
   GraphicalModelManipulator<GM>::GraphicalModelManipulator(GM& gm)
      : gm_(gm), locked_(false),  validModel_(false), validSubModels_(false),
        fixVariable_(std::vector<bool>(gm.numberOfVariables(),false)),
        fixVariableLabel_(std::vector<LabelType>(gm.numberOfVariables(),0)),
        var2subProblem_(std::vector<LabelType>(gm.numberOfVariables(),0))
   {
      return;
   }
   
/// \brief return the original graphical model
   template<class GM>
   inline const typename GraphicalModelManipulator<GM>::OGM &
   GraphicalModelManipulator<GM>::getOriginalModel() const 
   {
      return gm_;
   }
   
/// \brief return the modified graphical model
   template<class GM>
   inline const typename GraphicalModelManipulator<GM>::MGM &
   GraphicalModelManipulator<GM>::getModifiedModel() const
   {
      OPENGM_ASSERT(isLocked() && validModel_);
      return mgm_;
   }

/// \brief return the i-th modified sub graphical model
   template<class GM>
   inline const typename GraphicalModelManipulator<GM>::MGM &
   GraphicalModelManipulator<GM>::getModifiedSubModel(size_t i) const
   {
      OPENGM_ASSERT(isLocked() && validSubModels_);
      OPENGM_ASSERT(i < submodels_.size());
      return submodels_[i];
   }

/// \brief return the number of submodels
   template<class GM>
   size_t GraphicalModelManipulator<GM>::numberOfSubmodels() const
   { 
      OPENGM_ASSERT(isLocked());
      return submodels_.size();
   }

/// \brief unlock model
   template<class GM>
   void GraphicalModelManipulator<GM>::unlock()
   {
      locked_=false;
      validSubModels_=false;
      validModel_=false;
      submodels_.clear();
      freeAllVariables();
   }

/// \brief lock model
   template<class GM>
   void GraphicalModelManipulator<GM>::lock()
   {
      locked_=true;
   }


/// \brief return true if model is locked 
   template<class GM>
   bool GraphicalModelManipulator<GM>::isLocked() const
   { 
      return locked_;
   }
   
/// \brief fix label for variable
   template<class GM>
   void GraphicalModelManipulator<GM>::fixVariable(const typename GM::IndexType var, const typename GM::LabelType l)
   {
      OPENGM_ASSERT(!isLocked());
      if(!isLocked()){
         fixVariable_[var]=true;
         fixVariableLabel_[var]=l;
      }
   }

/// \brief remove fixed label for variable
   template<class GM>
   void GraphicalModelManipulator<GM>::freeVariable(const  typename GM::IndexType var)
   {
      OPENGM_ASSERT(!isLocked());
      if(!isLocked()){
         fixVariable_[var]=false;
      }
   }

/// \brief remove fixed label for all variable
   template<class GM>
   void GraphicalModelManipulator<GM>::freeAllVariables()
   {
      OPENGM_ASSERT(!isLocked())

         if(!isLocked()){
         for(IndexType var=0; var<fixVariable_.size(); ++var)
            fixVariable_[var]=false;
      }
   }
 
/// \brief transforming label of the modified to the labeling of the original problem 
   template<class GM>
   void GraphicalModelManipulator<GM>::modifiedState2OriginalState(const std::vector<LabelType>& ml, std::vector<LabelType>& ol) const
   {
      OPENGM_ASSERT(isLocked());  
      OPENGM_ASSERT(ml.size()==mgm_.numberOfVariables());
      
      if(isLocked() && ml.size()==mgm_.numberOfVariables()){
         ol.resize(gm_.numberOfVariables());
         size_t c = 0;
         for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
            if(fixVariable_[var]){
               ol[var] = fixVariableLabel_[var];
            }else{
               ol[var] = ml[c++];
            }
         }
      }
   }

/// \brief transforming label of the modified subproblems to the labeling of the original problem 
   template<class GM>
   void GraphicalModelManipulator<GM>::modifiedSubStates2OriginalState(const std::vector<std::vector<LabelType> >& subconf, std::vector<LabelType>& conf) const
   {  
      conf.resize(gm_.numberOfVariables());
      std::vector<IndexType> varCount(submodels_.size(),0);
      for(IndexType i=0;i<submodels_.size(); ++i){
          OPENGM_ASSERT(submodels_[i].numberOfVariables()==subconf[i].size());
      }
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         if(fixVariable_[var]){
            conf[var] = fixVariableLabel_[var];
         }else{
            const IndexType sp=var2subProblem_[var];
            conf[var] = subconf[sp][varCount[sp]++];
         }
      }
   }
 
/// \brief build modified model
   template<class GM>
   void
   GraphicalModelManipulator<GM>::buildModifiedModel()
   {
      locked_ = true;
      validModel_ = true;
      IndexType numberOfVariables = 0;
      std::vector<IndexType> varMap(gm_.numberOfVariables(),0);
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         if(fixVariable_[var]==false){
            varMap[var] = numberOfVariables++;
         }
      }
      std::vector<LabelType> shape(numberOfVariables,0);
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         if(fixVariable_[var]==false){
            shape[varMap[var]] = gm_.numberOfLabels(var);
         }
      }
      MSpaceType space(shape.begin(),shape.end());
      mgm_ = MGM(space);

      std::vector<PositionAndLabel<IndexType,LabelType> > fixedVars;
      std::vector<IndexType> MVars;


      ValueType constant;
      GM::OperatorType::neutral(constant);
      for(IndexType f=0; f<gm_.numberOfFactors();++f){
         fixedVars.resize(0); 
         MVars.resize(0);
         for(IndexType i=0; i<gm_[f].numberOfVariables(); ++i){
            const IndexType var = gm_[f].variableIndex(i);
            if(fixVariable_[var]){
               fixedVars.push_back(PositionAndLabel<IndexType,LabelType>(i,fixVariableLabel_[var]));
            }else{
               MVars.push_back(varMap[var]);
            }
         }
         if(fixedVars.size()==0){//non fixed
            const ViewFunction<GM> func(gm_[f]);
            mgm_.addFactor(mgm_.addFunction(func),MVars.begin(), MVars.end());
         }else if(fixedVars.size()==gm_[f].numberOfVariables()){//all fixed
            std::vector<LabelType> fixedStates(gm_[f].numberOfVariables(),0);
            for(IndexType i=0; i<gm_[f].numberOfVariables(); ++i){
               fixedStates[i]=fixVariableLabel_[ gm_[f].variableIndex(i)];
            }     
            GM::OperatorType::op(gm_[f](fixedStates.begin()),constant);       
         }else{
            const ViewFixVariablesFunction<GM> func(gm_[f], fixedVars);
            mgm_.addFactor(mgm_.addFunction(func),MVars.begin(), MVars.end());
         }
      }
      {
         LabelType temp;
         ConstantFunction<ValueType, IndexType, LabelType> func(&temp, &temp, constant);
         mgm_.addFactor(mgm_.addFunction(func),MVars.begin(), MVars.begin());
      }
   }  

/// \brief build modified sub-models 
   template<class GM>
   void
   GraphicalModelManipulator<GM>::buildModifiedSubModels()
   {
      locked_ = true; 
      validSubModels_ = true;
      
      //Find Connected Components
      std::vector<bool> closedVar   = fixVariable_;
      IndexType numberOfSubproblems = 0;
      for(IndexType var=0 ; var<gm_.numberOfVariables(); ++var){
         if(closedVar[var])
            continue;
         else{
            expand(var, numberOfSubproblems, closedVar); 
         }
         ++numberOfSubproblems;
      }

      submodels_.resize(numberOfSubproblems); 
      std::vector<IndexType> numberOfVariables(numberOfSubproblems,0);
      std::vector<IndexType> varMap(gm_.numberOfVariables(),0);
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         if(fixVariable_[var]==false){
            varMap[var] = numberOfVariables[var2subProblem_[var]]++;
         }
      }
      std::vector<std::vector<LabelType> > shape(numberOfSubproblems);
      for (size_t i=0; i<numberOfSubproblems; ++i){
         shape[i] = std::vector<LabelType>(numberOfVariables[i],0);
      }
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         if(fixVariable_[var]==false){
            shape[var2subProblem_[var]][varMap[var]] = gm_.numberOfLabels(var);
         }
      }
      for (size_t i=0; i<numberOfSubproblems; ++i){
         MSpaceType space(shape[i].begin(),shape[i].end());
         submodels_[i] = MGM(space); 
      }
      
      std::vector<PositionAndLabel<IndexType,LabelType> > fixedVars;
      std::vector<IndexType> MVars;

      ValueType constant;
      GM::OperatorType::neutral(constant);

      for(IndexType f=0; f<gm_.numberOfFactors();++f){
         IndexType subproblem = 0;
         fixedVars.resize(0); 
         MVars.resize(0);
         for(IndexType i=0; i<gm_[f].numberOfVariables(); ++i){
            const IndexType var = gm_[f].variableIndex(i);
            if(fixVariable_[var]){
               fixedVars.push_back(PositionAndLabel<IndexType,LabelType>(i,fixVariableLabel_[var]));
            }else{
               MVars.push_back(varMap[var]);
               subproblem = var2subProblem_[var];
            }
         }
         if(MVars.size()==0){ //constant, all fixed
            std::vector<LabelType> fixedStates(gm_[f].numberOfVariables(),0);
            for(IndexType i=0; i<gm_[f].numberOfVariables(); ++i){
               fixedStates[i]=fixVariableLabel_[ gm_[f].variableIndex(i)];
            }     
            GM::OperatorType::op(gm_[f](fixedStates.begin()),constant);  
         }
         else if(fixedVars.size()==0){//non fixed
            const ViewFunction<GM> func(gm_[f]);
            submodels_[subproblem].addFactor(submodels_[subproblem].addFunction(func),MVars.begin(), MVars.end());    
         }else{
            const ViewFixVariablesFunction<GM> func(gm_[f], fixedVars);
             submodels_[subproblem].addFactor(submodels_[subproblem].addFunction(func),MVars.begin(), MVars.end());
         }
      }
      {
         LabelType temp;
         ConstantFunction<ValueType, IndexType, LabelType> func(&temp, &temp, constant);
         submodels_[0].addFactor( submodels_[0].addFunction(func),MVars.begin(), MVars.begin());
      }
   }
 
////////////////////
// Private Methods
////////////////////    
   template<class GM>
   void
   GraphicalModelManipulator<GM>::expand(IndexType var, IndexType CCN,  std::vector<bool>& closedVar)
   {
      if(closedVar[var])
         return;
      else{
         closedVar[var]       = true;
         var2subProblem_[var] = CCN;
         for( typename GM::ConstFactorIterator itf = gm_.factorsOfVariableBegin(var); itf!=gm_.factorsOfVariableEnd(var); ++itf){
            for( typename  GM::ConstVariableIterator itv = gm_.variablesOfFactorBegin(*itf); itv!=gm_.variablesOfFactorEnd(*itf);++itv){
               expand(*itv, CCN, closedVar);
            }
         }
      }
    }

   
} //namespace opengm

#endif // #ifndef OPENGM_GRAPHICALMODEL_HXX
