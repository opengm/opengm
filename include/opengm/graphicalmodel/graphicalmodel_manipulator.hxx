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

#include "opengm/inference/dynamicprogramming.hxx"
#include "opengm/inference/messagepassing/messagepassing.hxx"

#include <iostream>

namespace opengm {

/// \brief GraphicalModelManipulator
///
/// Implementation of the core part of reduction techniques proposed in
/// J.H. Kappes, M. Speth, G. Reinelt, and C. Schnörr: Towards Efficient and Exact MAP-Inference for Large Scale Discrete Computer Vision Problems via Combinatorial Optimization, CVPR 2013
///
/// it provides:
/// * modelreduction for fixed variables
/// * construction of seperate, independent subparts of the objective
/// * preoptimization of acyclic subproblems
/// 
/// it extends the published version by
/// * support for higher order models
///
/// it requires:
/// * no external dependencies (those are in reducedinference)
///
/// Corresponding author: Jörg Hendrik Kappes
///
/// Invariant: Order of the variables in the modified subgraphs is the same as in the original graph
/// See also: reducedinference.hxx
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

      enum ManipulationMode {
         FIX, //fix variables in factors that are fixed.
         DROP //drop factors that include fixed variables.
      };

      typedef typename opengm::DiscreteSpace<IndexType, LabelType> MSpaceType;
      typedef typename meta::TypeListGenerator< 
	ViewFixVariablesFunction<GM>, 
	ViewFunction<GM>, 
	ConstantFunction<ValueType, IndexType, LabelType>,
	ExplicitFunction<ValueType, IndexType, LabelType> >::type MFunctionTypeList;
      typedef GraphicalModel<ValueType, typename GM::OperatorType, MFunctionTypeList, MSpaceType> MGM;

      GraphicalModelManipulator(const GM& gm, const ManipulationMode mode = FIX);

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
      template<class ACC> void lockAndTentacelElimination(); 
      bool isFixed(const typename GM::IndexType)const;

   private:
      void expand(IndexType, IndexType,  std::vector<bool>&);

      //General Members
      const OGM& gm_;                            // original model
      bool locked_;                              // if true no more manipulation is allowed 
      std::vector<bool> fixVariable_;            // flag if variables are fixed
      std::vector<LabelType> fixVariableLabel_;  // label of fixed variables (otherwise undefined)
      ManipulationMode mode_;

      //Modified Model
      bool validModel_;                          // true if modified model is valid
      MGM mgm_;                                  // modified model

      //Modified SubModels
      bool validSubModels_;                      // true if themodified submodels are valid            
      std::vector<MGM> submodels_;               // modified submodels           
      std::vector<IndexType> var2subProblem_;    // subproblem of variable (for fixed variables undefined)

      //Tentacles
     std::vector<IndexType> tentacleRoots_;                                                    // Root-node of the tentacles 
     std::vector<opengm::ExplicitFunction<ValueType,IndexType,LabelType> > tentacleFunctions_; // functions that replace the tentacles 
     std::vector<std::vector<std::vector<LabelType> > > tentacleLabelCandidates_;              // Candidates for labeling of the tentacles
     std::vector<std::vector<IndexType> > tentacleVars_;                                       // Variable-list of the tentacles
     std::vector<bool> tentacleFactor_;                                                        // factor is included in a tentacle
     ValueType tentacleConstValue_;


   };
  

   template<class GM>
   bool GraphicalModelManipulator<GM>::isFixed(const typename GM::IndexType vi) const
   { 
      return fixVariable_[vi];
   }

   template<class GM>
   GraphicalModelManipulator<GM>::GraphicalModelManipulator(const GM& gm, const ManipulationMode mode)
      : gm_(gm), 
        locked_(false), 
        fixVariable_(std::vector<bool>(gm.numberOfVariables(),false)),
        fixVariableLabel_(std::vector<LabelType>(gm.numberOfVariables(),0)), 
        mode_(mode),
        validModel_(false), 
        validSubModels_(false),
        var2subProblem_(std::vector<IndexType>(gm.numberOfVariables(),0))
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
      tentacleFactor_.resize(gm_.numberOfFactors(),false);
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
	 // Get labels for tentacle variables
	 for (size_t i=0; i<tentacleRoots_.size();++i){
	   const LabelType l = ol[tentacleRoots_[i]];
	   for(size_t k=0; k<tentacleVars_[i].size();++k){
	     ol[tentacleVars_[i][k]] = tentacleLabelCandidates_[i][l][k];
             //std::cout <<tentacleVars_[i][k] << " <- "<<tentacleLabelCandidates_[i][l][k]<<std::endl;
	   }
           OPENGM_ASSERT( l == ol[tentacleRoots_[i]]);
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
      // Get labels for tentacle variables
      for (size_t i=0; i<tentacleRoots_.size();++i){
	const LabelType l = conf[tentacleRoots_[i]];
	for(size_t k=0; k<tentacleVars_[i].size();++k){
	  conf[tentacleVars_[i][k]] = tentacleLabelCandidates_[i][l][k];
	}
        OPENGM_ASSERT( l == conf[tentacleRoots_[i]] );
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

      //building variable mapping between input and output models
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         if(fixVariable_[var]==false){
            varMap[var] = numberOfVariables++;
         }
      } 

      //construction of label space for non-fixed variables
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
	 if(tentacleFactor_[f]) continue;
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
         if(mode_==FIX){
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
         } else if(mode_==DROP){
            if(fixedVars.size()==0){//non fixed
               const ViewFunction<GM> func(gm_[f]);
               mgm_.addFactor(mgm_.addFunction(func),MVars.begin(), MVars.end());
            }
         } else{
            throw std::runtime_error("Unsupported manipulation mode");
         } 
      }
      if(mode_==FIX){
         // Add Tentacle nodes
         for(size_t i=0; i<tentacleRoots_.size(); ++i){
            IndexType var = varMap[tentacleRoots_[i]];
            mgm_.addFactor(mgm_.addFunction(tentacleFunctions_[i]), &var, &var+1);
         }
         
         // Add constant
         { 
            //std::cout <<"* Const= "<<constant<<std::endl;
            LabelType temp;
            ConstantFunction<ValueType, IndexType, LabelType> func(&temp, &temp, constant);
            mgm_.addFactor(mgm_.addFunction(func),MVars.begin(), MVars.begin());
         } 
         //std::cout << "* numvars : " << mgm_.numberOfVariables() <<std::endl;
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
      if(numberOfSubproblems==0) numberOfSubproblems=1;
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
	 if(tentacleFactor_[f]) continue; 
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
         if(mode_==FIX){
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
         } else if(mode_==DROP){
            if(fixedVars.size()==0){//non fixed
               const ViewFunction<GM> func(gm_[f]);
               submodels_[subproblem].addFactor(submodels_[subproblem].addFunction(func),MVars.begin(), MVars.end());    
            }
         }else{
            throw std::runtime_error("Unsupported manipulation mode"); 
         }
      } 
      if(mode_==FIX){
         // Add Tentacle nodes
         for(size_t i=0; i<tentacleRoots_.size(); ++i){
            IndexType var = varMap[tentacleRoots_[i]];
            IndexType subproblem = var2subProblem_[tentacleRoots_[i]];
            submodels_[subproblem].addFactor(submodels_[subproblem].addFunction(tentacleFunctions_[i]), &var, &var+1);
         }
         {
            //std::cout <<"Const= "<<constant<<std::endl;
            LabelType temp;
            ConstantFunction<ValueType, IndexType, LabelType> func(&temp, &temp, constant);
            submodels_[0].addFactor( submodels_[0].addFunction(func),MVars.begin(), MVars.begin());
         }  
      }
      //std::cout << " numvars : " << submodels_[0].numberOfVariables() <<std::endl;
   }

//////////////////////
// ACC Manipulation
/////////////////////

  //  std::vector<bool> fixVariable_;            // flag if variables are fixed
  //  std::vector<LabelType> fixVariableLabel_;  // label of fixed variables (otherwise undefined)
 
   template<class GM>
   template<class ACC>
   void GraphicalModelManipulator<GM>::lockAndTentacelElimination()                                     
   {
      if(locked_)    return;
      if(mode_!=FIX) throw std::runtime_error ("lockAndTentacelElimination only supports the mode FIX, yet");
    
      locked_ = true; 
      tentacleFactor_.resize(gm_.numberOfFactors(),false);
      std::vector<bool> tFactor(gm_.numberOfFactors(),false);

      //std::cout << "start detecting tentacles" << std::endl;

      std::vector<IndexType>            variableDegree(gm_.numberOfVariables(), 0);
      std::vector<IndexType>            factorDegree(gm_.numberOfFactors(), 0);
      std::vector<bool>                 isRoot(gm_.numberOfVariables(), false); 
      std::vector<bool>                 isInTentacle(gm_.numberOfVariables(), false); 
      std::vector<IndexType>            leafs; 
  
      //SETUP DEGREE
      for(IndexType factor = 0 ; factor < gm_.numberOfFactors() ; ++factor){
         //Factor degree 
         for(typename GM::ConstVariableIterator vit=gm_.variablesOfFactorBegin(factor); vit!=gm_.variablesOfFactorEnd(factor); ++vit){
            if(!fixVariable_[*vit]){
               ++factorDegree[factor];
            }
         }
         if(factorDegree[factor]>1){ 
            for(typename GM::ConstVariableIterator vit=gm_.variablesOfFactorBegin(factor); vit!=gm_.variablesOfFactorEnd(factor); ++vit){
               if(!fixVariable_[*vit]){
                  variableDegree[*vit] += 1;
               }
            }
         }
      }
      
      //SETUP LEAFS 
      std::vector<bool>  pushed2Leafs(gm_.numberOfFactors(), false);
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         if(variableDegree[var] <= 1 && !fixVariable_[var]){
            leafs.push_back(var);
            pushed2Leafs[var] = true;
            isInTentacle[var] = true;
         }
      }
      
      //std::cout << "Found  "<<leafs.size()<<" leafs."<<std::endl;
      if(leafs.size()==0) return;
      //
      
      // Find Tentacle variables
      std::map<IndexType, IndexType> representives;
      typename std::set<typename GM::IndexType>::iterator it;
      typename std::set<typename GM::IndexType>::iterator fi;
    
      while(!leafs.empty()){
         IndexType var=leafs.back();
         leafs.pop_back();
         OPENGM_ASSERT(isInTentacle[var]);
         // Reduce factor order 
         for(typename GM::ConstFactorIterator fit=gm_.factorsOfVariableBegin(var); fit !=gm_.factorsOfVariableEnd(var); ++fit){
            OPENGM_ASSERT(factorDegree[*fit]>0);
            --factorDegree[*fit]; 
            if(factorDegree[*fit]<=1){
               tFactor[*fit]=true;
            }
         }
         // Check for new vars
         for(typename GM::ConstFactorIterator fit=gm_.factorsOfVariableBegin(var); fit !=gm_.factorsOfVariableEnd(var); ++fit){
            if(factorDegree[*fit]==1){
               for(typename GM::ConstVariableIterator vit=gm_.variablesOfFactorBegin(*fit); vit!=gm_.variablesOfFactorEnd(*fit); ++vit){
                  if(!fixVariable_[*vit]){
                     OPENGM_ASSERT(variableDegree[*vit]>0);
                     --variableDegree[*vit];
                     if(variableDegree[*vit]==1 && !pushed2Leafs[*vit] ){
                        leafs.push_back(*vit); 
                        pushed2Leafs[*vit] = true;
                     }
                     isInTentacle[*vit] = true;           
                  }
               }
            }
         }
      }

    
      IndexType numTentacleVars = 0; 
      IndexType numRootVars = 0;
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         if( isInTentacle[var] )
            ++numTentacleVars;
         if( isInTentacle[var] && variableDegree[var]>0){
            ++numRootVars;
            isRoot[var]=true; 
            OPENGM_ASSERT_OP(variableDegree[var],>,0);
         }
         if( isInTentacle[var] && variableDegree[var]==0){
            for(typename GM::ConstFactorIterator fit=gm_.factorsOfVariableBegin(var); fit !=gm_.factorsOfVariableEnd(var); ++fit){ 
               OPENGM_ASSERT(tFactor[*fit]);
            }  
         }
      }
      for(IndexType factor=0; factor<gm_.numberOfFactors();++factor){
         if(tFactor[factor]){
            size_t count =0;
            for(typename GM::ConstVariableIterator vit=gm_.variablesOfFactorBegin(factor); vit!=gm_.variablesOfFactorEnd(factor); ++vit){
               if(!fixVariable_[*vit] && variableDegree[*vit]>0) ++count;
            } 
            OPENGM_ASSERT_OP(factorDegree[factor],<=,count);
            OPENGM_ASSERT_OP(factorDegree[factor]+1,>=,count);
            OPENGM_ASSERT_OP(factorDegree[factor],<=,1);
            
         }
      }

      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         if( isRoot[var])
            OPENGM_ASSERT(variableDegree[var]>0);
      }
      //std::cout << "Found  "<<numTentacleVars<<" tentacle variables and "<< numRootVars <<" root variables."<<std::endl;
      tentacleRoots_.reserve(numRootVars);
      tentacleFunctions_.reserve(numRootVars);
      tentacleLabelCandidates_.reserve(numRootVars);
      tentacleVars_.reserve(numRootVars);
    
    
      // Find Tentacels 
      size_t numTentacles = 0;
      std::vector<IndexType> gmvar2ttvar(gm_.numberOfVariables());
      std::vector<bool>  visitedVar(gm_.numberOfVariables(), false); 
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         if(!isInTentacle[var] || visitedVar[var]) continue; 

         std::vector<IndexType> varList;
         size_t i = 0;
         bool hasRoot = false;
         IndexType root = 0;
         varList.push_back(var);
         visitedVar[var] = true;
         while(i<varList.size()){
            IndexType v = varList[i];
            if(isRoot[v]){ 
               OPENGM_ASSERT_OP(variableDegree[v],>,0);
               OPENGM_ASSERT(!hasRoot);
               hasRoot = true;
               root = v;
            }else{
               OPENGM_ASSERT_OP(variableDegree[v],==,0);
            }

            for(typename GM::ConstFactorIterator fit=gm_.factorsOfVariableBegin(v); fit !=gm_.factorsOfVariableEnd(v); ++fit){	
               if(tFactor[*fit]){
                  for(typename GM::ConstVariableIterator vit=gm_.variablesOfFactorBegin(*fit); vit!=gm_.variablesOfFactorEnd(*fit); ++vit){
                     if(!fixVariable_[*vit] && !visitedVar[*vit]){
                        visitedVar[*vit] = true; 
                        varList.push_back(*vit); 
                     }
                  }
               }
               else{
                  OPENGM_ASSERT(hasRoot);
               }
            }
            ++i;	
         } 
         
         // ** Solve tentacle ** 
         //std::cout << "Tentacle "<<numTentacles<<" has "<<varList.size()<<" variables and " << hasRoot << " roots."<<std::endl;
         std::sort(varList.begin(),varList.end());
         //setup model
         std::vector<LabelType> numStates(varList.size());
         for(size_t i=0; i<varList.size(); ++i){
            numStates[i]              = gm_.numberOfLabels(varList[i]); 
            gmvar2ttvar[varList[i]] = i;
            //std::cout << varList[i]<<" ";
         }
         //std::cout<<std::endl;
       
         MGM gmt(typename MGM::SpaceType(numStates.begin(),numStates.end()));
         // Find factors an add those 
         std::vector<PositionAndLabel<IndexType,LabelType> > fixedVars;
         std::vector<IndexType> freeVars; 
       
         for(typename std::vector<IndexType>::iterator it=varList.begin(); it!= varList.end(); ++it){
            for(typename GM::ConstFactorIterator fit=gm_.factorsOfVariableBegin(*it); fit !=gm_.factorsOfVariableEnd(*it); ++fit){	
               if( tFactor[*fit] ){
                  tFactor[*fit] = false;
                  //if(hasRoot) continue;
                  //factor is only a tentacle factor if the tencale has a root, othrewise it can be fixed by DP
                  tentacleFactor_[*fit]=hasRoot;
                  fixedVars.resize(0); 
                  freeVars.resize(0);
                  IndexType pos = 0;
                  for(typename GM::ConstVariableIterator vit=gm_.variablesOfFactorBegin(*fit); vit!=gm_.variablesOfFactorEnd(*fit); ++vit){
                     if(fixVariable_[*vit]){
                        fixedVars.push_back(PositionAndLabel<IndexType,LabelType>(pos,fixVariableLabel_[*vit]));
                     }
                     else if(isInTentacle[*vit]){
                        freeVars.push_back(gmvar2ttvar[*vit]);
                     }
                     else{//no tentacle factor
                        OPENGM_ASSERT(hasRoot);
                     }
                     ++pos;
                  } 
                  if(fixedVars.size()==0){//non fixed
                     const ViewFunction<GM> func(gm_[*fit]);
                     gmt.addFactor(gmt.addFunction(func),freeVars.begin(), freeVars.end());    
                  }else{
                     const ViewFixVariablesFunction<GM> func(gm_[*fit], fixedVars);
                     gmt.addFactor(gmt.addFunction(func),freeVars.begin(), freeVars.end());
                  }  
               }
            }
         }
         //if(hasRoot) continue;
               
         // Infer
         if(gmt.maxFactorOrder()<=2){
            //std::cout << "DP" <<std::endl;
            typedef opengm::DynamicProgramming<MGM,ACC>  DP;
            typename DP::Parameter dpPara;
          
            if(hasRoot){
               dpPara.roots_ = std::vector<IndexType>(1,gmvar2ttvar[root]);
               DP dp(gmt,dpPara);
               dp.infer();  
               std::vector<ValueType>  values; 
               std::vector<IndexType>  nodes; 
               std::vector<std::vector<LabelType> >  substates;
               dp.getNodeInfo(dpPara.roots_[0], values ,substates ,nodes );
               OPENGM_ASSERT_OP( gm_.numberOfLabels(root), ==, substates.size());
             
               std::vector<std::vector<LabelType> >  orderedSubstates(substates.size(),std::vector<LabelType>(varList.size(),0));  
               for(size_t i=0; i<orderedSubstates.size();++i){
                  OPENGM_ASSERT_OP(varList.size(),==,substates[i].size()+1); 
                  OPENGM_ASSERT_OP(varList.size(),==,orderedSubstates[i].size());
                  orderedSubstates[i][dpPara.roots_[0]] = i;
                  for (size_t n=0; n<substates[i].size(); ++n)
                     orderedSubstates[i][nodes[n]]=substates[i][n];
               } 

               //tentacleRoots_.push_back(dpPara.roots_[0]);
               tentacleRoots_.push_back(root);
               LabelType shape = gm_.numberOfLabels(root); 
               opengm::ExplicitFunction<ValueType,IndexType,LabelType> func(&shape,  &shape+1);
               for(LabelType l=0; l<shape; ++l)
                  func(&l) = values[l];
               tentacleFunctions_.push_back(func);
               tentacleLabelCandidates_.push_back(orderedSubstates);
               tentacleVars_.push_back(varList);
               for(size_t i=0; i<varList.size();++i){
                  if(varList[i]==root) continue;
                  fixVariable_[varList[i]]=true;
                  fixVariableLabel_[varList[i]]= 0; //only a dummy entry
               }
            }
            else{            
               dpPara.roots_ = std::vector<IndexType>(1,0);
               DP dp(gmt,dpPara); 
               dp.infer();  
           
/*
               ValueType v;
               std::vector<ValueType>  values;
               std::vector<IndexType>  nodes; 
               std::vector<std::vector<LabelType> >  substates; 
               dp.getNodeInfo(dpPara.roots_[0], values ,substates ,nodes );  
               ValueType optl=0;
               for(LabelType l=1; l<values.size();++l)
                  if(ACC::bop(values[optl],values[l]))
                     optl=l;
               for(size_t i=0; i<substates[optl].size();++i){
                  fixVariable_[varList[i]]=true;
                  fixVariableLabel_[varList[i]]=substates[optl][i];
               }
*/

               
               std::vector<LabelType> arg(gmt.numberOfVariables(),0);
               dp.arg(arg);            
               for(size_t i=0; i<arg.size();++i){
                  fixVariable_[varList[i]]=true;
                  fixVariableLabel_[varList[i]]=arg[i];
               }
               
            }
         }
         else{ 
            //std::cout << "BP" <<std::endl;
            typedef opengm::BeliefPropagationUpdateRules<MGM,ACC> UpdateRulesType;
            typedef opengm::MessagePassing<MGM, ACC,UpdateRulesType, opengm::MaxDistance> BP;
            typename BP::Parameter para;
            typename BP::IndependentFactorType m;
            para.useNormalization_=false;
            if(hasRoot){ 
 
               std::vector<ValueType>  valuesdp; 
               std::vector<std::vector<LabelType> >  substatesdp(gm_.numberOfLabels(root),std::vector<LabelType>(gmt.numberOfVariables(),0));  
             
               //std::cout << "with root" <<std::endl; 
               // In order to use constrainedOptimum, we have to calculate all marginals, i.e. no tree-tricks can be used!
               para.isAcyclic_ = opengm::Tribool::False; 
               para.inferSequential_= false;
               para.maximumNumberOfSteps_ = gmt.numberOfVariables()-1;
               BP bp(gmt,para);
               bp.infer();
               //bp.marginal(gmvar2ttvar[root], m); 
               LabelType shape = gm_.numberOfLabels(root); 
               opengm::ExplicitFunction<ValueType,IndexType,LabelType> func(&shape,  &shape+1);
               std::vector<std::vector<LabelType> >  substates(shape);
               std::vector<IndexType> rootIndex(1,gmvar2ttvar[root]);
               std::vector<LabelType> rootLabel(1,0);
   
               for(size_t i=0;i<shape;++i){  
                  substates[i].resize(gmt.numberOfVariables(),0);
                  rootLabel[0]=i;  
                  bp.constrainedOptimum(rootIndex,rootLabel,substates[i]);
                  OPENGM_ASSERT_OP(substates[i].size(),==,gmt.numberOfVariables());
                  func(&i) = gmt.evaluate(substates[i]);
                  OPENGM_ASSERT_OP(substates[i][gmvar2ttvar[root]],==,i);
               } 
               //tentacleRoots_.push_back(gmvar2ttvar[root]);
               tentacleRoots_.push_back(root);
               tentacleFunctions_.push_back(func);
               tentacleLabelCandidates_.push_back(substates);
               tentacleVars_.push_back(varList);
               for(size_t i=0; i<varList.size();++i){
                  if(varList[i]==root) continue;
                  fixVariable_[varList[i]]=true;
                  fixVariableLabel_[varList[i]]= 0; //only a dummy entry
               }
            }
            else{
               //std::cout << "with no root" <<std::endl;
               para.useNormalization_ = true;
               para.isAcyclic_ = opengm::Tribool::True; 
               para.inferSequential_= true;
               BP bp(gmt,para);
               //Bruteforce<MGM,ACC> bp(gmt);

               bp.infer(); 
               std::vector<LabelType> arg;
               bp.arg(arg);
               for(IndexType i=0; i<gmt.numberOfVariables();++i){
                  fixVariable_[varList[i]]=true;
                  fixVariableLabel_[varList[i]]=arg[i];
               }
            }
         }
         //std::cout <<"x "<< std::endl;
         ++numTentacles;  
      } 
      //std::cout <<"done "<< std::endl;
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
