#pragma once
#ifndef OPENGM_MOVEMAKER_HXX
#define OPENGM_MOVEMAKER_HXX

#include <algorithm>
#include <vector>

#include "opengm/operations/multiplier.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/utilities/sorting.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/vector_view_space.hxx"
#include "opengm/functions/view.hxx"
#include "opengm/functions/view_fix_variables_function.hxx"
#include "opengm/datastructures/buffer_vector.hxx"
#include "opengm/inference/bruteforce.hxx"

namespace opengm {

/// A fremework for move making algorithms
template<class GM>
class Movemaker {
public:
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef typename std::vector<LabelType>::const_iterator LabelIterator;
   /// \cond HIDDEN_SYMBOLS
   typedef typename opengm::meta::TypeListGenerator<ViewFunction<GM>, ViewFixVariablesFunction<GM> >::type FunctionTypeList;
   typedef opengm::VectorViewSpace<IndexType, LabelType> SubGmSpace;
   typedef opengm::GraphicalModel<ValueType, OperatorType, FunctionTypeList, SubGmSpace> SubGmType;
   /// \endcond

   Movemaker(const GraphicalModelType&); 
   template<class StateIterator>
      Movemaker(const GraphicalModelType&, StateIterator); 
   ValueType value() const;
   template<class IndexIterator, class StateIterator>
      ValueType valueAfterMove(IndexIterator, IndexIterator, StateIterator);
   const LabelType& state(const size_t) const;
   LabelIterator stateBegin() const;
   LabelIterator stateEnd() const;
   void reset();
   template<class StateIterator>
      void initialize(StateIterator);
   template<class IndexIterator, class StateIterator>
      ValueType move(IndexIterator, IndexIterator, StateIterator);
   template<class ACCUMULATOR, class IndexIterator>
      ValueType moveOptimally(IndexIterator, IndexIterator);
   template<class ACCUMULATOR, class IndexIterator>
      ValueType moveOptimallyWithAllLabelsChanging(IndexIterator, IndexIterator);
   //template<class ACCUMULATOR, class IndexIterator>
      //ValueType moveAstarOptimally(IndexIterator, IndexIterator);
   template<class INFERENCE_TYPE, class INFERENCE_PARAMETER, class INDEX_ITERATOR, class STATE_ITERATOR>
      void proposeMoveAccordingToInference(const INFERENCE_PARAMETER&, INDEX_ITERATOR, INDEX_ITERATOR, std::vector<LabelType>&)const;

private:
   typedef PositionAndLabel<IndexType, LabelType > PositionAndLabelType;
   typedef opengm::BufferVector<PositionAndLabelType> PositionAndLabelVector;

   /// \cond HIDDEN_SYMBOLS
   template<class INDEX_ITERATOR>
      void addFactorsToSubGm(INDEX_ITERATOR, INDEX_ITERATOR, SubGmType&)const;
   /// \endcond
   void addSingleSide(const IndexType, const IndexType, SubGmType &, std::set<IndexType>&)const;
   void addHigherOrderBorderFactor(const IndexType, const opengm::BufferVector<IndexType>&, const PositionAndLabelVector &, SubGmType &, std::set<IndexType> &)const;
   void addHigherOrderInsideFactor(const IndexType, const opengm::BufferVector<IndexType>&, SubGmType &, std::set<IndexType> &)const;
   template<class FactorIndexIterator>
      ValueType evaluateFactors(FactorIndexIterator, FactorIndexIterator, const std::vector<LabelType>&) const;

   const GraphicalModelType& gm_;
   std::vector<std::set<size_t> > factorsOfVariable_;
   std::vector<LabelType> state_;
   std::vector<LabelType> stateBuffer_; // always equal to state_ (invariant)
   ValueType energy_; // energy of state state_ (invariant)
};

/*
template<class GM>
template<class ACCUMULATOR, class IndexIterator>
inline typename Movemaker<GM>::ValueType
Movemaker<GM>::moveAstarOptimally
(
   IndexIterator variableIndicesBegin,
   IndexIterator variableIndicesEnd
) {
   typedef opengm::AStar<SubGmType, ACCUMULATOR> SubGmInferenceType;
   typedef typename SubGmInferenceType::Parameter SubGmInferenceParameterType;
   SubGmInferenceParameterType para;
   para.heuristic_ = para.STANDARDHEURISTIC;
   std::vector<LabelType> states(std::distance(variableIndicesBegin, variableIndicesEnd));
   this-> template proposeMoveAccordingToInference<
      SubGmInferenceType, SubGmInferenceParameterType, IndexIterator, typename std::vector<LabelType>::iterator
      > (para, variableIndicesBegin, variableIndicesEnd, states);
   return this->move(variableIndicesBegin, variableIndicesEnd, states.begin());
}
*/
template<class GM>
template<class INFERENCE_TYPE, class INFERENCE_PARAMETER, class INDEX_ITERATOR, class STATE_ITERATOR>
inline void
Movemaker<GM>::proposeMoveAccordingToInference
(
   const INFERENCE_PARAMETER& inferenceParam,
   INDEX_ITERATOR variablesBegin,
   INDEX_ITERATOR variablesEnd,
   std::vector<LabelType>& states
)const {
   OPENGM_ASSERT(opengm::isSorted(variablesBegin, variablesEnd));
   const size_t numberOfVariables = std::distance(variablesBegin, variablesEnd);
   std::vector<LabelType> spaceVector(numberOfVariables);
   for (size_t v = 0; v < numberOfVariables; ++v)
      spaceVector[v] = gm_.numberOfLabels(variablesBegin[v]);
   SubGmSpace subGmSpace(spaceVector);
   SubGmType subGm(subGmSpace);
   this->addFactorsToSubGm(variablesBegin, variablesEnd, subGm);
   INFERENCE_TYPE subGmInference(subGm, inferenceParam);
   subGmInference.infer();
   subGmInference.arg(states);
}

template<class GM>
inline void Movemaker<GM>::addSingleSide
(
   const typename Movemaker<GM>::IndexType gmFactorIndex,
   const typename Movemaker<GM>::IndexType subGmVarIndex,
   typename Movemaker<GM>::SubGmType & subGm,
   std::set<typename Movemaker<GM>::IndexType> & addedFactors
)const {
   const size_t var1Index[] = {subGmVarIndex};
   ViewFunction<GM> function = (gm_[gmFactorIndex]);
   typename GM::FunctionIdentifier fid = subGm.addFunction(function);
   subGm.addFactor(fid, var1Index, var1Index + 1);
   addedFactors.insert(gmFactorIndex);
}

template<class GM>
inline void Movemaker<GM>::addHigherOrderInsideFactor
(
   const typename Movemaker<GM>::IndexType gmFactorIndex,
   const opengm::BufferVector<typename Movemaker<GM>::IndexType> & subGmFactorVi,
   typename Movemaker<GM>::SubGmType & subGm,
   std::set<typename Movemaker<GM>::IndexType> & addedFactors
)const {
   ViewFunction<GM> function(gm_[gmFactorIndex]);
   typename GM::FunctionIdentifier fid = subGm.addFunction(function);
   subGm.addFactor(fid, subGmFactorVi.begin(), subGmFactorVi.end());
   addedFactors.insert(gmFactorIndex);
}

template<class GM>
inline void Movemaker<GM>::addHigherOrderBorderFactor
(
   const typename Movemaker<GM>::IndexType gmFactorIndex,
   const opengm::BufferVector<typename Movemaker<GM>::IndexType> & subGmFactorVi,
   const typename Movemaker<GM>::PositionAndLabelVector & factorFixVi,
   typename Movemaker<GM>::SubGmType & subGm,
   std::set<typename Movemaker<GM>::IndexType> & addedFactors
)const {
   ViewFixVariablesFunction<GM> function(gm_[gmFactorIndex], factorFixVi);
   typename GM::FunctionIdentifier fid = subGm.addFunction(function);
   subGm.addFactor(fid, subGmFactorVi.begin(), subGmFactorVi.end());
   addedFactors.insert(gmFactorIndex);
}

template<class GM>
template<class INDEX_ITERATOR >
inline void Movemaker<GM>::addFactorsToSubGm
(
   INDEX_ITERATOR variablesBegin,
   INDEX_ITERATOR variablesEnd,
   typename Movemaker<GM>::SubGmType & subGm
)const {
   std::set<IndexType> addedFactors;
   opengm::BufferVector<IndexType> subGmFactorVi;
   opengm::BufferVector<opengm::PositionAndLabel<IndexType, LabelType > >factorFixVi;
   subGm.reserveFactors(subGm.numberOfVariables()*7);
   for (IndexType subGmVi = 0; subGmVi < subGm.numberOfVariables(); ++subGmVi) {
      for (size_t f = 0; f < gm_.numberOfFactors(variablesBegin[subGmVi]); ++f) {
         const size_t factorIndex = gm_.factorOfVariable(variablesBegin[subGmVi], f);
         // if the factor has not been added
         if (addedFactors.find(factorIndex) == addedFactors.end()) {
            if (gm_[factorIndex].numberOfVariables() == 0) {
            } else if (gm_[factorIndex].numberOfVariables() == 1)
               this->addSingleSide(factorIndex, subGmVi, subGm, addedFactors);
            else {
               // find if all variables of the factor are in the subgraph or not:
               subGmFactorVi.clear();
               factorFixVi.clear();
               for (IndexType vv = 0; vv < gm_[factorIndex].numberOfVariables(); ++vv) {
                  bool foundVarIndex = false;
                  IndexType varIndexSubGm = 0;
                  foundVarIndex = findInSortedSequence(variablesBegin, subGm.numberOfVariables(), gm_[factorIndex].variableIndex(vv), varIndexSubGm);
                  if (foundVarIndex == false) // variable is outside the subgraph
                     factorFixVi.push_back(opengm::PositionAndLabel<IndexType, LabelType > (vv, this->state(gm_[factorIndex].variableIndex(vv))));
                  else // variable is inside the subgraph
                     subGmFactorVi.push_back(varIndexSubGm);
               }
               if (factorFixVi.size() == 0) // all variables are in the subgraph
                  this->addHigherOrderInsideFactor(factorIndex, subGmFactorVi, subGm, addedFactors);
               else // not all are in the subgraph
                  this->addHigherOrderBorderFactor(factorIndex, subGmFactorVi, factorFixVi, subGm, addedFactors);
            }
         }
      }
   }
}

template<class GM>
Movemaker<GM>::Movemaker
(
   const GraphicalModelType& gm
)
:  gm_(gm),
   factorsOfVariable_(gm.numberOfVariables()),
   state_(gm.numberOfVariables()),
   stateBuffer_(gm.numberOfVariables()),
   energy_(gm.evaluate(state_.begin()))
{
   for (size_t f = 0; f < gm.numberOfFactors(); ++f) {
      for (size_t v = 0; v < gm[f].numberOfVariables(); ++v) {
         factorsOfVariable_[gm[f].variableIndex(v)].insert(f);
      }
   }
}

template<class GM>
template<class StateIterator>
Movemaker<GM>::Movemaker
(
   const GraphicalModelType& gm,
   StateIterator it
)
:  gm_(gm),
   factorsOfVariable_(gm.numberOfVariables()),
   state_(gm.numberOfVariables()),
   stateBuffer_(gm.numberOfVariables()),
   energy_(gm.evaluate(it)) // fails if *it is out of bounds
{
   for (size_t j = 0; j < gm.numberOfVariables(); ++j, ++it) {
      state_[j] = *it;
      stateBuffer_[j] = *it;
   }
   for (size_t f = 0; f < gm.numberOfFactors(); ++f) {
      for (size_t v = 0; v < gm[f].numberOfVariables(); ++v) {
         factorsOfVariable_[gm[f].variableIndex(v)].insert(f);
      }
   }
}

template<class GM>
template<class StateIterator>
void Movemaker<GM>::initialize
(
   StateIterator it
) {
   energy_ = gm_.evaluate(it); // fails if *it is out of bounds
   for (size_t j = 0; j < gm_.numberOfVariables(); ++j, ++it) {
      state_[j] = *it;
      stateBuffer_[j] = *it;
   }
}

template<class GM>
void
Movemaker<GM>::reset() {
   for (size_t j = 0; j < gm_.numberOfVariables(); ++j) {
      state_[j] = 0;
      stateBuffer_[j] = 0;
   }
   energy_ = gm_.evaluate(state_.begin());
}

template<class GM>
inline typename Movemaker<GM>::ValueType
Movemaker<GM>::value() const {
   return energy_;
}

template<class GM>
template<class IndexIterator, class StateIterator>
typename Movemaker<GM>::ValueType
Movemaker<GM>::valueAfterMove
(
   IndexIterator begin,
   IndexIterator end,
   StateIterator destinationState
) { 
   ValueType destinationValue;
   if(meta::Compare<OperatorType, opengm::Multiplier>::value){
      //Partial update for multiplication is not numrical stabel! That why recalculate the objective 

      // set stateBuffer_ to destinationState, and determine factors to recompute
      for (IndexIterator it = begin; it != end; ++it, ++destinationState) {
         stateBuffer_[*it] = *destinationState;
      }
      // evaluate destination state
      destinationValue = gm_.evaluate(stateBuffer_); 
      // restore stateBuffer_
      for (IndexIterator it = begin; it != end; ++it) {
         stateBuffer_[*it] = state_[*it];
      }
   }else{
      // do partial update 

      // set stateBuffer_ to destinationState, and determine factors to recompute
      std::set<size_t> factorsToRecompute;
      for (IndexIterator it = begin; it != end; ++it, ++destinationState) {
         OPENGM_ASSERT(*destinationState < gm_.numberOfLabels(*it));
         if (state_[*it] != *destinationState) {
            OPENGM_ASSERT(*destinationState < gm_.numberOfLabels(*it));
            stateBuffer_[*it] = *destinationState;
            std::set<size_t> tmpSet;
            std::set_union(factorsToRecompute.begin(), factorsToRecompute.end(),
                           factorsOfVariable_[*it].begin(), factorsOfVariable_[*it].end(),
                           std::inserter(tmpSet, tmpSet.begin()));
            factorsToRecompute.swap(tmpSet);
         }
      }
      // \todo consider buffering the values of ALL factors at the current state!
      destinationValue = energy_;
      for (std::set<size_t>::const_iterator it = factorsToRecompute.begin(); it != factorsToRecompute.end(); ++it) {
         OPENGM_ASSERT(*it < gm_.numberOfFactors());
         // determine current and destination state of the current factor
         std::vector<size_t> currentFactorState(gm_[*it].numberOfVariables());
         std::vector<size_t> destinationFactorState(gm_[*it].numberOfVariables());
         for (size_t j = 0; j < gm_[*it].numberOfVariables(); ++j) {
            currentFactorState[j] = state_[gm_[*it].variableIndex(j)];
            OPENGM_ASSERT(currentFactorState[j] < gm_[*it].numberOfLabels(j));
            destinationFactorState[j] = stateBuffer_[gm_[*it].variableIndex(j)];
            OPENGM_ASSERT(destinationFactorState[j] < gm_[*it].numberOfLabels(j));
         }
         OperatorType::op(destinationValue, gm_[*it](destinationFactorState.begin()), destinationValue);
         OperatorType::iop(destinationValue, gm_[*it](currentFactorState.begin()), destinationValue);
      }
      // restore stateBuffer_
      for (IndexIterator it = begin; it != end; ++it) {
         stateBuffer_[*it] = state_[*it];
      }
   }
   return destinationValue;
}

template<class GM>
template<class IndexIterator, class StateIterator>
inline typename Movemaker<GM>::ValueType
Movemaker<GM>::move
(
   IndexIterator begin,
   IndexIterator end,
   StateIterator sit
) {
   energy_ = valueAfterMove(begin, end, sit); // tests for assertions
   while (begin != end) {
      state_[*begin] = *sit;
      stateBuffer_[*begin] = *sit;
      ++begin;
      ++sit;
   }
   return energy_;
}


/// for a subset of variables, move to a labeling that is optimal w.r.t. ACCUMULATOR
/// \param variableIndices random access iterator to the beginning of a sequence of variable indices
/// \param variableIndicesEnd random access iterator to the end of a sequence of variable indices
/// \return new value
template<class GM>
template<class ACCUMULATOR, class IndexIterator>
inline typename Movemaker<GM>::ValueType
Movemaker<GM>::moveOptimally
(
   IndexIterator variableIndices,
   IndexIterator variableIndicesEnd
) {
   // determine factors to recompute
   std::set<size_t> factorsToRecompute;
   for (IndexIterator it = variableIndices; it != variableIndicesEnd; ++it) {
      std::set<size_t> tmpSet;
      std::set_union(factorsToRecompute.begin(), factorsToRecompute.end(),
         factorsOfVariable_[*it].begin(), factorsOfVariable_[*it].end(),
         std::inserter(tmpSet, tmpSet.begin()));
      factorsToRecompute.swap(tmpSet);
   }

   // find an optimal move and the corresponding energy of factors to recompute
   size_t numberOfVariables = std::distance(variableIndices, variableIndicesEnd);
   ValueType initialEnergy = evaluateFactors(
      factorsToRecompute.begin(),
      factorsToRecompute.end(),
      state_);
   ValueType bestEnergy = initialEnergy;
   std::vector<size_t> bestState(numberOfVariables);
   for (size_t j=0; j<numberOfVariables; ++j) {
      const size_t vi = variableIndices[j];
      stateBuffer_[vi] = 0;
   }
   for (;;) {
      // compute energy
      ValueType energy = evaluateFactors(
         factorsToRecompute.begin(),
         factorsToRecompute.end(),
         stateBuffer_);
      if(ACCUMULATOR::bop(energy, bestEnergy)) {
         // update energy and state
         bestEnergy = energy;
         for (size_t j = 0; j < numberOfVariables; ++j) {
            bestState[j] = stateBuffer_[variableIndices[j]];
         }
      }
      // increment buffered state
      for (size_t j = 0; j < numberOfVariables; ++j) {
         const size_t vi = variableIndices[j];
         if (stateBuffer_[vi] < gm_.numberOfLabels(vi) - 1) {
            ++stateBuffer_[vi];
            break;
         } else {
            if (j < numberOfVariables - 1) {
               stateBuffer_[vi] = 0;
            } else {
               goto overflow;
            }
         }
      }
   }
overflow:
   ;

   if (ACCUMULATOR::bop(bestEnergy, initialEnergy)) {
      // update state_ and stateBuffer_
      for (size_t j = 0; j < numberOfVariables; ++j) {
         const size_t vi = variableIndices[j];
         state_[vi] = bestState[j];
         stateBuffer_[vi] = bestState[j];
      }
      // update energy
      if(meta::And<
      meta::Compare<ACCUMULATOR, opengm::Maximizer>::value,
      meta::Compare<OperatorType, opengm::Multiplier>::value
      >::value && energy_ == static_cast<ValueType> (0)) {
         OPENGM_ASSERT(state_.size() == gm_.numberOfVariables());
         energy_ = gm_.evaluate(state_.begin());
      }
      else {
         OperatorType::iop(initialEnergy, energy_); // energy_ -= initialEnergy
         OperatorType::op(bestEnergy, energy_); // energy_ += bestEnergy
      }
   } else {
      // restore stateBuffer_
      for (size_t j = 0; j < numberOfVariables; ++j) {
         const size_t vi = variableIndices[j];
         stateBuffer_[vi] = state_[vi];
      }
   }

   return energy_;
}


/// \todo get rid of redundancy with moveOptimally
template<class GM>
template<class ACCUMULATOR, class IndexIterator>
inline typename Movemaker<GM>::ValueType
Movemaker<GM>::moveOptimallyWithAllLabelsChanging
(
   IndexIterator variableIndices,
   IndexIterator variableIndicesEnd
) {
   // determine factors to recompute
   std::set<size_t> factorsToRecompute;
   for (IndexIterator it = variableIndices; it != variableIndicesEnd; ++it) {
      std::set<size_t> tmpSet;
      std::set_union(factorsToRecompute.begin(), factorsToRecompute.end(),
         factorsOfVariable_[*it].begin(), factorsOfVariable_[*it].end(),
         std::inserter(tmpSet, tmpSet.begin()));
      factorsToRecompute.swap(tmpSet);
   }

   // find an optimal move and the corresponding energy of factors to recompute
   size_t numberOfVariables = std::distance(variableIndices, variableIndicesEnd);
   ValueType initialEnergy = evaluateFactors(
      factorsToRecompute.begin(),
      factorsToRecompute.end(),
      state_);
   ValueType bestEnergy = initialEnergy;
   std::vector<size_t> bestState(numberOfVariables);
   // set initial labeling
   for(size_t j=0; j<numberOfVariables; ++j) {
      if(gm_.space().numberOfLabels(variableIndices[j]) == 1) {
         // restore stateBuffer_
         for(size_t k=0; k<j; ++k) {
            stateBuffer_[k] = state_[k];
         }
         return energy_;
      }
      else {
         const size_t vi = variableIndices[j];
         if(state_[vi] == 0) {
            stateBuffer_[vi] = 1;
         }
         else {
            stateBuffer_[vi] = 0;
         }
      }
   }
   for (;;) {
#     ifndef NDEBUG
      for(size_t j=0; j<numberOfVariables; ++j) {
         const size_t vi = variableIndices[j];
         OPENGM_ASSERT(stateBuffer_[vi] != state_[vi]);
      }
#     endif
      // compute energy
      ValueType energy = evaluateFactors(
         factorsToRecompute.begin(),
         factorsToRecompute.end(),
         stateBuffer_);
      if(ACCUMULATOR::bop(energy, bestEnergy)) {
         // update energy and state
         bestEnergy = energy;
         for (size_t j = 0; j < numberOfVariables; ++j) {
            bestState[j] = stateBuffer_[variableIndices[j]];
         }
      }
      // increment buffered state
      for (size_t j=0; j<numberOfVariables; ++j) {
         const size_t vi = variableIndices[j];
         if(stateBuffer_[vi] < gm_.numberOfLabels(vi) - 1) {
            if(stateBuffer_[vi] + 1 != state_[vi]) {
               ++stateBuffer_[vi];
               break;
            }
            else if(stateBuffer_[vi] + 1 < gm_.numberOfLabels(vi) - 1) {
               stateBuffer_[vi] += 2; // skip current label
               break;
            }
            else {
               if (j < numberOfVariables - 1) {
                  if(state_[vi] == 0) {
                     stateBuffer_[vi] = 1;
                  }
                  else {
                     stateBuffer_[vi] = 0;
                  }
               } else {
                  goto overflow2;
               }
            }
         } else {
            if (j < numberOfVariables - 1) {
               if(state_[vi] == 0) {
                  stateBuffer_[vi] = 1;
               }
               else {
                  stateBuffer_[vi] = 0;
               }
            } else {
               goto overflow2;
            }
         }
      }
   }
overflow2:
   ;

   if (ACCUMULATOR::bop(bestEnergy, initialEnergy)) {
      // update state_ and stateBuffer_
      for (size_t j = 0; j < numberOfVariables; ++j) {
         const size_t vi = variableIndices[j];
         state_[vi] = bestState[j];
         stateBuffer_[vi] = bestState[j];
      }
      // update energy
      if(meta::And<
      meta::Compare<ACCUMULATOR, opengm::Maximizer>::value,
      meta::Compare<OperatorType, opengm::Multiplier>::value
      >::value && energy_ == static_cast<ValueType> (0)) {
         energy_ = gm_.evaluate(state_.begin());
      }
      else {
         OperatorType::iop(initialEnergy, energy_); // energy_ -= initialEnergy
         OperatorType::op(bestEnergy, energy_); // energy_ += bestEnergy
      }
   } else {
      // restore stateBuffer_
      for (size_t j = 0; j < numberOfVariables; ++j) {
         const size_t vi = variableIndices[j];
         stateBuffer_[vi] = state_[vi];
      }
   }

   return energy_;
}

template<class GM>
inline const typename Movemaker<GM>::LabelType&
Movemaker<GM>::state
(
   const size_t variableIndex
) const {
   OPENGM_ASSERT(variableIndex < state_.size());
   return state_[variableIndex];
}

template<class GM>
inline typename Movemaker<GM>::LabelIterator
Movemaker<GM>::stateBegin() const {
   return state_.begin();
}

template<class GM>
inline typename Movemaker<GM>::LabelIterator
Movemaker<GM>::stateEnd() const {
   return state_.end();
}

template<class GM>
template<class FactorIndexIterator>
inline typename Movemaker<GM>::ValueType
Movemaker<GM>::evaluateFactors
(
   FactorIndexIterator begin,
   FactorIndexIterator end,
   const std::vector<LabelType>& state
) const {
   ValueType value = OperatorType::template neutral<ValueType>();
   for(; begin != end; ++begin) {
      std::vector<size_t> currentFactorState(gm_[*begin].numberOfVariables());
      for (size_t j=0; j<gm_[*begin].numberOfVariables(); ++j) {
         currentFactorState[j] = state[gm_[*begin].variableIndex(j)];
      }
      OperatorType::op(value, gm_[*begin](currentFactorState.begin()), value);
   }
   return value;
}

} // namespace opengm

#endif // #ifndef OPENGM_MOVEMAKER_HXX
