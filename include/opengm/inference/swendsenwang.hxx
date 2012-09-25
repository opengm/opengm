#pragma once
#ifndef OPENGM_SWENDSENWANG_HXX
#define OPENGM_SWENDSENWANG_HXX

#include <vector>
#include <set>
#include <stack>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/multiplier.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/utilities/random.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/datastructures/randomaccessset.hxx"
#include "opengm/datastructures/partition.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/functions/view_convert_function.hxx"

namespace opengm {

/// \cond suppress doxygen
namespace detail_swendsenwang {
   template<class OPERATOR, class ACCUMULATOR, class PROBABILITY>
   struct ValueToProbability;

   template<class PROBABILITY>
   struct ValueToProbability<Multiplier, Maximizer, PROBABILITY>
   {
      typedef PROBABILITY ProbabilityType;
      template<class T>
         static ProbabilityType convert(const T x)
            { return static_cast<ProbabilityType>(x); }
   };

   template<class PROBABILITY>
   struct ValueToProbability<Adder, Minimizer, PROBABILITY>
   {
      typedef PROBABILITY ProbabilityType;
      template<class T>
         static ProbabilityType convert(const T x)
            { return static_cast<ProbabilityType>(std::exp(-x)); }
   };
}
/// \endcond no longer suppress doxygen

/// \brief Visitor
template<class SW>
class SwendsenWangEmptyVisitor {
public:
   typedef SW SwendsenWangType;

   void operator()(const SwendsenWangType&, const size_t, const size_t,
      const bool, const bool) const;
};

/// \brief Visitor
template<class SW>
class SwendsenWangVerboseVisitor
 {
public:
   typedef SW SwendsenWangType;

   void operator()(const SwendsenWangType&, const size_t, const size_t,
      const bool, const bool) const;
};

/// \brief Visitor
template<class SW>
class SwendsenWangMarginalVisitor {
public:
   typedef SW SwendsenWangType;
   typedef typename SwendsenWangType::ValueType ValueType;
   typedef typename SwendsenWangType::GraphicalModelType GraphicalModelType;
   typedef typename GraphicalModelType::IndependentFactorType IndependentFactorType;

   // construction
   SwendsenWangMarginalVisitor();
   SwendsenWangMarginalVisitor(const GraphicalModelType&);
   void assign(const GraphicalModelType&);

   // manipulation
   template<class VariableIndexIterator>
      size_t addMarginal(VariableIndexIterator, VariableIndexIterator);
   size_t addMarginal(const size_t);
   void operator()(const SwendsenWangType&, const size_t, const size_t,
      const bool, const bool);

   // query
   size_t numberOfSamples() const;
   size_t numberOfAcceptedSamples() const;
   size_t numberOfRejectedSamples() const;
   size_t numberOfMarginals() const;
   const IndependentFactorType& marginal(const size_t) const;

private:
   const GraphicalModelType* gm_;
   size_t numberOfSamples_;
   size_t numberOfAcceptedSamples_;
   size_t numberOfRejectedSamples_;
   std::vector<IndependentFactorType> marginals_;
   std::vector<typename GraphicalModelType::LabelType> stateCache_;
};

/// \brief Generalized Swendsen-Wang sampling\n\n
/// A. Barbu, S. Zhu, "Generalizing swendsen-wang to sampling arbitrary posterior probabilities", PAMI 27:1239-1253, 2005
/// 
/// \ingroup inference 
template<class GM, class ACC>
class SwendsenWang 
: public Inference<GM, ACC> {
public:
   typedef GM GraphicalModelType;
   typedef ACC AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef double ProbabilityType;
   typedef SwendsenWangEmptyVisitor<SwendsenWang<GM, ACC> > EmptyVisitorType;
   typedef SwendsenWangVerboseVisitor<SwendsenWang<GM, ACC> > VerboseVisitorType;
   typedef TimingVisitor<SwendsenWang<GM, ACC> > TimingVisitorType;

   struct Parameter
   {
      Parameter
      (
         const size_t maxNumberOfSamplingSteps = 1e5,
         const size_t numberOfBurnInSteps = 1e5,
         ProbabilityType lowestAllowedProbability = 1e-6,
         const std::vector<LabelType>& initialState = std::vector<LabelType>()
      )
      :  maxNumberOfSamplingSteps_(maxNumberOfSamplingSteps),
         numberOfBurnInSteps_(numberOfBurnInSteps),
         lowestAllowedProbability_(lowestAllowedProbability),
         initialState_(initialState)
      {}

      size_t maxNumberOfSamplingSteps_;
      size_t numberOfBurnInSteps_;
      ProbabilityType lowestAllowedProbability_;
      std::vector<LabelType> initialState_;
   };

   SwendsenWang(const GraphicalModelType&, const Parameter& param = Parameter());
   virtual std::string name() const;
   virtual const GraphicalModelType& graphicalModel() const;
   virtual void reset();
   virtual InferenceTermination infer();
   template<class VISITOR>
      InferenceTermination infer(VISITOR&);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;

   LabelType markovState(const size_t) const;
   ValueType markovValue() const;
   LabelType currentBestState(const size_t) const;
   ValueType currentBestValue() const;

private:
   void computeEdgeProbabilities();
   void cluster(Partition<size_t>&) const;
   template<bool BURNED_IN, class VARIABLE_ITERATOR, class STATE_ITERATOR>
      bool move(VARIABLE_ITERATOR, VARIABLE_ITERATOR, STATE_ITERATOR);

   Parameter parameter_;
   const GraphicalModelType& gm_;
   Movemaker<GraphicalModelType> movemaker_;
   std::vector<RandomAccessSet<size_t> > variableAdjacency_;
   std::vector<std::vector<ProbabilityType> > edgeProbabilities_;
   std::vector<LabelType> currentBestState_;
   ValueType currentBestValue_;
};

template<class GM, class ACC>
inline
SwendsenWang<GM, ACC>::SwendsenWang
(
   const typename SwendsenWang<GM, ACC>::GraphicalModelType& gm,
   const typename SwendsenWang<GM, ACC>::Parameter& param
)
:  parameter_(param),
   gm_(gm),
   movemaker_(param.initialState_.size() == gm.numberOfVariables() ? Movemaker<GM>(gm, param.initialState_.begin()) : Movemaker<GM>(gm)),
   variableAdjacency_(gm.numberOfVariables()),
   edgeProbabilities_(gm.numberOfVariables()),
   currentBestState_(gm.numberOfVariables()),
   currentBestValue_(movemaker_.value())
{
   if(parameter_.initialState_.size() != 0 && parameter_.initialState_.size() != gm.numberOfVariables()) {
      throw RuntimeError("The size of the initial state does not match the number of variables.");
   }
   gm.variableAdjacencyList(variableAdjacency_);
   for(size_t j=0; j<gm_.numberOfVariables(); ++j) {
      edgeProbabilities_[j].resize(variableAdjacency_[j].size());
   }
   computeEdgeProbabilities();
}

template<class GM, class ACC>
inline void
SwendsenWang<GM, ACC>::reset()
{
   if(parameter_.initialState_.size() == gm_.numberOfVariables()) {
      movemaker_.initialize(parameter_.initialState_.begin());
      currentBestState_.assign(parameter_.initialState_.begin(),parameter_.initialState_.end());
   }
   else if(parameter_.initialState_.size() != 0) {
      throw RuntimeError("The size of the initial state does not match the number of variables.");
   }
   else{
      movemaker_.reset();
      std::fill(currentBestState_.begin(),currentBestState_.end(),0);
   }
   currentBestValue_ = movemaker_.value();
   computeEdgeProbabilities();
}

template<class GM, class ACC>
inline std::string
SwendsenWang<GM, ACC>::name() const
{
   return "SwendsenWang";
}

template<class GM, class ACC>
inline const typename SwendsenWang<GM, ACC>::GraphicalModelType&
SwendsenWang<GM, ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC>
template<class VISITOR>
InferenceTermination
SwendsenWang<GM, ACC>::infer
(
   VISITOR& visitor
)
{
   Partition<size_t> partition(gm_.numberOfVariables());
   std::vector<size_t> representatives(gm_.numberOfVariables());
   std::vector<bool> visited(gm_.numberOfVariables());
   std::stack<size_t> stack;
   std::vector<size_t> variablesInCluster;
   std::vector<size_t> variablesAroundCluster;
   for(size_t j=0; j<parameter_.numberOfBurnInSteps_ + parameter_.maxNumberOfSamplingSteps_; ++j) {
      // cluster the variable adjacency graph by randomly removing edges
      cluster(partition);

      // draw one cluster at random
      partition.representatives(representatives.begin());
      RandomUniform<size_t> randomNumberGeneratorCluster(0, partition.numberOfSets());
      const size_t representative = representatives[randomNumberGeneratorCluster()];
      // collect all variables in and around the drawn cluster
      variablesInCluster.clear();
      variablesAroundCluster.clear();
      visited[representative] = true;
      stack.push(representative);
      while(!stack.empty()) {
         const size_t variable = stack.top();
         stack.pop();
         variablesInCluster.push_back(variable);
         for(size_t k=0; k<variableAdjacency_[variable].size(); ++k) {
            const size_t adjacentVariable = variableAdjacency_[variable][k];
            if(!visited[adjacentVariable]) {
               visited[adjacentVariable] = true;
               if(partition.find(adjacentVariable) == representative) { // if in cluster
                  stack.push(adjacentVariable);
               }
               else {
                  variablesAroundCluster.push_back(adjacentVariable);
               }
            }
         }
      }

      // clean vector visited
      for(size_t k=0; k<variablesInCluster.size(); ++k) {
         visited[variablesInCluster[k]] = false;
      }
      for(size_t k=0; k<variablesAroundCluster.size(); ++k) {
         visited[variablesAroundCluster[k]] = false;
      }

      // assertion testing
      if(!NO_DEBUG) {
         for(size_t k=0; k<visited.size(); ++k) {
            OPENGM_ASSERT(!visited[k]);
         }
         for(size_t k=0; k<variablesInCluster.size(); ++k) {
            OPENGM_ASSERT(gm_.numberOfLabels(variablesInCluster[k]) == gm_.numberOfLabels(representative));
         }
      }

      // draw a new label at random
      RandomUniform<size_t> randomNumberGeneratorState(0, gm_.numberOfLabels(representative));
      size_t targetLabel = randomNumberGeneratorState();
      std::vector<size_t> targetLabels(variablesInCluster.size(), targetLabel); // TODO add simpler function to movemaker

      if(j < parameter_.numberOfBurnInSteps_) {
         move<false>(variablesInCluster.begin(), variablesInCluster.end(), targetLabels.begin());
         visitor(*this, j, variablesInCluster.size(), true, true);
         continue;
      }

      // evaluate probability density function
      const ProbabilityType currentPDF =
         detail_swendsenwang::ValueToProbability<OperatorType, AccumulationType, ProbabilityType>::convert
            (movemaker_.value());
      const ProbabilityType targetPDF =
         detail_swendsenwang::ValueToProbability<OperatorType, AccumulationType, ProbabilityType>::convert
            (movemaker_.valueAfterMove(variablesInCluster.begin(), variablesInCluster.end(), targetLabels.begin()));

      // evaluate proposal density
      ProbabilityType currentValueProposal = 1;
      ProbabilityType targetValueProposal = 1;
      for(std::vector<size_t>::const_iterator vi = variablesAroundCluster.begin(); vi != variablesAroundCluster.end(); ++vi) {
         if(movemaker_.state(*vi) == movemaker_.state(representative)) { // *vi has old label
            for(size_t k=0; k<variableAdjacency_[*vi].size(); ++k) {
               const size_t nvi = variableAdjacency_[*vi][k];
               if(partition.find(nvi) == representative) { // if *nvi is in cluster
                  currentValueProposal *= (1.0 - edgeProbabilities_[*vi][k]);
               }
            }
         }
         else if(movemaker_.state(*vi) == targetLabel) { // *vi has new label
            for(size_t k=0; k<variableAdjacency_[*vi].size(); ++k) {
               const size_t nvi = variableAdjacency_[*vi][k];
               if(partition.find(nvi) == representative) { // if *nvi is in cluster
                  targetValueProposal *= (1.0 - edgeProbabilities_[*vi][k]);
               }
            }
         }
      }

      // accept or reject re-labeling
      const ProbabilityType metropolisHastingsProbability = (targetValueProposal / currentValueProposal) * (targetPDF / currentPDF);
      OPENGM_ASSERT(metropolisHastingsProbability > 0);
      if(metropolisHastingsProbability >= 1) { // accept
         move<true>(variablesInCluster.begin(), variablesInCluster.end(), targetLabels.begin());
         visitor(*this, j, variablesInCluster.size(), true, false);
      }
      else {
         RandomUniform<ProbabilityType> randomNumberGeneratorAcceptance(0, 1);
         if(metropolisHastingsProbability >= randomNumberGeneratorAcceptance()) { // accept
            move<true>(variablesInCluster.begin(), variablesInCluster.end(), targetLabels.begin());
            visitor(*this, j, variablesInCluster.size(), true, false);
         }
         else { // reject
            visitor(*this, j, variablesInCluster.size(), false, false);
         }
      }
   }

   return NORMAL;
}

template<class GM, class ACC>
inline InferenceTermination
SwendsenWang<GM, ACC>::infer()
{
   EmptyVisitorType visitor;
   return infer(visitor);
}

template<class GM, class ACC>
inline InferenceTermination
SwendsenWang<GM, ACC>::arg
(
   std::vector<LabelType>& x,
   const size_t N
) const {
   if(N == 1) {
      x = currentBestState_;
      return NORMAL;
   }
   else {
      return UNKNOWN;
   }
}

template<class GM, class ACC>
inline typename SwendsenWang<GM, ACC>::LabelType
SwendsenWang<GM, ACC>::markovState
(
   const size_t j
) const
{
   OPENGM_ASSERT(j < gm_.numberOfVariables());
   return movemaker_.state(j);
}

template<class GM, class ACC>
inline typename SwendsenWang<GM, ACC>::ValueType
SwendsenWang<GM, ACC>::markovValue() const
{
   return movemaker_.value();
}

template<class GM, class ACC>
inline typename SwendsenWang<GM, ACC>::LabelType
SwendsenWang<GM, ACC>::currentBestState
(
   const size_t j
) const
{
   OPENGM_ASSERT(j < gm_.numberOfVariables());
   return currentBestState_[j];
}

template<class GM, class ACC>
inline typename SwendsenWang<GM, ACC>::ValueType
SwendsenWang<GM, ACC>::currentBestValue() const
{
   return currentBestValue_;
}

template<class GM, class ACC>
template<bool BURNED_IN, class VARIABLE_ITERATOR, class STATE_ITERATOR>
inline bool SwendsenWang<GM, ACC>::move
(
   VARIABLE_ITERATOR begin,
   VARIABLE_ITERATOR end,
   STATE_ITERATOR it
)
{
   movemaker_.move(begin, end, it);
   if(BURNED_IN) {
      if(ACC::bop(movemaker_.value(), currentBestValue_)) {
         currentBestValue_ = movemaker_.value();
         std::copy(movemaker_.stateBegin(), movemaker_.stateEnd(), currentBestState_.begin());
         return true;
      }
   }
   return false;
}

template<class GM, class ACC>
void
SwendsenWang<GM, ACC>::computeEdgeProbabilities()
{
   std::set<size_t> factors;
   std::set<size_t> connectedVariables;
   size_t variables[] = {0, 0};
   for(variables[0] = 0; variables[0] < gm_.numberOfVariables(); ++variables[0]) {
      for(size_t j = 0; j < variableAdjacency_[variables[0]].size(); ++j) {
         variables[1] = variableAdjacency_[variables[0]][j];
         if(gm_.numberOfLabels(variables[0]) == gm_.numberOfLabels(variables[1])) {
            // for all pairs of connected variables, variables[0] and variables[1],
            // that have the same number of states, identify
            // - all factors connected to variables[0] or variables[1] (or both)
            // - all variables connected to these factors
            factors.clear();
            connectedVariables.clear();

            // factors that depend on at least variables[0] OR variables[1]
            for(size_t k = 0; k < 2; ++k) {
               for(typename GraphicalModelType::ConstFactorIterator it = gm_.factorsOfVariableBegin(variables[k]);
               it != gm_.factorsOfVariableEnd(variables[k]); ++it) {
                  factors.insert(*it);
                  for(size_t m = 0; m < gm_[*it].numberOfVariables(); ++m) {
                     connectedVariables.insert(gm_[*it].variableIndex(m));
                  }
               }
            }

            // factors that depend on at least variables[0] AND variables[1]
            /*
            for(typename GraphicalModelType::ConstFactorIterator it = gm_.factorsOfVariableBegin(variables[0]);
            it != gm_.factorsOfVariableEnd(variables[0]); ++it) {
               for(size_t k = 0; k<gm_[*it].numberOfVariables(); ++k) {
                  if(gm_[*it].variableIndex(k) == variables[1]) {
                     factors.insert(*it);
                     for(size_t m = 0; m < gm_[*it].numberOfVariables(); ++m) {
                        connectedVariables.insert(gm_[*it].variableIndex(m));
                     }
                     break;
                  }
               }
            }
            */

            // operate all found factors up
            IndependentFactorType localFactor(gm_,
               connectedVariables.begin(),
               connectedVariables.end(),
               OperatorType::template neutral<ValueType>());
            for(std::set<size_t>::const_iterator it = factors.begin(); it != factors.end(); ++it) {
               OperatorType::op(gm_[*it], localFactor);
            }

            // marginalize
            size_t indices[] = {0, 0};
            for(size_t k = 0; k < localFactor.numberOfVariables(); ++k) {
               if(localFactor.variableIndex(k) == variables[0]) {
                  indices[0] = k;
               }
               else if(localFactor.variableIndex(k) == variables[1]) {
                  indices[1] = k;
               }
            }
            ProbabilityType probEqual = 0;
            ProbabilityType probUnequal = 0;
            ShapeWalker< typename IndependentFactorType::ShapeIteratorType>
               walker(localFactor.shapeBegin(), localFactor.numberOfVariables());
            for(size_t k = 0; k < localFactor.size(); ++k, ++walker) {
               const ValueType value = localFactor(walker.coordinateTuple().begin());
               const ProbabilityType p = detail_swendsenwang::ValueToProbability<OperatorType, AccumulationType, ProbabilityType>::convert(value);
               if(walker.coordinateTuple()[indices[0]] == walker.coordinateTuple()[indices[1]]) {
                  probEqual += p;
               }
               else {
                  probUnequal += p;
               }
            }

            // normalize
            ProbabilityType sum = probEqual + probUnequal;
            if(sum == 0.0) {
               throw RuntimeError("Some local probabilities are exactly zero.");
            }
            probEqual /= sum;
            probUnequal /= sum;
            if(probEqual < parameter_.lowestAllowedProbability_ || probUnequal < parameter_.lowestAllowedProbability_) {
               throw RuntimeError("Marginal probabilities are smaller than the allowed minimum.");
            }

            edgeProbabilities_[variables[0]][j] = probUnequal;
         }
      }
   }
}

template<class GM, class ACC>
void
SwendsenWang<GM, ACC>::cluster
(
   Partition<size_t>& out
) const
{
   // randomly merge variables
   out.reset(gm_.numberOfVariables());
   opengm::RandomUniform<ProbabilityType> randomNumberGenerator(0.0, 1.0);
   size_t variables[] = {0, 0};
   for(variables[0] = 0; variables[0] < gm_.numberOfVariables(); ++variables[0]) {
      for(size_t j = 0; j < variableAdjacency_[variables[0]].size(); ++j) {
         variables[1] = variableAdjacency_[variables[0]][j];
         if(variables[0] < variables[1]) { // only once for each pair
            if(movemaker_.state(variables[0]) == movemaker_.state(variables[1])) {
               if(edgeProbabilities_[variables[0]][j] > randomNumberGenerator()) {
                  // turn edge on with probability edgeProbabilities_[variables[0]][variables[1]]
                  out.merge(variables[0], variables[1]);
               }
            }
         }
      }
   }
}

template<class SW>
inline void
SwendsenWangEmptyVisitor<SW>::operator()(
   const typename SwendsenWangEmptyVisitor<SW>::SwendsenWangType& sw,
   const size_t iteration,
   const size_t clusterSize,
   const bool accepted,
   const bool burningIn
) const {
}

template<class SW>
inline void
SwendsenWangVerboseVisitor<SW>::operator()(
   const typename SwendsenWangVerboseVisitor<SW>::SwendsenWangType& sw,
   const size_t iteration,
   const size_t clusterSize,
   const bool accepted,
   const bool burningIn
) const {
   std::cout << "Step " << iteration
      << ": " << "V_opt=" << sw.currentBestValue()
      << ", " << "V_markov=" << sw.markovValue()
      << ", " << "cs=" << clusterSize
      << ", " << (accepted ? "accepted" : "rejected")
      << ", " << (burningIn ? "burning in" : "sampling")
      << std::endl;
   //std::cout << "   arg_opt: ";
   //for(size_t j=0; j<sw.graphicalModel().numberOfVariables(); ++j) {
   //   std::cout << sw.currentBestState(j) << ' ';
   //}
   //std::cout << std::endl;
   //
   //std::cout << "   arg_markov: ";
   //for(size_t j=0; j<sw.graphicalModel().numberOfVariables(); ++j) {
   //   std::cout << sw.markovState(j) << ' ';
   //std::cout << std::endl;
}

template<class SW>
inline
SwendsenWangMarginalVisitor<SW>::SwendsenWangMarginalVisitor()
:  gm_(NULL),
   numberOfSamples_(0),
   numberOfAcceptedSamples_(0),
   numberOfRejectedSamples_(0),
   marginals_(),
   stateCache_()
{}

template<class SW>
inline
SwendsenWangMarginalVisitor<SW>::SwendsenWangMarginalVisitor(
   const typename SwendsenWangMarginalVisitor<SW>::GraphicalModelType& gm
)
:  gm_(&gm),
   numberOfSamples_(0),
   numberOfAcceptedSamples_(0),
   numberOfRejectedSamples_(0),
   marginals_(),
   stateCache_()
{}

template<class SW>
inline void
SwendsenWangMarginalVisitor<SW>::assign(
    const typename SwendsenWangMarginalVisitor<SW>::GraphicalModelType& gm
)
{
    gm_ = &gm;
}

template<class SW>
inline void
SwendsenWangMarginalVisitor<SW>::operator()(
   const typename SwendsenWangMarginalVisitor<SW>::SwendsenWangType& sw,
   const size_t iteration,
   const size_t clusterSize,
   const bool accepted,
   const bool burningIn
) {
   if(!burningIn) {
      ++numberOfSamples_;
      if(accepted) {
         ++numberOfAcceptedSamples_;
      }
      else {
         ++numberOfRejectedSamples_;
      }
      for(size_t j = 0; j < marginals_.size(); ++j) {
         for(size_t k = 0; k < marginals_[j].numberOfVariables(); ++k) {
            stateCache_[k] = sw.markovState(marginals_[j].variableIndex(k));
         }
         ++marginals_[j](stateCache_.begin());
      }
   }
}

template<class SW>
template<class VariableIndexIterator>
inline size_t
SwendsenWangMarginalVisitor<SW>::addMarginal(
   VariableIndexIterator begin,
   VariableIndexIterator end
) {
   marginals_.push_back(IndependentFactorType(*gm_, begin, end));
   if(marginals_.back().numberOfVariables() > stateCache_.size()) {
      stateCache_.resize(marginals_.back().numberOfVariables());
   }
   return marginals_.size() - 1;
}

template<class SW>
inline size_t
SwendsenWangMarginalVisitor<SW>::addMarginal(
   const size_t variableIndex
) {
   size_t variableIndices[] = {variableIndex};
   return addMarginal(variableIndices, variableIndices + 1);
}

template<class SW>
inline size_t
SwendsenWangMarginalVisitor<SW>::numberOfSamples() const {
   return numberOfSamples_;
}

template<class SW>
inline size_t
SwendsenWangMarginalVisitor<SW>::numberOfAcceptedSamples() const {
   return numberOfAcceptedSamples_;
}

template<class SW>
inline size_t
SwendsenWangMarginalVisitor<SW>::numberOfRejectedSamples() const {
   return numberOfRejectedSamples_;
}

template<class SW>
inline size_t
SwendsenWangMarginalVisitor<SW>::numberOfMarginals() const {
   return marginals_.size();
}

template<class SW>
inline const typename SwendsenWangMarginalVisitor<SW>::IndependentFactorType&
SwendsenWangMarginalVisitor<SW>::marginal(
   const size_t setIndex
) const {
   return marginals_[setIndex];
}

} // namespace opengm

#endif // #ifndef OPENGM_SWENDSENWANG_HXX
