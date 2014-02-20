#pragma once
#ifndef OPENGM_GIBBS_HXX
#define OPENGM_GIBBS_HXX

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <typeinfo>

#include "opengm/opengm.hxx"
#include "opengm/utilities/random.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/multiplier.hxx"
#include "opengm/operations/integrator.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS
namespace detail_gibbs {

   template<class OPERATOR, class ACCUMULATOR, class PROBABILITY>
   struct ValuePairToProbability;

   template<class PROBABILITY>
   struct ValuePairToProbability<Multiplier, Maximizer, PROBABILITY>
   {
      typedef PROBABILITY ProbabilityType;
      template<class T>
         static ProbabilityType convert(const T newValue, const T oldValue)
            { return static_cast<ProbabilityType>(newValue) / static_cast<ProbabilityType>(oldValue); }
   };

   template<class PROBABILITY>
   struct ValuePairToProbability<Adder, Minimizer, PROBABILITY>
   {
      typedef PROBABILITY ProbabilityType;
      template<class T>
         static ProbabilityType convert(const T newValue, const T oldValue)
            { return static_cast<ProbabilityType>(std::exp(oldValue - newValue)); }
   };
}
/// \endcond

/// \brief Visitor for the Gibbs sampler to compute arbitrary marginal probabilities
///
/// \ingroup inference
template<class GIBBS>
class GibbsMarginalVisitor {
public:
   typedef GIBBS GibbsType;
   typedef typename GibbsType::ValueType ValueType;
   typedef typename GibbsType::GraphicalModelType GraphicalModelType;
   typedef typename GraphicalModelType::IndependentFactorType IndependentFactorType;

   // construction
   GibbsMarginalVisitor();
   GibbsMarginalVisitor(const GraphicalModelType&);
   void assign(const GraphicalModelType&);

   // manipulation
   template<class VariableIndexIterator>
      size_t addMarginal(VariableIndexIterator, VariableIndexIterator);
   size_t addMarginal(const size_t);
   void operator()(const GibbsType&, const ValueType, const ValueType, const size_t, const bool, const bool);

   // query
   void begin(const GibbsType&, const ValueType, const ValueType) const {}
   void end(const GibbsType&, const ValueType, const ValueType) const {}
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
   std::vector<size_t> stateCache_;

};

/// \brief Gibbs sampling
template<class GM, class ACC>
class Gibbs 
: public Inference<GM, ACC> {
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef Movemaker<GraphicalModelType> MovemakerType;
   typedef visitors::VerboseVisitor<Gibbs<GM, ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<Gibbs<GM, ACC> > EmptyVisitorType;
   typedef visitors::TimingVisitor<Gibbs<GM, ACC> > TimingVisitorType;
   typedef double ProbabilityType;

   class Parameter {
   public:
      enum VariableProposal {RANDOM, CYCLIC};

      Parameter(
         const size_t maxNumberOfSamplingSteps = 1e5,
         const size_t numberOfBurnInSteps = 1e5,
         const bool useTemp=false,
         const ValueType tmin=0.0001,
         const ValueType tmax=1,
         const IndexType periods=10,
         const VariableProposal variableProposal = RANDOM,
         const std::vector<size_t>& startPoint = std::vector<size_t>()
      )
      :  maxNumberOfSamplingSteps_(maxNumberOfSamplingSteps), 
         numberOfBurnInSteps_(numberOfBurnInSteps), 
         variableProposal_(variableProposal),
         startPoint_(startPoint),
         useTemp_(useTemp),
         tempMin_(tmin),
         tempMax_(tmax),
         periods_(periods){
         p_=static_cast<ValueType>(maxNumberOfSamplingSteps_/periods_);
      }
      bool useTemp_;
      ValueType tempMin_;
      ValueType tempMax_;
      size_t periods_;
      ValueType p_;
      size_t maxNumberOfSamplingSteps_;
      size_t numberOfBurnInSteps_;
      VariableProposal variableProposal_;
      std::vector<size_t> startPoint_;
      
      
   };

   Gibbs(const GraphicalModelType&, const Parameter& param = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   void reset();
   InferenceTermination infer();
   template<class VISITOR>
      InferenceTermination infer(VISITOR&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;

   LabelType markovState(const size_t) const;
   ValueType markovValue() const;
   LabelType currentBestState(const size_t) const;
   ValueType currentBestValue() const;

private:
   ValueType cosTemp(const ValueType arg,const ValueType periode,const ValueType min,const ValueType max)const{
      return static_cast<ValueType>(((std::cos(arg/periode)+1.0)/2.0)*(max-min))+min;
      //if(v<
   }

   ValueType getTemperature(const size_t step)const{
      return cosTemp( 
         static_cast<ValueType>(step),
         parameter_.p_,
         parameter_.tempMin_,
         parameter_.tempMax_
      );
   }
   Parameter parameter_;
   const GraphicalModelType& gm_;
   MovemakerType movemaker_;
   std::vector<size_t> currentBestState_;
   ValueType currentBestValue_;
   bool inInference_;
};

template<class GM, class ACC>
inline
Gibbs<GM, ACC>::Gibbs
(
   const GraphicalModelType& gm, 
   const Parameter& parameter
)
:  parameter_(parameter), 
   gm_(gm), 
   movemaker_(gm), 
   currentBestState_(gm.numberOfVariables()),
   currentBestValue_()
{
   inInference_=false;
   ACC::ineutral(currentBestValue_);
   if(parameter.startPoint_.size() != 0) {
      if(parameter.startPoint_.size() == gm.numberOfVariables()) {
         movemaker_.initialize(parameter.startPoint_.begin());
         currentBestState_ = parameter.startPoint_;
         currentBestValue_ = movemaker_.value();
      }
      else {
         throw RuntimeError("parameter.startPoint_.size() is neither zero nor equal to the number of variables.");
      }
   }
}

template<class GM, class ACC>
inline void
Gibbs<GM, ACC>::reset() {
   if(parameter_.startPoint_.size() != 0) {
      if(parameter_.startPoint_.size() == gm_.numberOfVariables()) {
         movemaker_.initialize(parameter_.startPoint_.begin());
         currentBestState_ = parameter_.startPoint_;
         currentBestValue_ = movemaker_.value();
      }
      else {
         throw RuntimeError("parameter.startPoint_.size() is neither zero nor equal to the number of variables.");
      }
   }
   else {
     movemaker_.reset();
     std::fill(currentBestState_.begin(), currentBestState_.end(), 0);
   }
}

template<class GM, class ACC>
inline void
Gibbs<GM, ACC>::setStartingPoint
(
   typename std::vector<typename Gibbs<GM, ACC>::LabelType>::const_iterator begin
) {
   try{
      movemaker_.initialize(begin);

      for(IndexType vi=0;vi<static_cast<IndexType>(gm_.numberOfVariables());++vi ){
         currentBestState_[vi]=movemaker_.state(vi);
      }
      currentBestValue_ = movemaker_.value();

   }
   catch(...) {
      throw RuntimeError("unsuitable starting point");
   }
}

template<class GM, class ACC>
inline std::string
Gibbs<GM, ACC>::name() const
{
   return "Gibbs";
}

template<class GM, class ACC>
inline const typename Gibbs<GM, ACC>::GraphicalModelType&
Gibbs<GM, ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC>
inline InferenceTermination
Gibbs<GM, ACC>::infer()
{
   EmptyVisitorType visitor;
   return infer(visitor);
}

template<class GM, class ACC>
template<class VISITOR>
InferenceTermination Gibbs<GM, ACC>::infer(
   VISITOR& visitor
) {
   inInference_=true;
   visitor.begin(*this);
   opengm::RandomUniform<size_t> randomVariable(0, gm_.numberOfVariables());
   opengm::RandomUniform<ProbabilityType> randomProb(0, 1);
   
   if(parameter_.useTemp_==false){
      for(size_t iteration = 0; iteration < parameter_.maxNumberOfSamplingSteps_ + parameter_.numberOfBurnInSteps_; ++iteration) {
         // select variable
         size_t variableIndex = 0;
         if(this->parameter_.variableProposal_ == Parameter::RANDOM) {
            variableIndex = randomVariable();
         }
         else if(this->parameter_.variableProposal_ == Parameter::CYCLIC) {
            variableIndex < gm_.numberOfVariables() - 1 ? ++variableIndex : variableIndex = 0;
         }

         // draw label
         opengm::RandomUniform<size_t> randomLabel(0, gm_.numberOfLabels(variableIndex));
         const size_t label = randomLabel();

         // move
         const bool burningIn = (iteration < parameter_.numberOfBurnInSteps_);
         if(label != movemaker_.state(variableIndex)) {
            const ValueType oldValue = movemaker_.value();
            const ValueType newValue = movemaker_.valueAfterMove(&variableIndex, &variableIndex + 1, &label);
            if(AccumulationType::bop(newValue, oldValue)) {
               movemaker_.move(&variableIndex, &variableIndex + 1, &label);
               if(AccumulationType::bop(newValue, currentBestValue_) && newValue != currentBestValue_) {
                  currentBestValue_ = newValue;
                  for(size_t k = 0; k < currentBestState_.size(); ++k) {
                     currentBestState_[k] = movemaker_.state(k);
                  }
               }
               visitor(*this);
               //visitor(*this, newValue, currentBestValue_, iteration, true, burningIn);
            }
            else {
               const ProbabilityType pFlip =
                  detail_gibbs::ValuePairToProbability<
                     OperatorType, AccumulationType, ProbabilityType
                  >::convert(newValue, oldValue);
               if(randomProb() < pFlip) {
                  movemaker_.move(&variableIndex, &variableIndex + 1, &label); 
                  visitor(*this);
                  //visitor(*this, newValue, currentBestValue_, iteration, true, burningIn);
               }
               else {
                  visitor(*this);
                 // visitor(*this, newValue, currentBestValue_, iteration, false, burningIn);
               }
            }
            ++iteration;
         }
      }
   }
   else {
      for(size_t iteration = 0; iteration < parameter_.maxNumberOfSamplingSteps_ + parameter_.numberOfBurnInSteps_; ++iteration) {
         // select variable
         size_t variableIndex = 0;
         if(this->parameter_.variableProposal_ == Parameter::RANDOM) {
            variableIndex = randomVariable();
         }
         else if(this->parameter_.variableProposal_ == Parameter::CYCLIC) {
            variableIndex < gm_.numberOfVariables() - 1 ? ++variableIndex : variableIndex = 0;
         }

         // draw label
         opengm::RandomUniform<size_t> randomLabel(0, gm_.numberOfLabels(variableIndex));
         const size_t label = randomLabel();

         // move
         const bool burningIn = (iteration < parameter_.numberOfBurnInSteps_);
         if(label != movemaker_.state(variableIndex)) {
            const ValueType oldValue = movemaker_.value();
            const ValueType newValue = movemaker_.valueAfterMove(&variableIndex, &variableIndex + 1, &label);
            if(AccumulationType::bop(newValue, oldValue)) {
               movemaker_.move(&variableIndex, &variableIndex + 1, &label);
               if(AccumulationType::bop(newValue, currentBestValue_) && newValue != currentBestValue_) {
                  currentBestValue_ = newValue;
                  for(size_t k = 0; k < currentBestState_.size(); ++k) {
                     currentBestState_[k] = movemaker_.state(k);
                  }
               }
               visitor(*this);
               //visitor(*this, newValue, currentBestValue_, iteration, true, burningIn);
            }
            else {
               const ProbabilityType pFlip =
                  detail_gibbs::ValuePairToProbability<
                     OperatorType, AccumulationType, ProbabilityType
                  >::convert(newValue, oldValue);
               if(randomProb() < pFlip*this->getTemperature(iteration)){
                  //std::cout<<"temp="<<this->getTemperature(iteration)<<"\n";
                  movemaker_.move(&variableIndex, &variableIndex + 1, &label); 
                  visitor(*this);
                  //visitor(*this, newValue, currentBestValue_, iteration, true, burningIn);
               }
               else {
                  //std::cout<<"temp="<<this->getTemperature(iteration)<<"\n";
                  visitor(*this);
                  //visitor(*this, newValue, currentBestValue_, iteration, false, burningIn);
               }
            }
            ++iteration;
         }
      }
   }
   //visitor.end(*this, currentBestValue_, currentBestValue_);
   visitor.end(*this);
   inInference_=false;
   return NORMAL;
}

template<class GM, class ACC>
inline InferenceTermination
Gibbs<GM, ACC>::arg
(
   std::vector<LabelType>& x, 
   const size_t N
) const {
   if(N == 1) {
      x.resize(gm_.numberOfVariables());
      for(size_t j = 0; j < x.size(); ++j) {
         if(!inInference_)
            x[j] = currentBestState_[j];
         else{
            x[j] = movemaker_.state(j);
         }
      }
      return NORMAL;
   }
   else {
      return UNKNOWN;
   }
}

template<class GM, class ACC>
inline typename Gibbs<GM, ACC>::LabelType
Gibbs<GM, ACC>::markovState
(
   const size_t j
) const
{
   OPENGM_ASSERT(j < gm_.numberOfVariables());
   return movemaker_.state(j);
}

template<class GM, class ACC>
inline typename Gibbs<GM, ACC>::ValueType
Gibbs<GM, ACC>::markovValue() const
{
   return movemaker_.value();
}

template<class GM, class ACC>
inline typename Gibbs<GM, ACC>::LabelType
Gibbs<GM, ACC>::currentBestState
(
   const size_t j
) const
{
   OPENGM_ASSERT(j < gm_.numberOfVariables());
   return currentBestState_[j];
}

template<class GM, class ACC>
inline typename Gibbs<GM, ACC>::ValueType
Gibbs<GM, ACC>::currentBestValue() const
{
   return currentBestValue_;
}

template<class GIBBS>
inline
GibbsMarginalVisitor<GIBBS>::GibbsMarginalVisitor()
:  gm_(NULL), 
   numberOfSamples_(0), 
   numberOfAcceptedSamples_(0), 
   numberOfRejectedSamples_(0), 
   marginals_(), 
   stateCache_()
{}

template<class GIBBS>
inline
GibbsMarginalVisitor<GIBBS>::GibbsMarginalVisitor(
   const typename GibbsMarginalVisitor<GIBBS>::GraphicalModelType& gm
)
:  gm_(&gm), 
   numberOfSamples_(0), 
   numberOfAcceptedSamples_(0), 
   numberOfRejectedSamples_(0), 
   marginals_(), 
   stateCache_()
{}

template<class GIBBS>
inline void
GibbsMarginalVisitor<GIBBS>::assign(
   const typename GibbsMarginalVisitor<GIBBS>::GraphicalModelType& gm
)
{
    gm_ = &gm;
}

template<class GIBBS>
inline void
GibbsMarginalVisitor<GIBBS>::operator()(
   const typename GibbsMarginalVisitor<GIBBS>::GibbsType& gibbs, 
   const typename GibbsMarginalVisitor<GIBBS>::ValueType currentValue, 
   const typename GibbsMarginalVisitor<GIBBS>::ValueType bestValue, 
   const size_t iteration, 
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
            stateCache_[k] = gibbs.markovState(marginals_[j].variableIndex(k));
         }
         ++marginals_[j](stateCache_.begin());
      }
   }
}

template<class GIBBS>
template<class VariableIndexIterator>
inline size_t
GibbsMarginalVisitor<GIBBS>::addMarginal(
   VariableIndexIterator begin, 
   VariableIndexIterator end
) {
   marginals_.push_back(IndependentFactorType(*gm_, begin, end));
   if(marginals_.back().numberOfVariables() > stateCache_.size()) {
      stateCache_.resize(marginals_.back().numberOfVariables());
   }
   return marginals_.size() - 1;
}

template<class GIBBS>
inline size_t
GibbsMarginalVisitor<GIBBS>::addMarginal(
   const size_t variableIndex
) {
   size_t variableIndices[] = {variableIndex};
   return addMarginal(variableIndices, variableIndices + 1);
}

template<class GIBBS>
inline size_t
GibbsMarginalVisitor<GIBBS>::numberOfSamples() const {
   return numberOfSamples_;
}

template<class GIBBS>
inline size_t
GibbsMarginalVisitor<GIBBS>::numberOfAcceptedSamples() const {
   return numberOfAcceptedSamples_;
}

template<class GIBBS>
inline size_t
GibbsMarginalVisitor<GIBBS>::numberOfRejectedSamples() const {
   return numberOfRejectedSamples_;
}

template<class GIBBS>
inline size_t
GibbsMarginalVisitor<GIBBS>::numberOfMarginals() const {
   return marginals_.size();
}

template<class GIBBS>
inline const typename GibbsMarginalVisitor<GIBBS>::IndependentFactorType&
GibbsMarginalVisitor<GIBBS>::marginal(
   const size_t setIndex
) const {
   return marginals_[setIndex];
}

} // namespace opengm

#endif // #ifndef OPENGM_GIBBS_HXX
