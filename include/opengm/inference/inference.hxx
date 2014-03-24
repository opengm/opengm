#pragma once
#ifndef OPENGM_INFERENCE_HXX
#define OPENGM_INFERENCE_HXX

#include <vector>
#include <string>
#include <list>
#include <limits>
#include <exception>

#include "opengm/opengm.hxx"

#define OPENGM_GM_TYPE_TYPEDEFS                                                      \
   typedef typename GraphicalModelType::LabelType LabelType;                         \
   typedef typename GraphicalModelType::IndexType IndexType;                         \
   typedef typename GraphicalModelType::ValueType ValueType;                         \
   typedef typename GraphicalModelType::OperatorType OperatorType;                   \
   typedef typename GraphicalModelType::FactorType FactorType;                       \
   typedef typename GraphicalModelType::IndependentFactorType IndependentFactorType; \
   typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier        \

namespace opengm {

enum InferenceTermination {
   UNKNOWN=0, 
   NORMAL=1, 
   TIMEOUT=2, 
   CONVERGENCE=3, 
   INFERENCE_ERROR=4
};

/// Inference algorithm interface
template <class GM, class ACC>
class Inference
{
public:
   typedef GM GraphicalModelType;
   typedef ACC AccumulationType;
   typedef typename GraphicalModelType::LabelType LabelType;
   typedef typename GraphicalModelType::IndexType IndexType;
   typedef typename GraphicalModelType::ValueType ValueType;
   typedef typename GraphicalModelType::OperatorType OperatorType;
   typedef typename GraphicalModelType::FactorType FactorType;
   typedef typename GraphicalModelType::IndependentFactorType IndependentFactorType;
   typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;

   virtual ~Inference() {}

   virtual std::string name() const = 0;
   virtual const GraphicalModelType& graphicalModel() const = 0;
   virtual InferenceTermination infer() = 0;
   /// \todo
   /// virtual void reset() = 0;
   /// virtual InferenceTermination update() = 0;

   // member functions with default definition
   virtual void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   virtual InferenceTermination args(std::vector<std::vector<LabelType> >&) const;
   virtual InferenceTermination marginal(const size_t, IndependentFactorType&) const;
   virtual InferenceTermination factorMarginal(const size_t, IndependentFactorType&) const;
   virtual ValueType bound() const;
   virtual ValueType value() const;
   InferenceTermination constrainedOptimum(std::vector<IndexType>&,std::vector<LabelType>&, std::vector<LabelType>&) const;
   InferenceTermination modeFromMarginal(std::vector<LabelType>&) const;
   InferenceTermination modeFromFactorMarginal(std::vector<LabelType>&) const;
};

/// \brief output a solution
/// \param[out] arg labeling
/// \param argIndex solution index (1=best, 2=second best, etc.)
template<class GM, class ACC>
inline InferenceTermination
Inference<GM, ACC>::arg(
   std::vector<LabelType>& arg,
   const size_t argIndex
) const
{
   return UNKNOWN;
}
   
/// \brief set initial labeling
/// \param begin iterator to the beginning of a sequence of labels
template<class GM, class ACC>   
inline void 
Inference<GM, ACC>::setStartingPoint(
   typename std::vector<LabelType>::const_iterator begin
) 
{}
   
template<class GM, class ACC>
inline InferenceTermination
Inference<GM, ACC>::args(
   std::vector<std::vector<LabelType> >& out
) const
{
   return UNKNOWN;
}

/// \brief output a solution for a marginal for a specific variable
/// \param variableIndex index of the variable
/// \param[out] out the marginal
template<class GM, class ACC>
inline InferenceTermination
Inference<GM, ACC>::marginal(
   const size_t variableIndex,
   IndependentFactorType& out
   ) const
{
   return UNKNOWN;
}

/// \brief output a solution for a marginal for all variables connected to a factor
/// \param factorIndex index of the factor
/// \param[out] out the marginal
template<class GM, class ACC>
inline InferenceTermination
Inference<GM, ACC>::factorMarginal(
   const size_t factorIndex,
   IndependentFactorType& out
) const
{
   return UNKNOWN;
}


template<class GM, class ACC>
InferenceTermination
Inference<GM, ACC>::constrainedOptimum(
   std::vector<IndexType>& variableIndices,
   std::vector<LabelType>& givenLabels,
   std::vector<LabelType>& conf
) const
{
   const GM& gm = graphicalModel();
   std::vector<IndexType> waitingVariables;
   size_t variableId = 0;
   size_t numberOfVariables = gm.numberOfVariables();
   size_t numberOfFixedVariables = 0;
   conf.assign(gm.numberOfVariables(),std::numeric_limits<LabelType>::max());
   OPENGM_ASSERT(variableIndices.size()>=givenLabels.size());
   for(size_t i=0; i<givenLabels.size() ;++i) {
      OPENGM_ASSERT( variableIndices[i]<gm.numberOfVariables());
      OPENGM_ASSERT( givenLabels[i]<gm.numberOfLabels(variableIndices[i]));
      conf[variableIndices[i]] = givenLabels[i];
      waitingVariables.push_back(variableIndices[i]);
      ++numberOfFixedVariables;
   }
   while(variableId<gm.numberOfVariables() && numberOfFixedVariables<numberOfVariables) {
      while(waitingVariables.size()>0 && numberOfFixedVariables<numberOfVariables) {
         size_t var = waitingVariables.back();
         waitingVariables.pop_back();

         //Search unset neighbourd variable
         for(size_t i=0; i<gm.numberOfFactors(var); ++i) { 
            size_t var2=var;
            size_t afactorId = gm.factorOfVariable(var,i);
            for(size_t n=0; n<gm[afactorId].numberOfVariables();++n) {
               if(conf[gm[afactorId].variableIndex(n)] == std::numeric_limits<LabelType>::max()) {
                  var2=gm[afactorId].variableIndex(n);
                  break;
               }
            }
            if(var2 != var) { 
               //Set this variable
               IndependentFactorType t;
               //marginal(var2, t);
               for(size_t i=0; i<gm.numberOfFactors(var2); ++i) {
                  size_t factorId = gm.factorOfVariable(var2,i);
                  if(factorId != afactorId) continue;
                  std::vector<IndexType> knownVariables;
                  std::vector<LabelType> knownStates;
                  std::vector<IndexType> unknownVariables; 
                  IndependentFactorType out;
                  InferenceTermination term = factorMarginal(factorId, out);
                  if(NORMAL != term) {
                     return term;
                  }
                  for(size_t n=0; n<gm[factorId].numberOfVariables();++n) {
                     if(gm[factorId].variableIndex(n)!=var2) {
                        if(conf[gm[factorId].variableIndex(n)] < std::numeric_limits<LabelType>::max()) {
                           knownVariables.push_back(gm[factorId].variableIndex(n));
                           knownStates.push_back(conf[gm[factorId].variableIndex(n)]);
                        }else{
                           unknownVariables.push_back(gm[factorId].variableIndex(n));
                        }
                     }
                  } 
                     
                  out.fixVariables(knownVariables.begin(), knownVariables.end(), knownStates.begin()); 
                  if(unknownVariables.size()>0)
                     out.template accumulate<AccumulationType>(unknownVariables.begin(),unknownVariables.end());
                  OperatorType::op(out,t); 
               } 
               ValueType value;
               std::vector<LabelType> state(t.numberOfVariables());
               t.template accumulate<AccumulationType>(value,state);
               conf[var2] = state[0];
               ++numberOfFixedVariables;
               waitingVariables.push_back(var2);
            }
         }
      }
      if(conf[variableId]==std::numeric_limits<LabelType>::max()) {
         //Set variable state
         IndependentFactorType out;
         InferenceTermination term = marginal(variableId, out);
         if(NORMAL != term) {
            return term;
         } 
         ValueType value;
         std::vector<LabelType> state(out.numberOfVariables());
         out.template accumulate<AccumulationType>(value,state);
         conf[variableId] = state[0];
         waitingVariables.push_back(variableId);
      }
      ++variableId;
   }
   return NORMAL;
}

/*
template<class GM, class ACC>
InferenceTermination
Inference<GM, ACC>::constrainedOptimum(
   std::vector<IndexType>& variableIndices,
   std::vector<LabelType>& givenLabels,
   std::vector<LabelType>& conf
) const
{
   const GM& gm = graphicalModel();
   std::vector<IndexType> waitingVariables;
   size_t variableId = 0;
   size_t numberOfVariables = gm.numberOfVariables();
   size_t numberOfFixedVariables = 0;
   conf.assign(gm.numberOfVariables(),std::numeric_limits<LabelType>::max());
   OPENGM_ASSERT(variableIndices.size()>=givenLabels.size());
   for(size_t i=0; i<givenLabels.size() ;++i) {
      OPENGM_ASSERT( variableIndices[i]<gm.numberOfVariables());
      OPENGM_ASSERT( givenLabels[i]<gm.numberOfLabels(variableIndices[i]));
      conf[variableIndices[i]] = givenLabels[i];
      waitingVariables.push_back(variableIndices[i]);
      ++numberOfFixedVariables;
   }
   while(variableId<gm.numberOfVariables() && numberOfFixedVariables<numberOfVariables) {
      while(waitingVariables.size()>0 && numberOfFixedVariables<numberOfVariables) {
         size_t var = waitingVariables.back();
         waitingVariables.pop_back();

         //Search unset neighbourd variable
         for(size_t i=0; i<gm.numberOfFactors(var); ++i) { 
            size_t var2=var;
            size_t afactorId = gm.factorOfVariable(var,i);
            for(size_t n=0; n<gm[afactorId].numberOfVariables();++n) {
               if(conf[gm[afactorId].variableIndex(n)] == std::numeric_limits<LabelType>::max()) {
                  var2=gm[afactorId].variableIndex(n);
                  break;
               }
            }
            if(var2 != var) { 
               //Set this variable
               IndependentFactorType t;
               //marginal(var2, t);
               for(size_t i=0; i<gm.numberOfFactors(var2); ++i) {
                  size_t factorId = gm.factorOfVariable(var2,i);
                  if(factorId != afactorId) continue;
                  std::vector<IndexType> knownVariables;
                  std::vector<LabelType> knownStates;
                  std::vector<IndexType> unknownVariables; 
                  IndependentFactorType out;
                  InferenceTermination term = factorMarginal(factorId, out);
                  if(NORMAL != term) {
                     return term;
                  }
                  for(size_t n=0; n<gm[factorId].numberOfVariables();++n) {
                     if(gm[factorId].variableIndex(n)!=var2) {
                        if(conf[gm[factorId].variableIndex(n)] < std::numeric_limits<LabelType>::max()) {
                           knownVariables.push_back(gm[factorId].variableIndex(n));
                           knownStates.push_back(conf[gm[factorId].variableIndex(n)]);
                        }else{
                           unknownVariables.push_back(gm[factorId].variableIndex(n));
                        }
                     }
                  } 
                     
                  out.fixVariables(knownVariables.begin(), knownVariables.end(), knownStates.begin()); 
                  if(unknownVariables.size()>0)
                     out.template accumulate<AccumulationType>(unknownVariables.begin(),unknownVariables.end());
                  OperatorType::op(out,t); 
               } 
               ValueType value;
               std::vector<LabelType> state(t.numberOfVariables());
               t.template accumulate<AccumulationType>(value,state);
               conf[var2] = state[0];
               ++numberOfFixedVariables;
               waitingVariables.push_back(var2);
            }
         }
      }
      if(conf[variableId]==std::numeric_limits<LabelType>::max()) {
         //Set variable state
         IndependentFactorType out;
         InferenceTermination term = marginal(variableId, out);
         if(NORMAL != term) {
            return term;
         } 
         ValueType value;
         std::vector<LabelType> state(out.numberOfVariables());
         out.template accumulate<AccumulationType>(value,state);
         conf[variableId] = state[0];
         waitingVariables.push_back(variableId);
      }
      ++variableId;
   }
   return NORMAL;
}
*/

template<class GM, class ACC>
InferenceTermination
Inference<GM, ACC>::modeFromMarginal(
   std::vector<LabelType>& conf
   ) const
{
   const GM&         gm = graphicalModel();
   //const space_type& space = gm.space();
   size_t            numberOfNodes = gm.numberOfVariables();
   conf.resize(gm.numberOfVariables());
   IndependentFactorType out;
   for(size_t node=0; node<numberOfNodes; ++node) {
      InferenceTermination term = marginal(node, out);
      if(NORMAL != term) {
         return term;
      }
      ValueType value = out(0);
      size_t state = 0;
      for(size_t i=1; i<gm.numberOfLabels(node); ++i) {
         if(ACC::bop(out(i), value)) {
            value = out(i);
            state = i;
         }
      }
      conf[node] = state;
   }
   return NORMAL;
}

template<class GM, class ACC>
InferenceTermination
Inference<GM, ACC>::modeFromFactorMarginal(
   std::vector<LabelType>& conf
) const
{
   const GM& gm = graphicalModel();
   std::vector<IndexType> knownVariables;
   std::vector<LabelType> knownStates;
   IndependentFactorType out;
   for(size_t node=0; node<gm.numberOfVariables(); ++node) {
      InferenceTermination term = marginal(node, out);
      if(NORMAL != term) {
         return term;
      }
      ValueType value = out(0);
      size_t state = 0;
      bool unique = true;
      for(size_t i=1; i<gm.numberOfLabels(node); ++i) {

         //ValueType q = out(i)/value;
         //if(q<1.001 && q>0.999) {
         //   unique=false;
         //}
         if(fabs(out(i) - value)<0.00001) {
            unique=false;
         }
         else if(ACC::bop(out(i), value)) {
            value = out(i);
            state = i;
            unique=true;
         }
      }
      if(unique) {
         knownVariables.push_back(node);
         knownStates.push_back(state);
      }
   }
   return constrainedOptimum( knownVariables, knownStates, conf);
}

/// \brief return the solution (value)
template<class GM, class ACC>
typename GM::ValueType
Inference<GM, ACC>::value() const 
{
   if(ACC::hasbop()){ 
      // Default implementation if ACC defines an ordering  
      std::vector<LabelType> s;
      const GM& gm = graphicalModel();
      if(NORMAL == arg(s)) {
         return gm.evaluate(s);
      }
      else {
         return ACC::template neutral<ValueType>();
      }
   }else{
      //TODO: Maybe throw an exception here 
      //throw std::runtime_error("There is no default implementation for this type of semi-ring");
      return std::numeric_limits<ValueType>::quiet_NaN();
   }
}

/// \brief return a bound on the solution
template<class GM, class ACC>
typename GM::ValueType
Inference<GM, ACC>::bound() const { 
   if(ACC::hasbop()){
      // Default implementation if ACC defines an ordering
      return ACC::template ineutral<ValueType>();
   }else{
      //TODO: Maybe throw an exception here 
      //throw std::runtime_error("There is no default implementation for this type of semi-ring");
      return std::numeric_limits<ValueType>::quiet_NaN();
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_INFERENCE_HXX
