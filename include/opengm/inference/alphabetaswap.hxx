#pragma once
#ifndef OPENGM_ALPHABEATSWAP_HXX
#define OPENGM_ALPHABETASWAP_HXX

#include <vector>

#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// Alpha-Beta-Swap Algorithm
/// \ingroup inference
template<class GM, class INF>
class AlphaBetaSwap : public Inference<GM, typename INF::AccumulationType> {
public:
   typedef GM GraphicalModelType;
   typedef INF InferenceType;
   typedef typename INF::AccumulationType AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef opengm::visitors::VerboseVisitor<AlphaBetaSwap<GM,INF> > VerboseVisitorType;
   typedef opengm::visitors::EmptyVisitor<AlphaBetaSwap<GM,INF> >   EmptyVisitorType;
   typedef opengm::visitors::TimingVisitor<AlphaBetaSwap<GM,INF> >  TimingVisitorType;

   struct Parameter {
      Parameter() {
         maxNumberOfIterations_ = 1000;
      }

      typename InferenceType::Parameter parameter_; 
      size_t maxNumberOfIterations_; 
   };

   AlphaBetaSwap(const GraphicalModelType&, Parameter = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   template<class VISITOR>
   InferenceTermination infer(VISITOR & );
   void reset();
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;

private:
   const GraphicalModelType& gm_;
   Parameter parameter_;
   std::vector<LabelType> label_;
   size_t alpha_;
   size_t beta_;
   size_t maxState_;
   void increment();
   void addUnary(INF&, const size_t var, const ValueType v0, const ValueType v1);
   void addPairwise(INF&, const size_t var1, const size_t var2, const ValueType v0, const ValueType v1, const ValueType v2, const ValueType v3);
};

// reset assumes that the structure of the graphical model has not changed
template<class GM, class INF>
inline void
AlphaBetaSwap<GM, INF>::reset() {
   alpha_ = 0;
   beta_ = 0;
   std::fill(label_.begin(),label_.end(),0);
}

template<class GM, class INF>
inline void
AlphaBetaSwap<GM, INF>::increment() {
   if (++beta_ >= maxState_) {
      if (++alpha_ >= maxState_ - 1) {
         alpha_ = 0;
      }
      beta_ = alpha_ + 1;
   }
   OPENGM_ASSERT(alpha_ < maxState_);
   OPENGM_ASSERT(beta_ < maxState_);
   OPENGM_ASSERT(alpha_ < beta_);
}

template<class GM, class INF>
inline std::string
AlphaBetaSwap<GM, INF>::name() const {
   return "Alpha-Beta-Swap";
}

template<class GM, class INF>
inline const typename AlphaBetaSwap<GM, INF>::GraphicalModelType&
AlphaBetaSwap<GM, INF>::graphicalModel() const {
   return gm_;
}

template<class GM, class INF>
inline AlphaBetaSwap<GM, INF>::AlphaBetaSwap
(
   const GraphicalModelType& gm,
   Parameter para
)
:  gm_(gm)
{
   parameter_ = para;
   label_.resize(gm_.numberOfVariables(), 0);
   alpha_ = 0;
   beta_ = 0;
   for (size_t j = 0; j < gm_.numberOfFactors(); ++j) {
      if (gm_[j].numberOfVariables() > 2) {
         throw RuntimeError("This implementation of Alpha-Beta-Swap supports only factors of order <= 2.");
      }
   }
   maxState_ = 0;
   for (size_t i = 0; i < gm_.numberOfVariables(); ++i) {
      size_t numSt = gm_.numberOfLabels(i);
      if (numSt > maxState_)
         maxState_ = numSt;
   }
}

template<class GM, class INF>
inline void
AlphaBetaSwap<GM,INF>::setStartingPoint
(
   typename std::vector<typename AlphaBetaSwap<GM,INF>::LabelType>::const_iterator begin
) {
   try{
      label_.assign(begin, begin+gm_.numberOfVariables());
   }
   catch(...) {
      throw RuntimeError("unsuitable starting point");
   }
}

template<class GM, class INF>
inline void
AlphaBetaSwap<GM, INF>::addUnary
(
   INF& inf,
   const size_t var1,
   const ValueType v0,
   const ValueType v1
) {
   const size_t shape[] = {2};
   const size_t vars[] = {var1};
   opengm::IndependentFactor<ValueType,IndexType,LabelType> fac(vars, vars + 1, shape, shape + 1);
   fac(0) = v0;
   fac(1) = v1;
   inf.addFactor(fac);
}

template<class GM, class INF>
inline void
AlphaBetaSwap<GM, INF>::addPairwise
(
   INF& inf,
   const size_t var1,
   const size_t var2,
   const ValueType v0,
   const ValueType v1,
   const ValueType v2,
   const ValueType v3
) {
   const size_t shape[] = {2, 2};
   const size_t vars[] = {var1, var2};
   opengm::IndependentFactor<ValueType,IndexType,LabelType> fac(vars, vars + 2, shape, shape + 2);
   fac(0, 0) = v0;
   fac(0, 1) = v1;
   fac(1, 0) = v2;
   fac(1, 1) = v3;
   OPENGM_ASSERT(v1 + v2 - v0 - v3 >= 0);
   inf.addFactor(fac);
}
template<class GM, class INF>
InferenceTermination
AlphaBetaSwap<GM, INF>::infer() {
   EmptyVisitorType v;
   return infer(v);
}

template<class GM, class INF>
template<class VISITOR>
InferenceTermination
AlphaBetaSwap<GM, INF>::infer
(
   VISITOR & visitor
) {
   bool exitInf=false;
   visitor.begin(*this);
   size_t it = 0;
   size_t countUnchanged = 0;
   size_t numberOfVariables = gm_.numberOfVariables();
   std::vector<size_t> variable2Node(numberOfVariables, 0);
   ValueType energy = gm_.evaluate(label_);
   size_t vecA[1];
   size_t vecB[1];
   size_t vecAA[2];
   size_t vecAB[2];
   size_t vecBA[2];
   size_t vecBB[2];
   size_t vecAX[2];
   size_t vecBX[2];
   size_t vecXA[2];
   size_t vecXB[2];
   size_t numberOfLabelPairs = maxState_*(maxState_ - 1)/2;
   while (it++ < parameter_.maxNumberOfIterations_ && countUnchanged < numberOfLabelPairs && exitInf == false) {
      increment();
      size_t counter = 0;
      std::vector<size_t> numFacDim(4, 0);
      for (size_t i = 0; i < numberOfVariables; ++i) {
         if (label_[i] == alpha_ || label_[i] == beta_) {
            variable2Node[i] = counter++;
         }
      }
      if (counter == 0) {
         continue;
      }
      INF inf(counter, numFacDim);
      vecA[0] = alpha_;
      vecB[0] = beta_;
      vecAA[0] = alpha_;
      vecAA[1] = alpha_;
      vecBB[0] = beta_;
      vecBB[1] = beta_;
      vecBA[0] = beta_;
      vecBA[1] = alpha_;
      vecAB[0] = alpha_;
      vecAB[1] = beta_;
      vecAX[0] = alpha_;
      vecBX[0] = beta_;
      vecXA[1] = alpha_;
      vecXB[1] = beta_;
      for (size_t k = 0; k < gm_.numberOfFactors(); ++k) {
         const FactorType& factor = gm_[k];
         if (factor.numberOfVariables() == 1) {
            size_t var = factor.variableIndex(0);
            size_t node = variable2Node[var];
            if (label_[var] == alpha_ || label_[var] == beta_) {
               OPENGM_ASSERT(alpha_ < gm_.numberOfLabels(var));
               OPENGM_ASSERT(beta_ < gm_.numberOfLabels(var));
               addUnary(inf, node, factor(vecA), factor(vecB));
               //inf.addUnary(node, factor(vecA), factor(vecB));
            }
         } else if (factor.numberOfVariables() == 2) {
            size_t var1 = factor.variableIndex(0);
            size_t var2 = factor.variableIndex(1);
            size_t node1 = variable2Node[var1];
            size_t node2 = variable2Node[var2];

            if ((label_[var1] == alpha_ || label_[var1] == beta_) && (label_[var2] == alpha_ || label_[var2] == beta_)) {
               addPairwise(inf, node1, node2, factor(vecAA), factor(vecAB), factor(vecBA), factor(vecBB));
               //inf.addPairwise(node1, node2, factor(vecAA), factor(vecAB), factor(vecBA), factor(vecBB));
            } else if ((label_[var1] == alpha_ || label_[var1] == beta_) && (label_[var2] != alpha_ && label_[var2] != beta_)) {
               vecAX[1] = vecBX[1] = label_[var2];
               addUnary(inf, node1, factor(vecAX), factor(vecBX));
               //inf.addUnary(node1, factor(vecAX), factor(vecBX));
            } else if ((label_[var2] == alpha_ || label_[var2] == beta_) && (label_[var1] != alpha_ && label_[var1] != beta_)) {
               vecXA[0] = vecXB[0] = label_[var1];
               addUnary(inf, node2, factor(vecXA), factor(vecXB));
               //inf.addUnary(node2, factor(vecXA), factor(vecXB));
            }
         }
      }
      std::vector<LabelType> state; //(counter);
      inf.infer();
      inf.arg(state);
      OPENGM_ASSERT(state.size() == counter);
      for (size_t var = 0; var < numberOfVariables; ++var) {
         if (label_[var] == alpha_ || label_[var] == beta_) {
            if (state[variable2Node[var]] == 0)
               label_[var] = alpha_;
            else
               label_[var] = beta_;
         } else {
            //do nothing
         }
      }
      ValueType energy2 = gm_.evaluate(label_);
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
         exitInf=true;
      }
      OPENGM_ASSERT(!AccumulationType::ibop(energy2, energy));
      if (AccumulationType::bop(energy2, energy)) {
         energy = energy2;
      } else {
         ++countUnchanged;
      }
   }
   visitor.end(*this);
   return NORMAL;
}

template<class GM, class INF>
inline InferenceTermination
AlphaBetaSwap<GM, INF>::arg(std::vector<LabelType>& arg, const size_t n) const {
   if (n > 1) {
      return UNKNOWN;
   } else {
      OPENGM_ASSERT(label_.size() == gm_.numberOfVariables());
      arg.resize(label_.size());
      for (size_t i = 0; i < label_.size(); ++i)
         arg[i] = label_[i];
      return NORMAL;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_ALPHABEATSWAP_HXX
