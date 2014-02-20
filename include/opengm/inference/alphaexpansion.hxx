#pragma once
#ifndef OPENGM_ALPHAEXPANSION_HXX
#define OPENGM_ALPHAEXPANSION_HXX

#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// Alpha-Expansion Algorithm
/// \ingroup inference
template<class GM, class INF>
class AlphaExpansion
: public Inference<GM, typename INF::AccumulationType>
{
public:
   typedef GM GraphicalModelType;
   typedef INF InferenceType; 
   typedef typename INF::AccumulationType AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<AlphaExpansion<GM,INF> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<AlphaExpansion<GM,INF> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<AlphaExpansion<GM,INF> >  TimingVisitorType;

   struct Parameter {
      typedef typename InferenceType::Parameter InferenceParameter;
      enum LabelingIntitialType {DEFAULT_LABEL, RANDOM_LABEL, LOCALOPT_LABEL, EXPLICIT_LABEL};
      enum OrderType {DEFAULT_ORDER, RANDOM_ORDER, EXPLICIT_ORDER};

      Parameter
      (
         const size_t maxNumberOfSteps  = 1000,
         const InferenceParameter& para = InferenceParameter()
      )
      :  parameter_(para),
         maxNumberOfSteps_(maxNumberOfSteps),
         labelInitialType_(DEFAULT_LABEL),
         orderType_(DEFAULT_ORDER),
         randSeedOrder_(0),
         randSeedLabel_(0),
         labelOrder_(),
         label_()
      {}

      InferenceParameter parameter_;
      size_t maxNumberOfSteps_;
      LabelingIntitialType labelInitialType_;
      OrderType orderType_;
      unsigned int randSeedOrder_;
      unsigned int randSeedLabel_;
      std::vector<LabelType> labelOrder_;
      std::vector<LabelType> label_;
   };

   AlphaExpansion(const GraphicalModelType&, Parameter para = Parameter());

   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   template<class StateIterator>
      void setState(StateIterator, StateIterator);
   InferenceTermination infer();
   void reset();
   template<class Visitor>
      InferenceTermination infer(Visitor& visitor);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;

private:
   const GraphicalModelType& gm_;
   Parameter parameter_;
   std::vector<LabelType> label_;
   std::vector<LabelType> labelList_;
   size_t maxState_;
   size_t alpha_;
   size_t counter_;
   void incrementAlpha();
   void setLabelOrder(std::vector<LabelType>& l);
   void setLabelOrderRandom(unsigned int);
   void setInitialLabel(std::vector<LabelType>& l);
   void setInitialLabelLocalOptimal();
   void setInitialLabelRandom(unsigned int);
   void addUnary(INF&, const size_t var, const ValueType v0, const ValueType v1);
   void addPairwise(INF&, const size_t var1, const size_t var2, const ValueType v0, const ValueType v1, const ValueType v2, const ValueType v3);
};

template<class GM, class INF>
inline std::string
AlphaExpansion<GM, INF>::name() const
{
   return "Alpha-Expansion";
}

template<class GM, class INF>
inline const typename AlphaExpansion<GM, INF>::GraphicalModelType&
AlphaExpansion<GM, INF>::graphicalModel() const
{
   return gm_;
}

template<class GM, class INF>
template<class StateIterator>
inline void
AlphaExpansion<GM, INF>::setState
(
   StateIterator begin,
   StateIterator end
)
{
   label_.assign(begin, end);
}

template<class GM, class INF>
inline void
AlphaExpansion<GM,INF>::setStartingPoint
(
   typename std::vector<typename AlphaExpansion<GM,INF>::LabelType>::const_iterator begin
) {
   try{
      label_.assign(begin, begin+gm_.numberOfVariables());
   }
   catch(...) {
      throw RuntimeError("unsuitable starting point");
   }
}

template<class GM, class INF>
inline
AlphaExpansion<GM, INF>::AlphaExpansion
(
   const GraphicalModelType& gm,
   Parameter para
)
:  gm_(gm),
   parameter_(para),
   maxState_(0)
{
   for(size_t j=0; j<gm_.numberOfFactors(); ++j) {
      if(gm_[j].numberOfVariables() > 2) {
         throw RuntimeError("This implementation of Alpha-Expansion supports only factors of order <= 2.");
      }
   }
   for(size_t i=0; i<gm_.numberOfVariables(); ++i) {
      size_t numSt = gm_.numberOfLabels(i);
      if(numSt > maxState_) {
         maxState_ = numSt;
      }
   }

   if(parameter_.labelInitialType_ == Parameter::RANDOM_LABEL) {
      setInitialLabelRandom(parameter_.randSeedLabel_);
   }
   else if(parameter_.labelInitialType_ == Parameter::LOCALOPT_LABEL) {
      setInitialLabelLocalOptimal();
   }
   else if(parameter_.labelInitialType_ == Parameter::EXPLICIT_LABEL) {
      setInitialLabel(parameter_.label_);
   }
   else{
      label_.resize(gm_.numberOfVariables(), 0);
   }


   if(parameter_.orderType_ == Parameter::RANDOM_ORDER) {
      setLabelOrderRandom(parameter_.randSeedOrder_);
   }
   else if(parameter_.orderType_ == Parameter::EXPLICIT_ORDER) {
      setLabelOrder(parameter_.labelOrder_);
   }
   else{
      labelList_.resize(maxState_);
      for(size_t i=0; i<maxState_; ++i)
         labelList_[i] = i;
   }

   counter_ = 0;
   alpha_   = labelList_[counter_];
}

// reset assumes that the structure of
// the graphical model has not changed
template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::reset() {
   if(parameter_.labelInitialType_ == Parameter::RANDOM_LABEL) {
      setInitialLabelRandom(parameter_.randSeedLabel_);
   }
   else if(parameter_.labelInitialType_ == Parameter::LOCALOPT_LABEL) {
      setInitialLabelLocalOptimal();
   }
   else if(parameter_.labelInitialType_ == Parameter::EXPLICIT_LABEL) {
      setInitialLabel(parameter_.label_);
   }
   else{
      std::fill(label_.begin(),label_.end(),0);
   }


   if(parameter_.orderType_ == Parameter::RANDOM_ORDER) {
      setLabelOrderRandom(parameter_.randSeedOrder_);
   }
   else if(parameter_.orderType_ == Parameter::EXPLICIT_ORDER) {
      setLabelOrder(parameter_.labelOrder_);
   }
   else{
      for(size_t i=0; i<maxState_; ++i)
         labelList_[i] = i;
   }
   counter_ = 0;
   alpha_   = labelList_[counter_];
}

template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::addUnary
(
   INF& inf,
   const size_t var1,
   const ValueType v0,
   const ValueType v1
) {
   const size_t shape[] = {2};
   size_t vars[] = {var1};
   opengm::IndependentFactor<ValueType,IndexType,LabelType> fac(vars, vars+1, shape, shape+1);
   fac(0) = v0;
   fac(1) = v1;
   inf.addFactor(fac);
}

template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::addPairwise
(
   INF& inf,
   const size_t var1,
   const size_t var2,
   const ValueType v0,
   const ValueType v1,
   const ValueType v2,
   const ValueType v3
) {
   const LabelType shape[] = {2, 2};
   const IndexType vars[]  = {var1, var2};
   opengm::IndependentFactor<ValueType,IndexType,LabelType> fac(vars, vars+2, shape, shape+2);
   fac(0, 0) = v0;
   fac(0, 1) = v1;
   fac(1, 0) = v2;
   fac(1, 1) = v3;
   inf.addFactor(fac);
}

template<class GM, class INF>
inline InferenceTermination
AlphaExpansion<GM, INF>::infer()
{
   EmptyVisitorType visitor;
   return infer(visitor);
}

template<class GM, class INF>
template<class Visitor>
InferenceTermination
AlphaExpansion<GM, INF>::infer
(
   Visitor& visitor
)
{
   bool exitInf = false;
   size_t it = 0;
   size_t countUnchanged = 0;
   size_t numberOfVariables = gm_.numberOfVariables();
   std::vector<size_t> variable2Node(numberOfVariables);
   ValueType energy = gm_.evaluate(label_);
   visitor.begin(*this);
   LabelType vecA[1];
   LabelType vecX[1];
   LabelType vecAA[2];
   LabelType vecAX[2];
   LabelType vecXA[2];
   LabelType vecXX[2];
   while(it++ < parameter_.maxNumberOfSteps_ && countUnchanged < maxState_ && exitInf == false) {
      size_t numberOfAuxiliaryNodes = 0;
      for(size_t k=0 ; k<gm_.numberOfFactors(); ++k) {
         const FactorType& factor = gm_[k];
         if(factor.numberOfVariables() == 2) {
            size_t var1 = factor.variableIndex(0);
            size_t var2 = factor.variableIndex(1);
            if(label_[var1] != label_[var2] && label_[var1] != alpha_ && label_[var2] != alpha_ ) {
               ++numberOfAuxiliaryNodes;
            }
         }
      }
      std::vector<size_t> numFacDim(4, 0);
      INF inf(numberOfVariables + numberOfAuxiliaryNodes, numFacDim, parameter_.parameter_);
      size_t varX = numberOfVariables;
      size_t countAlphas = 0;
      for (size_t k=0 ; k<gm_.numberOfVariables(); ++k) {
         if (label_[k] == alpha_ ) {
            addUnary(inf, k, 0, std::numeric_limits<ValueType>::infinity());
            ++countAlphas;
         }
      }
      if(countAlphas < gm_.numberOfVariables()) {
         for (size_t k=0 ; k<gm_.numberOfFactors(); ++k) {
            const  FactorType& factor = gm_[k];
            if(factor.numberOfVariables() == 1) {
               size_t var = factor.variableIndex(0);
               vecA[0] = alpha_;
               vecX[0] = label_[var];
               if (label_[var] != alpha_ ) {
                  addUnary(inf, var, factor(vecA), factor(vecX));
               }
            }
            else if (factor.numberOfVariables() == 2) {
               size_t var1 = factor.variableIndex(0);
               size_t var2 = factor.variableIndex(1);
               std::vector<IndexType> vars(2); vars[0]=var1;vars[1]=var2;
               vecAA[0] = vecAA[1] = alpha_;
               vecAX[0] = alpha_;       vecAX[1] = label_[var2];
               vecXA[0] = label_[var1]; vecXA[1] = alpha_;
               vecXX[0] = label_[var1]; vecXX[1] = label_[var2];
               if(label_[var1]==alpha_ && label_[var2]==alpha_) {
                  continue;
               }
               else if(label_[var1]==alpha_) {
                  addUnary(inf, var2, factor(vecAA), factor(vecAX));
               }
               else if(label_[var2]==alpha_) {
                  addUnary(inf, var1, factor(vecAA), factor(vecXA));
               }
               else if(label_[var1]==label_[var2]) {
                  addPairwise(inf, var1, var2, factor(vecAA), factor(vecAX), factor(vecXA), factor(vecXX));
               }
               else{
                  OPENGM_ASSERT(varX < numberOfVariables + numberOfAuxiliaryNodes);
                  addPairwise(inf, var1, varX, 0, factor(vecAX), 0, 0);
                  addPairwise(inf, var2, varX, 0, factor(vecXA), 0, 0);
                  addUnary(inf, varX, factor(vecAA), factor(vecXX));
                  ++varX;
               }
            }
         }
         std::vector<LabelType> state;
         inf.infer();
         inf.arg(state);
         OPENGM_ASSERT(state.size() == numberOfVariables + numberOfAuxiliaryNodes);
         for(size_t var=0; var<numberOfVariables ; ++var) {
            if (label_[var] != alpha_ && state[var]==0) {
               label_[var] = alpha_;
            }
            OPENGM_ASSERT(label_[var] < gm_.numberOfLabels(var));
         }
      }
      OPENGM_ASSERT(gm_.numberOfVariables() == label_.size());
      ValueType energy2 = gm_.evaluate(label_);
      //visitor(*this,energy2,energy,alpha_);
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
         exitInf=true;
      }
      // OPENGM_ASSERT(!AccumulationType::ibop(energy2, energy));
      if(AccumulationType::bop(energy2, energy)) {
         energy=energy2;
         countUnchanged = 0;
      }
      else{
         ++countUnchanged;
      }
      incrementAlpha();
      OPENGM_ASSERT(alpha_ < maxState_);
   }
   visitor.end(*this);
   return NORMAL;
}

template<class GM, class INF>
inline InferenceTermination
AlphaExpansion<GM, INF>::arg
(
   std::vector<LabelType>& arg,
   const size_t n
) const
{
   if(n > 1) {
      return UNKNOWN;
   }
   else {
      OPENGM_ASSERT(label_.size() == gm_.numberOfVariables());
      arg.resize(label_.size());
      for(size_t i=0; i<label_.size(); ++i) {
         arg[i] = label_[i];
      }
      return NORMAL;
   }
}

template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::setLabelOrder
(
   std::vector<LabelType>& l
) {
   if(l.size() == maxState_) {
      labelList_=l;
   }
}

template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::setLabelOrderRandom
(
   unsigned int seed
) {
   srand(seed);
   labelList_.resize(maxState_);
   for (size_t i=0; i<maxState_;++i) {
      labelList_[i]=i;
   }
   random_shuffle(labelList_.begin(), labelList_.end());
}

template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::setInitialLabel
(
   std::vector<LabelType>& l
) {
   label_.resize(gm_.numberOfVariables());
   if(l.size() == label_.size()) {
      for(size_t i=0; i<l.size();++i) {
         if(l[i]>=gm_.numberOfLabels(i)) return;
      }
      for(size_t i=0; i<l.size();++i) {
         label_[i] = l[i];
      }
   }
}

template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::setInitialLabelLocalOptimal() {
   label_.resize(gm_.numberOfVariables(), 0);
   std::vector<size_t> accVec;
   for(size_t i=0; i<gm_.numberOfFactors();++i) {
      if(gm_[i].numberOfVariables()==1) {
         std::vector<size_t> state(1, 0);
         ValueType value = gm_[i](state.begin());
         for(state[0]=1; state[0]<gm_.numberOfLabels(i); ++state[0]) {
            if(AccumulationType::bop(gm_[i](state.begin()), value)) {
               value = gm_[i](state.begin());
               label_[i] = state[0];
            }
         }
      }
   }
}

template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::setInitialLabelRandom
(
   unsigned int seed
) {
   srand(seed);
   label_.resize(gm_.numberOfVariables());
   for(size_t i=0; i<gm_.numberOfVariables();++i) {
      label_[i] = rand() % gm_.numberOfLabels(i);
   }
}

template<class GM, class INF>
inline void
AlphaExpansion<GM, INF>::incrementAlpha() {
   counter_ = (counter_+1) % maxState_;
   alpha_ = labelList_[counter_];
}

} // namespace opengm

#endif // #ifndef OPENGM_ALPHAEXPANSION_HXX
