#pragma once
#ifndef OPENGM_ALPHAEXPANSIONFUSION_HXX
#define OPENGM_ALPHAEXPANSIONSUSION_HXX

#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/inference/fix-fusion/fusion-move.hpp"
#include "QPBO.h"

namespace opengm {

/// Alpha-Expansion-Fusion Algorithm
/// uses the code of Alexander Fix to reduce the higer order moves to binary pairwise problems which are solved by QPBO as described in
/// Alexander Fix, Artinan Gruber, Endre Boros, Ramin Zabih:  A Graph Cut Algorithm for Higher Order Markov Random Fields, ICCV 2011
///
/// Corresponding author: Joerg Hendrik Kappes
///
/// \ingroup inference
template<class GM, class ACC>
class AlphaExpansionFusion : public Inference<GM, ACC>
{
public:
   typedef GM GraphicalModelType; 
   typedef ACC AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<AlphaExpansionFusion<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<AlphaExpansionFusion<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<AlphaExpansionFusion<GM,ACC> >  TimingVisitorType;

   struct Parameter {
      enum LabelingIntitialType {DEFAULT_LABEL, RANDOM_LABEL, LOCALOPT_LABEL, EXPLICIT_LABEL};
      enum OrderType {DEFAULT_ORDER, RANDOM_ORDER, EXPLICIT_ORDER};

      Parameter
      (
         const size_t maxNumberOfSteps  = 1000
      )
      :  maxNumberOfSteps_(maxNumberOfSteps),
         labelInitialType_(DEFAULT_LABEL),
         orderType_(DEFAULT_ORDER),
         randSeedOrder_(0),
         randSeedLabel_(0),
         labelOrder_(),
         label_()
      {}

      size_t maxNumberOfSteps_;
      LabelingIntitialType labelInitialType_;
      OrderType orderType_;
      unsigned int randSeedOrder_;
      unsigned int randSeedLabel_;
      std::vector<LabelType> labelOrder_;
      std::vector<LabelType> label_;
   };

   AlphaExpansionFusion(const GraphicalModelType&, Parameter para = Parameter());

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
   static const size_t maxOrder_ =10;
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
   template<class INF>
   void addUnary(INF&, const size_t var, const ValueType v0, const ValueType v1);
   template<class INF>
   void addPairwise(INF&, const size_t var1, const size_t var2, const ValueType v0, const ValueType v1, const ValueType v2, const ValueType v3);
};

template<class GM, class ACC>
inline std::string
AlphaExpansionFusion<GM, ACC>::name() const
{
   return "Alpha-Expansion-Fusion";
}

template<class GM, class ACC>
inline const typename AlphaExpansionFusion<GM, ACC>::GraphicalModelType&
AlphaExpansionFusion<GM, ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC>
template<class StateIterator>
inline void
AlphaExpansionFusion<GM, ACC>::setState
(
   StateIterator begin,
   StateIterator end
)
{
   label_.assign(begin, end);
}

template<class GM, class ACC>
inline void
AlphaExpansionFusion<GM,ACC>::setStartingPoint
(
   typename std::vector<typename AlphaExpansionFusion<GM,ACC>::LabelType>::const_iterator begin
) {
   try{
      label_.assign(begin, begin+gm_.numberOfVariables());
   }
   catch(...) {
      throw RuntimeError("unsuitable starting point");
   }
}

template<class GM, class ACC>
inline
AlphaExpansionFusion<GM, ACC>::AlphaExpansionFusion
(
   const GraphicalModelType& gm,
   Parameter para
)
:  gm_(gm),
   parameter_(para),
   maxState_(0)
{
   for(size_t j=0; j<gm_.numberOfFactors(); ++j) {
      if(gm_[j].numberOfVariables() > maxOrder_) {
         throw RuntimeError("This implementation of Alpha-Expansion-Fusion supports only factors of this order! Increase the constant maxOrder_!");
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
template<class GM, class ACC>
inline void
AlphaExpansionFusion<GM, ACC>::reset() {
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

template<class GM, class ACC>
template<class INF>
inline void
AlphaExpansionFusion<GM, ACC>::addUnary
(
   INF& inf,
   const size_t var1,
   const ValueType v0,
   const ValueType v1
) {
   inf.AddUnaryTerm((int) (var1), v0, v1);
}

template<class GM, class ACC>
template<class INF>
inline void
AlphaExpansionFusion<GM, ACC>::addPairwise
(
   INF& inf,
   const size_t var1,
   const size_t var2,
   const ValueType v0,
   const ValueType v1,
   const ValueType v2,
   const ValueType v3
) {
   inf.AddPairwiseTerm((int) (var1), (int)(var2),v0,v1,v2,v3);
}

template<class GM, class ACC>
inline InferenceTermination
AlphaExpansionFusion<GM, ACC>::infer()
{
   EmptyVisitorType visitor;
   return infer(visitor);
}

template<class GM, class ACC>
template<class Visitor>
InferenceTermination
AlphaExpansionFusion<GM, ACC>::infer
(
   Visitor& visitor
)
{
   bool exitInf = false;
   size_t it = 0;
   size_t countUnchanged = 0;
//   size_t numberOfVariables = gm_.numberOfVariables();
//   std::vector<size_t> variable2Node(numberOfVariables);
   //ValueType energy = gm_.evaluate(label_);
   //visitor.begin(*this,energy,this->bound(),0);
   visitor.begin(*this);
/*
   LabelType vecA[1];
   LabelType vecX[1];
   LabelType vecAA[2];
   LabelType vecAX[2];
   LabelType vecXA[2];
   LabelType vecXX[2];
*/
   while(it++ < parameter_.maxNumberOfSteps_ && countUnchanged < maxState_ && exitInf == false) {
      // DO MOVE 
      unsigned int maxNumAssignments = 1 << maxOrder_;
      std::vector<ValueType> coeffs(maxNumAssignments);
      std::vector<LabelType> cliqueLabels(maxOrder_);

      HigherOrderEnergy<ValueType, maxOrder_> hoe;
      hoe.AddVars(gm_.numberOfVariables());
      for(IndexType f=0; f<gm_.numberOfFactors(); ++f){
         IndexType size = gm_[f].numberOfVariables();
         if (size == 0) {
            continue;
         } else if (size == 1) {
            IndexType var = gm_[f].variableIndex(0);
            ValueType e0 = gm_[f](&label_[var]);
            ValueType e1 = gm_[f](&alpha_);
            hoe.AddUnaryTerm(var, e1 - e0);
         } else {

            // unsigned int numAssignments = std::pow(2,size);
            unsigned int numAssignments = 1 << size;
            // -- // ValueType coeffs[numAssignments];
            for (unsigned int subset = 1; subset < numAssignments; ++subset) {
               coeffs[subset] = 0;
            }
            // For each boolean assignment, get the clique energy at the 
            // corresponding labeling
            // -- // LabelType cliqueLabels[size];
            for(unsigned int assignment = 0;  assignment < numAssignments; ++assignment){
               for (unsigned int i = 0; i < size; ++i) {
                  // only true for each second assigment?!?
                  //if (    assignment%2 ==  (std::pow(2,i))%2  )
                  if (assignment & (1 << i)) { 
                     cliqueLabels[i] = alpha_;
                  } else {
                     cliqueLabels[i] = label_[gm_[f].variableIndex(i)];
                  }
               }
               ValueType energy = gm_[f](cliqueLabels.begin());
               for (unsigned int subset = 1; subset < numAssignments; ++subset){
                  // if (assigment%2 != subset%2)
                  if (assignment & ~subset) {
                     continue;
                  } 
                  //(assigment%2 == subset%2)
                  else {
                     int parity = 0;
                     for (unsigned int b = 0; b < size; ++b) {
                        parity ^=  (((assignment ^ subset) & (1 << b)) != 0);
                     }
                     coeffs[subset] += parity ? -energy : energy;
                  }
               }
            }
            typename HigherOrderEnergy<ValueType, maxOrder_> ::VarId vars[maxOrder_];
            for (unsigned int subset = 1; subset < numAssignments; ++subset) {
               int degree = 0;
               for (unsigned int b = 0; b < size; ++b) {
                  if (subset & (1 << b)) {
                     vars[degree++] = gm_[f].variableIndex(b);
                  }
               }
               std::sort(vars, vars+degree);
               hoe.AddTerm(coeffs[subset], degree, vars);
            }
         }
      }  
      kolmogorov::qpbo::QPBO<ValueType>  qr(gm_.numberOfVariables(), 0); 
      hoe.ToQuadratic(qr);
      qr.Solve();
      IndexType numberOfChangedVariables = 0;
      for (IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
         int label = qr.GetLabel(i);
         if (label == 1) {
            label_[i] = alpha_;
            ++numberOfChangedVariables;
         } 
      }
      
      OPENGM_ASSERT(gm_.numberOfVariables() == label_.size());
      //ValueType energy2 = gm_.evaluate(label_);
      if(numberOfChangedVariables>0){
         //energy=energy2;
         countUnchanged = 0;
      }else{
         ++countUnchanged;
      }
      //visitor(*this,energy2,this->bound(),"alpha",alpha_);
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
         exitInf = true;
      }
      // OPENGM_ASSERT(!AccumulationType::ibop(energy2, energy));
      incrementAlpha();
      OPENGM_ASSERT(alpha_ < maxState_);
   } 
   //visitor.end(*this,energy,this->bound(),0);
   visitor.end(*this);
   return NORMAL; 
   /*
      while(it++ < parameter_.maxNumberOfSteps_ && countUnchanged < maxState_) {
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

         kolmogorov::qpbo::QPBO<ValueType >  inf(numberOfVariables + numberOfAuxiliaryNodes, gm_.numberOfFactors()); 
         inf.AddNode(numberOfVariables + numberOfAuxiliaryNodes);
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
                     addUnary(inf, var, factor(vecX), factor(vecA));
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
                     addUnary(inf, var2, factor(vecAX), factor(vecAA));
                  }
                  else if(label_[var2]==alpha_) {
                     addUnary(inf, var1, factor(vecXA), factor(vecAA));
                  }
                  else if(label_[var1]==label_[var2]) {
                     addPairwise(inf, var1, var2, factor(vecXX), factor(vecXA), factor(vecAX), factor(vecAA));
                  }
                  else{
                     OPENGM_ASSERT(varX < numberOfVariables + numberOfAuxiliaryNodes);
                     addPairwise(inf, var1, varX, 0, factor(vecXA), 0, 0);
                     addPairwise(inf, var2, varX, 0, factor(vecAX), 0, 0);
                     addUnary(inf, varX, factor(vecXX), factor(vecAA));
                     ++varX;
                  }
               }
            }
            inf.MergeParallelEdges();
            inf.Solve();
       
            for(size_t var=0; var<numberOfVariables ; ++var) {
               int b = inf.GetLabel(var);
               if (label_[var] != alpha_ && b==1) {
                  label_[var] = alpha_;
               }
               OPENGM_ASSERT(label_[var] < gm_.numberOfLabels(var));
            }
         }
         OPENGM_ASSERT(gm_.numberOfVariables() == label_.size());
         ValueType energy2 = gm_.evaluate(label_);
         visitor(*this,energy,this->bound(),alpha_);
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
   }
   visitor.end(*this,energy,this->bound(),0);
   return NORMAL; 
*/
}

template<class GM, class ACC>
inline InferenceTermination
AlphaExpansionFusion<GM, ACC>::arg
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

template<class GM, class ACC>
inline void
AlphaExpansionFusion<GM, ACC>::setLabelOrder
(
   std::vector<LabelType>& l
) {
   if(l.size() == maxState_) {
      labelList_=l;
   }
}

template<class GM, class ACC>
inline void
AlphaExpansionFusion<GM, ACC>::setLabelOrderRandom
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

template<class GM, class ACC>
inline void
AlphaExpansionFusion<GM, ACC>::setInitialLabel
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

template<class GM, class ACC>
inline void
AlphaExpansionFusion<GM, ACC>::setInitialLabelLocalOptimal() {
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

template<class GM, class ACC>
inline void
AlphaExpansionFusion<GM, ACC>::setInitialLabelRandom
(
   unsigned int seed
) {
   srand(seed);
   label_.resize(gm_.numberOfVariables());
   for(size_t i=0; i<gm_.numberOfVariables();++i) {
      label_[i] = rand() % gm_.numberOfLabels(i);
   }
}

template<class GM, class ACC>
inline void
AlphaExpansionFusion<GM, ACC>::incrementAlpha() {
   counter_ = (counter_+1) % maxState_;
   alpha_ = labelList_[counter_];
}

} // namespace opengm

#endif // #ifndef OPENGM_ALPHAEXPANSIONFUSION_HXX
