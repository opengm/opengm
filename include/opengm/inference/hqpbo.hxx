#pragma once
#ifndef OPENGM_HQPBO_HXX
#define OPENGM_HQPBO_HXX

#include "opengm/graphicalmodel/graphicalmodel_factor.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

#include "opengm/inference/external/qpbo.hxx"
#include "opengm/inference/fix-fusion/fusion-move.hpp"

namespace opengm {
   
/// HQPBO Algorithm\n\n
/// 
///
/// \ingroup inference 
template<class GM, class ACC>
class HQPBO : public Inference<GM, opengm::Minimizer>
{
public:
   typedef GM GraphicalModelType;
   typedef ACC AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<HQPBO<GM,ACC> > VerboseVisitorType;
   typedef visitors::TimingVisitor<HQPBO<GM,ACC> > TimingVisitorType;
   typedef visitors::EmptyVisitor<HQPBO<GM,ACC> > EmptyVisitorType;

   struct Parameter {};

   HQPBO(const GraphicalModelType&, Parameter = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   template<class VISITOR>
      InferenceTermination infer(VISITOR &);
   InferenceTermination arg(std::vector<LabelType>&, const size_t& = 1) const;
   void setStartingPoint(typename std::vector<LabelType>::const_iterator begin );
private:
   const GraphicalModelType& gm_;
   ValueType constV_;
   HigherOrderEnergy<ValueType, 10> hoe_;
   std::vector<LabelType> conf_;
   ValueType bound_;
}; 
 
template<class GM, class ACC>
inline void 
HQPBO<GM,ACC>::setStartingPoint
(
   typename std::vector<typename HQPBO<GM,ACC>::LabelType>::const_iterator begin
) {
   for (size_t i=0; i<gm_.numberOfVariables(); ++i)
      conf_[i] = *(begin+i);
}

template<class GM,class ACC>
HQPBO<GM,ACC>::HQPBO
(
   const GM & gm,
   typename HQPBO<GM,ACC>::Parameter
)
   :  gm_(gm), constV_(0.0), conf_(std::vector<LabelType>(gm.numberOfVariables(),0))
{
   hoe_.AddVars(gm_.numberOfVariables());
   for (IndexType f = 0; f < gm_.numberOfFactors(); ++f)
   {
      IndexType size = gm_[f].numberOfVariables();
      const LabelType l0 = 0;
      const LabelType l1 = 1;
      if (size == 0)
      {
         constV_ += gm_[f](&l0);
         continue;
      }
      else if (size == 1)
      {
         IndexType var = gm_[f].variableIndex(0);
         const ValueType e0 = gm_[f](&l0);
         const ValueType e1 = gm_[f](&l1);
         hoe_.AddUnaryTerm(var, e1 - e0);
      }
      else
      {
         unsigned int numAssignments = 1 << size;
         ValueType coeffs[numAssignments];
         for (unsigned int subset = 1; subset < numAssignments; ++subset)
         {
            coeffs[subset] = 0;
         }
         // For each boolean assignment, get the clique energy at the
         // corresponding labeling
         LabelType cliqueLabels[size];
         for (unsigned int assignment = 0;  assignment < numAssignments; ++assignment)
         {
            for (unsigned int i = 0; i < size; ++i)
            {
               if (assignment & (1 << i))
               {
                  cliqueLabels[i] = l1;
               }
               else
               {
                  cliqueLabels[i] = l0;
               }
            }
            ValueType energy = gm_[f](cliqueLabels);
            for (unsigned int subset = 1; subset < numAssignments; ++subset)
            {
               if (assignment & ~subset)
               {
                  continue;
               }
               else
               {
                  int parity = 0;
                  for (unsigned int b = 0; b < size; ++b)
                  {
                     parity ^=  (((assignment ^ subset) & (1 << b)) != 0);
                  }
                  coeffs[subset] += parity ? -energy : energy;
               }
            }
         }
         typename HigherOrderEnergy<ValueType, 10>::VarId vars[10];
         for (unsigned int subset = 1; subset < numAssignments; ++subset)
         {
            int degree = 0;
            for (unsigned int b = 0; b < size; ++b)
            {
               if (subset & (1 << b))
               {
                  vars[degree++] = gm_[f].variableIndex(b);
               }
            }
            std::sort(vars, vars + degree);
            hoe_.AddTerm(coeffs[subset], degree, vars);
         }
      }
   } 
}

template<class GM,class ACC>
inline std::string
HQPBO<GM,ACC>::name() const
{
   return "HQPBO";
}

template<class GM,class ACC>
inline const typename HQPBO<GM,ACC>::GraphicalModelType&
HQPBO<GM,ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM,class ACC>
inline InferenceTermination
HQPBO<GM,ACC>::infer() {
   EmptyVisitorType v;
   return infer(v);
}

template<class GM,class ACC>
template<class VISITOR>
inline InferenceTermination
HQPBO<GM,ACC>::infer(VISITOR & visitor) 
{
   visitor.begin(*this);
   kolmogorov::qpbo::QPBO<ValueType>  qr(gm_.numberOfVariables(), 0);
   hoe_.ToQuadratic(qr);
   qr.Solve();
   IndexType numberOfChangedVariables = 0;
   for (IndexType i = 0; i < gm_.numberOfVariables(); ++i)
   {
      int label = qr.GetLabel(i);
      if (label == 0 )
      {
         conf_[i] = 0;
      }
      else if (label == 1)
      {
         conf_[i] = 1;
      }
      else
      {
         //conf_[i] = 0;
      }
   }
   bound_ = constV_ + 0.5 * qr.ComputeTwiceLowerBound();
   visitor.end(*this);
   return NORMAL;
}

template<class GM,class ACC>
inline InferenceTermination
HQPBO<GM,ACC>::arg
(
   std::vector<LabelType>& arg,
   const size_t& n
   ) const
{
   if(n > 1) {
      return UNKNOWN;
   }
   else {
      arg.resize(gm_.numberOfVariables());
      for (IndexType i = 0; i < gm_.numberOfVariables(); ++i)
         arg[i] =conf_[i];
      return NORMAL;
   }
}


} // namespace opengm

#endif // #ifndef OPENGM_HQPBO_HXX
