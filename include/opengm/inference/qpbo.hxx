#pragma once
#ifndef OPENGM_QPBO_HXX
#define OPENGM_QPBO_HXX

#include "opengm/graphicalmodel/graphicalmodel_factor.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {
   
/// QPBO Algorithm\n\n
/// C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer, "Optimizing binary MRFs via extended roof duality", CVPR 2007
///
/// \ingroup inference 
template<class GM, class MIN_ST_CUT>
class QPBO : public Inference<GM, opengm::Minimizer>
{
public:
   typedef GM GraphicalModelType;
   typedef opengm::Minimizer AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<QPBO<GM,MIN_ST_CUT> > VerboseVisitorType;
   typedef visitors::TimingVisitor<QPBO<GM,MIN_ST_CUT> > TimingVisitorType;
   typedef visitors::EmptyVisitor<QPBO<GM,MIN_ST_CUT> > EmptyVisitorType;

   struct Parameter {};

   QPBO(const GraphicalModelType&, Parameter = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   template<class VISITOR>
      InferenceTermination infer(VISITOR &);
   InferenceTermination arg(std::vector<LabelType>&, const size_t& = 1) const;
   double partialOptimality(std::vector<bool>&) const;

private:
   void addUnaryFactorType(const FactorType& factor);
   void addUnaryFactorType(size_t var, ValueType value0, ValueType value1);
   void addEdgeCapacity(size_t v,size_t w, ValueType val);
   void addPairwiseFactorType(const FactorType& factor);
   void addPairwiseFactorType(size_t var0,size_t var1,ValueType A,ValueType B,ValueType C,ValueType D);

   // get the index of the opposite literal in the graph_
   size_t neg(size_t var) const { return (var+numVars_)%(2*numVars_); }

   const GraphicalModelType& gm_;
   //std::vector<LabelType> state_;
   std::vector<bool> stateBool_;
   size_t numVars_;
   ValueType constTerm_;
   MIN_ST_CUT minStCut_;
   ValueType tolerance_;
   size_t source_;
   size_t sink_;
};

template<class GM,class MIN_ST_CUT>
QPBO<GM,MIN_ST_CUT>::QPBO
(
   const GM & gm,
   typename QPBO<GM,MIN_ST_CUT>::Parameter
)
:  gm_(gm),
   numVars_(gm_.numberOfVariables()),
   minStCut_(2*gm_.numberOfVariables()+2, 6*gm_.numberOfVariables()) /// now many edges?
{
   constTerm_ = 0;
   source_ = 2*numVars_;
   sink_   = 2*numVars_ + 1;

   // add pairwise factors
   for(size_t j=0; j<gm_.numberOfFactors(); ++j) {
      switch (gm_[j].numberOfVariables()) {
      case 0:
      {
         size_t c[]={0};
         constTerm_ += gm_[j](c);
      }
      break;
      case 1:
         addUnaryFactorType(gm_[j]);
         break;
      case 2:
         addPairwiseFactorType(gm_[j]);
         break;
      default: throw std::runtime_error("This implementation of the QPBO optimizer does not support factors of order >2.");
      }
   }
}

template<class GM,class MIN_ST_CUT>
inline std::string
QPBO<GM,MIN_ST_CUT>::name() const
{
   return "QPBO";
}

template<class GM,class MIN_ST_CUT>
inline const typename QPBO<GM,MIN_ST_CUT>::GraphicalModelType&
QPBO<GM,MIN_ST_CUT>::graphicalModel() const
{
   return gm_;
}

template<class GM,class MIN_ST_CUT>
inline InferenceTermination
QPBO<GM,MIN_ST_CUT>::infer() {
   EmptyVisitorType v;
   return infer(v);
}

template<class GM,class MIN_ST_CUT>
template<class VISITOR>
inline InferenceTermination
QPBO<GM,MIN_ST_CUT>::infer(VISITOR & visitor) 
{
   visitor.begin(*this);
   minStCut_.calculateCut(stateBool_); 
   visitor.end(*this);
   return NORMAL;
}

template<class GM,class MIN_ST_CUT>
inline InferenceTermination
QPBO<GM,MIN_ST_CUT>::arg
(
   std::vector<LabelType>& arg,
   const size_t& n
   ) const
{
   if(n > 1) {
      return UNKNOWN;
   }
   else {
      arg.resize(numVars_);
      for(size_t j=0; j<arg.size(); ++j) {
         if (stateBool_[j+2] == true && stateBool_[neg(j)+2] == false)
            arg[j] = 1;
         else if (stateBool_[j+2] == false && stateBool_[neg(j)+2] == true)
            arg[j] = 0;
         else
            arg[j] = 0; // select 0 or 1
      }
      return NORMAL;
   }
}

template<class GM,class MIN_ST_CUT>
double
QPBO<GM,MIN_ST_CUT>::partialOptimality
(
   std::vector<bool>& optVec
   ) const
{
   double opt = 0;
   optVec.resize(numVars_);
   for(size_t j=0; j<optVec.size(); ++j)
      if (stateBool_[j+2] != stateBool_[neg(j)+2]) {
         optVec[j] = true;
         opt++;
      } else
         optVec[j] = false;

   return opt/gm_.numerOfVariables();
}

template<class GM,class MIN_ST_CUT>
void inline
QPBO<GM,MIN_ST_CUT>::addEdgeCapacity(size_t v, size_t w, ValueType val)
{
   minStCut_.addEdge((v+2)%(2*numVars_+2),(w+2)%(2*numVars_+2),val);
}

template<class GM,class MIN_ST_CUT>
void
QPBO<GM,MIN_ST_CUT>::addUnaryFactorType(const FactorType& factor)
{
   // indices of literal nodes in graph_
   size_t x_i  = factor.variableIndex(0);
   size_t nx_i = neg(x_i);

   // conversion to normal form on-the-fly: c_[n]x_i are the new
   // values of the unary factor.
   size_t c[]={0};
   ValueType c_nx_i = factor(c);
   c[0]=1;
   ValueType c_x_i  = factor(c);

   // has to be zero
   ValueType delta = std::min(c_nx_i, c_x_i);
   c_nx_i     -= delta;
   c_x_i      -= delta;
   constTerm_ += delta;

   addEdgeCapacity(x_i,    sink_, 0.5*c_nx_i);
   addEdgeCapacity(source_, nx_i, 0.5*c_nx_i);

   addEdgeCapacity(nx_i,   sink_, 0.5*c_x_i);
   addEdgeCapacity(source_,  x_i, 0.5*c_x_i);
}

template<class GM,class MIN_ST_CUT>
void
QPBO<GM,MIN_ST_CUT>::addPairwiseFactorType
(
   const FactorType& factor
) {
   // indices of literal nodes in graph_
   size_t x_i  = factor.variableIndex(0);
   size_t x_j  = factor.variableIndex(1);
   size_t nx_i = neg(x_i);
   size_t nx_j = neg(x_j);

   // conversion to normal form on-the-fly: c_[n]x_i_[n]x_j are the new
   // values of the pairwise factors. delta_c_[n]x_{i,j} are changes that have
   // to be made to the unary factors.

   size_t c[]={0,0};
   ValueType c_nx_i_nx_j = factor(c);
   c[1]=1;
   ValueType c_nx_i_x_j  = factor(c);
   c[0]=1;
   c[1]=0;
   ValueType c_x_i_nx_j  = factor(c);
   c[1]=1;
   ValueType c_x_i_x_j   = factor(c);

   ValueType delta_c_nx_j = 0;
   ValueType delta_c_x_j  = 0;
   ValueType delta_c_nx_i = 0;
   ValueType delta_c_x_i  = 0;

   // hast to be zero
   ValueType delta = std::min(c_nx_i_nx_j, c_x_i_nx_j);
   if (delta != 0) {

      c_nx_i_nx_j  -= delta;
      c_x_i_nx_j   -= delta;
      delta_c_nx_j += delta;
   }

   // has to be zero
   delta = std::min(c_nx_i_x_j, c_x_i_x_j);
   if (delta != 0) {

      c_nx_i_x_j  -= delta;
      c_x_i_x_j   -= delta;
      delta_c_x_j += delta;
   }

   // has to be zero
   delta = std::min(c_nx_i_nx_j, c_nx_i_x_j);
   if (delta != 0) {

      c_nx_i_nx_j  -= delta;
      c_nx_i_x_j   -= delta;
      delta_c_nx_i += delta;
   }

   // has to be zero
   delta = std::min(c_x_i_nx_j, c_x_i_x_j);
   if (delta != 0) {

      c_x_i_nx_j  -= delta;
      c_x_i_x_j   -= delta;
      delta_c_x_i += delta;
   }

   // for every non-zero c_[n]x_i_[n]x_j add two edges to the flow network

   if (c_nx_i_nx_j != 0) {
      addEdgeCapacity(x_i,  nx_j, 0.5*c_nx_i_nx_j);
      addEdgeCapacity(x_j,  nx_i, 0.5*c_nx_i_nx_j);
   }
   if (c_nx_i_x_j != 0) {
      addEdgeCapacity(x_i,   x_j, 0.5*c_nx_i_x_j);
      addEdgeCapacity(nx_j, nx_i, 0.5*c_nx_i_x_j);
   }
   if (c_x_i_nx_j != 0) {
      addEdgeCapacity(nx_i, nx_j, 0.5*c_x_i_nx_j);
      addEdgeCapacity(x_j,   x_i, 0.5*c_x_i_nx_j);
   }
   if (c_x_i_x_j != 0) {
      addEdgeCapacity(nx_i,  x_j, 0.5*c_x_i_x_j);
      addEdgeCapacity(nx_j,  x_i, 0.5*c_x_i_x_j);
   }

   // for every non-zero c_[n]x_{i,j} add two edges to the flow network

   if (delta_c_nx_j != 0) {
      addEdgeCapacity(x_j,    sink_, 0.5*delta_c_nx_j);
      addEdgeCapacity(source_, nx_j, 0.5*delta_c_nx_j);
   }
   if (delta_c_x_j != 0) {
      addEdgeCapacity(nx_j,   sink_, 0.5*delta_c_x_j);
      addEdgeCapacity(source_,  x_j, 0.5*delta_c_x_j);
   }
   if (delta_c_nx_i != 0) {
      addEdgeCapacity(x_i,    sink_, 0.5*delta_c_nx_i);
      addEdgeCapacity(source_, nx_i, 0.5*delta_c_nx_i);
   }
   if (delta_c_x_i != 0) {
      addEdgeCapacity(nx_i,   sink_, 0.5*delta_c_x_i);
      addEdgeCapacity(source_,  x_i, 0.5*delta_c_x_i);
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_EXTERNAL_QPBO_HXX
