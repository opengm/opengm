#pragma once
#ifndef OPENGM_GRAPHCUT_HXX
#define OPENGM_GRAPHCUT_HXX

#include <typeinfo>

#include "opengm/operations/adder.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// A framework for min st-cut algorithms.
///
/// \ingroup inference
template<class GM, class ACC, class MINSTCUT>
class GraphCut : public Inference<GM, ACC> {
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef MINSTCUT MinStCutType;
   typedef visitors::VerboseVisitor<GraphCut<GM, ACC, MINSTCUT> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<GraphCut<GM, ACC, MINSTCUT> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<GraphCut<GM, ACC, MINSTCUT> >  TimingVisitorType;
   struct Parameter {
      Parameter(const ValueType scale = 1)
         : scale_(scale) 
         {}
      ValueType scale_;
   };

   GraphCut(const GraphicalModelType&, const Parameter& = Parameter(), ValueType = static_cast<ValueType>(0.0));
   GraphCut(size_t numVar, std::vector<size_t> numFacDim, const Parameter& = Parameter(), ValueType = static_cast<ValueType>(0.0));
   ~GraphCut();

   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   template<class FACTOR>
      void addFactor(const FACTOR& factor);
   InferenceTermination infer();
   template<class VISITOR>
   InferenceTermination infer(VISITOR & visitor);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;

private:
   void addEdgeCapacity(const size_t, const size_t, const ValueType);
   size_t tripleId(std::vector<size_t>&);

   const GraphicalModelType& gm_;
   ValueType tolerance_;
   MinStCutType* minStCut_;
   Parameter parameter_;
   size_t numVariables_;
   std::vector<size_t> numFacDim_;
   std::list<std::vector<size_t> > tripleList;
   std::vector<bool> state_;
   std::vector<typename MINSTCUT::ValueType> sEdges_;
   std::vector<typename MINSTCUT::ValueType> tEdges_;
   bool inferenceDone_;
};

template<class GM, class ACC, class MINSTCUT>
inline std::string
GraphCut<GM, ACC, MINSTCUT>::name() const {
   return "GraphCut";
}

template<class GM, class ACC, class MINSTCUT>
inline const typename GraphCut<GM, ACC, MINSTCUT>::GraphicalModelType&
GraphCut<GM, ACC, MINSTCUT>::graphicalModel() const {
   return gm_;
}

template<class GM, class ACC, class MINSTCUT>
inline GraphCut<GM, ACC, MINSTCUT>::GraphCut
(
   const size_t numVariables,  
   std::vector<size_t> numFacDim, 
   const Parameter& para, 
   const ValueType tolerance
)
:  gm_(GM()), 
   tolerance_(fabs(tolerance))
{
   OPENGM_ASSERT(typeid(ACC) == typeid(opengm::Minimizer) || typeid(ACC) == typeid(opengm::Maximizer));
   OPENGM_ASSERT(typeid(typename GM::OperatorType) == typeid(opengm::Adder));
   OPENGM_ASSERT(numFacDim_.size() <= 3+1);
   parameter_ = para;
   numVariables_ = numVariables;
   numFacDim_ = numFacDim;
   numFacDim_.resize(4);
   minStCut_ = new MinStCutType(2 + numVariables_ + numFacDim_[3], 2*numVariables_ + numFacDim_[2] + 3*numFacDim_[3]);
   sEdges_.assign(numVariables_ + numFacDim_[3], 0);
   tEdges_.assign(numVariables_ + numFacDim_[3], 0);
   inferenceDone_=false;
   //std::cout << parameter_.scale_ <<std::endl;
}

template<class GM, class ACC, class MINSTCUT>
inline GraphCut<GM, ACC, MINSTCUT>::GraphCut
(
   const GraphicalModelType& gm, 
   const Parameter& para, 
   const ValueType tolerance
) 
:  gm_(gm), 
   tolerance_(fabs(tolerance))
{
   if(typeid(ACC) != typeid(opengm::Minimizer) && typeid(ACC) != typeid(opengm::Maximizer)) {
      throw RuntimeError("This implementation of the graph cut optimizer supports as accumulator only opengm::Minimizer and opengm::Maximizer.");
   }
   for(size_t j = 0; j < gm.numberOfVariables(); ++j) {
      if(gm.numberOfLabels(j) != 2) {
         throw RuntimeError("This implementation of the graph cut optimizer supports only binary variables.");
      }
   }
   for(size_t j = 0; j < gm.numberOfFactors(); ++j) {
      if(gm[j].numberOfVariables() > 3) {
         throw RuntimeError("This implementation of the graph cut optimizer supports only factors of order <= 3.");
      }
   }

   parameter_ = para;
   numVariables_ = gm.numberOfVariables();
   numFacDim_.resize(4, 0);
   for(size_t j = 0; j < gm.numberOfFactors(); ++j) {
      ++numFacDim_[gm[j].numberOfVariables()];
   }

   minStCut_ = new MinStCutType(2 + numVariables_ + numFacDim_[3], 2*numVariables_ + numFacDim_[2] + 3*numFacDim_[3]);
   sEdges_.assign(numVariables_ + numFacDim_[3], 0);
   tEdges_.assign(numVariables_ + numFacDim_[3], 0);

   for(size_t j = 0; j < gm.numberOfFactors(); ++j) {
      addFactor(gm[j]);
   }
   inferenceDone_=false;
   //std::cout << parameter_.scale_ <<std::endl;
}

template<class GM, class ACC, class MINSTCUT>
inline GraphCut<GM, ACC, MINSTCUT>::~GraphCut()
{
   delete minStCut_;
}

/// add a factor of the GraphicalModel to the min st-cut formulation of the solver MinStCutType
template<class GM, class ACC, class MINSTCUT>
template<class FACTOR>
inline void GraphCut<GM, ACC, MINSTCUT>::addFactor
(
   const FACTOR& factor
) {
   size_t numberOfVariables = factor.numberOfVariables();
   for(size_t i=0; i<numberOfVariables; ++i) {
      OPENGM_ASSERT(factor.numberOfLabels(i) == 2);
   }

   if(numberOfVariables == 0) {
      // do nothing
   }
   else if(numberOfVariables == 1) {
      const size_t var = factor.variableIndex(0);
      OPENGM_ASSERT(var < numVariables_);
      size_t i;
      i = 0; const ValueType v0 = factor(&i);
      i = 1; const ValueType v1 = factor(&i);
      if(typeid(ACC) == typeid(opengm::Minimizer)) {
         if(v0 <= v1) {
            addEdgeCapacity(0, var + 2, v1 - v0);
         }
         else {
            addEdgeCapacity(var + 2, 1, v0 - v1);
         }
      }
      else { //opengm::Maximizer
         if(v0 >= v1) {
            addEdgeCapacity(0, var + 2, -v1 + v0);
         }
         else {
            addEdgeCapacity(var + 2, 1, -v0 + v1);
         }
      }
   }
   else if(numberOfVariables == 2) {
      const size_t var0 = factor.variableIndex(0);
      const size_t var1 = factor.variableIndex(1);
      OPENGM_ASSERT(var0 < numVariables_);
      OPENGM_ASSERT(var1 < numVariables_);
      size_t i[] = {0, 0}; const ValueType A = factor(i);
      i[0] = 0; i[1] = 1;  const ValueType B = factor(i);
      i[0] = 1; i[1] = 0;  const ValueType C = factor(i);
      i[0] = 1; i[1] = 1;  const ValueType D = factor(i);
      if(typeid(ACC) == typeid(opengm::Minimizer)) {
         // first variabe
         if(C > A)
            addEdgeCapacity(0, var0 + 2, C - A);
         else if(C < A)
            addEdgeCapacity(var0 + 2, 1, A - C);
         // second variable
         if(D > C)
            addEdgeCapacity(0, var1 + 2, D - C);
         else if(D < C)
            addEdgeCapacity(var1 + 2, 1, C - D);
         // submodular term
         ValueType term = B + C - A - D;
         if((term < 0) && (term >= -tolerance_))
            term = 0.0;
         //if(term < 0.0) {
         //  throw RuntimeError("GraphCut<Factor>::addPairwisefactor(): non sub-modular factors cannot be processed.");
         //}
         addEdgeCapacity(var0 + 2, var1 + 2, term);
      }
      else{
         if(C < A)
            addEdgeCapacity(0, var0 + 2, -C + A);
         else if(C > A)
            addEdgeCapacity(var0 + 2, 1, -A + C);
         // second variable
         if(D < C)
            addEdgeCapacity(0, var1 + 2, -D + C);
         else if(D > C)
            addEdgeCapacity(var1 + 2, 1, -C + D);
         // submodular term
         ValueType term = B + C - A - D;
         if((term > 0) && (term <= tolerance_))
            term = 0.0;
         addEdgeCapacity(var0 + 2, var1 + 2, -term);
         //if(term > 0.0) {
         //  throw RuntimeError("GraphCut<Factor>::addPairwisefactor(): non sub-modular factors cannot be processed.");
         //}
      }
   }
   else if(numberOfVariables == 3) {
      const size_t var0 = factor.variableIndex(0);
      const size_t var1 = factor.variableIndex(1);
      const size_t var2 = factor.variableIndex(1);
      OPENGM_ASSERT(var0 < numVariables_);
      OPENGM_ASSERT(var1 < numVariables_);
      OPENGM_ASSERT(var2 < numVariables_);
      size_t i[] = {0, 0, 0};       const ValueType A = factor(i);
      i[0] = 0; i[1] = 0; i[2] = 1; const ValueType B = factor(i);
      i[0] = 0; i[1] = 1; i[2] = 0; const ValueType C = factor(i);
      i[0] = 0; i[1] = 1; i[2] = 1; const ValueType D = factor(i);
      i[0] = 1; i[1] = 0; i[2] = 0; const ValueType E = factor(i);
      i[0] = 1; i[1] = 0; i[2] = 1; const ValueType F = factor(i);
      i[0] = 1; i[1] = 1; i[2] = 0; const ValueType G = factor(i);
      i[0] = 1; i[1] = 1; i[2] = 1; const ValueType H = factor(i);

      if(typeid(ACC) == typeid(opengm::Minimizer)) {
         std::vector<size_t> triple(3);
         triple[0] = var0;
         triple[1] = var1;
         triple[2] = var2;
         size_t id = tripleId(triple);
         ValueType P = (A + D + F + G)-(B + C + E + H);
         if(P >= 0.0) {
            if(F-B>=0) addEdgeCapacity(0, var0+2, F - B);
            else       addEdgeCapacity(var0+2, 1, B - F);
            if(G-E>=0) addEdgeCapacity(0, var1+2, G - E);
            else       addEdgeCapacity(var1+2, 1, E - G);
            if(D-C>=0) addEdgeCapacity(0, var2+2, D - C);
            else       addEdgeCapacity(var2+2, 0, C - D);

            addEdgeCapacity(var1+2, var2+2, B + C - A - D);
            addEdgeCapacity(var2+2, var0+2, B + E - A - F);
            addEdgeCapacity(var0+2, var1+2, C + E - A - G);

            addEdgeCapacity(var0 + 2, id + 2, P);
            addEdgeCapacity(var1 + 2, id + 2, P);
            addEdgeCapacity(var2 + 2, id + 2, P);
            addEdgeCapacity(id, 1, P);
         }
         else {
            if(C-G>=0) addEdgeCapacity(var0+2, 1, C - G);
            else       addEdgeCapacity(0, var0+2, G - C);
            if(B-D>=0) addEdgeCapacity(var1+2, 1, B - D);
            else       addEdgeCapacity(0, var1+2, D - B);
            if(E-F>=0) addEdgeCapacity(var2+2, 1, E - F);
            else       addEdgeCapacity(0, var2+2, F - E);

            addEdgeCapacity(var2+2, var1+2, F + G - E - H);
            addEdgeCapacity(var0+2, var2+2, D + G - C - H);
            addEdgeCapacity(var1+2, var0+2, D + F - B - H);

            addEdgeCapacity(id + 2, var0 + 2, -P);
            addEdgeCapacity(id + 2, var1 + 2, -P);
            addEdgeCapacity(id + 2, var2 + 2, -P);
            addEdgeCapacity(0, id + 2, -P);
         };
      }
      else{
         throw RuntimeError("This implementation of the graph cut optimizer support 3rd order factors only in connection with opengm::Maximizer.");
      }
   }
   else {
      throw RuntimeError("This implementation of the graph cut optimizer does not support factors of order > 3.");
   }
}

template<class GM, class ACC, class MINSTCUT>
inline void 
GraphCut<GM, ACC, MINSTCUT>::addEdgeCapacity
(
   const size_t v, 
   const size_t w, 
   const ValueType val
) {
   typedef typename MINSTCUT::ValueType VType;
   typedef typename MINSTCUT::node_type NType;
   const NType n1   = static_cast<NType>(v);
   const NType n2   = static_cast<NType>(w);
   const VType cost = static_cast<VType>(parameter_.scale_*val);
   if(n1 == 0) {
      sEdges_[n2-2] += cost;
   }
   else if(n2 == 1) {
      tEdges_[n1-2] += cost;
   }
   else {
      minStCut_->addEdge(n1, n2, cost);
   }
}

template<class GM, class ACC, class MINSTCUT>
inline size_t 
GraphCut<GM, ACC, MINSTCUT>::tripleId
(
   std::vector<size_t>& triple
) {
   // search for triple in list
   std::list<std::vector<size_t> >::iterator it;
   size_t counter = numVariables_;
   for(it = tripleList.begin(); it != tripleList.end(); it++) {
      if(triple[0] == (*it)[0] && triple[1] == (*it)[1] && triple[2] == (*it)[2]) {
         return counter;
      }
      numVariables_++;
   }
   // add triple to list
   tripleList.push_back(triple);
   OPENGM_ASSERT(counter - numVariables_ < numFacDim_[3]);
   return counter;
}
   
template<class GM, class ACC, class MINSTCUT>
inline InferenceTermination 
GraphCut<GM, ACC, MINSTCUT>::infer() { 
   EmptyVisitorType v;
   return infer(v);
}
   
template<class GM, class ACC, class MINSTCUT>
template<class VISITOR>
inline InferenceTermination 
GraphCut<GM, ACC, MINSTCUT>::infer(VISITOR & visitor) { 
   visitor.begin(*this);
   for(size_t i=0; i<sEdges_.size(); ++i) {
      minStCut_->addEdge(0, i+2, sEdges_[i]);
      minStCut_->addEdge(i+2, 1, tEdges_[i]);
   }
   minStCut_->calculateCut(state_);
   inferenceDone_=true;
   visitor.end(*this);
   return NORMAL;
}

template<class GM, class ACC, class MINSTCUT>
inline InferenceTermination GraphCut<GM, ACC, MINSTCUT>::arg
(
   std::vector<LabelType>& arg, 
   const size_t n
) const {
   if(inferenceDone_==false){
      arg.resize(numVariables_,0);
      return UNKNOWN;
   }
   if(n > 1) {
      return UNKNOWN;
   } 
   else {
      // skip source and sink
      if(state_.size() > 2 + numFacDim_[3]) {
         arg.resize(state_.size() - 2 - numFacDim_[3]);
      }
      else {
         arg.resize(0);
      }

      for(size_t j = 0; j < arg.size(); ++j) {
         arg[j] = static_cast<LabelType>(state_[j + 2]);
      }
      return NORMAL;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_GRAPHCUT_HXX

