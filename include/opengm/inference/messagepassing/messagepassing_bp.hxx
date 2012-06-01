#pragma once
#ifndef OPENGM_BELIEFPROPAGATION_HXX
#define OPENGM_BELIEFPROPAGATION_HXX

#include <vector>
#include <map>
#include <list>
#include <set>

#include "opengm/opengm.hxx"
#include "opengm/utilities/tribool.hxx"
#include "opengm/operations/weightedoperations.hxx"
#include "opengm/operations/normalize.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/inference/messagepassing/messagepassing_operations.hxx"
#include "opengm/inference/messagepassing/messagepassing_buffer.hxx"

namespace opengm {

   /// \cond HIDDEN_SYMBOLD

   template<class GM, class BUFFER, class OP, class ACC>
   class VariableHullBP {
   public:
      typedef GM                                  GraphicalModelType;
      typedef BUFFER                              BufferType;
      typedef typename BUFFER::ArrayType          BufferArrayType;
      typedef typename GM::FactorType             FactorType;
      typedef typename GM::IndependentFactorType  IndependentFactorType;
      typedef typename GM::ValueType              ValueType;

      VariableHullBP();
      void assign(const GM&, const size_t, const meta::EmptyType*);
      BUFFER& connectFactorHullBP(const size_t, BUFFER&);
      size_t numberOfBuffers() const;
      void propagateAll(const GM&, const ValueType& = 0, const bool = false);
      void propagate(const GM&, const size_t, const ValueType& = 0, const bool = false);
      void marginal(const GM&, const size_t, IndependentFactorType&, const bool = true) const; 
      //typename GM::ValueType bound() const;
      template<class DIST> ValueType distance(const size_t) const;
      const typename BUFFER::ArrayType& outBuffer(const size_t) const;

   private:
      std::vector<BUFFER* > outBuffer_;
      std::vector<BUFFER > inBuffer_;
   };

   template<class GM, class BUFFER, class OP, class ACC>
   class FactorHullBP {
   public:
      typedef GM                                  GraphicalModelType;
      typedef BUFFER                              BufferType;
      typedef typename BUFFER::ArrayType          BufferArrayType;
      typedef typename GM::FactorType             FactorType;
      typedef typename GM::IndependentFactorType  IndependentFactorType;
      typedef typename GM::ValueType              ValueType;
  
      size_t numberOfBuffers() const        { return inBuffer_.size(); }
      void assign(const GM&, const size_t, std::vector<VariableHullBP<GM,BUFFER,OP,ACC> >&, const meta::EmptyType*);
      void propagateAll(const ValueType& = 0, const bool = true);
      void propagate(const size_t, const ValueType& = 0, const bool = true);
      void marginal(IndependentFactorType&, const bool = true) const;
      //typename GM::ValueType bound() const;
      template<class DIST> ValueType distance(const size_t) const;

   private:
      FactorType const* myFactor_;
      std::vector<BUFFER* > outBuffer_;
      std::vector<BUFFER > inBuffer_;
   };

   /// \endcond

   /// \brief Update rules for the MessagePassing framework
   template<class GM, class ACC, class BUFFER = MessageBuffer<marray::Marray<double> > >
   class BeliefPropagationUpdateRules {
   public:
      typedef GM                                  GraphicalModelType;
      typedef BUFFER                              BufferType;
      typedef typename BUFFER::ArrayType          BufferArrayType; 
      typedef typename GM::ValueType              ValueType; 
      typedef typename GM::IndependentFactorType  IndependentFactorType;
      typedef typename GM::FactorType             FactorType;
      typedef typename GM::OperatorType           OperatorType;
      /// \cond HIDDEN_SYMBOLS
      typedef FactorHullBP<GM, BufferType, OperatorType, ACC> FactorHullType;
      typedef VariableHullBP<GM, BufferType, OperatorType, ACC> VariableHullType;
      typedef meta::EmptyType SpecialParameterType;

      template<class MP_PARAM>
         static void initializeSpecialParameter(const GM& gm, MP_PARAM& mpParameter)
            {}
      /// \endcond
   };

   template<class GM, class BUFFER, class OP, class ACC>
   inline  VariableHullBP<GM, BUFFER, OP, ACC>::VariableHullBP()
   {}

   template<class GM, class BUFFER, class OP, class ACC>
   inline void VariableHullBP<GM, BUFFER, OP, ACC>::assign
   (
      const GM& gm,
      const size_t variableIndex,
      const meta::EmptyType* et
   ) {
      size_t numberOfFactors = gm.numberOfFactors(variableIndex);
      inBuffer_.resize(numberOfFactors);
      outBuffer_.resize(numberOfFactors);
      for(size_t j = 0; j < numberOfFactors; ++j) {
         inBuffer_[j].assign(gm.numberOfLabels(variableIndex), OP::template neutral<ValueType>());
      }
   }

   template<class GM, class BUFFER, class OP, class ACC>
   inline size_t VariableHullBP<GM, BUFFER, OP, ACC>::numberOfBuffers() const {
      return outBuffer_.size();
   }

   template<class GM, class BUFFER, class OP, class ACC>
   inline BUFFER& VariableHullBP<GM, BUFFER, OP, ACC>::connectFactorHullBP
   (
      const size_t bufferNumber,
      BUFFER& variableOutBuffer
   ) {
      OPENGM_ASSERT(bufferNumber < numberOfBuffers());
      outBuffer_[bufferNumber] = &variableOutBuffer;
      return inBuffer_[bufferNumber];
   }

   template<class GM, class BUFFER, class OP, class ACC >
   inline void VariableHullBP<GM, BUFFER, OP, ACC>::propagate
   (
      const GM& gm,
      const size_t bufferNumber,
      const ValueType& damping,
      const bool useNormalization
   ) {
      OPENGM_ASSERT(bufferNumber < numberOfBuffers());
      outBuffer_[bufferNumber]->toggle();
      if(inBuffer_.size() < 2) {
         return; // nothing to send
      }
      // initialize neutral message
      BufferArrayType& newMessage = outBuffer_[bufferNumber]->current();
      opengm::messagepassingOperations::operate<OP>(inBuffer_, bufferNumber, newMessage);

      // damp message
      if(damping != 0) {
         BufferArrayType& oldMessage = outBuffer_[bufferNumber]->old();
         opengm::messagepassingOperations::weightedMean<OP>(newMessage, oldMessage, damping, newMessage);
      }
      // normalize message
      if(useNormalization) {
         opengm::messagepassingOperations::normalize<OP,ACC>(newMessage);
      }
   }

  
   template<class GM, class BUFFER, class OP, class ACC>
   inline void VariableHullBP<GM, BUFFER, OP, ACC>::propagateAll
   (
      const GM& gm,
      const ValueType& damping,
      const bool useNormalization
   ) {
      for(size_t bufferNumber = 0; bufferNumber < numberOfBuffers(); ++bufferNumber) {
         propagate(gm, bufferNumber, damping, useNormalization);
      }
   }

   template<class GM, class BUFFER, class OP, class ACC >
   inline void VariableHullBP<GM, BUFFER, OP, ACC>::marginal
   (
      const GM& gm,
      const size_t variableIndex,
      IndependentFactorType& out,
      const bool useNormalization
   ) const {

      // set out to neutral
      out.assign(gm, &variableIndex, &variableIndex+1, OP::template neutral<ValueType>());
      opengm::messagepassingOperations::operate<OP>(inBuffer_, out);

      // normalize output
      if(useNormalization) {
         opengm::messagepassingOperations::normalize<OP,ACC>(out);
      }
   }
/*
   template<class GM, class BUFFER, class OP, class ACC>
   inline typename GM::ValueType VariableHullBP<GM, BUFFER, OP, ACC>::bound() const
   { 
      ValueType v;
      //OP::neutral(v);

      BufferArrayType a(inBuffer_[0].current().shapeBegin(),inBuffer_[0].current().shapeEnd());
      opengm::messagepassingOperations::operate<OP>(inBuffer_, a);

      if(typeid(ACC)==typeid(opengm::Minimizer) || typeid(ACC)==typeid(opengm::Maximizer)) {
         v = a(0);
         for(size_t n=1; n<a.size(); ++n) {
            ACC::op(a(n),v);
         }
      }
      else{
         ACC::ineutral(v);
      }  
      //ACC::ineutral(v);

      //v = opengm::messagepassingOperations::template boundOperation<ValueType,OP,ACC>(a,a); 
//      ACC::ineutral(v);
      return v;
   } 
*/

   template<class GM, class BUFFER, class OP, class ACC >
   template<class DIST>
   inline typename GM::ValueType VariableHullBP<GM, BUFFER, OP, ACC>::distance
   (
      const size_t bufferNumber
   ) const {
      return inBuffer_[bufferNumber].template dist<DIST > ();
   }

   template<class GM, class BUFFER, class OP, class ACC >
   inline const typename BUFFER::ArrayType& VariableHullBP<GM, BUFFER, OP, ACC>::outBuffer
   (
      const size_t bufferIndex
   ) const {
      OPENGM_ASSERT(bufferIndex < outBuffer_.size());
      return outBuffer_[bufferIndex]->current();
   }

   template<class GM, class BUFFER, class OP, class ACC>
   inline void FactorHullBP<GM, BUFFER, OP, ACC>::assign
   (
      const GM& gm,
      const size_t factorIndex,
      std::vector<VariableHullBP<GM, BUFFER, OP, ACC> >& variableHulls,
      const meta::EmptyType* et
   ) {
      myFactor_ = (FactorType *const)(&gm[factorIndex]);
      inBuffer_.resize(gm[factorIndex].numberOfVariables());
      outBuffer_.resize(gm[factorIndex].numberOfVariables());
      for(size_t n=0; n<gm.numberOfVariables(factorIndex); ++n) {
         size_t variableIndex = gm.variableOfFactor(factorIndex,n);
         inBuffer_[n].assign(gm.numberOfLabels(variableIndex), OP::template neutral<ValueType > ());
         size_t bufferNumber = 1000000;
         for(size_t i=0; i<gm.numberOfFactors(variableIndex); ++i) {
            if(gm.factorOfVariable(variableIndex,i) == factorIndex) {
               bufferNumber=i;
               break;
            }
         } 
         outBuffer_[n] =&(variableHulls[variableIndex].connectFactorHullBP(bufferNumber, inBuffer_[n]));
      }
   }

   template<class GM, class BUFFER, class OP, class ACC >
   inline void FactorHullBP<GM, BUFFER, OP, ACC>::propagate
   (
      const size_t id,
      const ValueType& damping,
      const bool useNormalization
   ) {
      OPENGM_ASSERT(id < outBuffer_.size());
      outBuffer_[id]->toggle();
      BufferArrayType& newMessage = outBuffer_[id]->current();
      opengm::messagepassingOperations::operateF<GM,ACC>(*myFactor_, inBuffer_, id, newMessage);

      // damp message
      if(damping != 0) {
         BufferArrayType& oldMessage = outBuffer_[id]->old();
         opengm::messagepassingOperations::weightedMean<OP>(newMessage, oldMessage, damping, newMessage);
      }
      // normalize message
      if(useNormalization) {
         opengm::messagepassingOperations::normalize<OP,ACC>(newMessage);
      }
   }

   template<class GM, class BUFFER, class OP, class ACC >
   inline void FactorHullBP<GM, BUFFER, OP, ACC>::propagateAll
   (
      const ValueType& damping,
      const bool useNormalization
   ) {
      for(size_t j = 0; j < inBuffer_.size(); ++j) {
         propagate(j, damping, useNormalization);
      }
   }

   template<class GM, class BUFFER, class OP, class ACC>
   inline void FactorHullBP<GM, BUFFER, OP, ACC>::marginal
   (
      IndependentFactorType& out,
      const bool useNormalization
   ) const 
   {  
      opengm::messagepassingOperations::operateF<GM>(*myFactor_, inBuffer_,out);

      if(useNormalization) {
         opengm::messagepassingOperations::normalize<OP,ACC>(out);
      }
   }
/*
   template<class GM, class BUFFER, class OP, class ACC>
   inline typename GM::ValueType FactorHullBP<GM, BUFFER, OP, ACC>::bound
   () const
   {
      //typename GM::IndependentFactorType a = myFactor_; 
      typename GM::IndependentFactorType a = *myFactor_;
      //opengm::messagepassingOperations::operateF<GM>(*myFactor_, inBuffer_,a);
      opengm::messagepassingOperations::operateFi<GM>(*myFactor_, outBuffer_, a);
      //return opengm::messagepassingOperations::boundOperation<ValueType,OP,ACC>(a,b);
      ValueType v;
      if(typeid(ACC)==typeid(opengm::Minimizer) || typeid(ACC)==typeid(opengm::Maximizer)) {
         v = a(0);
         for(size_t n=1; n<a.size(); ++n) {
            ACC::op(a(n),v);
         }
      }
      else{
         ACC::ineutral(v);
      }  
      return v;
   }
*/
   template<class GM, class BUFFER, class OP, class ACC >
   template<class DIST>
   inline typename GM::ValueType FactorHullBP<GM, BUFFER, OP, ACC>::distance
   (
      const size_t j
   ) const {
      return inBuffer_[j].template dist<DIST > ();
   }

} // namespace opengm

#endif // #ifndef OPENGM_BELIEFPROPAGATION_HXX

