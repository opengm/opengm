#pragma once
#ifndef OPENGM_TREEREWEIGTHEDBELIEFPROPAGATION_HXX
#define OPENGM_TREEREWEIGTHEDBELIEFPROPAGATION_HXX

#include <vector>
#include <map>
#include <list>
#include <set>

#include "opengm/opengm.hxx"
#include "opengm/inference/messagepassing/messagepassing_buffer.hxx"
#include "opengm/graphicalmodel/decomposition/graphicalmodeldecomposer.hxx"
#include "opengm/utilities/tribool.hxx"
#include "opengm/operations/weightedoperations.hxx"
#include "opengm/operations/normalize.hxx"
#include "opengm/inference/messagepassing/messagepassing_operations.hxx"

namespace opengm {
   /// \cond HIDDEN_SYMBOLS
   template<class GM, class BUFFER, class OP, class ACC>
   class VariableHullTRBP {
   public:
      typedef GM                                    GraphicalModelType;
      typedef BUFFER                                BufferType;
      typedef typename BUFFER::ArrayType            BufferArrayType;
      typedef typename GM::ValueType                ValueType;
      typedef typename GM::IndependentFactorType    IndependentFactorType;

      VariableHullTRBP();
      void assign(const GM&, const size_t, const std::vector<ValueType>*);
      BUFFER& connectFactorHullTRBP(const size_t, BUFFER&);
      size_t numberOfBuffers() const;
      void propagateAll(const GM&, const ValueType& = 0, const bool = false);
      void propagate(const GM&, const size_t, const ValueType& = 0, const bool = false);
      void marginal(const GM&, const size_t, IndependentFactorType&, const bool = true) const;
      //typename GM::ValueType bound() const;
      template<class DIST> ValueType distance(const size_t) const;

   private:
      std::vector<BUFFER* >  outBuffer_;
      std::vector<BUFFER >   inBuffer_;
      std::vector<ValueType> rho_;
   };

   // Wrapper class for factor nodes
   template<class GM, class BUFFER, class OP, class ACC>
   class FactorHullTRBP {
   public:
      typedef GM                                 GraphicalModelType;
      typedef BUFFER                             BufferType;
      typedef typename BUFFER::ArrayType         BufferArrayType;
      typedef typename GM::FactorType            FactorType;
      typedef typename GM::IndependentFactorType IndependentFactorType;
      typedef typename GM::ValueType             ValueType;

      FactorHullTRBP();
      size_t numberOfBuffers() const       { return inBuffer_.size(); }
      //size_t variableIndex(size_t i) const { return variableIndices_[i]; }
      void assign(const GM&, const size_t, std::vector<VariableHullTRBP<GM,BUFFER,OP,ACC> >&, const std::vector<ValueType>*);
      void propagateAll(const ValueType& = 0, const bool = true);
      void propagate(const size_t, const ValueType& = 0, const bool = true);
      void marginal(IndependentFactorType&, const bool = true) const; 
      //typename GM::ValueType bound() const;
      template<class DIST> ValueType distance(const size_t) const;

   private:
      FactorType*           myFactor_;
      std::vector<BUFFER* > outBuffer_;
      std::vector<BUFFER >  inBuffer_;
      ValueType             rho_;
   };
   /// \endcond

   /// \brief Update rules for the MessagePassing framework
   template<class GM, class ACC, class BUFFER = opengm::MessageBuffer<marray::Marray<double> > >
   class TrbpUpdateRules {
   public:
      typedef typename GM::ValueType ValueType;
      typedef typename GM::IndependentFactorType IndependentFactorType;
      typedef typename GM::FactorType FactorType;
      typedef typename GM::OperatorType OperatorType;
      typedef FactorHullTRBP<GM, BUFFER, OperatorType, ACC> FactorHullType;
      typedef VariableHullTRBP<GM, BUFFER, OperatorType, ACC> VariableHullType;
      typedef std::vector<ValueType> SpecialParameterType;

      template<class MP_PARAM>
      static void initializeSpecialParameter(const GM& gm,MP_PARAM& mpParameter) {
         // set rho if not set manually
         if (mpParameter.specialParameter_.size() == 0) {
            // set rho by tree decomposition
            opengm::GraphicalModelDecomposer<GM> decomposer;
            const opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoSpanningTrees(gm);
            OPENGM_ASSERT(decomposition.isValid(gm));
            typedef typename GraphicalModelDecomposition::SubFactorListType SubFactorListType;
            const std::vector<SubFactorListType>& subFactorList = decomposition.getFactorLists();
            mpParameter.specialParameter_.resize(gm.numberOfFactors());
            for (size_t factorId = 0; factorId < gm.numberOfFactors(); ++factorId) {
               mpParameter.specialParameter_[factorId] = 1.0 / subFactorList[factorId].size();
            }
         }
         else if (mpParameter.specialParameter_.size() != gm.numberOfFactors()) {
            throw RuntimeError("The parameter rho has been set incorrectly.");
         }
         if(!NO_DEBUG) {
            // test rho
            OPENGM_ASSERT(mpParameter.specialParameter_.size() == gm.numberOfFactors());
            for (size_t i = 0; i < gm.numberOfFactors(); ++i) {
               if(gm.numberOfVariables() < 2) { /// ???
                  OPENGM_ASSERT(mpParameter.specialParameter_[i] == 1); // ??? allow for numerical deviation
               }
               OPENGM_ASSERT(mpParameter.specialParameter_[i] > 0);
            }
         }
      }
   };

   template<class GM, class BUFFER, class OP, class ACC>
   inline VariableHullTRBP<GM, BUFFER, OP, ACC>::VariableHullTRBP()
   {}

   template<class GM, class BUFFER, class OP, class ACC>
   inline void VariableHullTRBP<GM, BUFFER, OP, ACC>::assign
   (
      const GM& gm,
      const size_t variableIndex,
      const std::vector<ValueType>* rho
   ) {
      size_t numberOfFactors = gm.numberOfFactors(variableIndex);
      rho_.resize(numberOfFactors);
      for(size_t j = 0; j < numberOfFactors; ++j) {
         rho_[j] = (*rho)[gm.factorOfVariable(variableIndex,j)];
      }
      inBuffer_.resize(numberOfFactors);
      outBuffer_.resize(numberOfFactors);
      // allocate input-buffer
      for(size_t j = 0; j < numberOfFactors; ++j) {
         inBuffer_[j].assign(gm.numberOfLabels(variableIndex), OP::template neutral<ValueType > ());
      }
   }


   template<class GM, class BUFFER, class OP, class ACC>
   inline size_t VariableHullTRBP<GM, BUFFER, OP, ACC>::numberOfBuffers() const {
      return inBuffer_.size();
   }

   template<class GM, class BUFFER, class OP, class ACC>
   inline BUFFER& VariableHullTRBP<GM, BUFFER, OP, ACC>::connectFactorHullTRBP(
      const size_t bufferNumber,
      BUFFER& variableOutBuffer
   ) {
      outBuffer_[bufferNumber] =&variableOutBuffer;
      return inBuffer_[bufferNumber];
   }

   template<class GM, class BUFFER, class OP, class ACC>
   void VariableHullTRBP<GM, BUFFER, OP, ACC>::propagate(
      const GM& gm,
      const size_t id,
      const ValueType& damping,
      const bool useNormalization
   ) {
      OPENGM_ASSERT(id < outBuffer_.size());
      outBuffer_[id]->toggle();
      if(numberOfBuffers() < 2) {
         return; // nothing to send
      }
      // initialize neutral message
      BufferArrayType& newMessage = outBuffer_[id]->current();
      opengm::messagepassingOperations::operateW<GM>(inBuffer_, id, rho_, newMessage); 

      // damp message 
      if(damping != 0) {
         BufferArrayType& oldMessage = outBuffer_[id]->old();
         if(useNormalization) {
            opengm::messagepassingOperations::normalize<OP,ACC>(newMessage); 
            opengm::messagepassingOperations::normalize<OP,ACC>(oldMessage);
         }
         opengm::messagepassingOperations::weightedMean<OP>(newMessage, oldMessage, damping, newMessage);
      }
      if(useNormalization) {
         opengm::messagepassingOperations::normalize<OP,ACC>(newMessage);
      }
   }



   template<class GM, class BUFFER, class OP, class ACC>
   inline void VariableHullTRBP<GM, BUFFER, OP, ACC>::propagateAll
   (
      const GM& gm,
      const ValueType& damping,
      const bool useNormalization
   ) {
      for(size_t j = 0; j < numberOfBuffers(); ++j) {
         propagate(gm, j, damping, useNormalization);
      }
   }

   template<class GM, class BUFFER, class OP, class ACC>
   inline void VariableHullTRBP<GM, BUFFER, OP, ACC>::marginal
   (
      const GM& gm,
      const size_t variableIndex,
      IndependentFactorType& out,
      const bool useNormalization
   ) const {
      // set out to neutral
      out.assign(gm, &variableIndex, &variableIndex+1, OP::template neutral<ValueType>());
      opengm::messagepassingOperations::operateW<GM>(inBuffer_, rho_, out);

      // Normalization::normalize output
      if(useNormalization) { 
         opengm::messagepassingOperations::normalize<OP,ACC>(out);
      }
   }
/*
   template<class GM, class BUFFER, class OP, class ACC>
   inline typename GM::ValueType VariableHullTRBP<GM, BUFFER, OP, ACC>::bound
   ()const
   {
     
      typename BUFFER::ArrayType a(inBuffer_[0].current().shapeBegin(),inBuffer_[0].current().shapeEnd());
      opengm::messagepassingOperations::operateW<GM>(inBuffer_, rho_, a);
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
    
      //return  opengm::messagepassingOperations::template boundOperation<ValueType,OP,ACC>(a,a); 
   } 
*/ 
   template<class GM, class BUFFER, class OP, class ACC>
   template<class DIST>
   inline typename GM::ValueType VariableHullTRBP<GM, BUFFER, OP, ACC>::distance
   (
      const size_t j
   ) const {
      return inBuffer_[j].template dist<DIST > ();
   }

   template<class GM, class BUFFER, class OP, class ACC>
   inline FactorHullTRBP<GM, BUFFER, OP, ACC>::FactorHullTRBP()
   {}

   template<class GM, class BUFFER, class OP, class ACC>
   inline void FactorHullTRBP<GM, BUFFER, OP, ACC>::assign
   (
      const GM& gm,
      const size_t factorIndex,
      std::vector<VariableHullTRBP<GM, BUFFER, OP, ACC> >& variableHulls,
      const std::vector<ValueType>* rho
   ) {
      rho_ = (*rho)[factorIndex];
      myFactor_ = (FactorType*) (&gm[factorIndex]);
      inBuffer_.resize(gm[factorIndex].numberOfVariables());
      outBuffer_.resize(gm[factorIndex].numberOfVariables());

      for(size_t n=0; n<gm.numberOfVariables(factorIndex); ++n) {
         size_t var = gm.variableOfFactor(factorIndex,n);
         inBuffer_[n].assign(gm.numberOfLabels(var), OP::template neutral<ValueType > ());
         size_t bufferNumber = 1000000;
         for(size_t i=0; i<gm.numberOfFactors(var); ++i) {
            if(gm.factorOfVariable(var,i)==factorIndex)
               bufferNumber=i;
         }
         OPENGM_ASSERT(bufferNumber!=1000000)
         outBuffer_[n] =&(variableHulls[var].connectFactorHullTRBP(bufferNumber, inBuffer_[n]));
      }
   }

   template<class GM, class BUFFER, class OP, class ACC>
   void FactorHullTRBP<GM, BUFFER, OP, ACC>::propagate
   (
      const size_t id,
      const ValueType& damping,
      const bool useNormalization
   ) {
      OPENGM_ASSERT(id < outBuffer_.size());
      outBuffer_[id]->toggle();
      BufferArrayType& newMessage = outBuffer_[id]->current();
      opengm::messagepassingOperations::operateWF<GM,ACC>(*myFactor_, rho_, inBuffer_, id, newMessage);

      // damp message
      if(damping != 0) { 
         BufferArrayType& oldMessage = outBuffer_[id]->old(); 
         if(useNormalization) {
            opengm::messagepassingOperations::normalize<OP,ACC>(newMessage);
            opengm::messagepassingOperations::normalize<OP,ACC>(oldMessage);
         }
         opengm::messagepassingOperations::weightedMean<OP>(newMessage, oldMessage, damping, newMessage);
      }
      // Normalization::normalize message
      if(useNormalization) {
         opengm::messagepassingOperations::normalize<OP,ACC>(newMessage);
      }
   }

 
   template<class GM, class BUFFER, class OP, class ACC>
   inline void FactorHullTRBP<GM, BUFFER, OP, ACC>::propagateAll
   (
      const ValueType& damping,
      const bool useNormalization
   ) {
      for(size_t j = 0; j < numberOfBuffers(); ++j) {
         propagate(j, damping, useNormalization);
      }
   }

   template<class GM, class BUFFER, class OP, class ACC>
   inline void FactorHullTRBP<GM, BUFFER, OP, ACC>::marginal
   (
      IndependentFactorType& out,
      const bool useNormalization
   ) const 
   {
      opengm::messagepassingOperations::operateWF<GM>(*(const_cast<FactorType*> (myFactor_)), rho_, inBuffer_, out);

      if(useNormalization) {
         opengm::messagepassingOperations::normalize<OP,ACC>(out);
      }
   }
/*
   template<class GM, class BUFFER, class OP, class ACC>
   inline typename GM::ValueType FactorHullTRBP<GM, BUFFER, OP, ACC>::bound
   () const
   {

      //typename GM::IndependentFactorType a = *myFactor_; 
      typename GM::IndependentFactorType a = *myFactor_;
      //opengm::messagepassingOperations::operateWF<GM>(*(const_cast<FactorType*> (myFactor_)), rho_, inBuffer_, a);
      opengm::messagepassingOperations::operateFiW<GM>(*myFactor_,outBuffer_, rho_, a);

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

      //return opengm::messagepassingOperations::template boundOperation<ValueType,OP,ACC>(a,b);

   }
*/
   template<class GM, class BUFFER, class OP, class ACC>
   template<class DIST>
   inline typename GM::ValueType FactorHullTRBP<GM, BUFFER, OP, ACC>::distance
   (
      const size_t j
   ) const {
      return inBuffer_[j].template dist<DIST > ();
   }

} // namespace opengm

#endif // #ifndef OPENGM_BELIEFPROPAGATION_HXX

