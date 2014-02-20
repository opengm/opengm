#pragma once
#ifndef OPENGM_MESSAGE_PASSING_HXX
#define OPENGM_MESSAGE_PASSING_HXX

#include <vector>
#include <map>
#include <list>
#include <set>

#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/messagepassing/messagepassing_trbp.hxx"
#include "opengm/inference/messagepassing/messagepassing_bp.hxx"
#include "opengm/utilities/tribool.hxx"
#include "opengm/utilities/metaprogramming.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// MaxDistance
/// \ingroup distances
struct MaxDistance {
      /// operation
      /// \param in1 factor 1
      /// \param in2 factor 1
      template<class M>
      static typename M::ValueType
      op(const M& in1, const M& in2)
      {
         typedef typename M::ValueType ValueType;
         ValueType v1,v2,d1,d2;
         Maximizer::neutral(v1); 
         Maximizer::neutral(v2);
         for(size_t n=0; n<in1.size(); ++n) {
            d1=in1(n)-in2(n);
            d2=-d1;
            Maximizer::op(d1,v1);
            Maximizer::op(d2,v2);
         }
         Maximizer::op(v2,v1);
         return v1;
   }
}; 

/// \brief A framework for message passing algorithms\n\n
/// Cf. F. R. Kschischang, B. J. Frey and H.-A. Loeliger, "Factor Graphs and the Sum-Product Algorithm", IEEE Transactions on Information Theory 47:498-519, 2001
template<class GM, class ACC, class UPDATE_RULES, class DIST=opengm::MaxDistance>
class MessagePassing : public Inference<GM, ACC> {
public:
   typedef GM GraphicalModelType;
   typedef ACC Accumulation;
   typedef ACC AccumulatorType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef DIST Distance;
   typedef typename UPDATE_RULES::FactorHullType FactorHullType;
   typedef typename UPDATE_RULES::VariableHullType VariableHullType;

   /// Visitor
   typedef visitors::VerboseVisitor<MessagePassing<GM, ACC, UPDATE_RULES, DIST> > VerboseVisitorType;
   /// Visitor
   typedef visitors::TimingVisitor<MessagePassing<GM, ACC, UPDATE_RULES, DIST> > TimingVisitorType;
   /// Visitor
   typedef visitors::EmptyVisitor<MessagePassing<GM, ACC, UPDATE_RULES, DIST> > EmptyVisitorType;

   struct Parameter {
      typedef typename  UPDATE_RULES::SpecialParameterType SpecialParameterType;
      Parameter
      (
         const size_t maximumNumberOfSteps = 100,
         const ValueType bound = static_cast<ValueType> (0.000000),
         const ValueType damping = static_cast<ValueType> (0),
         const SpecialParameterType & specialParameter =SpecialParameterType(),
         const opengm::Tribool isAcyclic = opengm::Tribool::Maybe
      )
      :  maximumNumberOfSteps_(maximumNumberOfSteps),
         bound_(bound),
         damping_(damping),
         inferSequential_(false),
         useNormalization_(true),
         specialParameter_(specialParameter),
         isAcyclic_(isAcyclic)
      {}

      size_t maximumNumberOfSteps_;
      ValueType bound_;
      ValueType damping_;
      bool inferSequential_;
      std::vector<size_t> sortedNodeList_;
      bool useNormalization_;
      SpecialParameterType specialParameter_;
      opengm::Tribool isAcyclic_;
   };

   /// \cond HIDDEN_SYMBOLS
   struct Message {
      Message()
         :  nodeId_(-1),
            internalMessageId_(-1)
         {}
      Message(const size_t nodeId, const size_t & internalMessageId)
         :  nodeId_(nodeId),
            internalMessageId_(internalMessageId)
         {}

      size_t nodeId_;
      size_t internalMessageId_;
   };
   /// \endcond

   MessagePassing(const GraphicalModelType&, const Parameter& = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination marginal(const size_t, IndependentFactorType& out) const;
   InferenceTermination factorMarginal(const size_t, IndependentFactorType & out) const;
   ValueType convergenceXF() const;
   ValueType convergenceFX() const;
   ValueType convergence() const;
   virtual void reset();
   InferenceTermination infer();
   template<class VisitorType>
      InferenceTermination infer(VisitorType&);
   void propagate(const ValueType& = 0);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   //InferenceTermination bound(ValueType&) const;
   //ValueType bound() const;
 
private:
   void inferAcyclic();
   void inferParallel();
   void inferSequential();
   template<class VisitorType>
      void inferParallel(VisitorType&);
   template<class VisitorType>
      void inferAcyclic(VisitorType&);
   template<class VisitorType>
      void inferSequential(VisitorType&);
private:
   const GraphicalModelType& gm_;
   Parameter parameter_;
   std::vector<FactorHullType> factorHulls_;
   std::vector<VariableHullType> variableHulls_;
};
  
template<class GM, class ACC, class UPDATE_RULES, class DIST>
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::MessagePassing
(
   const GraphicalModelType& gm,
   const typename  MessagePassing<GM, ACC, UPDATE_RULES, DIST>::Parameter& parameter
)
:  gm_(gm),
   parameter_(parameter)
{
   if(parameter_.sortedNodeList_.size() == 0) {
      parameter_.sortedNodeList_.resize(gm.numberOfVariables());
      for (size_t i = 0; i < gm.numberOfVariables(); ++i)
         parameter_.sortedNodeList_[i] = i;
   }
   OPENGM_ASSERT(parameter_.sortedNodeList_.size() == gm.numberOfVariables());

   UPDATE_RULES::initializeSpecialParameter(gm_,this->parameter_);
  
   // set hulls
   variableHulls_.resize(gm.numberOfVariables(), VariableHullType ());
   for (size_t i = 0; i < gm.numberOfVariables(); ++i) {
      variableHulls_[i].assign(gm, i, &parameter_.specialParameter_);
   } 
   factorHulls_.resize(gm.numberOfFactors(), FactorHullType ());
   for (size_t i = 0; i < gm.numberOfFactors(); i++) {
      factorHulls_[i].assign(gm, i, variableHulls_, &parameter_.specialParameter_);
   } 
}

template<class GM, class ACC, class UPDATE_RULES, class DIST>
void
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::reset()
{
   if(parameter_.sortedNodeList_.size() == 0) {
      parameter_.sortedNodeList_.resize(gm_.numberOfVariables());
      for (size_t i = 0; i < gm_.numberOfVariables(); ++i)
         parameter_.sortedNodeList_[i] = i;
   }
   OPENGM_ASSERT(parameter_.sortedNodeList_.size() == gm_.numberOfVariables());
   UPDATE_RULES::initializeSpecialParameter(gm_,this->parameter_);

   // set hulls
   variableHulls_.resize(gm_.numberOfVariables(), VariableHullType ());
   for (size_t i = 0; i < gm_.numberOfVariables(); ++i) {
      variableHulls_[i].assign(gm_, i, &parameter_.specialParameter_);
   }
   factorHulls_.resize(gm_.numberOfFactors(), FactorHullType ());
   for (size_t i = 0; i < gm_.numberOfFactors(); i++) {
      factorHulls_[i].assign(gm_, i, variableHulls_, &parameter_.specialParameter_);
   }
}

template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline std::string
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::name() const {
   return "MP";
}

template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline const typename MessagePassing<GM, ACC, UPDATE_RULES, DIST>::GraphicalModelType&
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::graphicalModel() const {
   return gm_;
}

template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline InferenceTermination
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::infer() {
   EmptyVisitorType v;
   return infer(v);
}

template<class GM, class ACC, class UPDATE_RULES, class DIST>
template<class VisitorType>
inline InferenceTermination
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::infer
(
   VisitorType& visitor
) {
   if (parameter_.isAcyclic_ == opengm::Tribool::True) {
      parameter_.useNormalization_=false;
      inferAcyclic(visitor);
   } else if (parameter_.isAcyclic_ == opengm::Tribool::False) {
      if (parameter_.inferSequential_) {
         inferSequential(visitor);
      } else {
         inferParallel(visitor);
      }
   } else { //triibool maby
      if (gm_.isAcyclic()) {
         parameter_.isAcyclic_ = opengm::Tribool::True;
         parameter_.useNormalization_=false;
         inferAcyclic(visitor);
      } else {
         parameter_.isAcyclic_ = opengm::Tribool::False;
         if (parameter_.inferSequential_) {
            inferSequential(visitor);
         } else {
            inferParallel(visitor);
         }
      }
   }
   return NORMAL;
}

/// \brief inference for acyclic graphs.
///
/// A message is sent from a variable (resp. from a factor) only if
/// all messages to that variable (factor) have been received.
///
template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline void
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::inferAcyclic() { 
   EmptyVisitorType v;
   return infer(v);
}

/// \brief inference for acyclic graphs.
//
/// A message is sent from a variable (resp. from a factor) only if
/// all messages to that variable (factor) have been received.
///
/// \param visitor
///
template<class GM, class ACC, class UPDATE_RULES, class DIST>
template<class VisitorType>
void
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::inferAcyclic
(
   VisitorType& visitor
) 
{
   OPENGM_ASSERT(gm_.isAcyclic());
   visitor.begin(*this);
   size_t numberOfVariables = gm_.numberOfVariables();
   size_t numberOfFactors = gm_.numberOfFactors();
   // number of messages which have not yet been recevied
   // but are required for sending
   std::vector<std::vector<size_t> > counterVar2FacMessage(numberOfVariables);
   std::vector<std::vector<size_t> > counterFac2VarMessage(numberOfFactors);
   // list of messages which are ready to send
   std::vector<Message> ready2SendVar2FacMessage;
   std::vector<Message> ready2SendFac2VarMessage;
   ready2SendVar2FacMessage.reserve(100);
   ready2SendFac2VarMessage.reserve(100);
   for (size_t fac = 0; fac < numberOfFactors; ++fac) {
      counterFac2VarMessage[fac].resize(gm_[fac].numberOfVariables(), gm_[fac].numberOfVariables() - 1);
   }
   for (size_t var = 0; var < numberOfVariables; ++var) {
      counterVar2FacMessage[var].resize(gm_.numberOfFactors(var));
      for (size_t i = 0; i < gm_.numberOfFactors(var); ++i) {
         counterVar2FacMessage[var][i] = gm_.numberOfFactors(var) - 1;
      }
   }
   // find all messages which are ready for sending
   for (size_t var = 0; var < numberOfVariables; ++var) {
      for (size_t i = 0; i < counterVar2FacMessage[var].size(); ++i) {
         if (counterVar2FacMessage[var][i] == 0) {
            --counterVar2FacMessage[var][i];
            ready2SendVar2FacMessage.push_back(Message(var, i));
         }
      }
   }
   for (size_t fac = 0; fac < numberOfFactors; ++fac) {
      for (size_t i = 0; i < counterFac2VarMessage[fac].size(); ++i) {
         if (counterFac2VarMessage[fac][i] == 0) {
            --counterFac2VarMessage[fac][i];
            ready2SendFac2VarMessage.push_back(Message(fac, i));
         }
      }
   }
   // send messages
   while (ready2SendVar2FacMessage.size() > 0 || ready2SendFac2VarMessage.size() > 0) {
      while (ready2SendVar2FacMessage.size() > 0) {
         Message m = ready2SendVar2FacMessage.back();
         size_t nodeId = m.nodeId_;
         size_t factorId = gm_.factorOfVariable(nodeId,m.internalMessageId_);
         // send message
         variableHulls_[nodeId].propagate(gm_, m.internalMessageId_, 0, false);
         ready2SendVar2FacMessage.pop_back();
         //check if new messages can be sent
         for (size_t i = 0; i < gm_[factorId].numberOfVariables(); ++i) {
            if (gm_[factorId].variableIndex(i) != nodeId) {
               if (--counterFac2VarMessage[factorId][i] == 0) {
                  ready2SendFac2VarMessage.push_back(Message(factorId, i));
               }
            }
         }
      }
      while (ready2SendFac2VarMessage.size() > 0) {
         Message m = ready2SendFac2VarMessage.back();
         size_t factorId = m.nodeId_;
         size_t nodeId = gm_[factorId].variableIndex(m.internalMessageId_);
         // send message
         factorHulls_[factorId].propagate(m.internalMessageId_, 0, parameter_.useNormalization_);
         ready2SendFac2VarMessage.pop_back();
         // check if new messages can be sent
         for (size_t i = 0; i < gm_.numberOfFactors(nodeId); ++i) {
            if (gm_.factorOfVariable(nodeId,i) != factorId) {
               if (--counterVar2FacMessage[nodeId][i] == 0) {
                  ready2SendVar2FacMessage.push_back(Message(nodeId, i));
               }
            }
         }
      }
      if(visitor(*this)!=0)
         break;
   }
   visitor.end(*this);
   
}

/// \brief invoke one iteration of message passing
template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline void MessagePassing<GM, ACC, UPDATE_RULES, DIST>::propagate
(
   const ValueType& damping
) {
   for (size_t i = 0; i < variableHulls_.size(); ++i) {
      variableHulls_[i].propagateAll(damping, false);
   }
   for (size_t i = 0; i < factorHulls_.size(); ++i) {
      factorHulls_[i].propagateAll(damping, parameter_.useNormalization_);
   }
}

/// \brief inference with parallel message passing.
template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline void MessagePassing<GM, ACC, UPDATE_RULES, DIST>::inferParallel() {
   EmptyVisitorType v;
   return infer(v);
}

/// \brief inference with parallel message passing.
/// \param visitor
template<class GM, class ACC, class UPDATE_RULES, class DIST>
template<class VisitorType>
inline void MessagePassing<GM, ACC, UPDATE_RULES, DIST>::inferParallel
(
   VisitorType& visitor
) 
{
   ValueType c = 0;
   ValueType damping = parameter_.damping_;
   visitor.begin(*this);
    
   // let all Factors with a order lower than 2 sending their Message
   for (size_t i = 0; i < factorHulls_.size(); ++i) {
      if (factorHulls_[i].numberOfBuffers() < 2) {
         factorHulls_[i].propagateAll(0, parameter_.useNormalization_);
         factorHulls_[i].propagateAll(0, parameter_.useNormalization_); // 2 times to fill both buffers
      }
   }
   for (unsigned long n = 0; n < parameter_.maximumNumberOfSteps_; ++n) {
      for (size_t i = 0; i < variableHulls_.size(); ++i) {
         variableHulls_[i].propagateAll(gm_, damping, false);
      }
      for (size_t i = 0; i < factorHulls_.size(); ++i) {
         if (factorHulls_[i].numberOfBuffers() >= 2)// messages from factors of order <2 do not change
            factorHulls_[i].propagateAll(damping, parameter_.useNormalization_);
      }
      if(visitor(*this)!=0)
         break;
      c = convergence();
      if (c < parameter_.bound_) {
         break;
      }
   }
   visitor.end(*this);
    
}

/// \brief inference with sequential message passing.
///
/// sequential message passing according to Kolmogorov (TRW-S) and
/// Tappen (BP-S). These algorithms are designed for factors of
/// order 2; we cannot guarantee the convergence properties for these
/// algorithms when applied to graphical models with higher order
/// factors.
///
template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline void MessagePassing<GM, ACC, UPDATE_RULES, DIST>::inferSequential() { 
   EmptyVisitorType v;
   return infer(v);
}

/// \brief inference with sequential message passing.
///
/// sequential message passing according to Kolmogorov (TRW-S) and
/// Tappen (BP-S). These algorithms are designed for factors of
/// order 2; we cannot guarantee the convergence properties for these
/// algorithms when applied to graphical models with higher order
/// factors.
///
/// \param visitor
///
template<class GM, class ACC, class UPDATE_RULES, class DIST>
template<class VisitorType>
inline void MessagePassing<GM, ACC, UPDATE_RULES, DIST>::inferSequential
(
   VisitorType& visitor
) {
   OPENGM_ASSERT(parameter_.sortedNodeList_.size() == gm_.numberOfVariables()); 
   visitor.begin(*this);
   ValueType damping = parameter_.damping_;

   // set nodeOrder
   std::vector<size_t> nodeOrder(gm_.numberOfVariables());
   for (size_t o = 0; o < gm_.numberOfVariables(); ++o) {
      nodeOrder[parameter_.sortedNodeList_[o]] = o;
   }

   // let all Factors with a order lower than 2 sending their Message
   for (size_t f = 0; f < factorHulls_.size(); ++f) {
      if (factorHulls_[f].numberOfBuffers() < 2) {
         factorHulls_[f].propagateAll(0, parameter_.useNormalization_);
         factorHulls_[f].propagateAll(0, parameter_.useNormalization_); //2 times to fill both buffers
      }
   }

   // calculate inverse positions
   std::vector<std::vector<size_t> > inversePositions(gm_.numberOfVariables());
   for(size_t var=0; var<gm_.numberOfVariables();++var) {
      for(size_t i=0; i<gm_.numberOfFactors(var); ++i) {
         size_t factorId = gm_.factorOfVariable(var,i);
         for(size_t j=0; j<gm_.numberOfVariables(factorId);++j) {
            if(gm_.variableOfFactor(factorId,j)==var) {
               inversePositions[var].push_back(j);
               break;
            }
         }
      }
   }


   // the following Code is not optimized and maybe too slow for small factors
   for (unsigned long itteration = 0; itteration < parameter_.maximumNumberOfSteps_; ++itteration) {
      if(itteration%2==0) {
         // in increasing ordering
         for (size_t o = 0; o < gm_.numberOfVariables(); ++o) {
            size_t variableId = parameter_.sortedNodeList_[o];
            // update messages to the variable node
            for(size_t i=0; i<gm_.numberOfFactors(variableId); ++i) {
               size_t factorId = gm_.factorOfVariable(variableId,i);
               factorHulls_[factorId].propagate(inversePositions[variableId][i], damping, parameter_.useNormalization_); 
            }

            // update messages from the variable node
            variableHulls_[variableId].propagateAll(gm_, damping, false);
         }
      }
      else{
         // in decreasing ordering
         for (size_t o = 0; o < gm_.numberOfVariables(); ++o) {
            size_t variableId = parameter_.sortedNodeList_[gm_.numberOfVariables() - 1 - o];
            // update messages to the variable node
            for(size_t i=0; i<gm_.numberOfFactors(variableId); ++i) {
               size_t factorId = gm_.factorOfVariable(variableId,i);
               factorHulls_[factorId].propagate(inversePositions[variableId][i], damping, parameter_.useNormalization_); 
            }
            // update messages from Variable
            variableHulls_[variableId].propagateAll(gm_, damping, false);
         }
      }
      if(visitor(*this)!=0)
         break;
    
   } 
   visitor.end(*this);
}

template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline InferenceTermination
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::marginal
(
   const size_t variableIndex,
   IndependentFactorType & out
) const {
   OPENGM_ASSERT(variableIndex < variableHulls_.size());
   variableHulls_[variableIndex].marginal(gm_, variableIndex, out, parameter_.useNormalization_);
   return NORMAL;
}

template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline InferenceTermination
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::factorMarginal
(
   const size_t factorIndex,
   IndependentFactorType &out
) const {
   typedef typename GM::OperatorType OP;
   OPENGM_ASSERT(factorIndex < factorHulls_.size());
   out.assign(gm_, gm_[factorIndex].variableIndicesBegin(), gm_[factorIndex].variableIndicesEnd(), OP::template neutral<ValueType>());
   factorHulls_[factorIndex].marginal(out, parameter_.useNormalization_);
   return NORMAL;
}

/// \brief cumulative distance between all pairs of messages from variables to factors (between the previous and the current interation)
template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline typename MessagePassing<GM, ACC, UPDATE_RULES, DIST>::ValueType
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::convergenceXF() const {
   ValueType result = 0;
   for (size_t j = 0; j < factorHulls_.size(); ++j) {
      for (size_t i = 0; i < factorHulls_[j].numberOfBuffers(); ++i) {
         ValueType d = factorHulls_[j].template distance<DIST > (i);
         if (d > result) {
            result = d;
         }
      }
   }
   return result;
}

/// \brief cumulative distance between all pairs of messages from factors to variables (between the previous and the current interation)
template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline typename MessagePassing<GM, ACC, UPDATE_RULES, DIST>::ValueType
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::convergenceFX() const {
   ValueType result = 0;
   for (size_t j = 0; j < variableHulls_.size(); ++j) {
      for (size_t i = 0; i < variableHulls_[j].numberOfBuffers(); ++i) {
         ValueType d = variableHulls_[j].template distance<DIST > (i);
         if (d > result) {
            result = d;
         }
      }
   }
   return result;
}

/// \brief cumulative distance between all pairs of messages (between the previous and the current interation)
template<class GM, class ACC, class UPDATE_RULES, class DIST>
inline typename MessagePassing<GM, ACC, UPDATE_RULES, DIST>::ValueType
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::convergence() const {
   return convergenceXF();
}

template<class GM, class ACC,class UPDATE_RULES, class DIST >
inline InferenceTermination
MessagePassing<GM, ACC, UPDATE_RULES, DIST>::arg
(
   std::vector<LabelType>& conf,
   const size_t N
) const {
   if (N != 1) {
      throw RuntimeError("This implementation of message passing cannot return the k-th optimal configuration.");
   }
   else {
      if (parameter_.isAcyclic_ == opengm::Tribool::True) {       
         return this->modeFromFactorMarginal(conf); 
      }
      else {
         return this->modeFromFactorMarginal(conf); 
         //return modeFromMarginal(conf);
      }
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_BELIEFPROPAGATION_HXX
