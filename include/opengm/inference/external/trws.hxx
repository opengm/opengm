/// OpenGM. Copyright (c) 2010 by Bjoern Andres and Joerg Hendrik Kappes.
///
/// Author(s) of this file: Joerg Hendrik Kappes
/// For further details see opengm/README.txt

#pragma once
#ifndef OPENGM_EXTERNAL_TRWS_HXX
#define OPENGM_EXTERNAL_TRWS_HXX

#include "opengm/inference/inference.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitor.hxx"

#include "typeView.h"
#include "MRFEnergy.h"
#include "instances.h"
#include "MRFEnergy.cpp"
#include "minimize.cpp"
#include "treeProbabilities.cpp"
#include "ordering.cpp"

namespace opengm {
   namespace external {
      /// \brief message passing (BP / TRBP): \n
      /// [?] 
      ///
      /// \ingroup inference
      /// \ingroup messagepassing_inference
      /// \ingroup external_inference
      /// Gibbs Sampling :
      /// - cite :[?]
      /// - Maximum factor order : \f$\infty\f$
      /// - Maximum number of labels : \f$\infty\f$
      /// - Restrictions : -
      /// - Convergent : convergent on trees
      template<class GM>
      class TRWS : public Inference<GM, opengm::Minimizer> {
      public:
         typedef GM                              GraphicalModelType;
         typedef opengm::Minimizer               AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef EmptyVisitor<TRWS<GM> > EmptyVisitorType;
         typedef VerboseVisitor<TRWS<GM> > VerboseVisitorType;
         typedef TimingVisitor<TRWS<GM> > TimingVisitorType;
         typedef size_t VariableIndex;
         ///Parameter
         struct Parameter {
            /// possible energy types for TRWS
            enum EnergyType {VIEW, TABLES, TL1, TL2/*, WEIGHTEDTABLE*/};
            /// number of iterations
            size_t numberOfIterations_;
            /// random starting message
            bool useRandomStart_;
            /// zero starting message
            bool useZeroStart_;
            /// use normal LBP
            bool doBPS_;
            /// selected energy type
            EnergyType energyType_;
            /// TRWS termintas if fabs(value - bound) / max(fabs(value), 1) < trwsTolerance_
            double tolerance_;
            /// \brief Constructor
            Parameter() {
               numberOfIterations_ = 1000;
               useRandomStart_ = false;
               useZeroStart_ = false;
               doBPS_ = false;
               energyType_ = VIEW;
               tolerance_ = 0.0;
            };
         };
         // construction
         TRWS(const GraphicalModelType& gm, const Parameter para = Parameter());
         // destruction
         ~TRWS();
         // query
         std::string name() const;
         const GraphicalModelType& graphicalModel() const;
         // inference
         template<class VISITOR>
         InferenceTermination infer(VISITOR & visitor);
         InferenceTermination infer();
         InferenceTermination arg(std::vector<LabelType>&, const size_t& = 1) const;
         typename GM::ValueType bound() const;  
         typename GM::ValueType value() const;
      private:
         const GraphicalModelType& gm_;
         Parameter parameter_;
         ValueType constTerm_;
         
         MRFEnergy<TypeView<GM> >* mrfView_;
         typename MRFEnergy<TypeView<GM> >::NodeId* nodesView_;
         MRFEnergy<TypeGeneral>* mrfGeneral_;
         MRFEnergy<TypeGeneral>::NodeId* nodesGeneral_;
         MRFEnergy<TypeTruncatedLinear>* mrfTL1_;
         MRFEnergy<TypeTruncatedLinear>::NodeId* nodesTL1_;
         MRFEnergy<TypeTruncatedQuadratic>* mrfTL2_;
         MRFEnergy<TypeTruncatedQuadratic>::NodeId* nodesTL2_;
         
         double runTime_;
         ValueType lowerBound_;
         ValueType value_;
         std::vector<LabelType> state_;
         const IndexType numNodes_;
         const IndexType numLabels_;
         bool hasSameLabelNumber() const;

         void generateMRFView();
         void generateMRFTables();
         void generateMRFTL1();
         void generateMRFTL2();
         //void generateMRFWeightedTable();

         ValueType getT(IndexType factor) const;

         // required for energy type tl1
         bool truncatedAbsoluteDifferenceFactors() const;

         // required for energy type tl2
         bool truncatedSquaredDifferenceFactors() const;

         template <class ENERGYTYPE>
         void addNodes(MRFEnergy<ENERGYTYPE>*& mrf, typename MRFEnergy<ENERGYTYPE>::NodeId*& nodes, typename ENERGYTYPE::REAL* D, typename ENERGYTYPE::GlobalSize globalSize, typename ENERGYTYPE::LocalSize localSize);

         template<class VISITOR, class ENERGYTYPE>
         InferenceTermination inferImpl(VISITOR & visitor, MRFEnergy<ENERGYTYPE>* mrf);
      };       

         template<class GM>
         TRWS<GM>::TRWS(
            const typename TRWS::GraphicalModelType& gm,
            const Parameter para
            )
            :  gm_(gm), parameter_(para), mrfView_(NULL), nodesView_(NULL), mrfGeneral_(NULL), nodesGeneral_(NULL),
               mrfTL1_(NULL), nodesTL1_(NULL), mrfTL2_(NULL), nodesTL2_(NULL), numNodes_(gm_.numberOfVariables()),
               numLabels_(gm_.numberOfLabels(0)) {
            // check label number
            if(!hasSameLabelNumber()) {
               throw(RuntimeError("TRWS only supports graphical models where each variable has the same number of states."));
            }

            // generate mrf model
            switch(parameter_.energyType_) {
               case Parameter::VIEW: {
                  generateMRFView();
                  break;
               }
               case Parameter::TABLES: {
                  generateMRFTables();
                  break;
               }
               case Parameter::TL1: {
                  generateMRFTL1();
                  break;
               }
               case Parameter::TL2: {
                  generateMRFTL2();
                  break;
               }
               /*case Parameter::WEIGHTEDTABLE: {
                  generateMRFWeightedTable();
                  break;
               }*/
               default: {
                  throw(RuntimeError("Unknown energy type."));
               }
            }

            // set initial value and lower bound
            AccumulationType::neutral(value_);
            AccumulationType::ineutral(lowerBound_);
         }

         template<class GM>
         TRWS<GM>::~TRWS() {
            if(mrfView_) {
               delete mrfView_;
            }
            if(nodesView_) {
               delete[] nodesView_;
            }

            if(mrfGeneral_) {
               delete mrfGeneral_;
            }
            if(nodesGeneral_) {
               delete[] nodesGeneral_;
            }

            if(mrfTL1_) {
               delete mrfTL1_;
            }
            if(nodesTL1_) {
               delete[] nodesTL1_;
            }

            if(mrfTL2_) {
               delete mrfTL2_;
            }
            if(nodesTL2_) {
               delete[] nodesTL2_;
            }
         }

         template<class GM>
         inline std::string
         TRWS<GM>
         ::name() const {
            return "TRWS";
         }

         template<class GM>
         inline const typename TRWS<GM>::GraphicalModelType&
         TRWS<GM>
         ::graphicalModel() const {
            return gm_;
         } 
      
         template<class GM>
         inline InferenceTermination
         TRWS<GM>::infer
         (
            ) {
            EmptyVisitorType visitor;
            return this->infer(visitor);
         }

         template<class GM>
         template<class VISITOR>
         inline InferenceTermination
         TRWS<GM>::infer
         (
            VISITOR & visitor
         ) {
            switch(parameter_.energyType_) {
               case Parameter::VIEW: {
                  return inferImpl(visitor, mrfView_);
                  break;
               }
               case Parameter::TABLES: {
                  return inferImpl(visitor, mrfGeneral_);
                  break;
               }
               case Parameter::TL1: {
                  return inferImpl(visitor, mrfTL1_);
                  break;
               }
               case Parameter::TL2: {
                  return inferImpl(visitor, mrfTL2_);
                  break;
               }
/*               case Parameter::WEIGHTEDTABLE: {
                  return inferImpl(visitor, mrf);
                  break;
               }*/
               default: {
                  throw(RuntimeError("Unknown energy type."));
               }
            }
         }

         template<class GM>
         inline InferenceTermination
         TRWS<GM>
         ::arg		(
            std::vector<LabelType>& arg,
            const size_t& n
            ) const {

            if(n > 1) {
               return UNKNOWN;
            }
            else {
               arg.resize(numNodes_);
               switch(parameter_.energyType_) {
                  case Parameter::VIEW: {
                     for(IndexType i = 0; i < numNodes_; i++) {
                        arg[i] = mrfView_->GetSolution(nodesView_[i]);
                     }
                     return NORMAL;
                     break;
                  }
                  case Parameter::TABLES: {
                     for(IndexType i = 0; i < numNodes_; i++) {
                        arg[i] = mrfGeneral_->GetSolution(nodesGeneral_[i]);
                     }
                     return NORMAL;
                     break;
                  }
                  case Parameter::TL1: {
                     for(IndexType i = 0; i < numNodes_; i++) {
                        arg[i] = mrfTL1_->GetSolution(nodesTL1_[i]);
                     }
                     return NORMAL;
                     break;
                  }
                  case Parameter::TL2: {
                     for(IndexType i = 0; i < numNodes_; i++) {
                        arg[i] = mrfTL2_->GetSolution(nodesTL2_[i]);
                     }
                     return NORMAL;
                     break;
                  }
   /*               case Parameter::WEIGHTEDTABLE: {
                     for(IndexType i = 0; i < numNodes_; i++) {
                        arg[i] = mrfGeneral_->GetSolution(nodesGeneral_[i]);
                     }
                     return NORMAL;
                     break;
                  }*/
                  default: {
                     throw(RuntimeError("Unknown energy type."));
                  }
               }
            }
         }

      template<class GM>
      inline typename GM::ValueType
      TRWS<GM>::bound() const {
         return lowerBound_;
      }
      template<class GM>
      inline typename GM::ValueType
      TRWS<GM>::value() const {
         return value_;
      }

      template<class GM>
      inline bool TRWS<GM>::hasSameLabelNumber() const {
         for(IndexType i = 1; i < gm_.numberOfVariables(); i++) {
            if(gm_.numberOfLabels(i) != numLabels_) {
               return false;
            }
         }
         return true;
      }

      template<class GM>
      inline void TRWS<GM>::generateMRFView() {
         mrfView_ = new MRFEnergy<TypeView<GM> >(typename TypeView<GM>::GlobalSize(numLabels_));
         nodesView_ = new typename MRFEnergy<TypeView<GM> >::NodeId[numNodes_];

         // add nodes
         for(IndexType i = 0; i < numNodes_; i++) {
            std::vector<typename GM::IndexType> factors;
            for(typename GM::ConstFactorIterator iter = gm_.factorsOfVariableBegin(i); iter != gm_.factorsOfVariableEnd(i); iter++) {
               if(gm_[*iter].numberOfVariables() == 1) {
                  factors.push_back(*iter);
               }
            }
            nodesView_[i] = mrfView_->AddNode(typename TypeView<GM>::LocalSize(numLabels_), typename TypeView<GM>::NodeData(gm_, factors));
         }

         // add edges
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_[i].numberOfVariables() == 2) {
               IndexType a = gm_[i].variableIndex(0);
               IndexType b = gm_[i].variableIndex(1);
               mrfView_->AddEdge(nodesView_[a], nodesView_[b], typename TypeView<GM>::EdgeData(gm_, i));
            }
         }
         // set random start message
         if(parameter_.useRandomStart_) {
            mrfView_->AddRandomMessages(1, 0.0, 1.0);
         } else if(parameter_.useZeroStart_) {
            mrfView_->ZeroMessages();
         }
      }

      template<class GM>
      inline void TRWS<GM>::generateMRFTables() {
         // add nodes
         typename TypeGeneral::REAL* D = new typename TypeGeneral::REAL[numLabels_];
         addNodes(mrfGeneral_, nodesGeneral_, D, TypeGeneral::GlobalSize(), TypeGeneral::LocalSize(numLabels_));
         delete[] D;

         // add edges
         typename TypeGeneral::REAL* V = new typename TypeGeneral::REAL[numLabels_ * numLabels_];
         IndexType index[2];
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_[i].numberOfVariables() == 2) {
               IndexType a = gm_[i].variableIndex(0);
               IndexType b = gm_[i].variableIndex(1);
               for(size_t j = 0; j < numLabels_; j++) {
                  for(size_t k = 0; k < numLabels_; k++) {
                     index[0] = j;
                     index[1] = k;
                     V[j + k * numLabels_] = gm_[i](index);
                  }
               }
               mrfGeneral_->AddEdge(nodesGeneral_[a], nodesGeneral_[b], TypeGeneral::EdgeData(TypeGeneral::GENERAL, V));
            }
         }
         delete[] V;

         // set random start message
         if(parameter_.useRandomStart_) {
            mrfGeneral_->AddRandomMessages(1, 0.0, 1.0);
         } else if(parameter_.useZeroStart_) {
            mrfGeneral_->ZeroMessages();
         }
      }

      template<class GM>
      inline void TRWS<GM>::generateMRFTL1() {
         OPENGM_ASSERT(truncatedAbsoluteDifferenceFactors());

         // add nodes
         typename TypeTruncatedLinear::REAL* D = new typename TypeTruncatedLinear::REAL[numLabels_];
         addNodes(mrfTL1_, nodesTL1_, D, TypeTruncatedLinear::GlobalSize(numLabels_), TypeTruncatedLinear::LocalSize());
         delete[] D;

         // add edges
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_[i].numberOfVariables() == 2) {
               // truncation
               ValueType t = getT(i);
               //std::cout << "t: " << t << std::endl;

               // weight
               IndexType index[] = {0, 1};
               ValueType w = gm_[i](index);
               //std::cout << "w: " << w << std::endl;

               // corresponding node IDs
               IndexType a = gm_[i].variableIndex(0);
               IndexType b = gm_[i].variableIndex(1);
               mrfTL1_->AddEdge(nodesTL1_[a], nodesTL1_[b], TypeTruncatedLinear::EdgeData(w, w * t));
            }
         }

         // set random start message
         if(parameter_.useRandomStart_) {
            mrfTL1_->AddRandomMessages(1, 0.0, 1.0);
         } else if(parameter_.useZeroStart_) {
            mrfTL1_->ZeroMessages();
         }
      }

      template<class GM>
      inline void TRWS<GM>::generateMRFTL2() {
         OPENGM_ASSERT(truncatedSquaredDifferenceFactors());

         // add nodes
         typename TypeTruncatedQuadratic::REAL* D = new typename TypeTruncatedQuadratic::REAL[numLabels_];
         addNodes(mrfTL2_, nodesTL2_, D, TypeTruncatedQuadratic::GlobalSize(numLabels_), TypeTruncatedQuadratic::LocalSize());
         delete[] D;

         // add edges
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_[i].numberOfVariables() == 2) {
               // truncation
               ValueType t = getT(i);
               //std::cout << "t: " << t << std::endl;

               // weight
               IndexType index[] = {0, 1};
               ValueType w = gm_[i](index);
               //std::cout << "w: " << w << std::endl;

               // corresponding node IDs
               IndexType a = gm_[i].variableIndex(0);
               IndexType b = gm_[i].variableIndex(1);
               mrfTL2_->AddEdge(nodesTL2_[a], nodesTL2_[b], TypeTruncatedQuadratic::EdgeData(w, w * t));
            }
         }

         //mrfTL2_->SetAutomaticOrdering();

         // set random start message
         if(parameter_.useRandomStart_) {
            mrfTL2_->AddRandomMessages(1, 0.0, 1.0);
         } else if(parameter_.useZeroStart_) {
            mrfTL2_->ZeroMessages();
         }
      }

/*      template<class GM>
      inline void TRWS<GM>::generateMRFWeightedTable() {

      }*/

      template<class GM>
      inline typename GM::ValueType TRWS<GM>::getT(IndexType factor) const {
         OPENGM_ASSERT(gm_.numberOfVariables(factor) == 2);

         IndexType index1[] = {0, 1};
         IndexType index0[] = {0, numLabels_-1};

         return gm_[factor](index0)/gm_[factor](index1);
      }

      template<class GM>
      inline bool TRWS<GM>::truncatedAbsoluteDifferenceFactors() const {
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               if(gm_[i].isTruncatedAbsoluteDifference() == false) {
                  return false;
               }
            }
         }
         return true;
      }

      template<class GM>
      inline bool TRWS<GM>::truncatedSquaredDifferenceFactors() const {
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               if(gm_[i].isTruncatedSquaredDifference() == false) {
                  return false;
               }
            }
         }
         return true;
      }

      template<class GM>
      template <class ENERGYTYPE>
      inline void TRWS<GM>::addNodes(MRFEnergy<ENERGYTYPE>*& mrf, typename MRFEnergy<ENERGYTYPE>::NodeId*& nodes, typename ENERGYTYPE::REAL* D, typename ENERGYTYPE::GlobalSize globalSize, typename ENERGYTYPE::LocalSize localSize) {
         mrf = new MRFEnergy<ENERGYTYPE>(globalSize);
         nodes = new typename MRFEnergy<ENERGYTYPE>::NodeId[numNodes_];
         for(IndexType i = 0; i < numNodes_; i++) {
            for(IndexType j = 0; j < numLabels_; j++) {
               D[j] = 0.0;
            }
            for(typename GM::ConstFactorIterator iter = gm_.factorsOfVariableBegin(i); iter != gm_.factorsOfVariableEnd(i); iter++) {
               if(gm_[*iter].numberOfVariables() == 1) {
                  for(IndexType j = 0; j < numLabels_; j++) {
                     D[j] += gm_[*iter](&j);
                  }
               }
            }
            nodes[i] = mrf->AddNode(localSize, typename ENERGYTYPE::NodeData(D));
         }
      }

      template<class GM>
      template<class VISITOR, class ENERGYTYPE>
      inline InferenceTermination TRWS<GM>::inferImpl(VISITOR & visitor, MRFEnergy<ENERGYTYPE>* mrf) {
         typename MRFEnergy<ENERGYTYPE>::Options options;
         options.m_iterMax = 1; // maximum number of iterations
         options.m_printIter = 2 * parameter_.numberOfIterations_;
         visitor.begin(*this);

         if(parameter_.doBPS_) {
            typename ENERGYTYPE::REAL v;
            for(size_t i = 0; i < parameter_.numberOfIterations_; ++i) {
               mrf->Minimize_BP(options, v);
               value_ = v;
               visitor(*this);
            }
         } else {
            typename ENERGYTYPE::REAL v;
            typename ENERGYTYPE::REAL b;
            for(size_t i = 0; i < parameter_.numberOfIterations_; ++i) {
               mrf->Minimize_TRW_S(options, b, v);
               lowerBound_ = b;
               value_ = v;
               visitor(*this);
               if(fabs(value_ - lowerBound_) / opengmMax(static_cast<double>(fabs(value_)), 1.0) < parameter_.tolerance_) {
                  break;
               }
            }
         }

         visitor.end(*this);
         return NORMAL;
      }

   } // namespace external
} // namespace opengm

#endif // #ifndef OPENGM_EXTERNAL_TRWS_HXX
