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
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/utilities/metaprogramming.hxx"

#include "typeView.h"
#include "MRFEnergy.h"
#include "instances.h"
#include "MRFEnergy.cpp"
#include "minimize.cpp"
#include "treeProbabilities.cpp"
#include "ordering.cpp"

namespace opengm {
   namespace external {
      /// \brief message passing (BPS, TRWS): \n
      /// [?] 
      ///
      /// \ingroup inference
      /// \ingroup messagepassing_inference
      /// \ingroup external_inference
      /// TRWS :
      /// - cite :[?]
      /// - Maximum factor order : \f$2\f$
      /// - Maximum number of labels : \f$\infty\f$
      /// - Restrictions : -
      /// - Convergent : convergent on trees
      template<class GM>
      class TRWS : public Inference<GM, opengm::Minimizer> {
      public:
         typedef GM                              GraphicalModelType;
         typedef opengm::Minimizer               AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef visitors::VerboseVisitor<TRWS<GM> > VerboseVisitorType;
         typedef visitors::EmptyVisitor<TRWS<GM> >   EmptyVisitorType;
         typedef visitors::TimingVisitor<TRWS<GM> >  TimingVisitorType;
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
            ///  TRWS termintas if fabs(bound(t)-bound(t+1)) < minDualChange_
            double minDualChange_;
            /// \brief Constructor
            Parameter() {
               numberOfIterations_ = 1000;
               useRandomStart_ = false;
               useZeroStart_ = false;
               doBPS_ = false;
               energyType_ = VIEW;
               tolerance_ = 0.0;
               minDualChange_ = 0.00001;
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
         IndexType maxNumLabels_;
         bool hasSameLabelNumber_;
         void checkLabelNumber();

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
         void addNodes(MRFEnergy<ENERGYTYPE>*& mrf, typename MRFEnergy<ENERGYTYPE>::NodeId*& nodes, typename ENERGYTYPE::REAL* D);

         template<class VISITOR, class ENERGYTYPE>
         InferenceTermination inferImpl(VISITOR & visitor, MRFEnergy<ENERGYTYPE>* mrf);
      };

      template<class GM, class ENERGYTYPE>
      struct createMRFEnergy{
         static void* create(typename GM::IndexType numLabels);
      };

      template<class GM>
      struct createMRFEnergy<GM, TypeView<GM> >{
         static void* create(typename GM::IndexType numLabels);
      };

      template<class GM>
      struct createMRFEnergy<GM, TypeGeneral>{
         static void* create(typename GM::IndexType numLabels);
      };

      template<class GM>
      struct createMRFEnergy<GM, TypeTruncatedLinear>{
         static void* create(typename GM::IndexType numLabels);
      };

      template<class GM>
      struct createMRFEnergy<GM, TypeTruncatedQuadratic>{
         static void* create(typename GM::IndexType numLabels);
      };

      template<class GM, class ENERGYTYPE>
      struct addMRFNode{
         static typename MRFEnergy<ENERGYTYPE>::NodeId add(MRFEnergy<ENERGYTYPE>* mrf, typename GM::IndexType numLabels, typename ENERGYTYPE::REAL* D);
      };

      template<class GM>
      struct addMRFNode<GM, TypeView<GM> >{
         static typename MRFEnergy<TypeView<GM> >::NodeId add(MRFEnergy<TypeView<GM> >* mrf, typename GM::IndexType numLabels, typename TypeView<GM>::REAL* D);
      };

      template<class GM>
      struct addMRFNode<GM, TypeGeneral>{
         static typename MRFEnergy<TypeGeneral>::NodeId add(MRFEnergy<TypeGeneral>* mrf, typename GM::IndexType numLabels, typename TypeGeneral::REAL* D);
      };

      template<class GM>
      struct addMRFNode<GM, TypeTruncatedLinear>{
         static typename MRFEnergy<TypeTruncatedLinear>::NodeId add(MRFEnergy<TypeTruncatedLinear>* mrf, typename GM::IndexType numLabels, typename TypeTruncatedLinear::REAL* D);
      };

      template<class GM>
      struct addMRFNode<GM, TypeTruncatedQuadratic>{
         static typename MRFEnergy<TypeTruncatedQuadratic>::NodeId add(MRFEnergy<TypeTruncatedQuadratic>* mrf, typename GM::IndexType numLabels, typename TypeTruncatedQuadratic::REAL* D);
      };

      template<class GM>
      TRWS<GM>::TRWS(
         const typename TRWS::GraphicalModelType& gm,
         const Parameter para
         )
         :  gm_(gm), parameter_(para), mrfView_(NULL), nodesView_(NULL), mrfGeneral_(NULL), nodesGeneral_(NULL),
            mrfTL1_(NULL), nodesTL1_(NULL), mrfTL2_(NULL), nodesTL2_(NULL), numNodes_(gm_.numberOfVariables()),
            maxNumLabels_(gm_.numberOfLabels(0)) {
         // check label number
         checkLabelNumber();

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
               if(!hasSameLabelNumber_) {
                  throw(RuntimeError("TRWS TL1 only supports graphical models where each variable has the same number of states."));
               }
               generateMRFTL1();
               break;
            }
            case Parameter::TL2: {
               if(!hasSameLabelNumber_) {
                  throw(RuntimeError("TRWS TL2 only supports graphical models where each variable has the same number of states."));
               }
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
         return lowerBound_+constTerm_;
      }
      template<class GM>
      inline typename GM::ValueType
      TRWS<GM>::value() const {
         return value_+constTerm_;
      }

      template<class GM>
      inline void TRWS<GM>::checkLabelNumber() {
         hasSameLabelNumber_ = true;
         for(IndexType i = 1; i < gm_.numberOfVariables(); i++) {
            if(gm_.numberOfLabels(i) != maxNumLabels_) {
               hasSameLabelNumber_ = false;
            }
            if(gm_.numberOfLabels(i) > maxNumLabels_) {
               maxNumLabels_ = gm_.numberOfLabels(i);
            }
         }
      }

      template<class GM>
      inline void TRWS<GM>::generateMRFView() {
         mrfView_ = new MRFEnergy<TypeView<GM> >(typename TypeView<GM>::GlobalSize());
         nodesView_ = new typename MRFEnergy<TypeView<GM> >::NodeId[numNodes_];

         // add nodes
         for(IndexType i = 0; i < numNodes_; i++) {
            std::vector<typename GM::IndexType> factors;
            for(typename GM::ConstFactorIterator iter = gm_.factorsOfVariableBegin(i); iter != gm_.factorsOfVariableEnd(i); iter++) {
               if(gm_[*iter].numberOfVariables() == 1) {
                  factors.push_back(*iter);
               }
            }
            nodesView_[i] = mrfView_->AddNode(typename TypeView<GM>::LocalSize(gm_.numberOfLabels(i)), typename TypeView<GM>::NodeData(gm_, factors));
         }
    
         // add edges
         constTerm_ = 0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) { 
            if(gm_[i].numberOfVariables() == 0){
               LabelType l = 0;
               constTerm_ += gm_[i](&l);
            }
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
         typename TypeGeneral::REAL* D = new typename TypeGeneral::REAL[maxNumLabels_];
         addNodes(mrfGeneral_, nodesGeneral_, D);
         delete[] D;

         // add edges
         IndexType index[2];
         constTerm_ = 0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_[i].numberOfVariables() == 0){
               LabelType l = 0;
               constTerm_ += gm_[i](&l);
            }
            if(gm_[i].numberOfVariables() == 2) {
               IndexType a = gm_[i].variableIndex(0);
               IndexType b = gm_[i].variableIndex(1);
               IndexType numLabels_a = gm_.numberOfLabels(a);
               IndexType numLabels_b = gm_.numberOfLabels(b);
               typename TypeGeneral::REAL* V = new typename TypeGeneral::REAL[numLabels_a * numLabels_b];
               for(size_t j = 0; j < numLabels_a; j++) {
                  for(size_t k = 0; k < numLabels_b; k++) {
                     index[0] = j;
                     index[1] = k;
                     V[j + k * numLabels_a] = gm_[i](index);
                  }
               }
               mrfGeneral_->AddEdge(nodesGeneral_[a], nodesGeneral_[b], TypeGeneral::EdgeData(TypeGeneral::GENERAL, V));
               delete[] V;
            }
         }

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
         typename TypeTruncatedLinear::REAL* D = new typename TypeTruncatedLinear::REAL[maxNumLabels_];
         addNodes(mrfTL1_, nodesTL1_, D);
         delete[] D;

         // add edges
         constTerm_=0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_[i].numberOfVariables() == 0){
               LabelType l = 0;
               constTerm_ += gm_[i](&l);
            }
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
         typename TypeTruncatedQuadratic::REAL* D = new typename TypeTruncatedQuadratic::REAL[maxNumLabels_];
         addNodes(mrfTL2_, nodesTL2_, D);
         delete[] D;

         // add edges
         constTerm_=0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) { 
            if(gm_[i].numberOfVariables() == 0){
               LabelType l = 0;
               constTerm_ += gm_[i](&l);
            }
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
         IndexType index0[] = {0, maxNumLabels_-1};

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
      inline void TRWS<GM>::addNodes(MRFEnergy<ENERGYTYPE>*& mrf, typename MRFEnergy<ENERGYTYPE>::NodeId*& nodes, typename ENERGYTYPE::REAL* D) {

         mrf = reinterpret_cast<MRFEnergy<ENERGYTYPE>*>(createMRFEnergy<GM, ENERGYTYPE>::create(maxNumLabels_));

         nodes = new typename MRFEnergy<ENERGYTYPE>::NodeId[numNodes_];
         for(IndexType i = 0; i < numNodes_; i++) {
            for(IndexType j = 0; j < gm_.numberOfLabels(i); j++) {
               D[j] = 0.0;
            }
            for(typename GM::ConstFactorIterator iter = gm_.factorsOfVariableBegin(i); iter != gm_.factorsOfVariableEnd(i); iter++) {
               if(gm_[*iter].numberOfVariables() == 1) {
                  for(IndexType j = 0; j < gm_.numberOfLabels(i); j++) {
                     D[j] += gm_[*iter](&j);
                  }
               }
            }
            nodes[i] = addMRFNode<GM, ENERGYTYPE>::add(mrf, gm_.numberOfLabels(i), D);
         }
      }

      template<class GM, class ENERGYTYPE>
      inline void* createMRFEnergy<GM, ENERGYTYPE>::create(typename GM::IndexType numLabels) {
         RuntimeError("Unsupported Energy Type!");
         return NULL;
      }

      template<class GM>
      inline void* createMRFEnergy<GM, TypeView<GM> >::create(typename GM::IndexType numLabels) {
         return reinterpret_cast<void*>(new MRFEnergy<TypeView<GM> >(typename TypeView<GM>::GlobalSize()));
      }

      template<class GM>
      inline void* createMRFEnergy<GM, TypeGeneral>::create(typename GM::IndexType numLabels) {
         return reinterpret_cast<void*>(new MRFEnergy<TypeGeneral>(typename TypeGeneral::GlobalSize()));
      }

      template<class GM>
      inline void* createMRFEnergy<GM, TypeTruncatedLinear>::create(typename GM::IndexType numLabels) {
         return reinterpret_cast<void*>(new MRFEnergy<TypeTruncatedLinear>(typename TypeTruncatedLinear::GlobalSize(numLabels)));
      }

      template<class GM>
      inline void* createMRFEnergy<GM, TypeTruncatedQuadratic>::create(typename GM::IndexType numLabels) {
         return reinterpret_cast<void*>(new MRFEnergy<TypeTruncatedQuadratic>(typename TypeTruncatedQuadratic::GlobalSize(numLabels)));
      }

      template<class GM, class ENERGYTYPE>
      inline typename MRFEnergy<ENERGYTYPE>::NodeId addMRFNode<GM, ENERGYTYPE>::add(MRFEnergy<ENERGYTYPE>* mrf, typename GM::IndexType numLabels, typename ENERGYTYPE::REAL* D) {
         RuntimeError("Unsupported Energy Type!");
         return NULL;
      }

      template<class GM>
      inline typename MRFEnergy<TypeView<GM> >::NodeId addMRFNode<GM, TypeView<GM> >::add(MRFEnergy<TypeView<GM> >* mrf, typename GM::IndexType numLabels, typename TypeView<GM>::REAL* D) {
         return mrf->AddNode(typename TypeView<GM>::LocalSize(numLabels), typename TypeView<GM>::NodeData(D));
      }

      template<class GM>
      inline typename MRFEnergy<TypeGeneral>::NodeId addMRFNode<GM, TypeGeneral>::add(MRFEnergy<TypeGeneral>* mrf, typename GM::IndexType numLabels, typename TypeGeneral::REAL* D) {
         return mrf->AddNode(typename TypeGeneral::LocalSize(numLabels), typename TypeGeneral::NodeData(D));
      }

      template<class GM>
      inline typename MRFEnergy<TypeTruncatedLinear>::NodeId addMRFNode<GM, TypeTruncatedLinear>::add(MRFEnergy<TypeTruncatedLinear>* mrf, typename GM::IndexType numLabels, typename TypeTruncatedLinear::REAL* D) {
         return mrf->AddNode(typename TypeTruncatedLinear::LocalSize(), typename TypeTruncatedLinear::NodeData(D));
      }

      template<class GM>
      inline typename MRFEnergy<TypeTruncatedQuadratic>::NodeId addMRFNode<GM, TypeTruncatedQuadratic>::add(MRFEnergy<TypeTruncatedQuadratic>* mrf, typename GM::IndexType numLabels, typename TypeTruncatedQuadratic::REAL* D) {
         return mrf->AddNode(typename TypeTruncatedQuadratic::LocalSize(), typename TypeTruncatedQuadratic::NodeData(D));
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
               if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) {
                  break;
               }
            }
         } else {
            typename ENERGYTYPE::REAL v;
            typename ENERGYTYPE::REAL b;
            typename ENERGYTYPE::REAL d;
            for(size_t i = 0; i < parameter_.numberOfIterations_; ++i) {
               mrf->Minimize_TRW_S(options, b, v);
               d = b-lowerBound_;
               lowerBound_ = b;
               value_ = v;
               if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) {
                  break;
               }
               if(fabs(value_ - lowerBound_) / opengmMax(static_cast<double>(fabs(value_)), 1.0) < parameter_.tolerance_) {
                  break;
               }
               if(d<parameter_.minDualChange_){
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
