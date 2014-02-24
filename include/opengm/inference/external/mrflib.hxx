#ifndef MRFLIB_HXX_
#define MRFLIB_HXX_

#include <algorithm>

#include "opengm/inference/inference.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

#include "mrflib.h"
/*
#include "MRF-v2.1/mrf.h"
#include "MRF-v2.1/ICM.h"
#include "MRF-v2.1/GCoptimization.h"
#include "MRF-v2.1/MaxProdBP.h"
#include "MRF-v2.1/TRW-S.h"
#include "MRF-v2.1/BP-S.h"
*/
namespace opengm {
   namespace external {

      /// MRFLIB
      /// MRFLIB inference algorithm class 
      /// \ingroup inference
      /// \ingroup external_inference
      ///
      //    MRFLIB
      /// - cite :[?]
      /// - Maximum factor order :2
      /// - Maximum number of labels : \f$\infty\f$
      /// - Restrictions : factor graph must be a 2d- grid
      /// - Convergent : ?
      template<class GM>
      class MRFLIB : public Inference<GM, opengm::Minimizer> {
      public:
         typedef GM                              GraphicalModelType;
         typedef opengm::Minimizer               AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef visitors::VerboseVisitor<MRFLIB<GM> > VerboseVisitorType;
         typedef visitors::EmptyVisitor<MRFLIB<GM> >   EmptyVisitorType;
         typedef visitors::TimingVisitor<MRFLIB<GM> >  TimingVisitorType;
         typedef size_t VariableIndex;
         ///Parameter
         struct Parameter {
            /// possible optimization algorithms for MRFLIB
            enum InferenceType {ICM, EXPANSION, SWAP, MAXPRODBP, TRWS, BPS};
            /// possible energy types for MRFLIB
            enum EnergyType {VIEW, TABLES, TL1, TL2, WEIGHTEDTABLE};
            /// selected optimization algorithm
            InferenceType inferenceType_;
            /// selected energy type
            EnergyType energyType_;
            /// number of iterations
            size_t numberOfIterations_;
            /// TRWS termintas if fabs(value - bound) / max(fabs(value), 1) < trwsTolerance_
            double trwsTolerance_;
            /// \brief Constructor
            Parameter(const InferenceType inferenceType = ICM, const EnergyType energyType = VIEW, const size_t numberOfIterations = 1000)
               : inferenceType_(inferenceType), energyType_(energyType), numberOfIterations_(numberOfIterations), trwsTolerance_(0.0) {
            }
         };
         // construction
         MRFLIB(const GraphicalModelType& gm, const Parameter& para = Parameter());
         // destruction
         ~MRFLIB();
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

          protected:
         const GraphicalModelType& gm_;
         IndexType sizeX_;
         IndexType sizeY_;
         IndexType numLabels_;
         marray::Matrix<size_t> grid_;
         Parameter parameter_;
         mrfLib::MRF* mrf_;
         mrfLib::MRF::CostVal* D_;
         mrfLib::MRF::CostVal* V_;
         mrfLib::MRF::CostVal* hCue_;
         mrfLib::MRF::CostVal* vCue_;
         mrfLib::DataCost *data_;
         mrfLib::SmoothnessCost* smooth_;
         mrfLib::EnergyFunction *energy_;
         void generateEnergyView();
         void generateEnergyTables();
         void generateEnergyTL1();
         void generateEnergyTL2();
         void generateEnergyWeightedTable();

         // required for energy type equal table
         void setD();
         void setV();
         void setWeightedTableWeights();
         bool hasSameLabelNumber() const;
         bool sameEnergyTable() const;
         bool symmetricEnergyTable() const;
         bool valueCheck() const;

         // required for energy type view
         static MRFLIB<GM>* mySelfView_;
         std::vector<std::vector<IndexType> > firstOrderFactorLookupTable_;
         std::vector<std::vector<IndexType> > horizontalSecondOrderFactorLookupTable_;
         std::vector<std::vector<IndexType> > verticalSecondOrderFactorLookupTable_;
         void generateFirstOrderFactorLookupTable_();
         void generateSecondOrderFactorLookupTables_();
         static mrfLib::MRF::CostVal firstOrderFactorViewAccess(int pix, int i);
         static mrfLib::MRF::CostVal secondOrderFactorViewAccess(int pix1, int pix2, int i, int j);

         // required for energy type tl1 and tl2
         ValueType getT(IndexType factor, ValueType e) const;
         bool sameT(ValueType T, ValueType e) const;
         void setWeights();
         bool equalWeights() const;

         // required for energy type tl1
         bool truncatedAbsoluteDifferenceFactors() const;

         // required for energy type tl2
         bool truncatedSquaredDifferenceFactors() const;

         // required for energy type tables
         static MRFLIB<GM>* mySelfTables_;
         std::vector<mrfLib::MRF::CostVal> firstOrderFactorValues;
         std::vector<mrfLib::MRF::CostVal> secondOrderFactorValues;
         static const IndexType right_ = 0;
         static const IndexType down_ = 1;
         void copyFactorValues();
         static mrfLib::MRF::CostVal firstOrderFactorTablesAccess(int pix, int i);
         static mrfLib::MRF::CostVal secondOrderFactorTablesAccess(int pix1, int pix2, int i, int j);
      };

      template<class GM>
      MRFLIB<GM>* MRFLIB<GM>::mySelfView_ = NULL;
      template<class GM>
      MRFLIB<GM>* MRFLIB<GM>::mySelfTables_ = NULL;

      template<class GM>
      MRFLIB<GM>::MRFLIB(const typename MRFLIB::GraphicalModelType& gm, const Parameter& para)
         : gm_(gm), parameter_(para), mrf_(NULL), D_(NULL), V_(NULL), hCue_(NULL), vCue_(NULL),
           data_(NULL), smooth_(NULL), energy_(NULL) {
         // check for grid structure
         bool isGrid = gm_.isGrid(grid_);
         if(!isGrid) {
            throw(RuntimeError("MRFLIB only supports graphical models which have a grid structure."));
         }
         sizeX_ = grid_.shape(0);
         sizeY_ = grid_.shape(1);

         // check label number
         numLabels_ = gm_.numberOfLabels(0);
         if(!hasSameLabelNumber()) {
            throw(RuntimeError("MRFLIB only supports graphical models where each variable has the same number of states."));
         }

         // generate energy function
         switch(parameter_.energyType_) {
            case Parameter::VIEW: {
               if(mySelfView_ != NULL) {
                  throw(RuntimeError("Singleton policy: MRFLIB only supports one instance with energy type \"VIEW\" at a time."));
               }
               mySelfView_ = this;
               generateEnergyView();
               break;
            }
            case Parameter::TABLES: {
               if(mySelfTables_ != NULL) {
                  throw(RuntimeError("Singleton policy: MRFLIB only supports one instance with energy type \"TABLES\" at a time."));
               }
               mySelfTables_ = this;
               generateEnergyTables();
               break;
            }
            case Parameter::TL1: {
               generateEnergyTL1();
               break;
            }
            case Parameter::TL2: {
               generateEnergyTL2();
               break;
            }
            case Parameter::WEIGHTEDTABLE: {
               generateEnergyWeightedTable();
               break;
            }
            default: {
               throw(RuntimeError("Unknown energy type."));
            }
         }

         // initialize selected algorithm
         switch(parameter_.inferenceType_) {
            case Parameter::ICM: {
               mrf_ = new mrfLib::ICM(sizeX_, sizeY_, numLabels_, energy_);
               break;
            }
            case Parameter::EXPANSION: {
               mrf_ = new mrfLib::Expansion(sizeX_, sizeY_, numLabels_, energy_);
               break;
            }
            case Parameter::SWAP: {
               mrf_ = new mrfLib::Swap(sizeX_, sizeY_, numLabels_, energy_);
               break;
            }
            case Parameter::MAXPRODBP: {
               mrf_ = new mrfLib::MaxProdBP(sizeX_, sizeY_, numLabels_, energy_);
               break;
            }
            case Parameter::TRWS: {
               mrf_ = new mrfLib::TRWS(sizeX_, sizeY_, numLabels_, energy_);
               break;
            }
            case Parameter::BPS: {
               mrf_ = new mrfLib::BPS(sizeX_, sizeY_, numLabels_, energy_);
               break;
            }
            default: {
               throw(RuntimeError("Unknown inference type."));
            }
         }

         mrf_->initialize();
         mrf_->clearAnswer();
      }

      template<class GM>
      MRFLIB<GM>::~MRFLIB() {
         if(parameter_.energyType_ == Parameter::VIEW) {
            mySelfView_ = NULL;
         } else if(parameter_.energyType_ == Parameter::TABLES) {
            mySelfTables_ = NULL;
         }
         if(mrf_) {
            delete mrf_;
         }
         if(D_) {
            delete[] D_;
         }
         if(V_) {
            delete[] V_;
         }
         if(hCue_) {
            delete[] hCue_;
         }
         if(vCue_) {
            delete[] vCue_;
         }
         if(data_) {
            delete data_;
         }
         if(smooth_) {
            delete smooth_;
         }
         if(energy_) {
            delete energy_;
         }
      }

      template<class GM>
      inline std::string MRFLIB<GM>::name() const {
         return "MRFLIB";
      }

      template<class GM>
      inline const typename MRFLIB<GM>::GraphicalModelType& MRFLIB<GM>::graphicalModel() const {
         return gm_;
      }

      template<class GM>
      inline InferenceTermination MRFLIB<GM>::infer() {
         EmptyVisitorType visitor;
         return this->infer(visitor);
      }

      template<class GM>
      template<class VISITOR>
      inline InferenceTermination MRFLIB<GM>::infer(VISITOR & visitor) {
         visitor.begin(*this);

         float t;

         // ICM, Expansion and Swap converge
         if((parameter_.inferenceType_ == Parameter::ICM) || (parameter_.inferenceType_ == Parameter::EXPANSION) || (parameter_.inferenceType_ == Parameter::SWAP)) {
            for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
               ValueType totalEnergyOld = mrf_->totalEnergy();
               mrf_->optimize(1, t);
               if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) {
                  break;
               }
               if(fabs(totalEnergyOld - mrf_->totalEnergy()) < OPENGM_FLOAT_TOL) {
                  break;
               }
            }
         // TRWS supports lower bound and thus early termination is possible if trwsTolerance is set
         } else if((parameter_.inferenceType_ == Parameter::TRWS) && (parameter_.trwsTolerance_ > 0.0)) {
            for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
               mrf_->optimize(1, t);
               if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) {
                  break;
               }
               if(fabs(mrf_->totalEnergy() - mrf_->lowerBound()) / std::max(fabs(mrf_->totalEnergy()), 1.0) < parameter_.trwsTolerance_) {
                  break;
               }
            }
         } else {

            for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
               mrf_->optimize(1, t);
               if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ) {
                  break;
               }
            }
         }

         visitor.end(*this);

         OPENGM_ASSERT(valueCheck());
         return NORMAL;
      }

      template<class GM>
      inline InferenceTermination MRFLIB<GM>::arg(std::vector<LabelType>& arg, const size_t& n) const {
         if(n > 1) {
            return UNKNOWN;
         }
         else { 
            arg.resize( gm_.numberOfVariables());
            for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
               arg[grid_(i)] = mrf_->getLabel(i);
            }
            return NORMAL;
         }
      }

      template<class GM>
      inline typename GM::ValueType MRFLIB<GM>::bound() const {
         if(parameter_.inferenceType_ == Parameter::TRWS) {
            return mrf_->lowerBound();
         } else {
            return Inference<GM, opengm::Minimizer>::bound();
         }
      }

      template<class GM>
      inline typename GM::ValueType MRFLIB<GM>::value() const {
         return mrf_->totalEnergy();
      }

      template<class GM>
      inline void MRFLIB<GM>::generateEnergyView() {
         generateFirstOrderFactorLookupTable_();
         generateSecondOrderFactorLookupTables_();

         data_ = new mrfLib::DataCost(firstOrderFactorViewAccess);
         smooth_ = new mrfLib::SmoothnessCost(secondOrderFactorViewAccess);
         energy_ = new mrfLib::EnergyFunction(data_,smooth_);
      }

      template<class GM>
      inline void MRFLIB<GM>::generateEnergyTables() {
         copyFactorValues();

         data_ = new mrfLib::DataCost(firstOrderFactorTablesAccess);
         smooth_ = new mrfLib::SmoothnessCost(secondOrderFactorTablesAccess);
         energy_ = new mrfLib::EnergyFunction(data_,smooth_);
      }

      template<class GM>
      inline void MRFLIB<GM>::generateEnergyTL1() {
         OPENGM_ASSERT(truncatedAbsoluteDifferenceFactors());

         ValueType t = 0.0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               t = getT(i, 1);
            }
         }
         std::cout << "T: " << t << std::endl;
         OPENGM_ASSERT(sameT(t, 1));

         setD();
         data_ = new mrfLib::DataCost(D_);

         setWeights();

         if(equalWeights()) {
            if(sizeX_ > 1) {
               std::cout << "lambda: " << hCue_[0] << std::endl;
               smooth_ = new mrfLib::SmoothnessCost(1, t, hCue_[0]);
            } else {
               std::cout << "lambda: " << vCue_[0] << std::endl;
               smooth_ = new mrfLib::SmoothnessCost(1, t, vCue_[0]);
            }
         } else {
            smooth_ = new mrfLib::SmoothnessCost(1, t, 1.0, hCue_, vCue_);
         }

         energy_ = new mrfLib::EnergyFunction(data_,smooth_);
      }

      template<class GM>
      inline void MRFLIB<GM>::generateEnergyTL2() {
         OPENGM_ASSERT(truncatedSquaredDifferenceFactors());

         ValueType t = 0.0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               t = getT(i, 2);
            }
         }
         std::cout << "T: " << t << std::endl;
         OPENGM_ASSERT(sameT(t, 2));

         setD();
         data_ = new mrfLib::DataCost(D_);

         setWeights();

         if(equalWeights()) {
            if(sizeX_ > 1) {
               std::cout << "lambda: " << hCue_[0] << std::endl;
               smooth_ = new mrfLib::SmoothnessCost(2, t, hCue_[0]);
            } else {
               std::cout << "lambda: " << vCue_[0] << std::endl;
               smooth_ = new mrfLib::SmoothnessCost(2, t, vCue_[0]);
            }
         } else {
            smooth_ = new mrfLib::SmoothnessCost(2, t, 1.0, hCue_, vCue_);
         }

         energy_ = new mrfLib::EnergyFunction(data_,smooth_);
      }

      template<class GM>
      inline void MRFLIB<GM>::generateEnergyWeightedTable() {
         setD();
         setV();
         setWeightedTableWeights();

         // check if energy table is symmetric. This is required by mrf.
         if(!symmetricEnergyTable()) {
            throw(RuntimeError("Energy table has to be symmetric."));
         }

         // check if all energy tables are Equal with respect to a scaling factor
         if(!sameEnergyTable()) {
            throw(RuntimeError("All energy tables have to be equal with respect to a scaling factor."));
         }

         data_ = new mrfLib::DataCost(D_);
         smooth_ = new mrfLib::SmoothnessCost(V_, hCue_, vCue_);
         energy_ = new mrfLib::EnergyFunction(data_,smooth_);
      }


      template<class GM>
      inline void MRFLIB<GM>::setD() {
         D_ = new mrfLib::MRF::CostVal[gm_.numberOfVariables() * numLabels_];
         for(IndexType i = 0; i < gm_.numberOfVariables() * numLabels_; i++) {
            D_[i] = 0;
         }

         for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
            IndexType gmVariableIndex = grid_(i);
            for(IndexType j = 0; j < gm_.numberOfFactors(gmVariableIndex); j++) {
               IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, j);
               if(gm_.numberOfVariables(gmFactorIndex) == 1) {
                  for(IndexType k = 0; k < numLabels_; k++) {
                     D_[i * numLabels_ + k] += gm_[gmFactorIndex](&k);
                  }
               }
            }
         }
      }

      template<class GM>
      inline void MRFLIB<GM>::setV() {
         V_ = new mrfLib::MRF::CostVal[numLabels_ * numLabels_];

         IndexType gmVariableIndex = grid_(0);
         for(IndexType i = 0; i < gm_.numberOfFactors(gmVariableIndex); i++) {
            IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, i);
            if(gm_.numberOfVariables(gmFactorIndex) == 2) {
               for(IndexType j = 0; j < numLabels_; j++) {
                  for(IndexType k = 0; k < numLabels_; k++) {
                     IndexType index[] = {j, k};
                     V_[(j * numLabels_) + k] = gm_[gmFactorIndex](index);
                  }
               }
            }
         }
      }

      template<class GM>
      inline void MRFLIB<GM>::setWeightedTableWeights() {
         hCue_ = new mrfLib::MRF::CostVal[sizeX_ * sizeY_];
         vCue_ = new mrfLib::MRF::CostVal[sizeX_ * sizeY_];

         for(IndexType i = 0; i < sizeX_; i++) {
            for(IndexType j = 0; j < sizeY_; j++) {
               IndexType gmVariableIndex = grid_(i, j);
               for(IndexType k = 0; k < gm_.numberOfFactors(gmVariableIndex); k++) {
                  IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, k);
                  if(gm_.numberOfVariables(gmFactorIndex) == 2) {
                     if((i < sizeX_ - 1) && gm_.variableFactorConnection(grid_(i + 1, j), gmFactorIndex)) {
                        // set hCue
                        hCue_[i + (j * sizeX_)] = 0;
                        for(IndexType l = 0; l < numLabels_; l++) {
                           IndexType m;
                           for(m = 0; m < numLabels_; m++) {
                              IndexType index[] = {l, m};
                              if((V_[(l * numLabels_) + m] != 0) && (gm_[gmFactorIndex](index) != 0)) {
                                 hCue_[i + (j * sizeX_)] = gm_[gmFactorIndex](index) / V_[(l * numLabels_) + m];
                                 break;
                              }
                           }
                           if(m != numLabels_) {
                              break;
                           }
                        }
                     } else if((j < sizeY_ -1 ) && gm_.variableFactorConnection(grid_(i, j + 1), gmFactorIndex)) {
                        // set vCue
                        vCue_[i + (j * sizeX_)] = 0;
                        for(IndexType l = 0; l < numLabels_; l++) {
                           IndexType m;
                           for(m = 0; m < numLabels_; m++) {
                              IndexType index[] = {l, m};
                              if((V_[(l * numLabels_) + m] != 0) && (gm_[gmFactorIndex](index) != 0)) {
                                 vCue_[i + (j * sizeX_)] = gm_[gmFactorIndex](index) / V_[(l * numLabels_) + m];
                                 break;
                              }
                           }
                           if(m != numLabels_) {
                              break;
                           }
                        }
                     } else if((i != 0) && gm_.variableFactorConnection(grid_(i - 1, j), gmFactorIndex)) {
                        continue;
                     } else if((j != 0) && gm_.variableFactorConnection(grid_(i, j - 1), gmFactorIndex)) {
                        continue;
                     } else {
                        // should never be reached as this can only happen if gm_ is not a grid which is checked during construction
                        OPENGM_ASSERT(false);
                     }
                  }
               }
            }
         }
      }

      template<class GM>
      inline bool MRFLIB<GM>::hasSameLabelNumber() const {
         for(IndexType i = 1; i < gm_.numberOfVariables(); i++) {
            if(gm_.numberOfLabels(i) != numLabels_) {
               return false;
            }
         }
         return true;
      }

      template<class GM>
      inline bool MRFLIB<GM>::sameEnergyTable() const {
         const double eps = OPENGM_FLOAT_TOL;
         for(IndexType i = 0; i < sizeX_ - 1; i++) {
            for(IndexType j = 0; j < sizeY_ - 1; j++) {
               IndexType gmVariableIndex = grid_(i, j);
               for(IndexType k = 0; k < gm_.numberOfFactors(gmVariableIndex); k++) {
                  IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, k);
                  if(gm_.numberOfVariables(gmFactorIndex) == 2) {
                     if(gm_.variableFactorConnection(grid_(i + 1, j), gmFactorIndex)) {
                        for(IndexType l = 0; l < numLabels_; l++) {
                           for(IndexType m = 0; m < numLabels_; m++) {
                              IndexType index[] = {l, m};
                              if((fabs(V_[(l * numLabels_) + m] * hCue_[i + (j * sizeX_)]) - gm_[gmFactorIndex](index)) > eps) {
                                 return false;
                              }
                           }
                        }
                     } else if(gm_.variableFactorConnection(grid_(i, j + 1), gmFactorIndex)) {
                        for(IndexType l = 0; l < numLabels_; l++) {
                           for(IndexType m = 0; m < numLabels_; m++) {
                              IndexType index[] = {l, m};
                              if(fabs((V_[(l * numLabels_) + m] * vCue_[i + (j * sizeX_)]) - gm_[gmFactorIndex](index)) > eps) {
                                 return false;
                              }
                           }
                        }
                     } else if((i != 0) && gm_.variableFactorConnection(grid_(i - 1, j), gmFactorIndex)) {
                        continue;
                     } else if((j != 0) && gm_.variableFactorConnection(grid_(i, j - 1), gmFactorIndex)) {
                        continue;
                     } else {
                        // should never be reached as this can only happen if gm_ is not a grid which is checked during construction
                        OPENGM_ASSERT(false);
                     }
                  }
               }
            }
         }
         return true;
      }

      template<class GM>
      inline bool MRFLIB<GM>::symmetricEnergyTable() const {
         for (IndexType i = 0; i < numLabels_; i++) {
            for (IndexType j = i; j < numLabels_; j++) {
               if (V_[(i * numLabels_) + j] != V_[(j * numLabels_) + i]) {
                   return false;
               }
            }
         }
         return true;
      }

      template<class GM>
      inline bool MRFLIB<GM>::valueCheck() const {
         std::vector<LabelType> state;
         arg(state);
         if(fabs(value() - gm_.evaluate(state)) < OPENGM_FLOAT_TOL) {
            return true;
         } else {
            return false;
         }
      }

      template<class GM>
      inline void MRFLIB<GM>::generateFirstOrderFactorLookupTable_() {
         firstOrderFactorLookupTable_.resize(gm_.numberOfVariables());
         for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
            IndexType gmVariableIndex = grid_(i);
            for(IndexType j = 0; j < gm_.numberOfFactors(gmVariableIndex); j++) {
               IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, j);
               if(gm_.numberOfVariables(gmFactorIndex) == 1) {
                  firstOrderFactorLookupTable_[i].push_back(gmFactorIndex);
               }
            }
         }
      }

      template<class GM>
      inline void MRFLIB<GM>::generateSecondOrderFactorLookupTables_() {
         horizontalSecondOrderFactorLookupTable_.resize(gm_.numberOfVariables());
         verticalSecondOrderFactorLookupTable_.resize(gm_.numberOfVariables());

         for(IndexType i = 0; i < sizeX_; i++) {
            for(IndexType j = 0; j < sizeY_; j++) {
               IndexType gmVariableIndex = grid_(i, j);
               for(IndexType k = 0; k < gm_.numberOfFactors(gmVariableIndex); k++) {
                  IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, k);
                  if(gm_.numberOfVariables(gmFactorIndex) == 2) {
                     if((i < sizeX_ - 1) && gm_.variableFactorConnection(grid_(i + 1, j), gmFactorIndex)) {
                        horizontalSecondOrderFactorLookupTable_[i + (j * sizeX_)].push_back(gmFactorIndex);
                     } else if((j < sizeY_ -1 ) && gm_.variableFactorConnection(grid_(i, j + 1), gmFactorIndex)) {
                        verticalSecondOrderFactorLookupTable_[i + (j * sizeX_)].push_back(gmFactorIndex);
                     } else if((i != 0) && gm_.variableFactorConnection(grid_(i - 1, j), gmFactorIndex)) {
                        continue;
                     } else if((j != 0) && gm_.variableFactorConnection(grid_(i, j - 1), gmFactorIndex)) {
                        continue;
                     } else {
                        // should never be reached as this can only happen if gm_ is not a grid which is checked during construction
                        OPENGM_ASSERT(false);
                     }
                  }
               }
            }
         }
      }

      template<class GM>
      inline mrfLib::MRF::CostVal MRFLIB<GM>::firstOrderFactorViewAccess(int pix, int i) {
         mrfLib::MRF::CostVal result = 0.0;

         typename std::vector<IndexType>::const_iterator iter;
         for(iter = mySelfView_->firstOrderFactorLookupTable_[pix].begin(); iter != mySelfView_->firstOrderFactorLookupTable_[pix].end(); iter++) {
            result += mySelfView_->gm_[*iter](&i);
         }
         return result;
      }

      template<class GM>
      inline mrfLib::MRF::CostVal MRFLIB<GM>::secondOrderFactorViewAccess(int pix1, int pix2, int i, int j) {
         OPENGM_ASSERT(pix1 != pix2);
         IndexType index[] = {i, j};

         mrfLib::MRF::CostVal result = 0.0;
         typename std::vector<IndexType>::const_iterator iter;
         if(pix1 < pix2) {
            if(pix2 == pix1 + 1) {
               // horizontal connection
               for(iter = mySelfView_->horizontalSecondOrderFactorLookupTable_[pix1].begin(); iter != mySelfView_->horizontalSecondOrderFactorLookupTable_[pix1].end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            } else {
               // vertical connection
               for(iter = mySelfView_->verticalSecondOrderFactorLookupTable_[pix1].begin(); iter != mySelfView_->verticalSecondOrderFactorLookupTable_[pix1].end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            }
         } else {
            if(pix1 == pix2 + 1) {
               // horizontal connection
               for(iter = mySelfView_->horizontalSecondOrderFactorLookupTable_[pix2].begin(); iter != mySelfView_->horizontalSecondOrderFactorLookupTable_[pix2].end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            } else {
               // vertical connection
               for(iter = mySelfView_->verticalSecondOrderFactorLookupTable_[pix2].begin(); iter != mySelfView_->verticalSecondOrderFactorLookupTable_[pix2].end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            }
         }
         return result;
      }

      template<class GM>
      inline bool MRFLIB<GM>::truncatedAbsoluteDifferenceFactors() const {
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
      inline bool MRFLIB<GM>::truncatedSquaredDifferenceFactors() const {
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
      inline typename GM::ValueType MRFLIB<GM>::getT(IndexType factor, ValueType e) const {
         OPENGM_ASSERT(gm_.numberOfVariables(factor) == 2);

         IndexType index1[] = {0, 1};
         IndexType index0[] = {0, numLabels_-1};

         return gm_[factor](index0)/gm_[factor](index1);
         /*
         //ValueType value = gm_[factor](index);
         ValueType w = gm_[factor](index1);
         for(size_t i = 1; i < numLabels_; i++) {
            index1[1] = i;
            index0[1] = i-1;
            //std::cout << "value: " << value << std::endl;
            if(fabs(gm_[factor](index1) - gm_[factor](index0)) < OPENGM_FLOAT_TOL) {
               return i;
            }
         }
         return numLabels_;
         */
      }

      template<class GM>
      inline bool MRFLIB<GM>::sameT(ValueType T, ValueType e) const {
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               if(fabs(getT(i, e) - T) < OPENGM_FLOAT_TOL) {
                  continue;
               } else {
                  return false;
               }
            }
         }
         return true;
      }

      template<class GM>
      inline void MRFLIB<GM>::setWeights() {
         hCue_ = new mrfLib::MRF::CostVal[sizeX_ * sizeY_];
         vCue_ = new mrfLib::MRF::CostVal[sizeX_ * sizeY_];

         for(IndexType i = 0; i < sizeX_; i++) {
            for(IndexType j = 0; j < sizeY_; j++) {
               IndexType gmVariableIndex = grid_(i, j);
               for(IndexType k = 0; k < gm_.numberOfFactors(gmVariableIndex); k++) {
                  IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, k);
                  if(gm_.numberOfVariables(gmFactorIndex) == 2) {
                     if((i < sizeX_ - 1) && gm_.variableFactorConnection(grid_(i + 1, j), gmFactorIndex)) {
                        // set hCue
                        IndexType index[] = {0, 1};
                        hCue_[i + (j * sizeX_)] = gm_[gmFactorIndex](index);
                     } else if((j < sizeY_ -1 ) && gm_.variableFactorConnection(grid_(i, j + 1), gmFactorIndex)) {
                        // set vCue
                        IndexType index[] = {0, 1};
                        vCue_[i + (j * sizeX_)] = gm_[gmFactorIndex](index);
                     } else if((i != 0) && gm_.variableFactorConnection(grid_(i - 1, j), gmFactorIndex)) {
                        continue;
                     } else if((j != 0) && gm_.variableFactorConnection(grid_(i, j - 1), gmFactorIndex)) {
                        continue;
                     } else {
                        // should never be reached as this can only happen if gm_ is not a grid which is checked during construction
                        OPENGM_ASSERT(false);
                     }
                  }
               }
            }
         }
      }

      template<class GM>
      inline bool MRFLIB<GM>::equalWeights() const {
         mrfLib::MRF::CostVal lambda;
         if(sizeX_ > 1) {
            lambda = hCue_[0];
         } else {
            lambda = vCue_[0];
         }
         for(IndexType i = 0; i < sizeX_; i++) {
            for(IndexType j = 0; j < sizeY_; j++) {
               if((i < sizeX_ - 1) && (fabs(hCue_[i + (j * sizeX_)] - lambda) > OPENGM_FLOAT_TOL)) {
                  return false;
               } else if((j < sizeY_ -1 ) && (fabs(vCue_[i + (j * sizeX_)] - lambda) > OPENGM_FLOAT_TOL)) {
                  return false;
               }
            }
         }
         return true;
      }

      template<class GM>
      inline void MRFLIB<GM>::copyFactorValues() {
         // first order
         firstOrderFactorValues.resize(gm_.numberOfVariables() * numLabels_, 0.0);
         for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
            IndexType gmVariableIndex = grid_(i);
            for(IndexType j = 0; j < gm_.numberOfFactors(gmVariableIndex); j++) {
               IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, j);
               if(gm_.numberOfVariables(gmFactorIndex) == 1) {
                  for(IndexType k = 0; k < numLabels_; k++) {
                     firstOrderFactorValues[(i * numLabels_) + k] += gm_[gmFactorIndex](&k);
                  }
               }
            }
         }

         // second order
         const size_t size = 2 * gm_.numberOfVariables() * numLabels_ * numLabels_;
         secondOrderFactorValues.resize(size, 0.0);

         for(IndexType i = 0; i < sizeX_; i++) {
            for(IndexType j = 0; j < sizeY_; j++) {
               IndexType gmVariableIndex = grid_(i, j);
               for(IndexType k = 0; k < gm_.numberOfFactors(gmVariableIndex); k++) {
                  IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, k);
                  if(gm_.numberOfVariables(gmFactorIndex) == 2) {
                     if((i < sizeX_ - 1) && gm_.variableFactorConnection(grid_(i + 1, j), gmFactorIndex)) {
                        // down
                        for(IndexType l = 0; l < numLabels_; l++) {
                           for(IndexType m = 0; m < numLabels_; m++) {
                              IndexType index[] = {l, m};
                              IndexType linearIndex = (l * numLabels_) + m;
                              secondOrderFactorValues[((2 * (i + (j * sizeX_))) + down_) * numLabels_ * numLabels_ + linearIndex] += gm_[gmFactorIndex](index);
                           }
                        }
                     } else if((j < sizeY_ -1 ) && gm_.variableFactorConnection(grid_(i, j + 1), gmFactorIndex)) {
                        // right
                        for(IndexType l = 0; l < numLabels_; l++) {
                           for(IndexType m = 0; m < numLabels_; m++) {
                              IndexType index[] = {l, m};
                              IndexType linearIndex = (l * numLabels_) + m;
                              secondOrderFactorValues[((2 * (i + (j * sizeX_))) + right_) * numLabels_ * numLabels_ + linearIndex] += gm_[gmFactorIndex](index);
                           }
                        }
                     } else if((i != 0) && gm_.variableFactorConnection(grid_(i - 1, j), gmFactorIndex)) {
                        // up
                        continue;
                     } else if((j != 0) && gm_.variableFactorConnection(grid_(i, j - 1), gmFactorIndex)) {
                        // left
                        continue;
                     } else {
                        // should never be reached as this can only happen if gm_ is not a grid which is checked during construction
                        OPENGM_ASSERT(false);
                     }
                  }
               }
            }
         }
      }

      template<class GM>
      inline mrfLib::MRF::CostVal MRFLIB<GM>::firstOrderFactorTablesAccess(int pix, int i) {
         return mySelfTables_->firstOrderFactorValues[(pix * mySelfTables_->numLabels_) + i];
      }

      template<class GM>
      inline mrfLib::MRF::CostVal MRFLIB<GM>::secondOrderFactorTablesAccess(int pix1, int pix2, int i, int j) {
         OPENGM_ASSERT(pix1 != pix2);

         IndexType linearIndex = (i * mySelfTables_->numLabels_) + j;

         if(pix1 < pix2) {
            if(pix2 == pix1 + 1) {
               // down
               return mySelfTables_->secondOrderFactorValues[((2 * pix1) + mySelfTables_->down_) * mySelfTables_->numLabels_ * mySelfTables_->numLabels_ + linearIndex];
            } else {
               // right
               return mySelfTables_->secondOrderFactorValues[((2 * pix1) + mySelfTables_->right_) * mySelfTables_->numLabels_ * mySelfTables_->numLabels_ + linearIndex];
            }
         } else {
            if(pix1 == pix2 + 1) {
               // up
               return mySelfTables_->secondOrderFactorValues[((2 * pix2) + mySelfTables_->down_) * mySelfTables_->numLabels_ * mySelfTables_->numLabels_ + linearIndex];
            } else {
               // left
               return mySelfTables_->secondOrderFactorValues[((2 * pix2) + mySelfTables_->right_) * mySelfTables_->numLabels_ * mySelfTables_->numLabels_ + linearIndex];
            }
         }
      }

   } // namespace external
} // namespace opengm
#endif /* MRFLIB_HXX_ */
