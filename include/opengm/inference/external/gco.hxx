#ifndef GCO_HXX_
#define GCO_HXX_

#include <map>

#include "opengm/inference/inference.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

#include "GCoptimization.h"

namespace opengm {
   namespace external {

      /// GCOLIB
      /// GCOLIB inference algorithm class
      /// \ingroup inference
      /// \ingroup external_inference
      ///
      //    GCOLIB
      /// - cite :[?]
      /// - Maximum factor order :2
      /// - Maximum number of labels : \f$\infty\f$
      /// - Restrictions : ?
      /// - Convergent : ?
      template<class GM>
      class GCOLIB : public Inference<GM, opengm::Minimizer> {
      public:
         typedef GM                              GraphicalModelType;
         typedef opengm::Minimizer               AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef visitors::VerboseVisitor<GCOLIB<GM> > VerboseVisitorType;
         typedef visitors::EmptyVisitor<GCOLIB<GM> >   EmptyVisitorType;
         typedef visitors::TimingVisitor<GCOLIB<GM> >  TimingVisitorType;
         ///Parameter
         struct Parameter {
            /// possible optimization algorithms for GCOLIB
            enum InferenceType {EXPANSION, SWAP};
            /// possible energy types for GCOLIB
            enum EnergyType {VIEW, TABLES, WEIGHTEDTABLE};
            /// selected optimization algorithm
            InferenceType inferenceType_;
            /// selected energy type
            EnergyType energyType_;
            /// number of iterations
            size_t numberOfIterations_;
            /// Enable random label order. By default, the labels for the swap and expansion algorithms are visited in not random order, but random label visitation might give better results.
            bool randomLabelOrder_;
            /// Use adaptive cycles for alpha-expansion
            bool useAdaptiveCycles_;
            /// Do not use grid structure
            bool doNotUseGrid_;
            Parameter(const InferenceType inferenceType = EXPANSION, const EnergyType energyType = VIEW, const size_t numberOfIterations = 1000)
               : inferenceType_(inferenceType), energyType_(energyType), numberOfIterations_(numberOfIterations), randomLabelOrder_(false), useAdaptiveCycles_(false), doNotUseGrid_(false) {
            }
         };
         // construction
         GCOLIB(const GraphicalModelType& gm, const Parameter& para);
         // destruction
         ~GCOLIB();
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
         typedef gcoLib::GCoptimization::EnergyTermType EnergyTermType;

         const GraphicalModelType& gm_;
         Parameter parameter_;
         bool isGrid_;
         IndexType sizeX_;
         IndexType sizeY_;
         const IndexType numNodes_;
         const LabelType numLabels_;
         marray::Matrix<size_t> grid_;
         gcoLib::GCoptimizationGeneralGraph* GCOGeneralGraph_;
         gcoLib::GCoptimizationGridGraph* GCOGridGraph_;

         // required for energy type weighted table
         void generateEnergyWeightedTable();
         EnergyTermType* D_;
         EnergyTermType* V_;
         EnergyTermType* hCue_;
         EnergyTermType* vCue_;
         void setD();
         void setV();
         void setWeightedTableWeights();
         bool hasSameLabelNumber() const;
         bool sameEnergyTable() const;
         bool symmetricEnergyTable() const;

         // required for energy type view
         void generateEnergyView();
         static GCOLIB<GM>* mySelfView_;
         std::vector<std::vector<IndexType> > firstOrderFactorLookupTable_;
         std::vector<std::vector<IndexType> > horizontalSecondOrderFactorLookupTable_;
         std::vector<std::vector<IndexType> > verticalSecondOrderFactorLookupTable_;
         std::map<std::pair<IndexType, IndexType>, std::vector<IndexType> > generalSecondOrderFactorLookupTable_;
         void generateFirstOrderFactorLookupTable();
         void generateSecondOrderFactorLookupTables();
         static EnergyTermType firstOrderFactorViewAccess(int pix, int i);
         static EnergyTermType secondOrderFactorViewGridAccess(int pix1, int pix2, int i, int j);
         // uses a std::map as lookup table ==> no constant access time.
         static EnergyTermType secondOrderFactorViewGeneralAccess(int pix1, int pix2, int i, int j);

         // required for energy type tables
         // only supported if graphical model is a grid.
         // A general graphical model would require to much memory to allocate all tables.
         void generateEnergyTables();
         static GCOLIB<GM>* mySelfTables_;
         std::vector<EnergyTermType> firstOrderFactorValues;
         std::vector<EnergyTermType> secondOrderFactorGridValues;
         static const IndexType right_ = 0;
         static const IndexType down_ = 1;
         void copyFactorValues();
         static EnergyTermType firstOrderFactorTablesAccess(int pix, int i);
         static EnergyTermType secondOrderFactorTablesGridAccess(int pix1, int pix2, int i, int j);

         bool valueCheck() const;
      };

      template<class GM>
      GCOLIB<GM>* GCOLIB<GM>::mySelfView_ = NULL;
      template<class GM>
      GCOLIB<GM>* GCOLIB<GM>::mySelfTables_ = NULL;

      template<class GM>
      GCOLIB<GM>::GCOLIB(const typename GCOLIB::GraphicalModelType& gm, const Parameter& para)
         : gm_(gm), parameter_(para), numNodes_(gm_.numberOfVariables()), numLabels_(gm_.numberOfLabels(0)), GCOGeneralGraph_(NULL),
           GCOGridGraph_(NULL), D_(NULL), V_(NULL), hCue_(NULL), vCue_(NULL) {

         // check label number
         if(!hasSameLabelNumber()) {
            throw(RuntimeError("GCOLIB only supports graphical models where each variable has the same number of states."));
         }

         // check for grid structure
         if(para.doNotUseGrid_){
            isGrid_ = false;
         }
         else{
            isGrid_ = gm_.isGrid(grid_);
         }

         // create graph
         if(isGrid_) {
            std::cout <<"GRID"<<std::endl;
            sizeX_ = grid_.shape(0);
            sizeY_ = grid_.shape(1);
            GCOGridGraph_ = new gcoLib::GCoptimizationGridGraph(sizeX_, sizeY_, numLabels_);
            GCOGridGraph_->setLabelOrder(parameter_.randomLabelOrder_);
         } else {
            std::cout <<"NO GRID"<<std::endl;
            GCOGeneralGraph_ = new gcoLib::GCoptimizationGeneralGraph(numNodes_, numLabels_);
            GCOGeneralGraph_->setLabelOrder(parameter_.randomLabelOrder_);
         }

         // generate energy function
         switch(parameter_.energyType_) {
            case Parameter::VIEW: {
               if(mySelfView_ != NULL) {
                  throw(RuntimeError("Singleton policy: GCOLIB only supports one instance with energy type \"VIEW\" at a time."));
               }
               mySelfView_ = this;
               generateEnergyView();
               break;
            }
            case Parameter::TABLES: {
               if(!isGrid_) {
                  throw(RuntimeError("GCOLIB only supports energy type \"TABLES\" if model is a grid."));
               }
               if(mySelfTables_ != NULL) {
                  throw(RuntimeError("Singleton policy: GCOLIB only supports one instance with energy type \"TABLES\" at a time."));
               }
               mySelfTables_ = this;
               generateEnergyTables();
               break;
            }
            case Parameter::WEIGHTEDTABLE: {
               if(!isGrid_) {
                  throw(RuntimeError("GCOLIB only supports energy type \"WEIGHTEDTABLE\" if model is a grid."));
               }
               generateEnergyWeightedTable();
               break;
            }
            default: {
               throw(RuntimeError("Unknown energy type."));
            }
         }
      }

      template<class GM>
      GCOLIB<GM>::~GCOLIB() {
         std::cout <<"~~"<<std::endl;
         if(parameter_.energyType_ == Parameter::VIEW) {
            mySelfView_ = NULL;
         } else if(parameter_.energyType_ == Parameter::TABLES) {
            mySelfTables_ = NULL;
         }
         if(GCOGeneralGraph_) {
            delete GCOGeneralGraph_;
         }
         if(GCOGridGraph_) {
            delete GCOGridGraph_;
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
      }

      template<class GM>
      inline std::string GCOLIB<GM>::name() const {
         return "GCOLIB";
      }

      template<class GM>
      inline const typename GCOLIB<GM>::GraphicalModelType& GCOLIB<GM>::graphicalModel() const {
         return gm_;
      }

      template<class GM>
      inline InferenceTermination GCOLIB<GM>::infer() {
         EmptyVisitorType visitor;
         return this->infer(visitor);
      }

      template<class GM>
      template<class VISITOR>
      inline InferenceTermination GCOLIB<GM>::infer(VISITOR & visitor) {
         visitor.begin(*this);

         if(GCOGeneralGraph_) {
            // Expansion and Swap converge
            if(parameter_.inferenceType_ == Parameter::EXPANSION) {
               if(parameter_.useAdaptiveCycles_) {
                  for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
                     ValueType totalEnergyOld = GCOGeneralGraph_->compute_energy();
                     ValueType totalEnergyNew = GCOGeneralGraph_->expansion(-1);
                     if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
                        break;
                     }
                     if(fabs(totalEnergyOld - totalEnergyNew) < OPENGM_FLOAT_TOL) {
                        break;
                     }
                  }
               } else {
                  for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
                     ValueType totalEnergyOld = GCOGeneralGraph_->compute_energy();
                     ValueType totalEnergyNew = GCOGeneralGraph_->expansion(1);
                     if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
                        break;
                     }
                     if(fabs(totalEnergyOld - totalEnergyNew) < OPENGM_FLOAT_TOL) {
                        break;
                     }
                  }
               }
            } else {
               for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
                  ValueType totalEnergyOld = GCOGeneralGraph_->compute_energy();
                  ValueType totalEnergyNew = GCOGeneralGraph_->swap(1);
                  if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
                     break;
                  }
                  if(fabs(totalEnergyOld - totalEnergyNew) < OPENGM_FLOAT_TOL) {
                     break;
                  }
               }
            }
         } else {
            // Expansion and Swap converge
            if(parameter_.inferenceType_ == Parameter::EXPANSION) {
               if(parameter_.useAdaptiveCycles_) {
                  for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
                     ValueType totalEnergyOld = GCOGridGraph_->compute_energy();
                     ValueType totalEnergyNew = GCOGridGraph_->expansion(-1);
                     if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
                        break;
                     }
                     if(fabs(totalEnergyOld - totalEnergyNew) < OPENGM_FLOAT_TOL) {
                        break;
                     }
                  }
               } else {
                  for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
                     ValueType totalEnergyOld = GCOGridGraph_->compute_energy();
                     ValueType totalEnergyNew = GCOGridGraph_->expansion(1);
                     if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
                        break;
                     }
                     if(fabs(totalEnergyOld - totalEnergyNew) < OPENGM_FLOAT_TOL) {
                        break;
                     }
                  }
               }
            } else {
               for (size_t i = 0; i <parameter_.numberOfIterations_; i++) {
                  ValueType totalEnergyOld = GCOGridGraph_->compute_energy();
                  ValueType totalEnergyNew = GCOGridGraph_->swap(1);
                  if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
                     break;
                  }
                  if(fabs(totalEnergyOld - totalEnergyNew) < OPENGM_FLOAT_TOL) {
                     break;
                  }
               }
            }
         }

         visitor.end(*this);

         OPENGM_ASSERT(valueCheck());
         return NORMAL;
      }

      template<class GM>
      inline InferenceTermination GCOLIB<GM>::arg(std::vector<LabelType>& arg, const size_t& n) const {
         if(n > 1) {
            return UNKNOWN;
         }
         else {
            arg.resize( gm_.numberOfVariables());
            if(GCOGridGraph_) {
               for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
                  arg[grid_(i)] = GCOGridGraph_->whatLabel(i);
               }
            } else {
               for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
                  arg[i] = GCOGeneralGraph_->whatLabel(i);
               }
            }

            return NORMAL;
         }
      }

      template<class GM>
      inline typename GM::ValueType GCOLIB<GM>::bound() const {
         return Inference<GM, opengm::Minimizer>::bound();
      }

      template<class GM>
      inline typename GM::ValueType GCOLIB<GM>::value() const {
         if(GCOGeneralGraph_) {
            return GCOGeneralGraph_->compute_energy();
         } else {
            return GCOGridGraph_->compute_energy();
         }
      }

      template<class GM>
      inline void GCOLIB<GM>::generateEnergyView() {
         generateFirstOrderFactorLookupTable();
         generateSecondOrderFactorLookupTables();

         if(isGrid_) {
            GCOGridGraph_->setDataCost(firstOrderFactorViewAccess);
            GCOGridGraph_->setSmoothCost(secondOrderFactorViewGridAccess);
         } else {
            // add edges
            for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
               if(gm_[i].numberOfVariables() == 2) {
                  IndexType a = gm_[i].variableIndex(0);
                  IndexType b = gm_[i].variableIndex(1);
                  GCOGeneralGraph_->setNeighbors(a, b);
               }
            }

            GCOGeneralGraph_->setDataCost(firstOrderFactorViewAccess);
            GCOGeneralGraph_->setSmoothCost(secondOrderFactorViewGeneralAccess);
         }
      }

      template<class GM>
      inline void GCOLIB<GM>::generateEnergyTables() {
         copyFactorValues();
         GCOGridGraph_->setDataCost(firstOrderFactorTablesAccess);
         GCOGridGraph_->setSmoothCost(secondOrderFactorTablesGridAccess);
      }

      template<class GM>
      inline void GCOLIB<GM>::generateEnergyWeightedTable() {
         setD();
         setV();

         // TODO check if this is a requirement only for mrf or also for gco.
         /*// check if energy table is symmetric. This is required by mrf.
         if(!symmetricEnergyTable()) {
            throw(RuntimeError("Energy table has to be symmetric."));
         }*/

         if(isGrid_) {
            setWeightedTableWeights();

            // check if all energy tables are Equal with respect to a scaling factor
            if(!sameEnergyTable()) {
               throw(RuntimeError("All energy tables have to be equal with respect to a scaling factor."));
            }

            GCOGridGraph_->setDataCost(D_);
            GCOGridGraph_->setSmoothCostVH(V_, vCue_, hCue_);
         } else {
            // add edges
            for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
               if(gm_[i].numberOfVariables() == 2) {
                  IndexType a = gm_[i].variableIndex(0);
                  IndexType b = gm_[i].variableIndex(1);

                  // compute weight
                  EnergyTermType weight;
                  for(IndexType l = 0; l < numLabels_; l++) {
                     IndexType m;
                     for(m = 0; m < numLabels_; m++) {
                        IndexType index[] = {l, m};
                        if((V_[(l * numLabels_) + m] != 0) && (gm_[i](index) != 0)) {
                           weight = gm_[i](index) / V_[(l * numLabels_) + m];
                           break;
                        }
                     }
                     if(m != numLabels_) {
                        break;
                     }
                  }

                  // check values
                  for(IndexType l = 0; l < numLabels_; l++) {
                     for(IndexType m = 0; m < numLabels_; m++) {
                        IndexType index[] = {l, m};
                        if(fabs((V_[(l * numLabels_) + m] * weight) - gm_[i](index)) > OPENGM_FLOAT_TOL) {
                           throw(RuntimeError("All energy tables have to be equal with respect to a scaling factor."));
                        }
                     }
                  }

                  // add edge
                  GCOGeneralGraph_->setNeighbors(a, b, weight);
               }
            }
            GCOGeneralGraph_->setDataCost(D_);
            GCOGeneralGraph_->setSmoothCost(V_);
         }

      }


      template<class GM>
      inline void GCOLIB<GM>::setD() {
         D_ = new EnergyTermType[gm_.numberOfVariables() * numLabels_];
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
      inline void GCOLIB<GM>::setV() {
         V_ = new EnergyTermType[numLabels_ * numLabels_];

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
      inline void GCOLIB<GM>::setWeightedTableWeights() {
         hCue_ = new EnergyTermType[sizeX_ * sizeY_];
         vCue_ = new EnergyTermType[sizeX_ * sizeY_];

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
      inline bool GCOLIB<GM>::hasSameLabelNumber() const {
         for(IndexType i = 1; i < gm_.numberOfVariables(); i++) {
            if(gm_.numberOfLabels(i) != numLabels_) {
               return false;
            }
         }
         return true;
      }

      template<class GM>
      inline bool GCOLIB<GM>::sameEnergyTable() const {
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
                              if(fabs((V_[(l * numLabels_) + m] * hCue_[i + (j * sizeX_)]) - gm_[gmFactorIndex](index)) > eps) {
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
      inline bool GCOLIB<GM>::symmetricEnergyTable() const {
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
      inline bool GCOLIB<GM>::valueCheck() const {
         std::vector<LabelType> state;
         arg(state);
         if(fabs(value() - gm_.evaluate(state)) < OPENGM_FLOAT_TOL) {
            return true;
         } else {
            return false;
         }
      }

      template<class GM>
      inline void GCOLIB<GM>::generateFirstOrderFactorLookupTable() {
         firstOrderFactorLookupTable_.resize(gm_.numberOfVariables());
         if(isGrid_) {
            for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
               IndexType gmVariableIndex = grid_(i);
               for(IndexType j = 0; j < gm_.numberOfFactors(gmVariableIndex); j++) {
                  IndexType gmFactorIndex = gm_.factorOfVariable(gmVariableIndex, j);
                  if(gm_.numberOfVariables(gmFactorIndex) == 1) {
                     firstOrderFactorLookupTable_[i].push_back(gmFactorIndex);
                  }
               }
            }
         } else {
            for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
               if(gm_[i].numberOfVariables() == 1) {
                  IndexType a = gm_[i].variableIndex(0);
                  firstOrderFactorLookupTable_[a].push_back(i);
               }
            }
         }
      }

      template<class GM>
      inline void GCOLIB<GM>::generateSecondOrderFactorLookupTables() {
         if(isGrid_) {
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
         } else {
            for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
               if(gm_[i].numberOfVariables() == 2) {
                  IndexType a = gm_[i].variableIndex(0);
                  IndexType b = gm_[i].variableIndex(1);
                  if(a <= b) {
                     const std::pair<IndexType, IndexType> variables(a, b);
                     generalSecondOrderFactorLookupTable_[variables].push_back(i);
                  } else {
                     const std::pair<IndexType, IndexType> variables(b, a);
                     generalSecondOrderFactorLookupTable_[variables].push_back(i);
                  }
               }
            }
         }
      }

      template<class GM>
      inline typename GCOLIB<GM>::EnergyTermType GCOLIB<GM>::firstOrderFactorViewAccess(int pix, int i) {
         EnergyTermType result = 0.0;

         typename std::vector<IndexType>::const_iterator iter;
         for(iter = mySelfView_->firstOrderFactorLookupTable_[pix].begin(); iter != mySelfView_->firstOrderFactorLookupTable_[pix].end(); iter++) {
            result += mySelfView_->gm_[*iter](&i);
         }
         return result;
      }

      template<class GM>
      inline typename GCOLIB<GM>::EnergyTermType GCOLIB<GM>::secondOrderFactorViewGridAccess(int pix1, int pix2, int i, int j) {
         OPENGM_ASSERT(pix1 != pix2);
         IndexType index[] = {i, j};

         EnergyTermType result = 0.0;
         typedef typename std::vector<IndexType>::const_iterator vecIter;
         if(pix1 < pix2) {
            if(pix2 == pix1 + 1) {
               // horizontal connection
               for(vecIter iter = mySelfView_->horizontalSecondOrderFactorLookupTable_[pix1].begin(); iter != mySelfView_->horizontalSecondOrderFactorLookupTable_[pix1].end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            } else {
               // vertical connection
               for(vecIter iter = mySelfView_->verticalSecondOrderFactorLookupTable_[pix1].begin(); iter != mySelfView_->verticalSecondOrderFactorLookupTable_[pix1].end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            }
         } else {
            if(pix1 == pix2 + 1) {
               // horizontal connection
               for(vecIter iter = mySelfView_->horizontalSecondOrderFactorLookupTable_[pix2].begin(); iter != mySelfView_->horizontalSecondOrderFactorLookupTable_[pix2].end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            } else {
               // vertical connection
               for(vecIter iter = mySelfView_->verticalSecondOrderFactorLookupTable_[pix2].begin(); iter != mySelfView_->verticalSecondOrderFactorLookupTable_[pix2].end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            }
         }
         return result;
      }

      template<class GM>
      inline typename GCOLIB<GM>::EnergyTermType GCOLIB<GM>::secondOrderFactorViewGeneralAccess(int pix1, int pix2, int i, int j) {
         OPENGM_ASSERT(pix1 != pix2);
         IndexType index[] = {i, j};

         EnergyTermType result = 0.0;
         typedef typename std::vector<IndexType>::const_iterator vecIter;
         typedef typename std::map<std::pair<IndexType, IndexType>, std::vector<IndexType> >::const_iterator mapIter;
         if(pix1 <= pix2) {
            const std::pair<IndexType, IndexType> variables(pix1, pix2);
            mapIter generalSecondOrderFactors = mySelfView_->generalSecondOrderFactorLookupTable_.find(variables);
            if(generalSecondOrderFactors != mySelfView_->generalSecondOrderFactorLookupTable_.end()) {
               for(vecIter iter = generalSecondOrderFactors->second.begin(); iter != generalSecondOrderFactors->second.end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            }
         } else {
            const std::pair<IndexType, IndexType> variables(pix2, pix1);
            mapIter generalSecondOrderFactors = mySelfView_->generalSecondOrderFactorLookupTable_.find(variables);
            if(generalSecondOrderFactors != mySelfView_->generalSecondOrderFactorLookupTable_.end()) {
               for(vecIter iter = generalSecondOrderFactors->second.begin(); iter != generalSecondOrderFactors->second.end(); iter++) {
                  result += mySelfView_->gm_[*iter](index);
               }
            }
         }

         return result;
      }

      template<class GM>
      inline void GCOLIB<GM>::copyFactorValues() {
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
         secondOrderFactorGridValues.resize(size, 0.0);

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
                              secondOrderFactorGridValues[((2 * (i + (j * sizeX_))) + down_) * numLabels_ * numLabels_ + linearIndex] += gm_[gmFactorIndex](index);
                           }
                        }
                     } else if((j < sizeY_ -1 ) && gm_.variableFactorConnection(grid_(i, j + 1), gmFactorIndex)) {
                        // right
                        for(IndexType l = 0; l < numLabels_; l++) {
                           for(IndexType m = 0; m < numLabels_; m++) {
                              IndexType index[] = {l, m};
                              IndexType linearIndex = (l * numLabels_) + m;
                              secondOrderFactorGridValues[((2 * (i + (j * sizeX_))) + right_) * numLabels_ * numLabels_ + linearIndex] += gm_[gmFactorIndex](index);
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
      inline typename GCOLIB<GM>::EnergyTermType GCOLIB<GM>::firstOrderFactorTablesAccess(int pix, int i) {
         return mySelfTables_->firstOrderFactorValues[(pix * mySelfTables_->numLabels_) + i];
      }

      template<class GM>
      inline typename GCOLIB<GM>::EnergyTermType GCOLIB<GM>::secondOrderFactorTablesGridAccess(int pix1, int pix2, int i, int j) {
         OPENGM_ASSERT(pix1 != pix2);

         IndexType linearIndex = (i * mySelfTables_->numLabels_) + j;

         if(pix1 < pix2) {
            if(pix2 == pix1 + 1) {
               // down
               return mySelfTables_->secondOrderFactorGridValues[((2 * pix1) + mySelfTables_->down_) * mySelfTables_->numLabels_ * mySelfTables_->numLabels_ + linearIndex];
            } else {
               // right
               return mySelfTables_->secondOrderFactorGridValues[((2 * pix1) + mySelfTables_->right_) * mySelfTables_->numLabels_ * mySelfTables_->numLabels_ + linearIndex];
            }
         } else {
            if(pix1 == pix2 + 1) {
               // up
               return mySelfTables_->secondOrderFactorGridValues[((2 * pix2) + mySelfTables_->down_) * mySelfTables_->numLabels_ * mySelfTables_->numLabels_ + linearIndex];
            } else {
               // left
               return mySelfTables_->secondOrderFactorGridValues[((2 * pix2) + mySelfTables_->right_) * mySelfTables_->numLabels_ * mySelfTables_->numLabels_ + linearIndex];
            }
         }
      }

   } // namespace external
} // namespace opengm

#endif /* GCO_HXX_ */
