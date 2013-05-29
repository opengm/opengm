#ifndef FASTPD_HXX_
#define FASTPD_HXX_

#include "opengm/inference/inference.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitor.hxx"

#include "Fast_PD.h"

namespace opengm {
   namespace external {

      /// FastPD
      /// FastPD inference algorithm class
      /// \ingroup inference
      /// \ingroup external_inference
      ///
      //    FastPD
      /// - cite :[?]
      /// - Maximum factor order : ?
      /// - Maximum number of labels : ?
      /// - Restrictions : ?
      /// - Convergent : ?
      template<class GM>
      class FastPD : public Inference<GM, opengm::Minimizer> {
      public:
         typedef GM                              GraphicalModelType;
         typedef opengm::Minimizer               AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef EmptyVisitor<FastPD<GM> > EmptyVisitorType;
         typedef VerboseVisitor<FastPD<GM> > VerboseVisitorType;
         typedef TimingVisitor<FastPD<GM> > TimingVisitorType;

         ///Parameter
         struct Parameter {
            /// \brief Constructor
            Parameter() : numberOfIterations_(1000) {
            }
            /// number of iterations
            size_t numberOfIterations_;
         };
         // construction
         FastPD(const GraphicalModelType& gm, const Parameter& para = Parameter());
         // destruction
         ~FastPD();
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
         Parameter parameter_;
         fastPDLib::CV_Fast_PD* pdInference_;
         ValueType value_;
         ValueType lowerBound_;

         fastPDLib::CV_Fast_PD::Real* labelCosts_;
         int numPairs_;
         int* pairs_;
         fastPDLib::CV_Fast_PD::Real* distance_;
         fastPDLib::CV_Fast_PD::Real* weights_;

         bool sameNumberOfLabels() const;
         void setLabelCosts();
         void getNumPairs();
         void setPairs();
         void setDistance();
         void setWeights();
         bool sameEnergyTable();
      };

      template<class GM>
      FastPD<GM>::FastPD(const typename FastPD<GM>::GraphicalModelType& gm, const Parameter& para)
         : gm_(gm), parameter_(para), pdInference_(NULL), labelCosts_(NULL),
           numPairs_(0), pairs_(NULL), distance_(NULL), weights_(NULL) {
         OPENGM_ASSERT(sameNumberOfLabels());
         OPENGM_ASSERT(gm_.maxFactorOrder(2));

         setLabelCosts();
         getNumPairs();
         setPairs();
         setDistance();
         setWeights();

         if(sameEnergyTable()==false){ 
              throw std::runtime_error("Error: Tables are not proportional");
         }

         pdInference_ = new fastPDLib::CV_Fast_PD(
               gm_.numberOfVariables(),
               gm_.numberOfLabels(0),
               labelCosts_,
               numPairs_,
               pairs_,
               distance_,
               parameter_.numberOfIterations_,
               weights_
               );

         // set initial value and lower bound
         AccumulationType::neutral(value_);
         AccumulationType::ineutral(lowerBound_);
      }

      template<class GM>
      FastPD<GM>::~FastPD() {
         if(pdInference_) {
            delete pdInference_;
         }
         if(labelCosts_) {
            delete[] labelCosts_;
         }
         if(pairs_) {
            delete[] pairs_;
         }
         if(distance_) {
            delete[] distance_;
         }
         if(weights_) {
            delete[] weights_;
         }
      }

      template<class GM>
      inline std::string FastPD<GM>::name() const {
         return "FastPD";
      }

      template<class GM>
      inline const typename FastPD<GM>::GraphicalModelType& FastPD<GM>::graphicalModel() const {
         return gm_;
      }

      template<class GM>
      inline InferenceTermination FastPD<GM>::infer() {
         EmptyVisitorType visitor;
         return this->infer(visitor);
      }

      template<class GM>
      template<class VISITOR>
      inline InferenceTermination FastPD<GM>::infer(VISITOR & visitor) {
         visitor.begin(*this);
         // TODO check for possible visitor injection method
         // TODO this is slow, check if fast_pd allows energy extraction
         if(pdInference_ != NULL) {
            pdInference_->run();
            std::vector<LabelType> result;
            arg(result);
            value_ = gm_.evaluate(result);
         }
         visitor.end(*this);
         return NORMAL;
      }

      template<class GM>
      inline InferenceTermination FastPD<GM>::arg(std::vector<LabelType>& arg, const size_t& n) const {
         OPENGM_ASSERT(pdInference_ != NULL);

         arg.resize(gm_.numberOfVariables());
         for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
            arg[i] = pdInference_->_pinfo[i].label;
         }
         return NORMAL;
      }

      template<class GM>
      inline typename GM::ValueType FastPD<GM>::bound() const {
         return lowerBound_;
      }

      template<class GM>
      inline typename GM::ValueType FastPD<GM>::value() const {
         return value_;
      }

      template<class GM>
      inline bool FastPD<GM>::sameNumberOfLabels() const {
         OPENGM_ASSERT(gm_.numberOfVariables() > 0);
         LabelType numLabels = gm_.numberOfLabels(0);
         for(IndexType i = 1; i < gm_.numberOfVariables(); i++) {
            if(gm_.numberOfLabels(i) != numLabels) {
               return false;
            }
         }
         return true;
      }

      template<class GM>
      inline void FastPD<GM>::setLabelCosts() {
         labelCosts_ = new fastPDLib::CV_Fast_PD::Real[gm_.numberOfVariables() * gm_.numberOfLabels(0)];
         for(IndexType i = 0; i < gm_.numberOfVariables() * gm_.numberOfLabels(0); i++) {
            labelCosts_[i] = 0;
         }

         for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
            for(IndexType j = 0; j < gm_.numberOfFactors(i); j++) {
               IndexType gmFactorIndex = gm_.factorOfVariable(i, j);
               if(gm_.numberOfVariables(gmFactorIndex) == 1) {
                  for(IndexType k = 0; k < gm_.numberOfLabels(0); k++) {
                     labelCosts_[k * gm_.numberOfVariables() + i ] += gm_[gmFactorIndex](&k);
                  }
               }
            }
         }
      }

      template<class GM>
      inline void FastPD<GM>::getNumPairs() {
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               numPairs_++;
            }
         }
      }

      template<class GM>
      inline void FastPD<GM>::setPairs() {
         pairs_ = new int[numPairs_ * 2];
         int currentPair = 0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               pairs_[currentPair * 2] = gm_[i].variableIndex(0);
               pairs_[(currentPair * 2) + 1] = gm_[i].variableIndex(1);
               currentPair++;
            }
         }
      }

      template<class GM>
      inline void FastPD<GM>::setDistance() {
         distance_ = new fastPDLib::CV_Fast_PD::Real[gm_.numberOfLabels(0) * gm_.numberOfLabels(0)];
         for(IndexType k = 0; k < gm_.numberOfLabels(0); k++) {
            for(IndexType l = 0; l < gm_.numberOfLabels(0); l++) {
               distance_[(l * gm_.numberOfLabels(0)) + k] = 0;
            }
         }
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               for(IndexType k = 0; k < gm_.numberOfLabels(0); k++) {
                  for(IndexType l = 0; l < gm_.numberOfLabels(0); l++) {
                     IndexType index[] = {k, l};
                     distance_[(l * gm_.numberOfLabels(0)) + k] = gm_[i](index);
                  }
               }
               break;
            }
         }
      }

      template<class GM>
      inline void FastPD<GM>::setWeights() {
         weights_ = new fastPDLib::CV_Fast_PD::Real[numPairs_];
         int currentPair = 0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               OPENGM_ASSERT(currentPair < numPairs_);
               IndexType k;
               for(k = 0; k < gm_.numberOfLabels(0); k++) {
                  IndexType l;
                  for(l = 0; l < gm_.numberOfLabels(0); l++) {
                     IndexType index[] = {k, l};
                     if((gm_[i](index) != 0) && (distance_[(l * gm_.numberOfLabels(0)) + k] != 0)) {
                        double currentWeight = static_cast<double>(gm_[i](index)) / static_cast<double>(distance_[(l * gm_.numberOfLabels(0)) + k]);
                        weights_[currentPair] = static_cast<fastPDLib::CV_Fast_PD::Real>(currentWeight);
                        if(fabs(currentWeight - static_cast<double>(weights_[currentPair])) > OPENGM_FLOAT_TOL) {
                           throw(RuntimeError("Function not supported"));
                        }
                        currentPair++;
                        break;
                     }
                  }
                  if(l != gm_.numberOfLabels(0)) {
                     break;
                  }
               }
               if(k == gm_.numberOfLabels(0)) {
                  weights_[currentPair] = 0;
                  currentPair++;
               }
            }
         }
         OPENGM_ASSERT(currentPair == numPairs_);
      }

      template<class GM>
      inline bool FastPD<GM>::sameEnergyTable() {
         int currentPair = 0;
         for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
            if(gm_.numberOfVariables(i) == 2) {
               for(IndexType k = 0; k < gm_.numberOfLabels(0); k++) {
                  for(IndexType l = 0; l < gm_.numberOfLabels(0); l++) {
                     IndexType index[] = {k, l};
                     if(fabs(gm_[i](index) - (distance_[(l * gm_.numberOfLabels(0)) + k] * weights_[currentPair])) > OPENGM_FLOAT_TOL) {
                        return false;
                     }
                  }
               }
               currentPair++;
            }
         }
         OPENGM_ASSERT(currentPair == numPairs_);
         return true;
      }

   } // namespace external
} // namespace opengm

#endif /* FASTPD_HXX_ */
