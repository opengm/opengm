#pragma once
#ifndef OPENGM_EXTERNAL_QPBO_HXX
#define OPENGM_EXTERNAL_QPBO_HXX

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
//#include "opengm/inference/alphabetaswap.hxx"
//#include "opengm/inference/alphaexpansion.hxx"

#include "QPBO.h"

namespace opengm {
   namespace external {

      /// \brief QPBO Algorithm
      ///
      /// C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer. "Optimizing binary MRFs via extended roof duality". CVPR 2007 
      ///
      /// \ingroup inference
      /// \ingroup external_inference
      template<class GM>
      class QPBO : public Inference<GM, opengm::Minimizer> {

      public:
         typedef GM GraphicalModelType;
         typedef opengm::Minimizer AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef visitors::VerboseVisitor<QPBO<GM> > VerboseVisitorType;
         typedef visitors::TimingVisitor<QPBO<GM> > TimingVisitorType;
         typedef visitors::EmptyVisitor<QPBO<GM> > EmptyVisitorType;
    
         ///TriBool
         enum TriBool {
            TB0, TB1, TBX
         };

         ///Parameter for opengm::external::QPBO
         struct Parameter {
            /// using probeing technique
            bool useProbeing_;
            /// forcing strong persistency
            bool strongPersistency_;
            /// using improving technique
            bool useImproveing_;
            /// initial configuration for improving
            std::vector<size_t> label_;
            /// \brief constructor

            Parameter() {
               strongPersistency_ = true;
               useImproveing_ = false;
               useProbeing_ = false;
            }
         };
         // construction
         QPBO(const GraphicalModelType& gm, const Parameter para = Parameter());
         ~QPBO();
         // query
         std::string name() const;
         const GraphicalModelType& graphicalModel() const;
         // inference
         InferenceTermination infer();
         template<class VisitorType>
         InferenceTermination infer(VisitorType&);
         InferenceTermination arg(std::vector<LabelType>&, const size_t& = 1) const;
         InferenceTermination arg(std::vector<TriBool>&, const size_t& = 1) const;
         virtual typename GM::ValueType bound() const;
         virtual typename GM::ValueType value() const; 
         double partialOptimality(std::vector<bool>&) const;

      private:
         const GraphicalModelType& gm_;
         Parameter parameter_;
         kolmogorov::qpbo::QPBO<ValueType>* qpbo_;
         ValueType constTerm_;
         ValueType bound_;
         
         int* label_;
         int* defaultLabel_;

      };
      // public interface
      /// \brief Construcor
      /// \param gm graphical model
      /// \param para belief propargation paramaeter

      template<class GM>
      QPBO<GM>
      ::QPBO(
         const typename QPBO::GraphicalModelType& gm,
         const Parameter para
         )
         : gm_(gm), bound_(-std::numeric_limits<ValueType>::infinity()) {
         parameter_ = para;
         label_ = new int[gm_.numberOfVariables()];
         defaultLabel_ = new int[gm_.numberOfVariables()];
         for(size_t i = 0; i < gm_.numberOfVariables(); ++i) {
            label_[i] = -1;
            defaultLabel_[i] = 0;
         }
         if(parameter_.label_.size() > 0) {
            for(size_t i = 0; i < parameter_.label_.size(); ++i) {
               defaultLabel_[i] = parameter_.label_[i];
            }
         }
         size_t numVariables = gm_.numberOfVariables();
         size_t numPairwiseFactors = 0;
         constTerm_ = 0;
         size_t vec0[] = {0};
         size_t vec1[] = {1};
         size_t vec00[] = {0, 0};
         size_t vec01[] = {0, 1};
         size_t vec10[] = {1, 0};
         size_t vec11[] = {1, 1};
         for(size_t j = 0; j < gm_.numberOfVariables(); ++j) {
            if(gm_.numberOfLabels(j) != 2) {
               throw RuntimeError("This implementation of QPBO supports only binary variables.");
            }
         }
         for(size_t j = 0; j < gm_.numberOfFactors(); ++j) {
            if(gm_[j].numberOfVariables() == 2) {
               ++numPairwiseFactors;
            }
            else if(gm_[j].numberOfVariables() > 2) {
               throw RuntimeError("This implementation of QPBO supports only factors of order <= 2.");
            }
         }
         qpbo_ = new kolmogorov::qpbo::QPBO<ValueType > (numVariables, numPairwiseFactors); // max number of nodes & edges
         qpbo_->AddNode(numVariables); // add two nodes
         for(size_t j = 0; j < gm_.numberOfFactors(); ++j) {
            if(gm_[j].numberOfVariables() == 0) {
               ; //constTerm_+= gm_[j](0);
            }
            else if(gm_[j].numberOfVariables() == 1) {
               qpbo_->AddUnaryTerm((int) (gm_[j].variableIndex(0)), gm_[j](vec0), gm_[j](vec1));
            }
            else if(gm_[j].numberOfVariables() == 2) {
               qpbo_->AddPairwiseTerm((int) (gm_[j].variableIndex(0)), (int) (gm_[j].variableIndex(1)),
                                      gm_[j](vec00), gm_[j](vec01), gm_[j](vec10), gm_[j](vec11));
            }
         }
         qpbo_->MergeParallelEdges();
      }

      template<class GM>
      QPBO<GM>
      ::~QPBO() {
         delete label_;
         delete defaultLabel_;
		 delete qpbo_;
      }

      template<class GM>
      inline std::string
      QPBO<GM>
      ::name() const {
         return "QPBO";
      }

      template<class GM>
      inline const typename QPBO<GM>::GraphicalModelType&
      QPBO<GM>
      ::graphicalModel() const {
         return gm_;
      }

      template<class GM>
      inline InferenceTermination
      QPBO<GM>
      ::infer() {
         EmptyVisitorType v;
         return infer(v);
      }

      template<class GM>
      template<class VisitorType>
      InferenceTermination 
      QPBO<GM>::infer(VisitorType& visitor)
      { 
         visitor.begin(*this);
         qpbo_->Solve();
         if(!parameter_.strongPersistency_) {
            qpbo_->ComputeWeakPersistencies();
         } 

         bound_ = constTerm_ + 0.5 * qpbo_->ComputeTwiceLowerBound();
         
         int countUnlabel = 0;
         int *listUnlabel = new int[gm_.numberOfVariables()];
         for(size_t i = 0; i < gm_.numberOfVariables(); ++i) {
            label_[i] = qpbo_->GetLabel(i);
            if(label_[i] < 0) {
               listUnlabel[countUnlabel++] = i;
            }
         }
        
         // Initialize mapping for probe
         int *mapping = new int[gm_.numberOfVariables()];
         for(int i = 0; i < static_cast<int>(gm_.numberOfVariables()); i++) {
            mapping[i] = i * 2;
         }

         /*PROBEING*/
         if(parameter_.useProbeing_ && countUnlabel > 0) {
            typename kolmogorov::qpbo::QPBO<ValueType>::ProbeOptions options;
            //options.C = 1000000000;
            //options.dilation = 1;
            options.weak_persistencies = 1;
            //options.iters = (int)(10);//parameter_.numberOfProbeingIterations_);

            int *new_mapping = new int[gm_.numberOfVariables()];
            qpbo_->Probe(new_mapping, options);
            qpbo_->MergeMappings(gm_.numberOfVariables(), mapping, new_mapping);
            qpbo_->ComputeWeakPersistencies();
            delete new_mapping;

            // Read out entire labelling again (as weak persistencies may have changed)
            countUnlabel = 0;
            for(IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
               label_[i] = qpbo_->GetLabel(mapping[i] / 2);
               if(label_[i] < 0)
                  listUnlabel[countUnlabel++] = i;
               else
                  label_[i] = (label_[i] + mapping[i]) % 2;
            }
         }
         if(parameter_.useImproveing_ && countUnlabel > 0) {
            int *improve_order = new int[countUnlabel];

            // Set the labels to the user-defined value
            for(size_t i = 0; static_cast<int>(i) < countUnlabel; i++) {
               improve_order[i] = mapping[listUnlabel[i]] / 2;
               qpbo_->SetLabel(improve_order[i], defaultLabel_[improve_order[i]]);
            }

            // Randomize order
            for(int i = 0; i < countUnlabel - 1; ++i) {
               int j = i + (int) (((double) rand() / ((double) RAND_MAX + 1)) * (countUnlabel - i));
               OPENGM_ASSERT(j < countUnlabel);
               int k = improve_order[j];
               improve_order[j] = improve_order[i];
               improve_order[i] = k;
            }

            // Run QPBO-I
            qpbo_->Improve(countUnlabel, improve_order);
            delete improve_order;

            // Read out the labels
            for(int i = 0; i < countUnlabel; ++i) {
               label_[listUnlabel[i]] = (qpbo_->GetLabel(mapping[listUnlabel[i]] / 2) + mapping[listUnlabel[i]]) % 2;
            }
         }
       
         visitor.end(*this);
         delete mapping;
	 delete listUnlabel;
         return NORMAL;
      }

      template<class GM>
      inline InferenceTermination
      QPBO<GM>
      ::arg(std::vector<LabelType>& arg, const size_t& n) const {
         if(n > 1) {
            return UNKNOWN;
         }
         else {
            arg.resize(gm_.numberOfVariables());
            for(size_t i = 0; i < gm_.numberOfVariables(); ++i) {
               if(label_[i] < 0) arg[i] = defaultLabel_[i];
               else arg[i] = label_[i];
            }
            return NORMAL;
         }
      }

      template<class GM>
      inline InferenceTermination
      QPBO<GM>
      ::arg(std::vector<TriBool>& arg, const size_t& n) const {
         if(n > 1) {
            return UNKNOWN;
         }
         else {
            arg.resize(gm_.numberOfVariables(), TBX);
            for(int i = 0; i < gm_.numberOfVariables(); ++i) {
               if(label_[i] < 0) arg[i] = TBX;
               if(label_[i] == 0) arg[i] = TB0;
               else arg[i] = TB1;
            }
            return NORMAL;
         }
      }

      template<class GM>
      double  QPBO<GM>::partialOptimality(std::vector<bool>& opt) const
      {
         double p=0; 
         opt.resize(gm_.numberOfVariables());
         for(IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
            if(label_[i] < 0) {opt[i] = 0;}
            else              {opt[i] = 1; ++p;}
         }
         return p/gm_.numberOfVariables();
      }


      template<class GM>
      inline typename GM::ValueType
      QPBO<GM>
      ::bound() const {
         return bound_;//constTerm_ + 0.5 * qpbo_->ComputeTwiceLowerBound();		
      }

      template<class GM>
      inline typename GM::ValueType
      QPBO<GM>
      ::value() const {
         std::vector<LabelType> c;
         arg(c);
         return gm_.evaluate(c);
         //return constTerm_ + 0.5 * qpbo_->ComputeTwiceEnergy();		
      }

   } // namespace external
} // namespace opengm

#endif // #ifndef OPENGM_EXTERNAL_QPBO_HXX

