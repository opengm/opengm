#ifndef GRANTE_HXX_
#define GRANTE_HXX_

#include <sstream>

#include "opengm/inference/inference.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

// grante includes
#include "FactorGraph.h"
#include "BruteForceExactInference.h"
#include "BeliefPropagation.h"
#include "DiffusionInference.h"
#include "SimulatedAnnealingInference.h"

namespace opengm {
   namespace external {

   /// GRANTE
   /// GRANTE inference algorithm class
   /// \ingroup inference
   /// \ingroup external_inference
   ///
   //    GRANTE
   /// - cite :[?]
   /// - Maximum factor order : ?
   /// - Maximum number of labels : ?
   /// - Restrictions : ?
   /// - Convergent : ?
   template<class GM>
   class GRANTE : public Inference<GM, opengm::Minimizer> {
   public:
      typedef GM                              GraphicalModelType;
      typedef opengm::Minimizer               AccumulationType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef visitors::VerboseVisitor<GRANTE<GM> > VerboseVisitorType;
      typedef visitors::EmptyVisitor<GRANTE<GM> >   EmptyVisitorType;
      typedef visitors::TimingVisitor<GRANTE<GM> >  TimingVisitorType;

      ///Parameter
      struct Parameter {
         enum InferenceType {BRUTEFORCE, BP, DIFFUSION, SA};
         InferenceType inferenceType_;
         /// number of iterations for Belief Propagation method
         size_t numberOfIterations_;
         // Used to define the threshold for stopping condition for Belief Propagation method
         double tolerance_;
         // Print iteration statistics for Belief Propagation method
         bool verbose_;

         // Select MessageSchedule type for Belief Propagation method
         Grante::BeliefPropagation::MessageSchedule BPSchedule_;

         // Number of simulated annealing distributions
         unsigned int SASteps_;
         // Initial Boltzmann temperature for simulated annealing.
         double SAT0_;
         // Final Boltzmann temperature for simulated annealing.
         double SATfinal_;

         /// \brief Constructor
         Parameter() : inferenceType_(BRUTEFORCE), numberOfIterations_(100),
               tolerance_(1.0e-6), verbose_(false),
               BPSchedule_(Grante::BeliefPropagation::Sequential), SASteps_(100),
               SAT0_(10.0), SATfinal_(0.05) {
         }
      };
      // construction
      GRANTE(const GraphicalModelType& gm, const Parameter& para);
      // destruction
      ~GRANTE();
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
      ValueType value_;
      ValueType lowerBound_;

      Grante::FactorGraphModel* granteModel_;
      Grante::FactorGraph* granteGraph_;
      Grante::InferenceMethod* granteInferenceMethod_;
      std::vector<unsigned int> granteState_;
      std::vector<Grante::FactorDataSource*> granteDataSourceCollector_;

      bool sanityCheck(ValueType value) const;

      void groupFactors(std::vector<std::vector<IndexType> >& groupedFactors) const;
      void groupFactorTypes(const std::vector<std::vector<IndexType> >& groupedFactors, std::vector<std::vector<IndexType> >& groupedFactorTypes) const;

      template<class T, class OBJECT>
      struct InsertFunctor {
          void operator()(const T v) {
             (*object_)[index_] = static_cast<double>(v);
             index_++;
          }
          int index_;
          OBJECT* object_;
      };
   };

   template<class GM>
   GRANTE<GM>::GRANTE(const typename GRANTE<GM>::GraphicalModelType& gm, const Parameter& para)
      : gm_(gm), parameter_(para), granteModel_(new Grante::FactorGraphModel()), granteGraph_(NULL),
        granteInferenceMethod_(NULL) {

      // group factors
      std::vector<std::vector<IndexType> > groupedFactors;
      groupFactors(groupedFactors);

      // group grante factor types
      std::vector<std::vector<IndexType> > groupedFactorTypes;
      groupFactorTypes(groupedFactors, groupedFactorTypes);

      // add factor types
      for(size_t i = 0; i < groupedFactorTypes.size(); i++) {
         // create unique factor type name
         std::stringstream ss;
         ss << i;
         std::string name = ss.str();

         // select representative factor
         IndexType currentFactor = groupedFactors[groupedFactorTypes[i][0]][0];

         // set number of labels for each variable
         std::vector<unsigned int> cardinalities;
         for(IndexType j = 0; j < gm_[currentFactor].numberOfVariables(); j++) {
            cardinalities.push_back(static_cast<unsigned int>(gm_.numberOfLabels(gm_[currentFactor].variableIndex(j))));
         }

         // add factor type to model
         granteModel_->AddFactorType(new Grante::FactorType(name, cardinalities, std::vector<double>()));
      }

      // get number of labels for all variables
      std::vector<unsigned int> cardinalities;
      for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
         cardinalities.push_back(gm_.numberOfLabels(i));
      }
      // create factor graph from model
      granteGraph_ = new Grante::FactorGraph(granteModel_, cardinalities);

      // add factors to graph
      for(size_t i = 0; i < groupedFactorTypes.size(); i++) {
         // create unique factor type name
         std::stringstream ss;
         ss << i;
         std::string name = ss.str();
         // get factor type by name
         Grante::FactorType* currentFactorType = granteModel_->FindFactorType(name);
         // add all factors with same factor type
         OPENGM_ASSERT(groupedFactorTypes[i].size() > 0);
         for(size_t j = 0; j < groupedFactorTypes[i].size(); j++) {
            OPENGM_ASSERT(groupedFactors[groupedFactorTypes[i][j]].size() > 0);
            if(groupedFactors[groupedFactorTypes[i][j]].size() == 1) {
               // single factor, no shared data
               IndexType currentFactor = groupedFactors[groupedFactorTypes[i][j]][0];
               // determine connected variables
               std::vector<unsigned int> var_index;
               for(IndexType k = 0; k < gm_[currentFactor].numberOfVariables(); k++) {
                  var_index.push_back(static_cast<unsigned int>(gm_[currentFactor].variableIndex(k)));
               }
               // copy data
               std::vector<double> data(currentFactorType->ProdCardinalities());
               ViewFunction<GM> function = gm_[currentFactor];
               InsertFunctor<ValueType, std::vector<double> > inserter;
               inserter.index_ = 0;
               inserter.object_ = &data;
               function.forAllValuesInOrder(inserter);

               // crate factor
               Grante::Factor* factor = new Grante::Factor(currentFactorType, var_index, data);
               // add factor to graph (graph takes ownership)
               granteGraph_->AddFactor(factor);
            } else {
               // multiple factors with shared data
               IndexType currentFactor = groupedFactors[groupedFactorTypes[i][j]][0];
               // create shared factor data
               std::vector<double> data(currentFactorType->ProdCardinalities());
               ViewFunction<GM> function = gm_[currentFactor];
               InsertFunctor<ValueType, std::vector<double> > inserter;
               inserter.index_ = 0;
               inserter.object_ = &data;
               function.forAllValuesInOrder(inserter);
               Grante::FactorDataSource* currentDataSource = new Grante::FactorDataSource(data);
               granteDataSourceCollector_.push_back(currentDataSource);
               // add all factors with shared data
               for(size_t k = 0; k < groupedFactors[groupedFactorTypes[i][j]].size(); k++) {
                  currentFactor = groupedFactors[groupedFactorTypes[i][j]][k];
                  // determine connected variables
                  std::vector<unsigned int> var_index;
                  for(IndexType l = 0; l < gm_[currentFactor].numberOfVariables(); l++) {
                     var_index.push_back(static_cast<unsigned int>(gm_[currentFactor].variableIndex(l)));
                  }
                  // crate factor
                  Grante::Factor* factor = new Grante::Factor(currentFactorType, var_index, currentDataSource);
                  // add factor to graph (graph takes ownership)
                  granteGraph_->AddFactor(factor);
               }
            }
         }
      }

      // Perform forward map: update energies upon model change
      granteGraph_->ForwardMap();

      // set inference method
      switch(parameter_.inferenceType_) {
      case Parameter::BRUTEFORCE : {
         granteInferenceMethod_ = new Grante::BruteForceExactInference(granteGraph_);
         break;
      }
      case Parameter::BP : {
         granteInferenceMethod_ = new Grante::BeliefPropagation(granteGraph_, parameter_.BPSchedule_);
         static_cast<Grante::BeliefPropagation*>(granteInferenceMethod_)->SetParameters(parameter_.verbose_, parameter_.numberOfIterations_, parameter_.tolerance_);
         break;
      }
      case Parameter::DIFFUSION : {
         granteInferenceMethod_ = new Grante::DiffusionInference(granteGraph_);
         static_cast<Grante::DiffusionInference*>(granteInferenceMethod_)->SetParameters(parameter_.verbose_, parameter_.numberOfIterations_, parameter_.tolerance_);
         break;
      }
      case Parameter::SA : {
         granteInferenceMethod_ = new Grante::SimulatedAnnealingInference(granteGraph_, parameter_.verbose_);
         static_cast<Grante::SimulatedAnnealingInference*>(granteInferenceMethod_)->SetParameters(parameter_.SASteps_, parameter_.SAT0_, parameter_.SATfinal_);
         break;
      }
      default: {
         throw(RuntimeError("Unknown inference type"));
      }
      }
      // set initial value and lower bound
      AccumulationType::neutral(value_);
      AccumulationType::ineutral(lowerBound_);
   }

   template<class GM>
   GRANTE<GM>::~GRANTE() {
      if(granteInferenceMethod_) {
         delete granteInferenceMethod_;
      }
      for(size_t i = 0; i < granteDataSourceCollector_.size(); i++) {
         delete granteDataSourceCollector_[i];
      }
      if(granteGraph_) {
         delete granteGraph_;
      }
      if(granteModel_) {
         delete granteModel_;
      }
    }

    template<class GM>
    inline std::string GRANTE<GM>::name() const {
       return "GRANTE";
    }

    template<class GM>
    inline const typename GRANTE<GM>::GraphicalModelType& GRANTE<GM>::graphicalModel() const {
       return gm_;
    }

    template<class GM>
    inline InferenceTermination GRANTE<GM>::infer() {
       EmptyVisitorType visitor;
       return this->infer(visitor);
    }

    template<class GM>
    template<class VISITOR>
    inline InferenceTermination GRANTE<GM>::infer(VISITOR & visitor) {
       visitor.begin(*this);
       value_ = granteInferenceMethod_->MinimizeEnergy(granteState_);
       visitor.end(*this);
       return NORMAL;
    }

    template<class GM>
    inline InferenceTermination GRANTE<GM>::arg(std::vector<LabelType>& arg, const size_t& n) const {
       arg.resize(gm_.numberOfVariables());
       for(IndexType i = 0; i < gm_.numberOfVariables(); i++) {
          arg[i] = static_cast<LabelType>(granteState_[i]);
       }
       return NORMAL;
    }

    template<class GM>
    inline typename GM::ValueType GRANTE<GM>::bound() const {
       return lowerBound_;
    }

    template<class GM>
    inline typename GM::ValueType GRANTE<GM>::value() const {
       //sanity check
       OPENGM_ASSERT(sanityCheck(value_));
       return value_;
    }

    template<class GM>
    inline bool GRANTE<GM>::sanityCheck(ValueType value) const {
       if(granteState_.size() > 0) {
          std::vector<LabelType> result;
          arg(result);
          return fabs(value - gm_.evaluate(result)) < OPENGM_FLOAT_TOL;
       } else {
          ValueType temp;
          AccumulationType::neutral(temp);
          return value == temp;
       }
    }

    template<class GM>
    inline void GRANTE<GM>::groupFactors(std::vector<std::vector<IndexType> >& groupedFactors) const {
       // Factors are grouped by function index and the cardinalities of the connected variables.
       groupedFactors.clear();
       typedef std::map<std::pair<IndexType, std::vector<LabelType> >, size_t> Map;
       Map lookupTable;
       for(IndexType i = 0; i < gm_.numberOfFactors(); i++) {
          IndexType currentFunctionIndex = gm_[i].functionIndex();
          std::vector<LabelType> currentCardinalities;
          for(IndexType j = 0; j < gm_[i].numberOfVariables(); j++) {
             currentCardinalities.push_back(gm_.numberOfLabels(gm_[i].variableIndex(j)));
          }
          std::pair<IndexType, std::vector<LabelType> > currentKey(currentFunctionIndex, currentCardinalities);
          typename Map::const_iterator iter = lookupTable.find(currentKey);
          if(iter != lookupTable.end()) {
             groupedFactors[iter->second].push_back(i);
          } else {
             std::vector<IndexType> newVec(1, i);
             groupedFactors.push_back(newVec);
             lookupTable[currentKey] = groupedFactors.size() - 1;
          }
       }
    }

    template<class GM>
    inline void GRANTE<GM>::groupFactorTypes(const std::vector<std::vector<IndexType> >& groupedFactors, std::vector<std::vector<IndexType> >& groupedFactorTypes) const {
       groupedFactorTypes.clear();
       typedef std::map<std::vector<LabelType>, size_t > Map;
       Map lookupTable;
       for(IndexType i = 0; i < groupedFactors.size(); i++) {
          IndexType currentNumberOfVariables = gm_[groupedFactors[i][0]].numberOfVariables();
          std::vector<LabelType> currentCardinalities;
          for(IndexType j = 0; j < currentNumberOfVariables; j++) {
             currentCardinalities.push_back(gm_.numberOfLabels(gm_[groupedFactors[i][0]].variableIndex(j)));
          }
          typename Map::const_iterator iter = lookupTable.find(currentCardinalities);
          if(iter != lookupTable.end()) {
             groupedFactorTypes[iter->second].push_back(i);
          } else {
             std::vector<IndexType> newVec(1, i);
             groupedFactorTypes.push_back(newVec);
             lookupTable[currentCardinalities] = groupedFactorTypes.size() - 1;
          }
       }
    }

   } // namespace external
} // namespace opengm

#endif /* GRANTE_HXX_ */
