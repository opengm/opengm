#ifndef OPENGM_EXTERNAL_SRMP_HXX_
#define OPENGM_EXTERNAL_SRMP_HXX_

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/utilities/indexing.hxx>

#include <srmp/SRMP.h>
#include <srmp/FactorTypes/PottsType.h>
#include <srmp/FactorTypes/GeneralType.h>

namespace opengm {
namespace external {

/*********************
 * class definitions *
 *********************/
template<class GM>
class SRMP : public Inference<GM, opengm::Minimizer> {
public:
   typedef GM                              GraphicalModelType;
   typedef opengm::Minimizer               AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<SRMP<GM> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<SRMP<GM> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<SRMP<GM> >  TimingVisitorType;

   struct Parameter : public srmpLib::Energy::Options {
      Parameter() : srmpLib::Energy::Options(), BLPRelaxation_(false),
            FullRelaxation_(false), FullRelaxationMethod_(0),
            FullDualRelaxation_(false), FullDualRelaxationMethod_(0) {
         // disable verbose mode per default
         verbose = false;
      }

      bool BLPRelaxation_;
      bool FullRelaxation_;
      int  FullRelaxationMethod_;         // method=0: add all possible pairs (A,B) with B \subset A, with the following exception:
                                          //           if there exists factor C with B\subset C \subset A then don't add (A,B)
                                          // method=1: move all costs to outer factors (converting them to general types first, if they are not already of these types).
                                          //           Then run method=0 and remove unnecesary edges (i.e. those that do not affect the relaxation).
                                          //           Note, all edges outgoing from outer factors will be kepts.
                                          // method=2: similar to method=1, but all edges {A->B, B->C} are replaced with {A->B, A->C} (so this results in a two-layer graph).
                                          //
                                          // Note, method=1 and method=2 merge duplicate factors while method=0 does not. For this reason the relaxation may be tighther.
                                          // (If there are no duplicate factors then the resulting relaxation should be the same in all three cases).
                                          //
                                          // method=3: run method=2 and then create a new Energy instance with unary and pairwise terms in which nodes correspond
                                          // to outer factors of the original energy, and pairwise terms with {0,+\infty} costs enforce consistency between them.
      bool FullDualRelaxation_;
      int  FullDualRelaxationMethod_;     // FullDualRelaxationMethod_ has the same meaning as in srmpLib::Energy::Options::sort_flag.
   };

   // construction
   SRMP(const GraphicalModelType& gm, const Parameter para = Parameter());
   // destruction
   ~SRMP();
   // query
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   // inference
   InferenceTermination infer();
   template<class VISITOR>
   InferenceTermination infer(VISITOR & visitor);
   InferenceTermination arg(std::vector<LabelType>& arg, const size_t& n = 1) const;
   typename GM::ValueType bound() const;
   typename GM::ValueType value() const;
private:
   const GraphicalModelType& gm_;
   Parameter parameter_;

   ValueType constTerm_;
   ValueType value_;
   ValueType lowerBound_;

   srmpLib::Energy::Options srmpOptions_;
   srmpLib::Energy          srmpSolver_;

   std::vector<srmpLib::PottsFactorType*> pottsFactorList_; // list of created potts functions which must be deleted when ~SRMP() is called
   std::vector<srmpLib::GeneralFactorType*> generalFactorList_; // list of created general functions which must be deleted when ~SRMP() is called

   void addUnaryFactor(const IndexType FactorID);
   void addPairwiseFactor(const IndexType FactorID);
   void addPottsFactor(const IndexType FactorID);
   void addGeneralFactor(const IndexType FactorID);
};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template<class GM>
inline SRMP<GM>::SRMP(const GraphicalModelType& gm, const Parameter para)
: gm_(gm), parameter_(para), constTerm_(0.0), value_(), lowerBound_(),
  srmpOptions_(), srmpSolver_(gm_.numberOfVariables()), pottsFactorList_(),
  generalFactorList_() {
   // set states of variables
   for(IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
      srmpSolver_.AddNode(gm_.numberOfLabels(i));
   }

   // set factors
   for(IndexType i = 0; i < gm_.numberOfFactors(); ++i) {
      if(gm_[i].numberOfVariables() == 0) {
         // constant factor not supported by srmp, hence handle constant term external from srmp solver
         LabelType l = 0;
         constTerm_ += gm_[i](&l);
      } else if(gm_[i].numberOfVariables() == 1) {
         // add unary factor
         addUnaryFactor(i);
      } else if(gm_[i].numberOfVariables() == 2) {
         if(gm_[i].numberOfLabels(0) == gm_[i].numberOfLabels(1) && gm_[i].isPotts()) {
            // add potts factor
            // srmp potts type does not support potts functions with more than
            // two variables or with different number of labels
            addPottsFactor(i);
         } else {
            // add pairwise factor
            addPairwiseFactor(i);
         }
      } else {
         // general factor
         // TODO srmp provides other function types which can be used instead of general type for some factors (SharedPairwiseType, PatternType, PairwiseDualType)
         addGeneralFactor(i);
      }
   }

   // set options
   srmpOptions_.method = parameter_.method;
   srmpOptions_.iter_max = parameter_.iter_max;
   srmpOptions_.time_max = parameter_.time_max;
   srmpOptions_.eps = parameter_.eps;
   srmpOptions_.compute_solution_period = parameter_.compute_solution_period;
   srmpOptions_.print_times = parameter_.print_times;
   srmpOptions_.sort_flag = parameter_.sort_flag;
   srmpOptions_.verbose = parameter_.verbose;
   srmpOptions_.TRWS_weighting = parameter_.TRWS_weighting;

   // set initial value and lower bound
   AccumulationType::neutral(value_);
   AccumulationType::ineutral(lowerBound_);
}

template<class GM>
inline SRMP<GM>::~SRMP() {
   for(size_t i = 0; i < pottsFactorList_.size(); ++i) {
      delete pottsFactorList_[i];
   }
   for(size_t i = 0; i < generalFactorList_.size(); ++i) {
      delete generalFactorList_[i];
   }
}

template<class GM>
inline std::string SRMP<GM>::name() const {
   return "SRMP";
}

template<class GM>
inline const typename SRMP<GM>::GraphicalModelType& SRMP<GM>::graphicalModel() const {
   return gm_;
}

template<class GM>
inline InferenceTermination SRMP<GM>::infer() {
   EmptyVisitorType visitor;
   return this->infer(visitor);
}

template<class GM>
template<class VISITOR>
inline InferenceTermination SRMP<GM>::infer(VISITOR & visitor) {
   visitor.begin(*this);

   if (parameter_.BLPRelaxation_) {
      srmpSolver_.SetMinimalEdges();
   } else if (parameter_.FullRelaxation_) {
      srmpSolver_.SetFullEdges(parameter_.FullRelaxationMethod_);
   } else if (parameter_.FullDualRelaxation_) {
      srmpSolver_.SetFullEdgesDual(parameter_.FullDualRelaxationMethod_);
   }

   // call solver
   lowerBound_ = srmpSolver_.Solve(srmpOptions_);
   std::vector<LabelType> l;
   arg(l);
   value_ = gm_.evaluate(l);

   visitor.end(*this);
   return NORMAL;
}

template<class GM>
inline InferenceTermination SRMP<GM>::arg(std::vector<LabelType>& arg, const size_t& n) const {
   if(n > 1) {
      return UNKNOWN;
   }
   else {
      arg.resize(gm_.numberOfVariables());
      for(IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
         arg[i] = srmpSolver_.GetSolution(i);
      }
      return NORMAL;
   }
}

template<class GM>
inline typename GM::ValueType SRMP<GM>::bound() const {
   return lowerBound_ + constTerm_;
}

template<class GM>
inline typename GM::ValueType SRMP<GM>::value() const {
   return value_;
   //return value_ + constTerm_;
}

template<class GM>
inline void SRMP<GM>::addUnaryFactor(const IndexType FactorID) {
   double* values = new double[gm_[FactorID].numberOfLabels(0)];
   LabelType label = 0;
   for(LabelType i = 0; i < gm_[FactorID].numberOfLabels(0); ++i) {
      values[i] = static_cast<double>(gm_[FactorID](&label));
      ++label;
   }
   srmpSolver_.AddUnaryFactor(static_cast<srmpLib::Energy::NodeId>(gm_[FactorID].variableIndex(0)), values);
   delete[] values;
}

template<class GM>
inline void SRMP<GM>::addPairwiseFactor(const IndexType FactorID) {
   double* values = new double[gm_[FactorID].numberOfLabels(0) * gm_[FactorID].numberOfLabels(1)];
   LabelType labeling[2] = {0, 0};
   for(LabelType i = 0; i < gm_[FactorID].numberOfLabels(0); ++i) {
      labeling[0] = i;
      for(LabelType j = 0; j < gm_[FactorID].numberOfLabels(1); ++j) {
         labeling[1] = j;
         values[(i * gm_[FactorID].numberOfLabels(1)) + j] = static_cast<double>(gm_[FactorID](labeling));
      }
   }
   srmpSolver_.AddPairwiseFactor(static_cast<srmpLib::Energy::NodeId>(gm_[FactorID].variableIndex(0)), static_cast<srmpLib::Energy::NodeId>(gm_[FactorID].variableIndex(1)), values);
   delete[] values;
}

template<class GM>
inline void SRMP<GM>::addPottsFactor(const IndexType FactorID) {
   ValueType valueEqual;
   ValueType valueNotEqual;

   LabelType labeling[2] = {0, 0};
   valueEqual = gm_[FactorID](labeling);
   for(IndexType j = 0; j < 2; ++j) {
      if(gm_[FactorID].numberOfLabels(j) > 1) {
         labeling[j] = 1;
         break;
      }
   }
   valueNotEqual = gm_[FactorID](labeling);

   srmpLib::PottsFactorType* pottsFactor = new srmpLib::PottsFactorType;
   pottsFactorList_.push_back(pottsFactor);

   // srmp potts type uses 0.0 as equal value, hence shift values
   double lambda = valueNotEqual - valueEqual;
   constTerm_ += valueEqual;

   srmpLib::Energy::NodeId nodes[2] = {static_cast<srmpLib::Energy::NodeId>(gm_[FactorID].variableIndex(0)), static_cast<srmpLib::Energy::NodeId>(gm_[FactorID].variableIndex(1))};

   srmpSolver_.AddFactor(2, nodes, &lambda, pottsFactor);
}

template<class GM>
inline void SRMP<GM>::addGeneralFactor(const IndexType FactorID) {
   double* values = new double[gm_[FactorID].size()];

   ShapeWalkerSwitchedOrder<typename FactorType::ShapeIteratorType> shapeWalker(gm_[FactorID].shapeBegin(), gm_[FactorID].dimension());
   for(size_t i = 0; i < gm_[FactorID].size(); ++i) {
      values[i] = gm_[FactorID](shapeWalker.coordinateTuple().begin());
      ++shapeWalker;
   }

   srmpLib::Energy::NodeId* nodes = new srmpLib::Energy::NodeId[gm_[FactorID].numberOfVariables()];
   for(IndexType i = 0; i < gm_[FactorID].numberOfVariables(); ++i) {
      nodes[i] = static_cast<srmpLib::Energy::NodeId>(gm_[FactorID].variableIndex(i));
   }

   srmpLib::GeneralFactorType* generalFactor = new srmpLib::GeneralFactorType;

   srmpSolver_.AddFactor(gm_[FactorID].numberOfVariables(), nodes, values, generalFactor);

   delete[] nodes;
   delete[] values;
}

} // namespace external
} // namespace opengm

#endif /* OPENGM_EXTERNAL_SRMP_HXX_ */
