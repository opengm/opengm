#ifndef OPENGM_EXTERNAL_MPLP_HXX_
#define OPENGM_EXTERNAL_MPLP_HXX_

#include <algorithm>
#include <sstream>

#include <opengm/inference/inference.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/inference.hxx>
#include "opengm/inference/visitors/visitors.hxx"

// mplp logic
#include <cycle.h>

namespace opengm {

namespace external {

/// MPLP
/// MPLP inference algorithm class
/// \ingroup inference
/// \ingroup external_inference
///
//    MPLP
/// - cite :[?]
/// - Maximum factor order : ?
/// - Maximum number of labels : ?
/// - Restrictions : ?
/// - Convergent : ?
template<class GM>
class MPLP : public Inference<GM, opengm::Minimizer> {
public:
   typedef GM                              GraphicalModelType;
   typedef opengm::Minimizer               AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<MPLP<GM> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<MPLP<GM> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<MPLP<GM> >  TimingVisitorType;

   struct Parameter {
      /// \brief Constructor
      Parameter(
         //const size_t maxTightIter = 1000000,
         //const size_t numIter = 1000,
         //const size_t numIterLater = 20,
         const size_t maxIterLP = 1000,
         const size_t maxIterTight = 100000,
         const size_t maxIterLater = 20,
         const double maxTime = 3600,
         const double maxTimeLP = 1200,
         const size_t numClusToAddMin = 5, 
         const size_t numClusToAddMax = 20,
         const double objDelThr = 0.0002, 
         const double intGapThr = 0.0002,
         const bool UAIsettings = false,
         const bool addEdgeIntersections = true,
         const bool doGlobalDecoding = false, 
         const bool useDecimation=false,
         const bool lookForCSPs = false, 
         //const double infTime = 0.0,
         const bool logMode = false,
         const std::string& logFile = std::string(), 
         const int seed = 0,
         const std::string& inputFile = std::string(),
         const std::string& evidenceFile = std::string()
         )
         : //maxTightIter_(maxTightIter), 
           //numIter_(numIter),
           //numIterLater_(numIterLater), 
           maxIterLP_(maxIterLP),
           maxIterTight_(maxIterTight),
           maxIterLater_(maxIterLater),
           maxTime_(maxTime),
           maxTimeLP_(maxTimeLP),
           numClusToAddMin_(numClusToAddMin),
           numClusToAddMax_(numClusToAddMax), 
           objDelThr_(objDelThr),
           intGapThr_(intGapThr), 
           UAIsettings_(UAIsettings),
           addEdgeIntersections_(addEdgeIntersections),
           doGlobalDecoding_(doGlobalDecoding), 
           useDecimation_(useDecimation),
           lookForCSPs_(lookForCSPs), 
           //infTime_(infTime), 
           logFile_(logFile),
           seed_(seed),
           inputFile_(inputFile),
           evidenceFile_(evidenceFile)
         { }

      // new parameters
      size_t maxIterLP_;    //maximum number of iterrations for the initial LP
      size_t maxIterTight_; //maximum number of rounds for tightening
      size_t maxIterLater_; //maximum number of iterrations after each tightening
      double maxTime_;      //overall time limit in seconds
      double maxTimeLP_;    //time limit for the initial LP in seconds

      //size_t maxTightIter_;
      //size_t numIter_;
      //size_t numIterLater_;
      size_t numClusToAddMin_;
      size_t numClusToAddMax_;
      double objDelThr_;
      double intGapThr_;

      // Settings for UAI inference competition override all others
      bool UAIsettings_;

      // defaults. UAIsettings modulates the value of many of these
      bool addEdgeIntersections_ ;
      bool doGlobalDecoding_;
      bool useDecimation_;
      bool lookForCSPs_;

      /* Note:
      *  Setting infTime_ to the total number of seconds allowed to run will
      *  result in global decoding being called once 1/3 through, and (if turned
      *  on) decimation being called 2/3 through (very helpful for CSP intances).
      */
      double infTime_;

      std::string logFile_;
      int seed_;
      std::string inputFile_;
      std::string evidenceFile_;
   };

   // construction
   MPLP(const GraphicalModelType& gm, const Parameter& para = Parameter());
   // destruction
   ~MPLP();
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

   FILE* mplpLogFile_;
   //double mplpTimeLimit_;
   clock_t mplpStart_;
   MPLPAlg* mplp_;

   bool valueCheck() const;
};

template<class GM>
inline MPLP<GM>::MPLP(const GraphicalModelType& gm, const Parameter& para)
   : gm_(gm), parameter_(para), mplpLogFile_(NULL), mplp_(NULL) {

  if(parameter_.UAIsettings_) {
     parameter_.doGlobalDecoding_ = true;
     parameter_.useDecimation_=true;
     parameter_.lookForCSPs_ = true;
  }

  // Log file
  if(!parameter_.logFile_.empty()) {
     mplpLogFile_ = fopen(parameter_.logFile_.c_str(), "w");
  }

  if (parameter_.maxTime_ <= 0.0) {
     parameter_.maxTime_ = 3600*24*30; //30 days
  }

  if(parameter_.maxTime_< parameter_.maxTimeLP_){
     parameter_.maxTimeLP_=  parameter_.maxTime_; 
  }

  if(MPLP_DEBUG_MODE) {
     std::cout << "Time limit = " << parameter_.maxTime_ << std::endl;
  }

  mplpStart_ = clock();

  // Load in the MRF and initialize GMPLP state
  if(!parameter_.inputFile_.empty()) {
     //mplp_ = new MPLPAlg(mplpStart_, mplpTimeLimit_, parameter_.inputFile_, parameter_.evidenceFile_, mplpLogFile_, parameter_.lookForCSPs_);
     mplp_ = new MPLPAlg(mplpStart_, parameter_.maxTime_, parameter_.inputFile_, parameter_.evidenceFile_, mplpLogFile_, parameter_.lookForCSPs_);
  } else {
     // fill vectors from opengm model
     std::vector<int> var_sizes(gm_.numberOfVariables());
     for(IndexType var = 0; var < gm_.numberOfVariables(); ++var){
        var_sizes[var] = static_cast<int>(gm_.numberOfLabels(var));
     }

     std::vector< std::vector<int> > all_factors(gm_.numberOfFactors());
     for(IndexType f = 0; f < gm_.numberOfFactors(); ++f){
        all_factors[f].resize(gm_[f].numberOfVariables());
        for(IndexType i = 0; i < gm_[f].numberOfVariables(); ++i){
           all_factors[f][i] = static_cast<int>(gm_[f].variableIndex(i));
        }
     }

     std::vector< std::vector<double> > all_lambdas(gm_.numberOfFactors());
     for(IndexType f = 0; f < gm_.numberOfFactors(); ++f){
        all_lambdas[f].resize(gm_[f].size());
        //gm_[f].copyValues(all_lambdas[f].begin());

        gm_[f].copyValuesSwitchedOrder(all_lambdas[f].begin());

        // TODO check if value transform (log or exp) is needed
        for(size_t i = 0; i < all_lambdas[f].size(); i++) {
           all_lambdas[f][i] = -all_lambdas[f][i];
        }
     }

     //mplp_ = new MPLPAlg(mplpStart_, mplpTimeLimit_, var_sizes, all_factors, all_lambdas, mplpLogFile_, parameter_.lookForCSPs_);
     mplp_ = new MPLPAlg(mplpStart_, parameter_.maxTime_, var_sizes, all_factors, all_lambdas, mplpLogFile_, parameter_.lookForCSPs_);
  }
}

template<class GM>
inline MPLP<GM>::~MPLP() {
   if(mplp_) {
      delete mplp_;
   }
}

template<class GM>
inline std::string MPLP<GM>::name() const {
   return "MPLP";
}

template<class GM>
inline const typename MPLP<GM>::GraphicalModelType& MPLP<GM>::graphicalModel() const {
   return gm_;
}

template<class GM>
inline InferenceTermination MPLP<GM>::infer() {
   EmptyVisitorType visitor;
   return this->infer(visitor);
}

template<class GM>
template<class VISITOR>
inline InferenceTermination MPLP<GM>::infer(VISITOR & visitor) {
   visitor.begin(*this);

   bool decimation_has_started = false;
   bool force_decimation = false;
   bool prevGlobalDecodingWas1 = true;

   // Keep track of triplets added so far
   std::map<std::vector<int>, bool> triplet_set;

   //if(MPLP_DEBUG_MODE) std::cout << "Random seed = " << parameter_.seed_ << std::endl;
   srand(parameter_.seed_);
/*
   if(!parameter_.logFile_.empty()) {
      std::stringstream stream;
      stream << "I niter=" << parameter_.numIter_ << ", niter_later=" << parameter_.numIterLater_ << ", nclus_to_add_min=" << parameter_.numClusToAddMin_ << ", nclus_to_add_max=" << parameter_.numClusToAddMax_ << ", obj_del_thr=" << parameter_.objDelThr_ << ", int_gap_thr=" << parameter_.intGapThr_ << "\n";
      fprintf(mplpLogFile_, "%s", stream.str().c_str());
   }

   if (MPLP_DEBUG_MODE) {
      std::cout << "niter=" << parameter_.numIter_ << "\nniter_later=" << parameter_.numIterLater_ << "\nnclus_to_add=" << parameter_.numClusToAddMin_ << "\nobj_del_thr=" << parameter_.objDelThr_ << "\nint_gap_thr=" << parameter_.intGapThr_ << "\n";
      std::cout << "Initially running MPLP for " << parameter_.numIter_ << " iterations\n";
   }
*/
   double value_old;
   for (size_t i=0; i<=parameter_.maxIterLP_;++i){
      value_old = mplp_->last_obj; 
      mplp_->RunMPLP(1, parameter_.objDelThr_, parameter_.intGapThr_);
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
         if(!parameter_.logFile_.empty()) fflush(mplpLogFile_);
         if(!parameter_.logFile_.empty()) fclose(mplpLogFile_);
         visitor.end(*this);
         return NORMAL;
      }
      if(((double)(clock() - mplpStart_) / CLOCKS_PER_SEC) >  parameter_.maxTimeLP_){
         std::cout << "stop because of timelimit for LP switching to tightening" <<std::endl;
         break;
      }
      if(((double)(clock() - mplpStart_) / CLOCKS_PER_SEC) >  parameter_.maxTime_){
         std::cout << "stop because of timelimit" <<std::endl;
         break;
      }
      if (std::fabs(value_old- mplp_->last_obj)<parameter_.objDelThr_ && i > 16){ 
         std::cout << "stop because small progress" <<std::endl;
         break;
      }
      if(std::fabs(value()-bound())<parameter_.intGapThr_){
         std::cout << "stop because small gap" <<std::endl;
         break;
      }
   }

   for(size_t iter=1; iter<=parameter_.maxIterTight_; iter++){  // Break when problem is solved
      // if(!parameter_.logFile_.empty()) fflush(mplpLogFile_);
      // if (MPLP_DEBUG_MODE) std::cout << "\n\nOuter loop iteration " << iter << "\n----------------------\n";

      // Is problem solved? If so, break.
      double int_gap = mplp_->last_obj - mplp_->m_best_val;
      if(int_gap < parameter_.intGapThr_){
         if (MPLP_DEBUG_MODE) std::cout << "Done! Integrality gap less than " << parameter_.intGapThr_ << "\n";
         break;
      } 
      double time_elapsed = (double)(clock() - mplpStart_)/ CLOCKS_PER_SEC;
      if (time_elapsed >  parameter_.maxTime_) {
         break;    // terminates if alreay running past time limit (this should be very conservative)
      }

      // Heuristic: when the integrality gap is sufficiently small, allow the algorithm
      // more time to run till convergence

      if(int_gap < 1){
         parameter_.maxIterLater_ = std::max(parameter_.maxIterLater_, static_cast<size_t>(600));  // TODO opt: don't hard code
         parameter_.objDelThr_ = std::min(parameter_.objDelThr_, 1e-5);
         if (MPLP_DEBUG_MODE) std::cout << "Int gap small, so setting niter_later to " << parameter_.maxIterLater_ << " and obj_del_thr to " << parameter_.objDelThr_ << "\n";
      }

      // Keep track of global decoding time and run this frequently, but at most 20% of total runtime
      if(parameter_.doGlobalDecoding_ && (((double)clock() - mplp_->last_global_decoding_end_time)/CLOCKS_PER_SEC >= mplp_->last_global_decoding_total_time*4)) {
         // Alternate between global decoding methods
         if(prevGlobalDecodingWas1) {
            mplp_->RunGlobalDecoding(false);
            prevGlobalDecodingWas1 = false;
         } else {
            mplp_->RunGlobalDecoding2(false);
            prevGlobalDecodingWas1 = true;
         }
      }

      // Tighten LP
      if (MPLP_DEBUG_MODE) std::cout << "Now attempting to tighten LP relaxation..." << std::endl;

      clock_t tightening_start_time = clock();
      double bound=0; double bound2 = 0;
      int nClustersAdded = 0;

      nClustersAdded += TightenTriplet(*mplp_, parameter_.numClusToAddMin_, parameter_.numClusToAddMax_, triplet_set, bound);
      nClustersAdded += TightenCycle(*mplp_, parameter_.numClusToAddMin_, triplet_set, bound2, 1);

      if(std::max(bound, bound2) < CLUSTER_THR) {
         if(MPLP_DEBUG_MODE) std::cout << "TightenCycle did not find anything useful! Re-running with FindPartition." << std::endl;

         nClustersAdded += TightenCycle(*mplp_, parameter_.numClusToAddMin_, triplet_set, bound2, 2);
      }

      // Check to see if guaranteed bound criterion was non-trivial.
      // TODO: these bounds are not for the cycles actually added (since many of the top ones are skipped, already being in the relaxation). Modify it to be so.
      bool noprogress = false;
      if(std::max(bound, bound2) < CLUSTER_THR) noprogress = true;

      clock_t tightening_end_time = clock();
      double tightening_total_time = (double)(tightening_end_time - tightening_start_time)/CLOCKS_PER_SEC;
      if (MPLP_DEBUG_MODE) {
         std::cout << " -- Added " << nClustersAdded << " clusters to relaxation. Took " << tightening_total_time << " seconds\n";
      }
      if(!parameter_.logFile_.empty()) {
         std::stringstream stream;
         stream << "I added " << nClustersAdded << " clusters. Took " << tightening_total_time << " seconds\n";
         fprintf(mplpLogFile_, "%s", stream.str().c_str());
      }

      // For CSP instances, 2/3 through run time, start decimation -- OR, when no progress being made
      if((mplp_->CSP_instance || noprogress) && ((double)(clock() - mplpStart_) / CLOCKS_PER_SEC) >  parameter_.maxTimeLP_ + (parameter_.maxTime_-parameter_.maxTimeLP_)/2){
         force_decimation = true;
      }
      /*
        We have done as much as we can with the existing edge intersection sets. Now
        add in all new edge intersection sets for large clusters.
      */
      if(nClustersAdded == 0 && parameter_.addEdgeIntersections_) {
         mplp_->AddAllEdgeIntersections();
         parameter_.addEdgeIntersections_ = false; // only makes sense to run this code once
      }

      // Not able to tighten relaxation further, so try to see if decoding is the problem
      // Do not run this too often!
      else if((!parameter_.addEdgeIntersections_ && nClustersAdded == 0) || force_decimation) {
         // Do one last push to try to find the global assignment!
         if(parameter_.doGlobalDecoding_ && (!parameter_.useDecimation_ || !decimation_has_started)) mplp_->RunGlobalDecoding3();

         // Do one step of decimation
         if (parameter_.useDecimation_) {
            decimation_has_started = true;

            bool fixed_node = mplp_->RunDecimation();
            if(!fixed_node) {
               if(MPLP_DEBUG_MODE) std::cout << "Decimation fixed all of the nodes it could... quiting." << std::endl;
               break;
            }
         }
      }

      if (MPLP_DEBUG_MODE) std::cout << "Running MPLP again for " << parameter_.maxIterLater_ << " more iterations\n";
      mplp_->RunMPLP(parameter_.maxIterLater_, parameter_.objDelThr_, parameter_.intGapThr_);

      if(parameter_.UAIsettings_) {
         // For UAI competition: time limit can be up to 1 hour, so kill process if still running.
         //double time_elapsed, /*time,*/ time_limit;
         double time_elapsed = (double)(clock() - mplpStart_)/ CLOCKS_PER_SEC;
         if (time_elapsed > 4000 && time_elapsed >  parameter_.maxTime_) {
            break;    // terminates if alreay running past time limit (this should be very conservative)
         }
      }

      if(!parameter_.logFile_.empty()) fflush(mplpLogFile_);
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
         break;
      }
   }

   if(!parameter_.logFile_.empty()) fflush(mplpLogFile_);
   if(!parameter_.logFile_.empty()) fclose(mplpLogFile_);


   visitor.end(*this);
   return NORMAL;
}

template<class GM>
inline InferenceTermination MPLP<GM>::arg(std::vector<LabelType>& arg, const size_t& n) const {
   if(n > 1) {
      return UNKNOWN;
   }
   else {
      if(parameter_.inputFile_.empty()) {
         OPENGM_ASSERT(mplp_->m_decoded_res.size() == gm_.numberOfVariables());
      }

      arg.resize(mplp_->m_decoded_res.size());
      for(size_t i = 0; i < arg.size(); i++) {
         arg[i] = static_cast<LabelType>(mplp_->m_decoded_res[i]);
      }
      return NORMAL;
   }
}

template<class GM>
inline typename GM::ValueType MPLP<GM>::bound() const {
   return -mplp_->last_obj;
   //return -mplp_->m_best_val;
}

template<class GM>
inline typename GM::ValueType MPLP<GM>::value() const {
   std::vector<LabelType> state;
   arg(state);
   return gm_.evaluate(state);
   // -mplp_->m_best_val is the best value so far and not the value of the current configuration!

   //OPENGM_ASSERT(valueCheck()); 
   //return -mplp_->m_best_val;
}

template<class GM>
inline bool MPLP<GM>::valueCheck() const {
   if(!parameter_.inputFile_.empty()) {
      return true;
   } else {
      static bool visited = false;
      if(visited) {
         std::vector<LabelType> state;
         arg(state);
         if(fabs(-mplp_->m_best_val - gm_.evaluate(state)) < OPENGM_FLOAT_TOL) {
            return true;
         } else {
            std::cout << "state: ";
            for(size_t i = 0; i < state.size(); i++) {
               std::cout << state[i] << "; ";
            }
            std::cout << std::endl;

            std::cout << "value: " << -mplp_->m_best_val << std::endl;
            std::cout << "expected: " << gm_.evaluate(state) << std::endl;
            return false;
         }
      } else {
         visited = true;
         return true;
      }
   }
}

} // namespace external

} // namespace opengm

#endif /* OPENGM_EXTERNAL_MPLP_HXX_ */
