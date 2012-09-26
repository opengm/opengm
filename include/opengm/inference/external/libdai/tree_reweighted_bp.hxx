#ifndef  OPENGM_LIBDAI_TRBP_HXX 
#define  OPENGM_LIBDAI_TRBP_HXX 

#include "opengm/inference/external/libdai/inference.hxx"

namespace opengm{
namespace external{
namespace libdai{  
   
/// \brief tree reweighted belief propagation : \n
/// [?] 
///
/// \ingroup inference
/// \ingroup messagepassing_inference
/// \ingroup external_inference
/// Tree Reweighted Belief Propagation :
/// - cite :[?]
/// - Maximum factor order :\f$\infty\f$
/// - Maximum number of labels : \f$\infty\f$
/// - Restrictions : ?
/// - Convergent : convergent on trees
template<class GM,class ACC>
class TreeReweightedBp : public opengm::external::libdai::LibDaiInference<GM,ACC>
{
      public:
         std::string name() const{
            return "libDAI-Tree-Reweighted-Bp";
         }
         struct Parameter{
         Parameter
         (
            const size_t maxIterations=100,
            const double damping=0.0,
            const double tolerance=0.000001,
            const size_t ntrees=0,
            BpUpdateRule updateRule= PARALL,
            const size_t verbose=0
         ) :maxIterations_(maxIterations),
            damping_(damping),
            tolerance_(tolerance),
            ntrees_(ntrees),
            updateRule_(updateRule),
            verbose_(verbose),
            logDomain_(0) {
            
         }
         std::string toString( )const{
            std::string ur;
            std::stringstream ss;
            if(updateRule_==PARALL)ur="PARALL";
            else if(updateRule_==SEQFIX)ur = "SEQFIX";
            else if(updateRule_==SEQMAX)ur = "SEQMAX";
            else if(updateRule_==SEQRND)ur = "SEQRND";
            ss <<"TRWBP["
               <<"updates="<<ur<<","
               <<"damping="<<damping_<<","
               <<"maxiter="<<maxIterations_<<","
               <<"tol="<<tolerance_<<","
               <<"ntrees="<<ntrees_<<","
               <<"logdomain="<<logDomain_<<","
               <<"inference="<< std::string(::opengm::meta::Compare<ACC,::opengm::Integrator>::value==true ? std::string("SUMPROD") : std::string("MAXPROD")  ) <<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }
         size_t maxIterations_;
         double damping_;
         double tolerance_;
         BpUpdateRule updateRule_;
         size_t ntrees_;
         size_t verbose_;
         size_t logDomain_;
         
      };
      TreeReweightedBp(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC>(gm,param.toString()) {
         
      }
};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

#endif // OPENGM_LIBDAI_TRBP_HXX 
