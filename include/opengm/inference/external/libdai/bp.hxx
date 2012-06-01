#ifndef  OPENGM_LIBDAI_BP_HXX 
#define  OPENGM_LIBDAI_BP_HXX

#include "opengm/inference/external/libdai/inference.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  
   
template<class GM,class ACC>
class Bp : public LibDaiInference<GM,ACC>{
   public:
      enum UpdateRule{
         PARALL,
         SEQFIX,
         SEQRND,
         SEQMAX
      };
      std::string name() {
         return "libDAI-Bp";
      }
      struct Parameter{
         Parameter
         (
            const size_t maxIterations=100,
            const double damping=0.0,
            const double tolerance=0.000001,
            UpdateRule updateRule= PARALL,
            const size_t verbose=0
         ) :maxIterations_(maxIterations),
            damping_(damping),
            tolerance_(tolerance),
            updateRule_(updateRule),
            verbose_(verbose),
            logDomain_(0) {

         }
         std::string toString()const{
            std::string ur;
            std::stringstream ss;
            if(updateRule_==PARALL)ur="PARALL";
            else if(updateRule_==SEQFIX)ur = "SEQFIX";
            else if(updateRule_==SEQMAX)ur = "SEQMAX";
            else if(updateRule_==SEQRND)ur = "SEQRND";
            ss <<"BP["
               <<"updates="<<ur<<","
               <<"damping="<<damping_<<","
               <<"maxiter="<<maxIterations_<<","
               <<"tol="<<tolerance_<<","
               <<"logdomain="<<logDomain_<<","
               <<"inference="<< std::string(::opengm::meta::Compare<ACC,::opengm::Integrator>::value==true ? std::string("SUMPROD") : std::string("MAXPROD")  ) <<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }

         size_t maxIterations_;
         double damping_;
         double tolerance_;
         UpdateRule updateRule_;
         size_t verbose_;
         size_t logDomain_;
      };
      Bp(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC>(gm,param.toString()) {
         
      }

};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

/// \endcond

#endif // OPENGM_LIBDAI_BP_HXX 