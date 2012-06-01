#ifndef OPENGM_LIBDAI_MEAN_FIELD_HXX 
#define OPENGM_LIBDAI_MEAN_FIELD_HXX

#include "opengm/inference/external/libdai/inference.hxx"

namespace opengm{
namespace external{
namespace libdai{  

/// \brief Mean Field : \n
/// [?] 
///
/// \ingroup inference
/// \ingroup external_inference
/// Mean Field :
/// - cite :[?]
/// - Maximum factor order :\f$\infty\f$
/// - Maximum number of labels : \f$\infty\f$
/// - Restrictions : ?
/// - Convergent : ?
template<class GM,class ACC>
class MeanField : public LibDaiInference<GM,ACC>{
   public:
      enum UpdateRule{
            NAIVE,
            HARDSPIN
         };
         enum Init{
            UNIFORM,
            RANDOM
         };
      std::string name() {
         return "libDAI-Mean-Field";
      }
      struct Parameter{
         Parameter
         (
            const size_t maxiter=10000,
            const double damping=0.0,
            const double tolerance=1e-9,
            const UpdateRule updateRule= NAIVE,
            const Init init=UNIFORM,
            const size_t verbose=0
         ) :maxiter_(maxiter),
            damping_(damping),
            tolerance_(tolerance),
            updateRule_(updateRule),
            init(init_),
            verbose_(verbose) {
         }
         std::string toString()const{
            std::stringstream ss;
            std::string ur,init;
            
            if(updateRule_==NAIVE)ur = "NAIVE";
            else if(updateRule_==HARDSPIN)ur = "HARDSPIN";
            
            if(init_==UNIFORM)init="UNIFORM";
            else if(init_==RANDOM)init = "RANDOM";

            ss <<"MF["
               <<"maxiter="<<maxiter_<<","
               <<"updates="<<ur<<","
               <<"init="<<init<<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }
         size_t maxiter_;
         double damping_;
         double tolerance_;  
         UpdateRule updateRule_;
         Heuristic init_;
         size_t verbose_;
         
      };
      MeanField(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC>(gm,param.toString()) {
         
      }

};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

#endif // OPENGM_LIBDAI_MEAN_FIELD_HXX 
