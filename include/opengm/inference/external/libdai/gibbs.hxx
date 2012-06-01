#ifndef OPENGM_LIBDAI_GIBBS_HXX 
#define OPENGM_LIBDAI_GIBBS_HXX

#include "opengm/inference/external/libdai/inference.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  

template<class GM,class ACC>
class Gibbs : public LibDaiInference<GM,ACC>{
   public:
      std::string name() {
         return "libDAI-Gibbs";
      }
      struct Parameter{
         Parameter
         (
            const size_t maxiter=10000,
            const size_t burnin=100,
            const size_t restart_=10000,
            const size_t verbose=0
         ) :maxiter_(maxiter),
            burnin_(burnin),
            restart_(restart),
            verbose_(verbose) {
         }
         std::string toString()const{
            std::stringstream ss;
            ss <<"GIBBS["
               <<"maxiter="<<maxiter_<<","
               <<"burnin="<<burnin_<<","
               <<"restart="<<restart_<<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }
         size_t maxiter_;
         size_t burnin_;
         size_t restart_;
         size_t verbose_;
         
      };
      Gibbs(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC>(gm,param.toString()) {
         
      }

};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

// \endcond

#endif // OPENGM_LIBDAI_GIBBS_HXX 
