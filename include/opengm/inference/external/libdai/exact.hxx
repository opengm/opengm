#ifndef OPENGM_LIBDAI_EXACT_HXX 
#define OPENGM_LIBDAI_EXACT_HXX

#include "opengm/inference/external/libdai/inference.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  
   
template<class GM,class ACC>
class Exact : public opengm::external::libdai::LibDaiInference<GM,ACC>
{
      public:
         enum UpdateRule{
            PARALL,
            SEQFIX,
            SEQRND,
            SEQMAX
         };
         std::string name() {
            return "libDAI-Exact";
         }
         struct Parameter{
         Parameter
         (
            const size_t verbose=0
         ) :
            verbose_(verbose),
            logDomain_(0) {
            
         }
         std::string toString()const{
            std::stringstream ss;
            ss <<"EXACT["
               <<"logdomain="<<logDomain_<<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }
         size_t logDomain_;
         size_t verbose_;
      };
      Exact(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC>(gm,param.toString()) {
         
      }
};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

/// \endcond

#endif // OPENGM_LIBDAI_EXACT_HXX 