#ifndef OPENGM_LIBDAI_DOUBLE_LOOP_GENERALIZED_BP_HXX 
#define OPENGM_LIBDAI_DOUBLE_LOOP_GENERALIZED_BP_HXX

#include "opengm/inference/external/libdai/inference.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  

enum Clusters{
MIN,
BETHE,
DELTA,
LOOP
};
enum Init{
   UNIFORM,
   RANDOM
};   

template<class GM,class ACC>
class DoubleLoopGeneralizedBP : public LibDaiInference<GM,ACC>{
   public:

      std::string name() const{
         return "Double-Loop-Generalized-BP";
      }
      struct Parameter{
         Parameter
         (
            const bool doubleloop=1,
            const Clusters clusters=BETHE,
            const size_t loopdepth = 3,
            const Init init=UNIFORM,
            const size_t maxiter=10000,
            const double tolerance=1e-9,
            const size_t verbose=0
         ) :
         doubleloop_(doubleloop),
         loopdepth_(loopdepth),
         clusters_(clusters),
         init_(init),
         maxiter_(maxiter),
         tolerance_(tolerance),
         verbose_(verbose) {
         }
         std::string toString()const{
            std::stringstream ss;
            std::string init,cluster;

            if(init_==UNIFORM)init="UNIFORM";
            else if(init_==RANDOM)init = "RANDOM";

            if(clusters_==MIN)cluster="MIN";
            else if(clusters_==BETHE)cluster = "BETHE";
            else if(clusters_==DELTA)cluster = "DELTA";
            else if(clusters_==LOOP)cluster = "LOOP";
            
            if(clusters_==LOOP) {
            ss <<"HAK["
               <<"doubleloop="<<doubleloop_<<","
               <<"clusters="<<"LOOP"<<","
               <<"init="<<init<<","
               <<"tol="<<tolerance_<<","
               <<"loopdepth="<<loopdepth_<<","
               <<"maxiter="<<maxiter_<<","
               <<"verbose="<<verbose_<<"]";
               return ss.str();
            }
            else{
             ss <<"HAK["
               <<"doubleloop="<<doubleloop_<<","
               <<"clusters="<<cluster<<","
               <<"init="<<init<<","
               <<"tol="<<tolerance_<<","
               <<"maxiter="<<maxiter_<<","
               <<"verbose="<<verbose_<<"]";
               return ss.str();              
            }
         }
         bool doubleloop_;
         size_t loopdepth_;
         Clusters clusters_;
         Init init_;
         size_t maxiter_;
         double tolerance_;
         size_t verbose_;
         
      };
      DoubleLoopGeneralizedBP(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC>(gm,param.toString()) {
         
      }

};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

/// \endcond

#endif // OPENGM_LIBDAI_DOUBLE_LOOP_GENERALIZED_BP_HXX 
