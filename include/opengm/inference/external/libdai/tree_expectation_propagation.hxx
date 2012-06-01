#ifndef OPENGM_LIBDAI_TREEEP_HXX 
#define OPENGM_LIBDAI_TREEEP_HXX

#include "opengm/inference/external/libdai/inference.hxx"

namespace opengm{
namespace external{
namespace libdai{  

/// \brief Tree Expectation Propagation : \n
/// [?] 
///
/// \ingroup inference
/// \ingroup external_inference
/// Tree Expectation Propagation :
/// - cite :[?]
/// - Maximum factor order :\f$\infty\f$
/// - Maximum number of labels : \f$\infty\f$
/// - Restrictions : ?
/// - Convergent : ?
template<class GM,class ACC>
class TreeExpectationPropagation : public LibDaiInference<GM,ACC>
{
   public:
      enum TreeEpType{
            ORG,
            ALT
         };
      std::string name() {
         return "libDAI-Tree-Expectation-Propagation";
      }
      struct Parameter{
         Parameter
         (
            TreeEpType treeEpTyp=ORG,
            const size_t maxiter=10000,
            const double tolerance=1e-9,
            size_t verbose=0
         ) :treeEpType_(treeEpTyp),
            maxiter_(maxiter),
            tolerance_(tolerance),
            verbose_(verbose) {
         }
         std::string toString()const{
            std::stringstream ss;
            std::string treeept,hr;
            
            if(treeEpType_==ORG)treeept = "ORG";
            else if(treeEpType_==ALT)treeept = "ALT";
            
            ss <<"TREEEP["
               <<"type="<<treeept<<","
               <<"tol="<<tolerance_<<","
               <<"maxiter="<<maxiter_<<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }
         TreeEpType treeEpType_;
         size_t maxiter_;
         double tolerance_;
         size_t verbose_;
      };
      TreeExpectationPropagation(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC>(gm,param.toString()) {
         
      }

};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

#endif // OPENGM_LIBDAI_TREEEP_HXX 
