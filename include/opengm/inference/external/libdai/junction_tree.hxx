#ifndef OPENGM_LIBDAI_JUNCTION_TREE_HXX 
#define OPENGM_LIBDAI_JUNCTION_TREE_HXX

#include "opengm/inference/external/libdai/inference.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  

/// \brief  junction tree  : \n
/// [?] 
///
/// \ingroup inference
/// \ingroup messagepassing_inference
/// \ingroup exact_inference
/// \ingroup external_inference
/// Junction Tree :
/// - cite :[?]
/// - Maximum factor order :\f$\infty\f$
/// - Maximum number of labels : \f$\infty\f$
/// - Restrictions : ?
/// - Convergent : yes
template<class GM,class ACC>
class JunctionTree : public LibDaiInference<GM,ACC>
{
   public:
      enum UpdateRule{
            HUGIN,
            SHSH
         };
         enum Heuristic{
            MINFILL,
            WEIGHTEDMINFILL,
            MINWEIGHT,
            MINNEIGHBORS
         };
      std::string name() {
         return "libDAI-Junction-Tree";
      }
      struct Parameter{
         Parameter
         (
            UpdateRule updateRule= HUGIN,
            Heuristic heuristic=MINWEIGHT,
            size_t verbose=0
         ) :updateRule_(updateRule),
            heuristic_(heuristic),
            verbose_(verbose) {
         }
         std::string toString()const{
            std::stringstream ss;
            std::string ur,hr;
            
            if(updateRule_==HUGIN)ur = "HUGIN";
            else if(updateRule_==SHSH)ur = "SHSH";
            
            if(heuristic_==MINFILL)hr="MINFILL";
            else if(heuristic_==WEIGHTEDMINFILL)hr = "WEIGHTEDMINFILL";
            else if(heuristic_==MINWEIGHT)hr = "MINWEIGHT";
            else if(heuristic_==MINNEIGHBORS)hr = "MINNEIGHBORS";
            
            ss <<"JTREE["
               <<"updates="<<ur<<","
               <<"heuristic="<<hr<<","
               <<"inference="<< std::string(::opengm::meta::Compare<ACC,::opengm::Integrator>::value==true ? std::string("SUMPROD") : std::string("MAXPROD")  ) <<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }
         UpdateRule updateRule_;
         Heuristic heuristic_;
         size_t verbose_;
      };
      JunctionTree(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC>(gm,param.toString()) {
         
      }

};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

/// \endcond

#endif // OPENGM_LIBDAI_JUNCTION_TREE_HXX 
