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
class JunctionTree : public LibDaiInference<GM,ACC,JunctionTree<GM,ACC> > , public opengm::Inference<GM,ACC>{
   public:
      typedef ACC AccumulationType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef opengm::visitors::VerboseVisitor< JunctionTree<GM,ACC> > VerboseVisitorType;
      typedef opengm::visitors::TimingVisitor<  JunctionTree<GM,ACC> > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<   JunctionTree<GM,ACC> > EmptyVisitorType;

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
      std::string name() const {
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
      :LibDaiInference<GM,ACC,JunctionTree<GM,ACC> >(gm,param.toString()) {
         
      }

      virtual const GraphicalModelType& graphicalModel() const{
         return this->graphicalModel_impl();
      }

      virtual void reset(){
         return this->reset_impl();
      }

      virtual InferenceTermination infer(){
         return this->infer_impl();
      }

      template<class VISITOR>
      InferenceTermination infer(VISITOR& visitor ){
         visitor.begin(*this);
         InferenceTermination infTerm = this->infer_impl();
         visitor.end(*this);
         return infTerm;
      }

      virtual InferenceTermination arg(std::vector<LabelType>& v, const size_t argnr=1)const{
         return this->arg_impl(v,argnr);
      }
      virtual InferenceTermination marginal(const size_t v, IndependentFactorType& m) const{
         return this->marginal_impl(v,m);
      }
      virtual InferenceTermination factorMarginal(const size_t f, IndependentFactorType& m) const{
         return this->factorMarginal_impl(f,m);
      }

};

} // end namespace libdai
} // end namespace external
} //end namespace opengm

/// \endcond

#endif // OPENGM_LIBDAI_JUNCTION_TREE_HXX 
