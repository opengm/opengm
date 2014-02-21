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
class TreeExpectationPropagation : public LibDaiInference<GM,ACC,TreeExpectationPropagation<GM,ACC> > , public opengm::Inference<GM,ACC>{
   public:
      typedef ACC AccumulationType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef opengm::visitors::VerboseVisitor< TreeExpectationPropagation<GM,ACC> > VerboseVisitorType;
      typedef opengm::visitors::TimingVisitor<  TreeExpectationPropagation<GM,ACC> > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<   TreeExpectationPropagation<GM,ACC> > EmptyVisitorType;

      enum TreeEpType{
            ORG,
            ALT
         };
      std::string name() const{
         return "libDAI-Tree-Expectation-Propagation";
      }
      struct Parameter{
         Parameter
         (
            TreeEpType treeEpTyp=ORG,
            const size_t maxiter=10000,
            const double maxtime=120,
            const double tolerance=1e-9,
            size_t verbose=0
         ) :treeEpType_(treeEpTyp),
            maxiter_(maxiter),
            maxtime_(maxtime),
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
               //<<"maxtime="<<maxtime_<<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }
         TreeEpType treeEpType_;
         size_t maxiter_;
         double maxtime_; // in seconds
         double tolerance_;
         size_t verbose_;
      };
      TreeExpectationPropagation(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC,TreeExpectationPropagation<GM,ACC> >(gm,param.toString()) {
         
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

#endif // OPENGM_LIBDAI_TREEEP_HXX 
