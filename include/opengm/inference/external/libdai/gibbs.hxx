#ifndef OPENGM_LIBDAI_GIBBS_HXX 
#define OPENGM_LIBDAI_GIBBS_HXX

#include "opengm/inference/external/libdai/inference.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  

template<class GM,class ACC>
class Gibbs : public LibDaiInference<GM,ACC,Gibbs<GM,ACC> >, public opengm::Inference<GM,ACC>{
   public:
      typedef ACC AccumulationType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef opengm::visitors::VerboseVisitor< Gibbs<GM,ACC> > VerboseVisitorType;
      typedef opengm::visitors::TimingVisitor<  Gibbs<GM,ACC> > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<   Gibbs<GM,ACC> > EmptyVisitorType;

      std::string name() const {
         return "libDAI-Gibbs";
      }
      struct Parameter{
         Parameter
         (
            const size_t maxiter=10000,
            const size_t burnin=100,
            const size_t restart=10000,
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
      :LibDaiInference<GM,ACC,Gibbs<GM,ACC> >(gm,param.toString()) {
         
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

// \endcond

#endif // OPENGM_LIBDAI_GIBBS_HXX 
