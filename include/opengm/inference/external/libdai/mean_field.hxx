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
class MeanField : public LibDaiInference<GM,ACC,MeanField<GM,ACC> >, public opengm::Inference<GM,ACC>{
   public:
      typedef ACC AccumulationType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      //typedef VerboseVisitor< MeanField<GM,ACC> > VerboseVisitorType; // verbose and timing visitors try to use arg(), and fails.
      //typedef TimingVisitor<  MeanField<GM,ACC> > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<   MeanField<GM,ACC> > EmptyVisitorType;

      enum UpdateRule{
            NAIVE,
            HARDSPIN
         };
         enum Init{
            UNIFORM,
            RANDOM
         };
      std::string name() const{
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
            init_(init),
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
               <<"tol="<<tolerance_<<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }
         size_t maxiter_;
         double damping_;
         double tolerance_;  
         UpdateRule updateRule_;
         Init init_;
         size_t verbose_;
         
      };
      MeanField(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference< GM,ACC,MeanField<GM,ACC> >(gm,param.toString()) {
         
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
         throw opengm::RuntimeError("MeanField implementation in libdai doesn't have member function findMaximum(), hence opengm::external::libdai::MeanField::arg() can't be used.");
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

#endif // OPENGM_LIBDAI_MEAN_FIELD_HXX 
