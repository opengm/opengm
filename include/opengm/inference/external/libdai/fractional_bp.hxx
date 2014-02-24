#ifndef OPENGM_LIBDAI_FRACTIONAL_BP_HXX 
#define OPENGM_LIBDAI_FRACTIONAL_BP_HXX

#include "opengm/inference/external/libdai/inference.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  
   
template<class GM,class ACC>
class FractionalBp : public LibDaiInference<GM,ACC, FractionalBp<GM,ACC> >, public opengm::Inference<GM,ACC>{
   public:
      typedef ACC AccumulationType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef opengm::visitors::VerboseVisitor< FractionalBp<GM,ACC> > VerboseVisitorType;
      typedef opengm::visitors::TimingVisitor<  FractionalBp<GM,ACC> > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<   FractionalBp<GM,ACC> > EmptyVisitorType;

      enum UpdateRule{
         PARALL,
         SEQFIX,
         SEQRND,
         SEQMAX
      };
      std::string name() const {
         return "libDAI-Fractional-Bp";
      }
      struct Parameter{
         Parameter
         (
            const size_t maxIterations=100,
            const double damping=0.0,
            const double tolerance=0.000001,
            UpdateRule updateRule= PARALL,
            const size_t verbose=0
         ) :maxIterations_(maxIterations),
            damping_(damping),
            tolerance_(tolerance),
            updateRule_(updateRule),
            verbose_(verbose),
            logDomain_(0) {

         }
         std::string toString()const{
            std::string ur;
            std::stringstream ss;
            if(updateRule_==PARALL)ur="PARALL";
            else if(updateRule_==SEQFIX)ur = "SEQFIX";
            else if(updateRule_==SEQMAX)ur = "SEQMAX";
            else if(updateRule_==SEQRND)ur = "SEQRND";
            ss <<"FBP["
               <<"updates="<<ur<<","
               <<"damping="<<damping_<<","
               <<"maxiter="<<maxIterations_<<","
               <<"tol="<<tolerance_<<","
               <<"logdomain="<<logDomain_<<","
               <<"inference="<< std::string(::opengm::meta::Compare<ACC,::opengm::Integrator>::value==true ? std::string("SUMPROD") : std::string("MAXPROD")  ) <<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }

         size_t maxIterations_;
         double damping_;
         double tolerance_;
         UpdateRule updateRule_;
         size_t verbose_;
         size_t logDomain_;
      };
      FractionalBp(const GM & gm,const Parameter param=Parameter())
      :LibDaiInference<GM,ACC, FractionalBp<GM,ACC> >(gm,param.toString()) {
         
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

#endif // OPENGM_LIBDAI_FRACTIONAL_BP_HXX 
