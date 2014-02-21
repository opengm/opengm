#ifndef OPENGM_LIBDAI_EXACT_HXX 
#define OPENGM_LIBDAI_EXACT_HXX

#include "opengm/inference/external/libdai/inference.hxx"

/// \cond HIDDEN_SYMBOLS

namespace opengm{
namespace external{
namespace libdai{  
   
template<class GM,class ACC>
class Exact : public opengm::external::libdai::LibDaiInference<GM,ACC,Exact<GM,ACC> >, public opengm::Inference<GM,ACC>{
   public:
      typedef ACC AccumulationType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef opengm::visitors::VerboseVisitor< Exact<GM,ACC> > VerboseVisitorType;
      typedef opengm::visitors::TimingVisitor<  Exact<GM,ACC> > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<   Exact<GM,ACC> > EmptyVisitorType;

         enum UpdateRule{
            PARALL,
            SEQFIX,
            SEQRND,
            SEQMAX
         };
         std::string name() const{
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
      :LibDaiInference<GM,ACC,Exact<GM,ACC> >(gm,param.toString()) {
         
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

#endif // OPENGM_LIBDAI_EXACT_HXX 
