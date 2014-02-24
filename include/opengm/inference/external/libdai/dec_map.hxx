#ifndef OPENGM_LIBDAI_DEC_MAP_HXX 
#define OPENGM_LIBDAI_DEC_MAP_HXX
#include "opengm/inference/external/libdai/inference.hxx"

namespace opengm{
namespace external{
namespace libdai{  
   
class None{
   struct Parameter{
      Parameter() {
      }
   };
};
 
template<class SUB_INFERENCE>
class DecMap : 
   public LibDaiInference<typename SUB_INFERENCE::GraphicalModelType,typename SUB_INFERENCE::AccumulationType,DecMap<SUB_INFERENCE> >, 
   public opengm::Inference<typename SUB_INFERENCE::GraphicalModelType,typename SUB_INFERENCE::AccumulationType>
{
   public:
      typedef typename SUB_INFERENCE::AccumulationType AccumulationType;
      typedef typename SUB_INFERENCE::GraphicalModelType GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef opengm::visitors::VerboseVisitor< DecMap<SUB_INFERENCE> > VerboseVisitorType;
      typedef opengm::visitors::TimingVisitor<  DecMap<SUB_INFERENCE> > TimingVisitorType;
      typedef opengm::visitors::EmptyVisitor<   DecMap<SUB_INFERENCE> > EmptyVisitorType;

      typedef typename SUB_INFERENCE::Parameter SubInferenceParameter;

      std::string name() const{
         return "libDAI-Dec-Map";
      }
      
      
      // DECMAP[
      // ianame=BP,
      // iaopts=[
      //    inference=MAXPROD,
      //    updates=SEQRND,logdomain=1,tol=1e-9,maxiter=10000,damping=0.1,verbose=0
      // ],
      //reinit=1,verbose=0]

      struct Parameter{
         Parameter
         (
            const SubInferenceParameter & subInferenceParameter =SubInferenceParameter(),
            const size_t reinit=1,
            const size_t verbose=0
         ) :   subInferenceParam_(subInferenceParameter),
               reinit_(reinit),
               verbose_(verbose) {
         }
         std::string toString()const{
            std::string ur,cav,subAiName,subAiOpts;
            std::string subAiAsString =subInferenceParam_.toString();
            if(opengm::meta::Compare<SUB_INFERENCE,opengm::external::libdai::None>::value) {
               subAiName="NONE";
               subAiOpts="[]";
            }
            else{
               size_t counter=0;
               while(subAiAsString[counter]!='[') {
                     subAiName.push_back(subAiAsString[counter]);
                     ++counter;
               }
               subAiOpts.reserve(subAiAsString.size()-subAiName.size());
               for(;counter<subAiAsString.size();++counter) {
                  subAiOpts.push_back(subAiAsString[counter]);
                  if(subAiAsString[counter]==']'){
                     break;
                  }
               }
            }
            
            std::stringstream ss;
            ss <<"DECMAP["
               <<"ianame="<<subAiName<<","
               <<"iaopts="<<subAiOpts<<","
               <<"reinit="<<reinit_<<","
               <<"verbose="<<verbose_<<"]";
            return ss.str();
         }

         size_t reinit_;
         SubInferenceParameter subInferenceParam_;
         size_t verbose_;
      };
      DecMap(const GraphicalModelType & gm,const Parameter param=Parameter())
      :LibDaiInference<GraphicalModelType,AccumulationType,DecMap<SUB_INFERENCE> >(gm,param.toString()) {
         
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

#endif // OPENGM_LIBDAI_DEC_MAP_HXX 
