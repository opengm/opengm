#ifndef MQPBO_CALLER_HXX_
#define MQPBO_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/mqpbo.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {
namespace interface {

template <class IO, class GM, class ACC>
class MQPBOCaller : public InferenceCallerBase<IO, GM, ACC, MQPBOCaller<IO, GM, ACC> > {
public:
   typedef MQPBO<GM, ACC> MQPBOType;
   typedef InferenceCallerBase<IO, GM, ACC, MQPBOCaller<IO, GM, ACC> > BaseClass;
   typedef typename MQPBOType::VerboseVisitorType VerboseVisitorType;
   typedef typename MQPBOType::EmptyVisitorType EmptyVisitorType;
   typedef typename MQPBOType::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   MQPBOCaller(IO& ioIn);
   std::vector<std::string> permittedPermutationTypes;
   std::string desiredPermutationType_;
   std::string useKovtunsMethod_;

protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;
   typename MQPBOType::Parameter mqpboParameter_;
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline MQPBOCaller<IO, GM, ACC>::MQPBOCaller(IO& ioIn) 
  : BaseClass(name_, "detailed description of MQPBO caller...", ioIn) {

   addArgument(VectorArgument<std::vector<typename GM::LabelType> >(mqpboParameter_.label_, "", "label", "location of the file containing the initial configuration"));
   //addArgument(BoolArgument(mqpboParameter_.probing_, "", "probing", "use probing"));
   addArgument(BoolArgument(mqpboParameter_.strongPersistency_, "", "strongPersistency", "enforce strong persistency"));

   addArgument(Size_TArgument<>(mqpboParameter_.rounds_, "", "rounds", "rounds of MQPBO"));
  
   std::vector<std::string> permittedPermutationTypes;
   permittedPermutationTypes.push_back("NONE");
   permittedPermutationTypes.push_back("RANDOM");
   permittedPermutationTypes.push_back("OPTMARG");
   addArgument(StringArgument<>(desiredPermutationType_, "", "permutation", "permutation used for label-ordering", permittedPermutationTypes.at(0), permittedPermutationTypes));
 
   std::vector<std::string> yesno; 
   yesno.push_back("yes");
   yesno.push_back("no");
   addArgument(StringArgument<>(useKovtunsMethod_, "", "useKovtunsMethod", "use Kovtuns method for partial optimaity first", yesno.at(0), yesno));   
}


template <class IO, class GM, class ACC>
inline void MQPBOCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running MQPBO caller" << std::endl;

   //check start point
   if(mqpboParameter_.label_.size() != model.numberOfVariables()){
      mqpboParameter_.label_.resize(model.numberOfVariables(),0);
   }

   //permutation
   if(desiredPermutationType_ == "NONE") {
      mqpboParameter_.permutationType_ =   MQPBOType::NONE; 
   } else if(desiredPermutationType_ == "RANDOM") {
      mqpboParameter_.permutationType_ =   MQPBOType::RANDOM; 
   } else if(desiredPermutationType_ == "OPTMARG") {
      mqpboParameter_.permutationType_ =   MQPBOType::MINMARG; 
   } else {
      throw RuntimeError("Unknown order type!");
   }
   if(useKovtunsMethod_ == "yes"){
      mqpboParameter_.useKovtunsMethod_=true;
   }
   else if(useKovtunsMethod_ == "no"){
      mqpboParameter_.useKovtunsMethod_=false;
   }
   else{
      throw RuntimeError("Unknown value - expect yes or no!");
   }

   this-> template infer<MQPBOType, TimingVisitorType, typename MQPBOType::Parameter>(model, output, verbose, mqpboParameter_);
}

template <class IO, class GM, class ACC>
const std::string MQPBOCaller<IO, GM, ACC>::name_ = "MQPBO";


} // namespace interface

} // namespace opengm

#endif /* MQPBO_CALLER_HXX_ */
