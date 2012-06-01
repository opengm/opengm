#ifndef DD_MERIT_CALLER_HXX_
#define DD_MERIT_CALLER_HXX_
#include <opengm/inference/dualdecomposition/dualdecomposition_merit.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/graphcut.hxx>
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

#include "dd_base_caller.hxx"


namespace opengm {
   namespace interface {

      template <class IO, class GM, class ACC>
      class DDMeritCaller : public DDBaseCaller<IO, GM, ACC> 
      {
      protected:  
         using InferenceCallerBase<IO, GM, ACC>::addArgument;
         using InferenceCallerBase<IO, GM, ACC>::io_;
 
         void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
      private:
         template<class DD> void internalRun(GM& model, StringArgument<>& outputfile, typename DD::Parameter);
         double damping_;
      public:
         const static std::string name_;
         DDMeritCaller(IO& ioIn);
      };

      template <class IO, class GM, class ACC>
      const std::string DDMeritCaller<IO, GM, ACC>::name_ = "DDMerit";

      template <class IO, class GM, class ACC>
      inline DDMeritCaller<IO, GM, ACC>::DDMeritCaller(IO& ioIn)
         : DDBaseCaller<IO, GM, ACC>(name_, "detailed description of DD-Merit caller...", ioIn) 
      {
         addArgument(DoubleArgument<>(damping_, "", "damping", "damping values (0,1)", 0.5));
       
      }


      template <class IO, class GM, class ACC> 
      template <class DD>
      inline void DDMeritCaller<IO, GM, ACC>::internalRun(GM& model, StringArgument<>& outputfile, typename DD::Parameter parameter)
      {  
         getParameter(&parameter);
         parameter.damping_ = damping_;
         
         DD dd(model, parameter); 
         DualDecompositionVisitor<DD> visitor;
         std::vector<size_t> states;
         std::cout << "Inferring!" << std::endl;
         if(!(dd.infer(visitor) == opengm::NORMAL)) {
            std::string error("DD-Merit did not solve the problem.");
            io_.errorStream() << error << std::endl;
            throw RuntimeError(error);
         }
         std::cout << "writing states in vector!" << std::endl;
         if(!(dd.arg(states) == opengm::NORMAL)) {
            std::string error("DD-Merit could not return optimal argument.");
            io_.errorStream() << error << std::endl;
            throw RuntimeError(error);
         }  
         storeVectorHDF5(outputfile,"states", states);
         storeVectorHDF5(outputfile,"values", visitor.values());
         storeVectorHDF5(outputfile,"bounds", visitor.bounds());
         storeVectorHDF5(outputfile,"times",  visitor.times());
         storeVectorHDF5(outputfile,"primalTimes",  visitor.primalTimes());
         storeVectorHDF5(outputfile,"dualTimes",  visitor.dualTimes());
      }



      template <class IO, class GM, class ACC>
      inline void DDMeritCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose)
      {
     
         std::cout << "running DD-Merit caller" << std::endl;
         
         typedef typename GM::ValueType                                                  ValueType;
         typedef opengm::DDDualVariableBlock2<marray::View<ValueType, false> >            DualBlockType;
         typedef typename opengm::DualDecompositionBase<GM,DualBlockType>::SubGmType     SubGmType;

         if((*this).subInfType_.compare("ILP")==0){
            typedef opengm::LPCplex<SubGmType, ACC>                            InfType;
            typedef opengm::DualDecompositionMerit<GM,InfType,DualBlockType>  DDType;              
            typename DDType::Parameter parameter;        
            parameter.subPara_.integerConstraint_  = true;
            internalRun<DDType>(model, outputfile, parameter);
         }
         else if((*this).subInfType_.compare("DPTree")==0){
            typedef opengm::DynamicProgramming<SubGmType, ACC>                 InfType;
            typedef opengm::DualDecompositionMerit<GM,InfType,DualBlockType>  DDType;              
            typename DDType::Parameter parameter;
            internalRun<DDType>(model, outputfile, parameter);        
         } 
         else if((*this).subInfType_.compare("GraphCut")==0){
#ifdef WITH_MAXFLOW
            typedef opengm::external::MinSTCutKolmogorov<size_t, double>        MinStCutType; 
            typedef opengm::GraphCut<SubGmType, ACC, MinStCutType>             InfType;
            typedef opengm::DualDecompositionMerit<GM,InfType,DualBlockType>  DDType;              
            typename DDType::Parameter parameter;
            internalRun<DDType>(model, outputfile, parameter);        
#else
            std::cout << "MaxFlow not enabled!!!" <<std::endl;
#endif
         }
         else{
            std::cout << "Unknown Sub-Inference-Algorithm !!!" <<std::endl;
         }
      }

   } // namespace interface
} // namespace opengm

#endif /* DDBUNDLE_CALLER_HXX_ */
