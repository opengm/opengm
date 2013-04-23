#ifndef DD_SUBGRADIENT_CALLER_HXX_
#define DD_SUBGRADIENT_CALLER_HXX_

#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#ifdef WITH_CPLEX
#include <opengm/inference/lpcplex.hxx>
#endif
#include <opengm/inference/graphcut.hxx>
#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"


namespace opengm {
   namespace interface {

      template <class IO, class GM, class ACC>
      class DDSubgradientCaller : public InferenceCallerBase<IO, GM, ACC>
      {
      protected:  
         using InferenceCallerBase<IO, GM, ACC>::addArgument;
         using InferenceCallerBase<IO, GM, ACC>::io_;
         using InferenceCallerBase<IO, GM, ACC>::infer; 
       
         double minimalAbsAccuracy_; 
         double minimalRelAccuracy_;
         size_t maximalDualOrder_;
         size_t numberOfBlocks_;
         size_t maximalNumberOfIterations_;
         size_t numberOfThreads_;

         std::string stepsizeRule_;
         std::string decomposition_;
         std::string subInf_;

         // Update Parameters
         double stepsizeStride_;    //updateStride_;
         double stepsizeScale_;     //updateScale_;
         double stepsizeExponent_;  //updateExponent_;
         double stepsizeMin_;       //updateMin_;
         double stepsizeMax_;       //updateMax_;
  
         virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);

      private:
         template<class Parameter> void setParameter(Parameter& p);
      public:
         const static std::string name_;
         DDSubgradientCaller(IO& ioIn);
      };

      template <class IO, class GM, class ACC>
      const std::string DDSubgradientCaller<IO, GM, ACC>::name_ = "DDSubgradient";

      template <class IO, class GM, class ACC>
      inline DDSubgradientCaller<IO, GM, ACC>::DDSubgradientCaller(IO& ioIn) : InferenceCallerBase<IO, GM, ACC>("DD-Subgradient", "detailed description of DD-Subgradient Parser...", ioIn)
      { 
         addArgument(Size_TArgument<>(maximalNumberOfIterations_, 
                                      "", "maxIt", "Maximum number of iterations.", size_t(100)));
         addArgument(DoubleArgument<>(minimalAbsAccuracy_, 
                                      "", "absStop", "Stop if primal-dual-gap is smaller than this value", double(0.0)));
         addArgument(DoubleArgument<>(minimalRelAccuracy_, 
                                      "", "relStop", "Stop if primal-dual-gap/(|dual|+1) is smale than this value", double(0.0)));
         addArgument(DoubleArgument<>(stepsizeStride_,
                                      "", "ssStride", "stride (s) of stepsize rule", double(1.0)));
         addArgument(DoubleArgument<>(stepsizeScale_, 
                                      "", "ssScale", "scale (a) of stepsize rule", double(1.0)));
         addArgument(DoubleArgument<>(stepsizeExponent_,
                                      "", "ssExponent", "exponent (e) of stepsize rule", double(0.5))); 
         addArgument(DoubleArgument<>(stepsizeMin_, 
                                      "", "ssMin", "minimal stepsize", double(0.0)));
         addArgument(DoubleArgument<>(stepsizeMax_, 
                                      "", "ssMax", "maximal stepsize", double(std::numeric_limits<double>::infinity())));
         addArgument(Size_TArgument<>(numberOfBlocks_, 
                                      "", "numBlocks", "Number of blocks (subproblems).", size_t(2)));       
         addArgument(Size_TArgument<>(numberOfThreads_, 
                                      "", "numThreads", "Number of Threads used for primal subproblems", size_t(1)));    

         std::vector<std::string> stepsizeRules;
         stepsizeRules.push_back("ProjectedAdaptive");
         stepsizeRules.push_back("Adaptive");
         stepsizeRules.push_back("StepLength");
         stepsizeRules.push_back("StepSize");
         addArgument(StringArgument<>(stepsizeRule_, 
                                      "", "stepsizeRule", "Stepsize rule for dual update \n \t\t\t\t\t\t\t* ProjectedAdaptive: primalDualGap/(1+ (a*i)^e)/|P(s)|\n \t\t\t\t\t\t\t* Adaptive:          primalDualGap/(1+ (a*i)^e)/|s|\n \t\t\t\t\t\t\t* StepLength:         s/(1+ (a*i)^e)/|P(s)|\n \t\t\t\t\t\t\t* StepSize:           s/(1+ (a*i)^e)", stepsizeRules[0], stepsizeRules));  
         std::vector<std::string> subInfs;
         subInfs.push_back("ILP");
         subInfs.push_back("DPTree"); 
         subInfs.push_back("GraphCut");
         addArgument(StringArgument<>(subInf_, 
                                      "", "subInf", "Algorithm used for subproblems", subInfs[0], subInfs));
         std::vector<std::string> decompositions;
         decompositions.push_back("Tree");
         decompositions.push_back("SpanningTrees");
         decompositions.push_back("Blocks");
         addArgument(StringArgument<>(decomposition_, 
                                      "", "decomp", "Type of used decomposition",  decompositions[0], decompositions));
        
      } 

      template <class IO, class GM, class ACC>
      template<class Parameter>
      void DDSubgradientCaller<IO, GM, ACC>::setParameter(Parameter& p)
      {
         p.minimalAbsAccuracy_=minimalAbsAccuracy_; 
         p.minimalRelAccuracy_=minimalRelAccuracy_;
         p.maximalDualOrder_=maximalDualOrder_;
         p.numberOfBlocks_=numberOfBlocks_;
         p.maximalNumberOfIterations_=maximalNumberOfIterations_;
         p.numberOfThreads_=numberOfThreads_;

         // Update Parameters
         p.stepsizeStride_=stepsizeStride_;    //updateStride_;
         p.stepsizeScale_=stepsizeScale_;     //updateScale_;
         p.stepsizeExponent_=stepsizeExponent_;  //updateExponent_;
         p.stepsizeMin_=stepsizeMin_;       //updateMin_;
         p.stepsizeMax_=stepsizeMax_;       //updateMax_;


         //UpdateRule
         if(stepsizeRule_.compare("ProjectedAdaptive")==0){
            p.useProjectedAdaptiveStepsize_=true;
            //p.stepsizePrimalDualGapStride_ = true;
            //p.stepsizeNormalizedSubgradient_ = true;
         }else if(stepsizeRule_.compare("Adaptive")==0){ 
            p.useAdaptiveStepsize_=true;
            //p.stepsizePrimalDualGapStride_ = true;
            //p.stepsizeNormalizedSubgradient_ = false;
         }else if(stepsizeRule_.compare("StepLength")==0){
            p.stepsizePrimalDualGapStride_ = false;
            p.stepsizeNormalizedSubgradient_ = true;
         }else if(stepsizeRule_.compare("StepSize")==0){
            p.stepsizePrimalDualGapStride_ = false;
            p.stepsizeNormalizedSubgradient_ = false;
         }else{
            std::cout << "Unknown stepsize rule !!! " << std::endl;
         } 

         //Decompositions
         if(decomposition_.compare("Tree")==0){
            p.decompositionId_=  opengm::DualDecompositionBaseParameter::TREE;
         }else if(decomposition_.compare("SpanningTrees")==0){
            p.decompositionId_= opengm::DualDecompositionBaseParameter::SPANNINGTREES;
         }else if(decomposition_.compare("Blocks")==0){
            p.decompositionId_= opengm::DualDecompositionBaseParameter::BLOCKS;
         }else{
            std::cout << "Unknown decomposition type !!! " << std::endl;
         }
      }

      template <class IO, class GM, class ACC> 
      inline void DDSubgradientCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) 
      {
         std::cout << "running DD-Subgradient caller" << std::endl;
         
         typedef typename GM::ValueType                                                  ValueType;
         typedef opengm::DDDualVariableBlock<marray::View<ValueType, false> >            DualBlockType;
         typedef typename opengm::DualDecompositionBase<GM,DualBlockType>::SubGmType     SubGmType;
 
         if((*this).subInf_.compare("ILP")==0){
#ifdef WITH_CPLEX
            typedef opengm::LPCplex<SubGmType, ACC>                                 InfType;
            typedef opengm::DualDecompositionSubGradient<GM,InfType,DualBlockType>  DDType; 
            typedef typename DDType::TimingVisitorType                              TimingVisitorType;             
            
            typename DDType::Parameter parameter; 
            setParameter(parameter);
            parameter.subPara_.integerConstraint_ = true; 
            this-> template infer<DDType, TimingVisitorType, typename DDType::Parameter>(model, outputfile, verbose, parameter);
#else
            std::cout << "CPLEX not enabled!!!" <<std::endl;
#endif
         }
         else if((*this).subInf_.compare("DPTree")==0){
            typedef opengm::DynamicProgramming<SubGmType, ACC>                      InfType;
            typedef opengm::DualDecompositionSubGradient<GM,InfType,DualBlockType>  DDType;
            typedef typename DDType::TimingVisitorType                              TimingVisitorType;             
                        
            typename DDType::Parameter parameter;
            setParameter(parameter);
            this-> template infer<DDType, TimingVisitorType, typename DDType::Parameter>(model, outputfile, verbose, parameter);       
         } 
         else if((*this).subInf_.compare("GraphCut")==0){
#ifdef WITH_MAXFLOW
            typedef opengm::external::MinSTCutKolmogorov<size_t, double>            MinStCutType; 
            typedef opengm::GraphCut<SubGmType, ACC, MinStCutType>                  InfType;
            typedef opengm::DualDecompositionSubGradient<GM,InfType,DualBlockType>  DDType;
            typedef typename DDType::TimingVisitorType                              TimingVisitorType;             
                        
            typename DDType::Parameter parameter;
            setParameter(parameter);
            this-> template infer<DDType, TimingVisitorType, typename DDType::Parameter>(model, outputfile, verbose, parameter); 
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
