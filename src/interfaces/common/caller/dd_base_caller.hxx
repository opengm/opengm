#ifndef DD_BASE_CALLER_HXX_
#define DD_BASE_CALLER_HXX_

#include <opengm/inference/dualdecomposition/dualdecomposition_base.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {
   namespace interface {

      template <class IO, class GM, class ACC>
      class DDBaseCaller : public InferenceCallerBase<IO, GM, ACC> 
      {
      protected:  
         using InferenceCallerBase<IO, GM, ACC>::addArgument;
         using InferenceCallerBase<IO, GM, ACC>::io_;     
         using InferenceCallerBase<IO, GM, ACC>::infer;

         double minimalAbsAccuracy_;
         double minimalRelAccuracy_;
         size_t maximumNumberOfIterations_;
         double stepsizeStride_;
         double stepsizeScale_;
         double stepsizeExponent_;
         double stepsizeMin_;
         double stepsizeMax_;
         bool   stepsizeUsePrimalDualGap_;
         bool   stepsizeNormalizedSubgradient_;   
         std::string subInfType_;  
         std::string decompositionType_;
         size_t numBlocks_;
         size_t numThreads_;
         void getParameter(DualDecompositionBaseParameter*);
      public:
         DDBaseCaller(const std::string& InferenceParserNameIn, const std::string& inferenceParserDescriptionIn, IO& ioIn, const size_t maxNumArguments = 50);
      };

      template <class IO, class GM, class ACC>
      inline DDBaseCaller<IO, GM, ACC>::DDBaseCaller(const std::string& InferenceParserNameIn, const std::string& inferenceParserDescriptionIn, IO& ioIn, const size_t maxNumArguments)
         : InferenceCallerBase<IO, GM, ACC>(InferenceParserNameIn, inferenceParserDescriptionIn, ioIn) 
      {
         addArgument(Size_TArgument<>(maximumNumberOfIterations_, "", "maxIt", "Maximum number of iterations.", size_t(100)));
         addArgument(DoubleArgument<>(minimalAbsAccuracy_, "", "absStop", "Stop if primal-dual-gap is smaller than this value", double(0.0)));
         addArgument(DoubleArgument<>(minimalRelAccuracy_, "", "relStop", "Stop if primal-dual-gap/(|dual|+1) is smale than this value", double(0.0)));
         addArgument(DoubleArgument<>(stepsizeStride_, "", "ssStride", "stride (s) of stepsize sequence [ s/(1+ (a*i)^e) ]", double(1.0)));
         addArgument(DoubleArgument<>(stepsizeScale_, "", "ssScale", "scale (a) of stepsize sequence [ s/(1+ (a*i)^e) ]", double(1.0)));
         addArgument(DoubleArgument<>(stepsizeExponent_, "", "ssExponent", "exponent (e) of stepsize sequence [ s/(1+ (a*i)^e) ]", double(0.5))); 
         addArgument(DoubleArgument<>(stepsizeMin_, "", "ssMin", "minimal stepsize", double(0.0)));
         addArgument(DoubleArgument<>(stepsizeMax_, "", "ssMax", "maximal stepsize", double(std::numeric_limits<double>::infinity())));
         addArgument(BoolArgument(stepsizeUsePrimalDualGap_, "", "usePDG", "Use primal-dual-gap for stepsize estimation"));
         addArgument(BoolArgument(stepsizeNormalizedSubgradient_, "", "useNSG", "Use subgradient normalization for stepsize estimation"));

         std::vector<std::string> subInfs;
         subInfs.push_back("ILP");
         subInfs.push_back("DPTree"); 
         subInfs.push_back("GraphCut");
         addArgument(StringArgument<>(subInfType_, "", "subInf", "Algorithm used for subproblems", subInfs[0], subInfs));
         std::vector<std::string> decompType;
         decompType.push_back("Tree");
         decompType.push_back("SpanningTrees");
         decompType.push_back("Blocks");
         addArgument(StringArgument<>(decompositionType_, "", "decomp", "Type of used decomposition",  decompType[0], decompType));
         addArgument(Size_TArgument<>(numBlocks_, "", "numBlocks", "Number of blocks (subproblems).", size_t(2)));       
         addArgument(Size_TArgument<>(numThreads_, "", "numThreads", "Number of Threads used for primal subproblems", size_t(1)));       
      }

      template <class IO, class GM, class ACC> 
      void DDBaseCaller<IO, GM, ACC>::getParameter(DualDecompositionBaseParameter* parameter)
      {     
         parameter->maximalNumberOfIterations_  = maximumNumberOfIterations_;
         parameter->stepsizeStride_=stepsizeStride_;
         parameter->stepsizeScale_=stepsizeScale_;
         parameter->stepsizeExponent_=stepsizeExponent_; 
         parameter->stepsizeMin_=stepsizeMin_;
         parameter->stepsizeMax_=stepsizeMax_;
         parameter->stepsizeNormalizedSubgradient_ = stepsizeNormalizedSubgradient_;
         parameter->numberOfThreads_ = numThreads_;
         if( stepsizeUsePrimalDualGap_ ){
            parameter->stepsizePrimalDualGapStride_   = stepsizeUsePrimalDualGap_;
            parameter->stepsizeNormalizedSubgradient_ = true;
         }

         parameter->numberOfBlocks_             = numBlocks_;      
         if(decompositionType_.compare("Tree")==0){
            parameter->decompositionId_=  opengm::DualDecompositionBaseParameter::TREE;
         }else if(decompositionType_.compare("SpanningTrees")==0){
            parameter->decompositionId_= opengm::DualDecompositionBaseParameter::SPANNINGTREES;
         }else if(decompositionType_.compare("Blocks")==0){
            parameter->decompositionId_= opengm::DualDecompositionBaseParameter::BLOCKS;
         }else{
            std::cout << "Unknown decomposition type !!! " << std::endl;
         }
      }
   } // namespace interface
} // namespace opengm

#endif /* DDBASE_CALLER_HXX_ */
