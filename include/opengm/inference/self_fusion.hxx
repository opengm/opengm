#ifndef OPENGM_SELF_FUSION_HXX
#define OPENGM_SELF_FUSION_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/inference/inference.hxx"


// Fusion Move Solver
#include "opengm/inference/lazyflipper.hxx"

#include "opengm/inference/visitors/visitors.hxx"


#ifdef WITH_CPLEX
#include "opengm/inference/lpcplex.hxx"
#endif
#ifdef WITH_QPBO
#include "QPBO.h"
#include "opengm/inference/reducedinference.hxx"
#include "opengm/inference/hqpbo.hxx"
#endif

// fusion move model generator
#include "opengm/inference/auxiliary/fusion_move/fusion_mover.hxx"


namespace opengm {
  
template<class INF,class SELF_FUSION,class SELF_FUSION_VISITOR>
struct FusionVisitor{

    typedef typename INF::AccumulationType AccumulationType;
    typedef typename INF::GraphicalModelType GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;

    typedef FusionMover<GraphicalModelType,AccumulationType>    FusionMoverType ;

    typedef typename FusionMoverType::SubGmType                 SubGmType;
    // sub-inf-lf
    typedef opengm::LazyFlipper<SubGmType,AccumulationType>     LazyFlipperSubInf;


    #ifdef WITH_QPBO
    typedef kolmogorov::qpbo::QPBO<double>                          QpboSubInf;
    typedef opengm::external::QPBO<SubGmType>                       QPBOSubInf;
    typedef opengm::HQPBO<SubGmType,AccumulationType>               HQPBOSubInf;
    #endif
    #ifdef WITH_CPLEX
    typedef opengm::LPCplex<SubGmType,AccumulationType>             CplexSubInf;
    #endif

    typedef SELF_FUSION SelfFusionType;
    typedef SELF_FUSION_VISITOR SelfFusionVisitorType;

    FusionVisitor(
            SelfFusionType &            selfFusion,
            SelfFusionVisitorType &     selfFusionVisitor,
            std::vector<LabelType> &    argBest,
            ValueType &                 value,
            ValueType &                 bound,
            UInt64Type                  fuseNth=1
        )
    :   gm_(selfFusion.graphicalModel()),
        selfFusion_(selfFusion),
        selfFusionVisitor_(selfFusionVisitor), 
        fusionMover_(selfFusion.graphicalModel()),
        iteration_(0),
        fuseNth_(fuseNth),
        value_(value),
                bound_(bound),
        argFromInf_(selfFusion.graphicalModel().numberOfVariables()),
        argBest_(argBest),
        argOut_(selfFusion.graphicalModel().numberOfVariables()),
        returnFlag_(visitors::VisitorReturnFlag::ContinueInf),
        numNoProgress_(0)
    {

    }

    void begin(
        INF  & inf
    ){
        returnFlag_ =   selfFusionVisitor_(selfFusion_);
        selfFusionVisitor_.log("infValue",inf.value());
    }
    void end(
        INF  & inf
    ){
    }

    size_t operator()(
        INF  & inf
    ){
        return this->fuseVisit(inf);
    }






    size_t fuseVisit(INF & inference){

        const typename SelfFusionType::Parameter & param = selfFusion_.parameter();

        ValueType oldValue = value_;

        if(iteration_==0 ){         
            inference.arg(argBest_);
            ValueType value = inference.value();
            if(AccumulationType::bop(value,value_)){
               std::copy(argOut_.begin(),argOut_.end(),argBest_.begin());
               value_ = value;
            }
            returnFlag_ =   selfFusionVisitor_(selfFusion_);
            selfFusionVisitor_.log("infValue",value);
        }
        else if(iteration_%fuseNth_==0){
          
            inference.arg(argFromInf_);

            const ValueType infValue = inference.value();
            bound_   = inference.bound();
            lastInfValue_=infValue;
            IndexType nLocalVar=0;
            for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
                if(argBest_[vi]!=argFromInf_[vi]){
                    ++nLocalVar;
                }
            }


            // setup which to labels should be fused and declare 
            // output label vector
            fusionMover_.setup(argBest_,argFromInf_,argOut_,value_,infValue);
            // get the number of fusion-move variables
            const IndexType nFuseMoveVar=fusionMover_.numberOfFusionMoveVariable();

            if(nFuseMoveVar>0){


                if(param.fusionSolver_==SelfFusionType::LazyFlipperFusion){
                    //std::cout<<"fuse with lazy flipper "<<param.maxSubgraphSize_<<"\n";
                    value_ = fusionMover_. template fuse<LazyFlipperSubInf> (
                        typename LazyFlipperSubInf::Parameter(param.maxSubgraphSize_),true
                    );

                }
                #ifdef WITH_CPLEX
                else if(param.fusionSolver_==SelfFusionType::CplexFusion ){
#ifdef WITH_QPBO
                   // NON reduced inference
                   if(param.reducedInf_==false){
#endif  
                      //std::cout <<"ILP"<<std::endl;
                      typename CplexSubInf::Parameter p;
                      p.integerConstraint_ = true;
                      p.numberOfThreads_   = 1;
                      p.timeLimit_         = param.fusionTimeLimit_;
                      value_ = fusionMover_. template fuse<CplexSubInf> (p,true);
 #ifdef WITH_QPBO
                   } 
                   // reduced inference
                   else{
                      //std::cout <<"RILP"<<std::endl;
                      typedef typename ReducedInferenceHelper<SubGmType>::InfGmType ReducedGmType;
                      typedef opengm::LPCplex<ReducedGmType, AccumulationType>      _CplexSubInf;
                      typedef ReducedInference<SubGmType,AccumulationType,_CplexSubInf>          CplexReducedSubInf; 
                      typename _CplexSubInf::Parameter _subInfParam;
                      _subInfParam.integerConstraint_ = true; 
                      _subInfParam.numberOfThreads_   = 1;
                      _subInfParam.timeLimit_         = param.fusionTimeLimit_; 
                      typename CplexReducedSubInf::Parameter subInfParam(true,param.tentacles_,param.connectedComponents_,_subInfParam);
                      value_ = fusionMover_. template fuse<CplexReducedSubInf> (subInfParam,true); 
                   }
                  
 #endif
                   
                }
                #endif

                #ifdef WITH_QPBO
                else if(param.fusionSolver_==SelfFusionType::QpboFusion ){
                    
                    if(selfFusion_.maxOrder()<=2){
                        //std::cout<<"fuse with qpbo\n";
                        value_ = fusionMover_. template fuseQpbo<QpboSubInf> ();
                        //typename QPBOSubInf::Parameter subInfParam;
                        //subInfParam.strongPersistency_ = false;
                        //subInfParam.label_ = argBest_;
                        //value_ = fusionMover_. template fuse<QPBOSubInf> (subInfParam,false); 
                    }
                    else{
                        //std::cout<<"fuse with fix-qpbo\n";
                        //value_ = fusionMover_. template fuseFixQpbo<QpboSubInf> ();
                        typename HQPBOSubInf::Parameter subInfParam;
                        value_ = fusionMover_. template fuse<HQPBOSubInf> (subInfParam,true);
                    }
                }
                #endif
                else{
                   throw std::runtime_error("Unknown Fusion Type! Maybe caused by missing linking!");
                }


                // write fusion result into best arg
                std::copy(argOut_.begin(),argOut_.end(),argBest_.begin());

                //std::cout<<"fusionValue "<<value_<<" infValue "<<infValue<<"\n";

                returnFlag_ =  selfFusionVisitor_(selfFusion_);
                                selfFusionVisitor_.log("infValue",infValue);
            } 

            else{
               returnFlag_ =   selfFusionVisitor_(selfFusion_);
               selfFusionVisitor_.log("infValue",value_);
            }
        }
        ++iteration_;

        if(oldValue == value_){
           ++numNoProgress_;
        }else{
           numNoProgress_=0; 
        }
        
        if(numNoProgress_>=param.numStopIt_)
           return visitors::VisitorReturnFlag::StopInfTimeout;

        return returnFlag_;
    } 




    const GraphicalModelType & gm_;
    SelfFusionType & selfFusion_;
    SelfFusionVisitorType & selfFusionVisitor_;
    
    FusionMoverType fusionMover_;

    UInt64Type iteration_;
    UInt64Type fuseNth_;

    ValueType & value_;
    ValueType & bound_;

    std::vector<LabelType>      argFromInf_;
    std::vector<LabelType> &    argBest_;
    std::vector<LabelType>      argOut_;

    ValueType lastInfValue_;
    size_t returnFlag_;
    size_t numNoProgress_;

};


template<class INFERENCE>
class SelfFusion : public Inference<typename INFERENCE::GraphicalModelType, typename INFERENCE::AccumulationType>
{
public:

    typedef typename INFERENCE::AccumulationType AccumulationType;
    typedef typename INFERENCE::GraphicalModelType GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;

   typedef visitors::VerboseVisitor< SelfFusion<INFERENCE> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<   SelfFusion<INFERENCE> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<  SelfFusion<INFERENCE> >  TimingVisitorType;

    typedef SelfFusion<INFERENCE> SelfType;

    typedef INFERENCE ToFuseInferenceType;

    enum FusionSolver{
        QpboFusion,
        CplexFusion,
        LazyFlipperFusion
    };


   class Parameter {
   public:
      Parameter(
        const UInt64Type fuseNth=1,
        const FusionSolver fusionSolver=LazyFlipperFusion,
        const typename INFERENCE::Parameter & infParam = typename INFERENCE::Parameter(),
        const UInt64Type maxSubgraphSize=2,
        const bool reducedInf = false,
        const bool tentacles = false,
        const bool connectedComponents = false,
        const double fusionTimeLimit = 100.0,
        const size_t numStopIt = 10
      )
      : fuseNth_(fuseNth),
        fusionSolver_(fusionSolver),
        infParam_(infParam),
        maxSubgraphSize_(maxSubgraphSize),
        reducedInf_(reducedInf),
        connectedComponents_(connectedComponents),
        tentacles_(tentacles),
        fusionTimeLimit_(fusionTimeLimit),
        numStopIt_(numStopIt)
      {

      }
      UInt64Type fuseNth_;
      FusionSolver fusionSolver_;
      typename INFERENCE::Parameter infParam_;
      UInt64Type maxSubgraphSize_;
      bool reducedInf_;
      bool connectedComponents_;
      bool tentacles_;
      double fusionTimeLimit_;
      size_t numStopIt_;
   };

   SelfFusion(const GraphicalModelType&, const Parameter& = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   template<class VisitorType>
   InferenceTermination infer(VisitorType&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;

    ValueType value()const{
        return value_;
    }

    ValueType bound()const{
        return bound_;
    }

    const Parameter & parameter()const{
        return param_;
    }
    const size_t maxOrder()const{
        return maxOrder_;
    }

private:

    Parameter param_;
    size_t maxOrder_;


    const GraphicalModelType& gm_;


    std::vector<LabelType> argBest_;
    ValueType value_;
    ValueType bound_;
};



template<class INFERENCE>
SelfFusion<INFERENCE>::SelfFusion
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gm_(gm),
   param_(parameter),
   argBest_(gm.numberOfVariables()),
   value_(),
   maxOrder_(gm.factorOrder())
{
    AccumulationType::neutral(value_);
}
      

   
template<class INFERENCE>
inline void 
SelfFusion<INFERENCE>::setStartingPoint
(
   typename std::vector<typename SelfFusion<INFERENCE>::LabelType>::const_iterator begin
) {

}
   
template<class INFERENCE>
inline std::string
SelfFusion<INFERENCE>::name() const
{
   return "SelfFusion";
}

template<class INFERENCE>
inline const typename SelfFusion<INFERENCE>::GraphicalModelType&
SelfFusion<INFERENCE>::graphicalModel() const
{
   return gm_;
}
  
template<class INFERENCE>
inline InferenceTermination
SelfFusion<INFERENCE>::infer()
{
   EmptyVisitorType v;
    //VerboseVisitorType v;
   return infer(v);
}

  
template<class INFERENCE>
template<class VisitorType>
InferenceTermination SelfFusion<INFERENCE>::infer
(
   VisitorType& visitor
)
{
   AccumulationType::ineutral(bound_);
   AccumulationType::neutral(value_);

   visitor.begin(*this);
   visitor.addLog("infValue");
   // the fusion visitor will do the job...
   FusionVisitor<INFERENCE,SelfType,VisitorType> fusionVisitor(*this,visitor,argBest_,value_,bound_,param_.fuseNth_);

   INFERENCE inf(gm_,param_.infParam_);
   inf.infer(fusionVisitor);
   visitor.end(*this);
   return NORMAL;
}

template<class INFERENCE>
inline InferenceTermination
SelfFusion<INFERENCE>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{
    x.resize(gm_.numberOfVariables());
    for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
        x[vi]=argBest_[vi];
    }
    return NORMAL;
}

} // namespace opengm

#endif // #ifndef OPENGM_SELF_FUSION_HXX
