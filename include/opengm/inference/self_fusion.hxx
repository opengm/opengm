#ifndef OPENGM_SELF_FUSION_HXX
#define OPENGM_SELF_FUSION_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"


// Fusion Move Solver
#include "opengm/inference/astar.hxx"
#include "opengm/inference/lazyflipper.hxx"
#include "opengm/inference/infandflip.hxx"
#include <opengm/inference/messagepassing/messagepassing.hxx>



#ifdef WITH_AD3
#include "opengm/inference/external/ad3.hxx"
#endif
#ifdef WITH_CPLEX
#include "opengm/inference/lpcplex.hxx"
#endif
#ifdef WITH_QPBO
#include "QPBO.h"
#endif

// fusion move model generator
#include "opengm/inference/auxiliary/fusion_move/fusion_mover.hxx"


namespace opengm {
  

class FromAnyType{
public:	
	FromAnyType(){

	}

	template<class T>
	FromAnyType(const T & any){

	}
};





template<class INF,class SELF_FUSION,class SELF_FUSION_VISITOR>
struct FusionVisitor{

	typedef typename INF::AccumulationType AccumulationType;
	typedef typename INF::GraphicalModelType GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;

	typedef FusionMover<GraphicalModelType,AccumulationType> 	FusionMoverType ;

	typedef typename FusionMoverType::SubGmType 				SubGmType;
	// sub-inf-astar
	typedef opengm::AStar<SubGmType,AccumulationType> 			AStarSubInf;
	// sub-inf-lf
	typedef opengm::LazyFlipper<SubGmType,AccumulationType> 	LazyFlipperSubInf;
	// sub-inf-bp
	typedef opengm::BeliefPropagationUpdateRules<SubGmType,AccumulationType> 						UpdateRulesType;
    typedef opengm::MessagePassing<SubGmType,AccumulationType,UpdateRulesType, opengm::MaxDistance> BpSubInf;
    // sub-inf-bp-lf
    typedef opengm::InfAndFlip<SubGmType,AccumulationType,BpSubInf>        BpLfSubInf;

   


	#ifdef WITH_AD3
	typedef opengm::external::AD3Inf<SubGmType,AccumulationType> 	Ad3SubInf;
	#endif
	#ifdef WITH_QPBO
	typedef kolmogorov::qpbo::QPBO<double> 			  				QpboSubInf;
	#endif
	#ifdef WITH_CPLEX
	typedef opengm::LPCplex<SubGmType,AccumulationType> 			CplexSubInf;
	#endif

	typedef SELF_FUSION SelfFusionType;
	typedef SELF_FUSION_VISITOR SelfFusionVisitorType;

	FusionVisitor(
			SelfFusionType & 			selfFusion,
			SelfFusionVisitorType & 	selfFusionVisitor,
			std::vector<LabelType> & 	argBest,
			ValueType & 				value,
			UInt64Type 					fuseNth=1
		)
	:	gm_(selfFusion.graphicalModel()),
		selfFusion_(selfFusion),
		selfFusionVisitor_(selfFusionVisitor), 
		fusionMover_(selfFusion.graphicalModel()),
		iteration_(0),
		fuseNth_(fuseNth),
		value_(value),
		argFromInf_(selfFusion.graphicalModel().numberOfVariables()),
		argBest_(argBest),
		argOut_(selfFusion.graphicalModel().numberOfVariables())
	{

	}

	void begin(
		INF  & inf,
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType()
	){
		ValueType nn;
		AccumulationType::neutral(nn);
		selfFusionVisitor_(selfFusion_,inf.value(),inf.bound(),nn);
	}
	void end(
		INF  & inf,
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType()
	){
		
	}

	void operator()(
		INF  & inf,
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType(),
		const FromAnyType = FromAnyType()
	){
		this->fuseVisit(inf);
	}






	void fuseVisit(INF & inference){

		const typename SelfFusionType::Parameter & param = selfFusion_.parameter();


		if(iteration_==0 ){
			inference.arg(argBest_);
			ValueType nn;
			AccumulationType::neutral(nn);
			selfFusionVisitor_(selfFusion_,inference.value(),inference.bound(),nn);
		}
		else if(iteration_%fuseNth_==0){

			inference.arg(argFromInf_);

			const ValueType infValue = inference.value();
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

			//std::cout<<"nFuseMoveVar "<<nFuseMoveVar<<"\n";

			if(nFuseMoveVar>0){


				if(param.fusionSolver_==SelfFusionType::LazyFlipperFusion){
					//std::cout<<"fuse with lazy flipper "<<param.maxSubgraphSize_<<"\n";
					value_ = fusionMover_. template fuse<LazyFlipperSubInf> (
						typename LazyFlipperSubInf::Parameter(param.maxSubgraphSize_),true
					);

				}
                else if(param.fusionSolver_==SelfFusionType::BpFusion){
                    typename BpSubInf::Parameter sParam(param.bpSteps_,0.001,param.damping_);
                    sParam.isAcyclic_=false;
                    value_ = fusionMover_. template fuse<BpSubInf> (sParam,false);
                }

                else if(param.fusionSolver_==SelfFusionType::BpLfFusion){
                    typename BpLfSubInf::Parameter sParam(param.maxSubgraphSize_);
                    typename BpSubInf::Parameter sParamBp(param.bpSteps_,0.001,param.damping_);
                    sParam.subPara_=sParamBp;
                    value_ = fusionMover_. template fuse<BpLfSubInf> (sParam,false);
                }
				#ifdef WITH_CPLEX
				else if(param.fusionSolver_==SelfFusionType::CplexFusion ){
					typename CplexSubInf::Parameter p;
					p.integerConstraint_;
					value_ = fusionMover_. template fuse<CplexSubInf> (p,false);
				}
				#endif

				#ifdef WITH_AD3
				else if(param.fusionSolver_==SelfFusionType::Ad3Fusion ){
					value_ = fusionMover_. template fuse<Ad3SubInf> (Ad3SubInf::AD3_ILP,false);
				}
				#endif

				#ifdef WITH_QPBO
				else if(param.fusionSolver_==SelfFusionType::QpboFusion ){
					
					if(selfFusion_.maxOrder()<=2){
						//std::cout<<"fuse with qpbo\n";
						value_ = fusionMover_. template fuseQpbo<QpboSubInf> ();
					}
					else{
						//std::cout<<"fuse with fix-qpbo\n";
						value_ = fusionMover_. template fuseFixQpbo<QpboSubInf> ();
					}
				}
				#endif



				// write fusion result into best arg
				std::copy(argOut_.begin(),argOut_.end(),argBest_.begin());

				//std::cout<<"fusionValue "<<value_<<" infValue "<<infValue<<"\n";

				selfFusionVisitor_(selfFusion_,value_,inference.bound(),infValue);
			}
		}
		++iteration_;
	} 




	const GraphicalModelType & gm_;
	SelfFusionType & selfFusion_;
	SelfFusionVisitorType & selfFusionVisitor_;
	
	FusionMoverType fusionMover_;

	UInt64Type iteration_;
	UInt64Type fuseNth_;

	ValueType & value_;

	std::vector<LabelType> 		argFromInf_;
	std::vector<LabelType> &	argBest_;
	std::vector<LabelType> 		argOut_;

	ValueType lastInfValue_;


};


template<class INFERENCE>
class SelfFusion : public Inference<typename INFERENCE::GraphicalModelType, typename INFERENCE::AccumulationType>
{
public:

	typedef typename INFERENCE::AccumulationType AccumulationType;
	typedef typename INFERENCE::GraphicalModelType GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;
	typedef VerboseVisitor<SelfFusion<INFERENCE> > VerboseVisitorType;
	typedef EmptyVisitor<SelfFusion<INFERENCE> > EmptyVisitorType;
	typedef TimingVisitor<SelfFusion<INFERENCE> > TimingVisitorType;

	typedef SelfFusion<INFERENCE> SelfType;

	typedef INFERENCE ToFuseInferenceType;

	enum FusionSolver{
		#ifdef WITH_QPBO
		QpboFusion,
		#endif
		#ifdef WITH_AD3
		Ad3Fusion,
		#endif
		#ifdef WITH_CPLEX
		CplexFusion,
		#endif
		AStarFusion,
		LazyFlipperFusion,
        BpFusion,
        BpLfFusion
	};


   class Parameter {
   public:
      Parameter(
      	const UInt64Type fuseNth=1,
      	const FusionSolver fusionSolver=LazyFlipperFusion,
      	typename INFERENCE::Parameter infParam = typename INFERENCE::Parameter(),
      	const UInt64Type maxSubgraphSize=2,
        const double damping=0.5,
        const UInt64Type bpSteps=10
      )
      :	fuseNth_(fuseNth),
      	fusionSolver_(fusionSolver),
      	infParam_(infParam),
      	maxSubgraphSize_(maxSubgraphSize),
        damping_(damping),
        bpSteps_(bpSteps)
      {

      }
      UInt64Type fuseNth_;
      FusionSolver fusionSolver_;
      typename INFERENCE::Parameter infParam_;
      UInt64Type maxSubgraphSize_;
      double damping_;
      UInt64Type bpSteps_;
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
   ValueType b,n;
   AccumulationType::ineutral(b);
   AccumulationType::neutral(n);

   visitor.begin(*this,value_,b,n);

   // the fusion visitor will do the job...
   FusionVisitor<INFERENCE,SelfType,VisitorType> fusionVisitor(*this,visitor,argBest_,value_,param_.fuseNth_);

   INFERENCE inf(gm_,param_.infParam_);
   inf.infer(fusionVisitor);
   visitor.end(*this,value_,b,fusionVisitor.lastInfValue_);
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
