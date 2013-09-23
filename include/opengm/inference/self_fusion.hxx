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
#ifdef WITH_AD3
#include "opengm/inference/external/ad3.hxx"
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

	typedef FusionMover<GraphicalModelType,AccumulationType> FusionMoverType ;

	typedef typename FusionMoverType::SubGmType SubGmType;

	typedef opengm::AStar<SubGmType,AccumulationType> AStarSubInf;
	typedef  opengm::LazyFlipper<SubGmType,AccumulationType> LazyFlipperSubInf;
	#ifdef WITH_AD3
	typedef opengm::external::AD3Inf<SubGmType,AccumulationType> Ad3SubInf;
	#endif
	#ifdef WITH_QPBO
	typedef kolmogorov::qpbo::QPBO<double> 			  QpboSubInf;
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

		typename SelfFusionType::Parameter & param = selfFusion_.param_;


		if(iteration_==0 ){
			inference.arg(argBest_);
		}
		else if(iteration_%fuseNth_==0){

			inference.arg(argFromInf_);

			const ValueType infValue = inference.value();

			IndexType nLocalVar=0;
			for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
				if(argBest_[vi]!=argFromInf_[vi]){
					++nLocalVar;
				}
			}


			#ifdef WITH_QPBO
			if(param.fusionSolver_==SelfFusionType::QpboFusion){
				if(selfFusion_.maxOrder_==2){
					//std::cout<<"qpbo fusion\n";
					value_ = fusionMover_. template fuseSecondOrderInplace<QpboSubInf>(argBest_,argFromInf_,argOut_,value_,infValue);
				}
				else{
					OPENGM_CHECK_OP(selfFusion_.maxOrder_,<=,9,"Qpbo Fusion Reduction does not support a factorOrder > 9");
					std::cout<<"reductionAndFuse * start *\n";
					value_ = fusionMover_.reductionAndFuse(argBest_,argFromInf_,argOut_,value_,infValue);
				}
			}
			#endif

			else if(
				param.fusionSolver_==SelfFusionType::AStarFusion || 
				#ifdef WITH_AD3 
				(param.fusionSolver_==SelfFusionType::Ad3Fusion && nLocalVar<=5 )
				#endif
			){
				std::cout<<"astar fusion\n";
				value_ = fusionMover_. template fuse<AStarSubInf>(
					typename AStarSubInf::Parameter(),argBest_,argFromInf_,argOut_,value_,infValue
				);
			}

			#ifdef WITH_AD3
			else if(param.fusionSolver_==SelfFusionType::Ad3Fusion){
				std::cout<<"ad3 fusion\n";
				value_ = fusionMover_. template fuseInplace<Ad3SubInf>(
					typename Ad3SubInf::Parameter(Ad3SubInf::AD3_ILP),
					argBest_,argFromInf_,argOut_,value_,infValue
				);
			}
			#endif
			else if(param.fusionSolver_==SelfFusionType::LazyFlipperFusion){
				std::cout<<"lf fusion\n";
				typedef LazyFlipperSubInf LfSubInf;
				value_ = fusionMover_. template fuse<LazyFlipperSubInf>(
					typename LfSubInf::Parameter(3),argBest_,argFromInf_,argOut_,value_,infValue,true
				);
			}




			// write fusion result into best arg
			std::copy(argOut_.begin(),argOut_.end(),argBest_.begin());

			std::cout<<"fusionValue "<<value_<<" infValue "<<infValue<<"\n";

			selfFusionVisitor_(selfFusion_,value_,inference.bound());
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
		AStarFusion,
		LazyFlipperFusion
	};


   class Parameter {
   public:
      Parameter(
      	const UInt64Type fuseNth=1,
      	const FusionSolver fusionSolver=LazyFlipperFusion,
      	typename INFERENCE::Parameter infParam = typename INFERENCE::Parameter()
      )
      :	fuseNth_(fuseNth),
      	fusionSolver_(fusionSolver),
      	infParam_(infParam)
      {

      }
      UInt64Type fuseNth_;
      FusionSolver fusionSolver_;
      typename INFERENCE::Parameter infParam_;
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

   Parameter param_;
   size_t maxOrder_;

private:
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
   ValueType b;
   AccumulationType::ineutral(b);
   visitor.begin(*this,value_,b);

   // the fusion visitor will do the job...
   FusionVisitor<INFERENCE,SelfType,VisitorType> fusionVisitor(*this,visitor,argBest_,value_,param_.fuseNth_);

   INFERENCE inf(gm_,param_.infParam_);
   inf.infer(fusionVisitor);
   visitor.end(*this,value_,b);
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
