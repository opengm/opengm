#pragma once
#ifndef OPENGM_SelfFusion_HXX
#define OPENGM_SelfFusion_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"

#include "opengm/inference/auxiliary/fusion_move/fusion_mover.hxx"



#ifndef WITH_AD3
   error("ad3 is needed");
#endif

#include "opengm/inference/external/ad3.hxx"
#include "opengm/inference/astar.hxx"
#include "opengm/inference/lazyflipper.hxx"

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

	typedef opengm::external::AD3Inf<SubGmType,AccumulationType> Ad3SubInf;
	typedef opengm::AStar<SubGmType,AccumulationType> AStarSubInf;

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


			// do the fusion move
			
			if(nLocalVar<=5){
				value_ = fusionMover_. template fuse<AStarSubInf>(
					typename AStarSubInf::Parameter(),
					argBest_,
					argFromInf_,
					argOut_,
					value_,
					infValue
				);
			}
			else if(nLocalVar<=50){
				value_ = fusionMover_. template fuseInplace<Ad3SubInf>(
					typename Ad3SubInf::Parameter(Ad3SubInf::AD3_ILP),
					argBest_,
					argFromInf_,
					argOut_,
					value_,
					infValue
				);
			}
			else{
				typedef opengm::LazyFlipper<SubGmType,AccumulationType> LfSubInf;
				value_ = fusionMover_. template fuse<LfSubInf>(
					typename LfSubInf::Parameter(2),
					argBest_,
					argFromInf_,
					argOut_,
					value_,
					infValue,
					true
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

   class Parameter {
   public:
      Parameter(
      	const UInt64Type fuseNth,
      	typename INFERENCE::Parameter infParam
      )
      :	fuseNth_(fuseNth),
      	infParam_(infParam)
      {

      }
      UInt64Type fuseNth_;
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


private:
      const GraphicalModelType& gm_;
      Parameter param_;

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
   param_(parameter)
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

#endif // #ifndef OPENGM_SelfFusion_HXX
