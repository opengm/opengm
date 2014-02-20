#pragma once
#ifndef OPENGM_INF_AND_FLIP_HXX
#define OPENGM_INF_AND_FLIP_HXX

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <stdexcept>
#include <list>

#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/lazyflipper.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/utilities/tribool.hxx"

namespace opengm {



/// \brief Inference and Flip\n\n
///
/// \ingroup inference 
template<class GM, class ACC, class INF>
class InfAndFlip : public Inference<GM, ACC> {
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<InfAndFlip<GM, ACC, INF> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<InfAndFlip<GM, ACC, INF> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<InfAndFlip<GM, ACC, INF> >  TimingVisitorType;

   struct Parameter
   {
      Parameter(const size_t maxSubgraphSize=2)
      : 
         maxSubgraphSize_(maxSubgraphSize),
         subPara_(),
         warmStartableInf_(false){
      }

      size_t maxSubgraphSize_;
      typename INF::Parameter subPara_;
      bool warmStartableInf_;
   };

   InfAndFlip(const GraphicalModelType&, typename InfAndFlip::Parameter param);
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   ValueType value() const;
   ValueType bound() const;
   void reset();
   InferenceTermination infer();
   template<class VisitorType>
      InferenceTermination infer(VisitorType&);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1)const;
   void setStartingPoint(typename std::vector<LabelType>::const_iterator sp){
      sp_.resize(gm_.numberOfVariables());
      sp_.assign(sp,sp+gm_.numberOfVariables());
      spValue_=gm_.evaluate(sp_.begin());
   }
private:
   const GraphicalModelType& gm_;
   Parameter para_; 
   std::vector<LabelType> state_;
   ValueType value_;
   ValueType bound_; 

   ValueType spValue_;
   std::vector<LabelType> sp_;
};



// implementation of InfAndFlip

template<class GM, class ACC, class INF>
inline
InfAndFlip<GM, ACC, INF>::InfAndFlip(
   const GraphicalModelType& gm,
   typename InfAndFlip<GM, ACC, INF>::Parameter param
   )
   :  gm_(gm), para_(param)
{
   if(gm_.numberOfVariables() == 0) {
      throw RuntimeError("The graphical model has no variables.");
   }
   value_ = ACC::template neutral<ValueType>();
   bound_ = ACC::template ineutral<ValueType>();
}

template<class GM, class ACC, class INF>
inline void
InfAndFlip<GM, ACC, INF>::reset()
{}


template<class GM, class ACC, class INF>
inline std::string
InfAndFlip<GM, ACC, INF>::name() const
{
   return "InfAndFlip";
}

template<class GM, class ACC, class INF>
inline const typename InfAndFlip<GM, ACC, INF>::GraphicalModelType&
InfAndFlip<GM, ACC, INF>::graphicalModel() const
{
   return gm_;
}


/// \brief start the algorithm
template<class GM, class ACC, class INF>
template<class VisitorType>
inline InferenceTermination
InfAndFlip<GM, ACC, INF>::infer(
   VisitorType& visitor
)
{
   INF inf(gm_,para_.subPara_);
   LazyFlipper<GM,ACC> lf(gm_);

   visitor.begin(*this);
   if(para_.warmStartableInf_ && !sp_.size()==0)
      inf.setStartingPoint(sp_.begin());
   inf.infer();
   inf.arg(state_);
   value_=inf.value();
   bound_=inf.bound();
   if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
      visitor.end(*this);
      return NORMAL;
   }

   if(para_.maxSubgraphSize_>0){
      lf.setMaxSubgraphSize(para_.maxSubgraphSize_);
      if(sp_.size()!=gm_.numberOfVariables())
         lf.setStartingPoint(state_.begin());
      else{
         if(ACC::bop(value_,spValue_))
            lf.setStartingPoint(state_.begin());
         else
            lf.setStartingPoint(sp_.begin());
      }
      std::cout << "start flipping ..."<<std::endl;
      lf.infer();
      lf.arg(state_);
      value_=lf.value(); 
   }
   visitor.end(*this);

   return NORMAL;
}

/// \brief start the algorithm
template<class GM, class ACC, class INF>
inline InferenceTermination
InfAndFlip<GM, ACC, INF>::infer()
{
   EmptyVisitorType visitor;
   return this->infer(visitor);
}


template<class GM, class ACC, class INF>
inline InferenceTermination
InfAndFlip<GM, ACC, INF>::arg(
   std::vector<LabelType>& arg,
   const size_t N
) const
{
   if(N==1) {
      arg.resize(gm_.numberOfVariables());
      for(size_t j=0; j<arg.size(); ++j) {
         arg[j] = state_[j];
      }
      return NORMAL;
   }
   else {
      return UNKNOWN;
   }
}

template<class GM, class ACC, class INF>
inline typename InfAndFlip<GM, ACC, INF>::ValueType
InfAndFlip<GM, ACC, INF>::value() const
{
   return value_;
}
template<class GM, class ACC, class INF>
inline typename InfAndFlip<GM, ACC, INF>::ValueType
InfAndFlip<GM, ACC, INF>::bound() const
{
   return bound_;
}
} // namespace opengm

#endif // #ifndef OPENGM_INFANDFLIP_HXX
