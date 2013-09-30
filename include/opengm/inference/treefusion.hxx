#pragma once
#ifndef OPENGM_TREE_FUSION_HXX
#define OPENGM_TREE_FUSION_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"


namespace opengm {
  
template<class GM, class ACC>
class TreeFusion : public Inference<GM, ACC>
{
public:

   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef Movemaker<GraphicalModelType> MovemakerType;
   typedef VerboseVisitor<TreeFusion<GM,ACC> > VerboseVisitorType;
   typedef EmptyVisitor<TreeFusion<GM,ACC> > EmptyVisitorType;
   typedef TimingVisitor<TreeFusion<GM,ACC> > TimingVisitorType;

   class Parameter {
   public:
      Parameter(
      )
      : {

      }

   };
   TreeFusion(const GraphicalModelType&, const Parameter& = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   template<class VisitorType>
   InferenceTermination infer(VisitorType&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;

   ValueType value()const{

   }

   ValueType bound()const{

   }

private:
      const GraphicalModelType& gm_;
      Parameter param_;
};



template<class GM, class ACC>
TreeFusion<GM, ACC>::TreeFusion
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gm_(gm),
   param_(parameter)
{

}
      

   
template<class GM, class ACC>
inline void 
TreeFusion<GM,ACC>::setStartingPoint
(
   typename std::vector<typename TreeFusion<GM,ACC>::LabelType>::const_iterator begin
) {

}
   
template<class GM, class ACC>
inline std::string
TreeFusion<GM, ACC>::name() const
{
   return "TreeFusion";
}

template<class GM, class ACC>
inline const typename TreeFusion<GM, ACC>::GraphicalModelType&
TreeFusion<GM, ACC>::graphicalModel() const
{
   return gm_;
}
  
template<class GM, class ACC>
inline InferenceTermination
TreeFusion<GM,ACC>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  
template<class GM, class ACC>
template<class VisitorType>
InferenceTermination TreeFusion<GM,ACC>::infer
(
   VisitorType& visitor
)
{
   visitor.begin(*this,movemaker_.value(), movemaker_.value());

   visitor.end(*this,movemaker_.value(), movemaker_.value());
   return NORMAL;
}

template<class GM, class ACC>
inline InferenceTermination
TreeFusion<GM,ACC>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{

}

} // namespace opengm

#endif // #ifndef OPENGM_TREE_FUSION_HXX
