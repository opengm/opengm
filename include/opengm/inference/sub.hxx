#pragma once
#ifndef OPENGM_ICM_HXX
#define OPENGM_ICM_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"


namespace opengm {
  
template<class GM, class ACC>
class ICM : public Inference<GM, ACC>
{
public:

   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef Movemaker<GraphicalModelType> MovemakerType;
   typedef VerboseVisitor<ICM<GM,ACC> > VerboseVisitorType;
   typedef EmptyVisitor<ICM<GM,ACC> > EmptyVisitorType;
   typedef TimingVisitor<ICM<GM,ACC> > TimingVisitorType;

   class Parameter {
   public:
      Parameter(
      )
      : {

      }

   };
   ICM(const GraphicalModelType&, const Parameter& = Parameter());
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
ICM<GM, ACC>::ICM
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
ICM<GM,ACC>::setStartingPoint
(
   typename std::vector<typename ICM<GM,ACC>::LabelType>::const_iterator begin
) {

}
   
template<class GM, class ACC>
inline std::string
ICM<GM, ACC>::name() const
{
   return "ICM";
}

template<class GM, class ACC>
inline const typename ICM<GM, ACC>::GraphicalModelType&
ICM<GM, ACC>::graphicalModel() const
{
   return gm_;
}
  
template<class GM, class ACC>
inline InferenceTermination
ICM<GM,ACC>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  
template<class GM, class ACC>
template<class VisitorType>
InferenceTermination ICM<GM,ACC>::infer
(
   VisitorType& visitor
)
{
   visitor.begin(*this,movemaker_.value(), movemaker_.value());
   /////////////////////////
   // INFERENCE CODE HERE //
   /////////////////////////
   visitor.end(*this,movemaker_.value(), movemaker_.value());
   return NORMAL;
}

template<class GM, class ACC>
inline InferenceTermination
ICM<GM,ACC>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{

}

} // namespace opengm

#endif // #ifndef OPENGM_ICM_HXX
