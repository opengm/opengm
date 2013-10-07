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
  



template<class GM>
class ActiveLabels{

public:
   typedef typename GM::LabelType LabelType;
   RandomAccessSet<LabelType> SetType;

   ActiveLabels(const LabelType numberOfLabels){
      activeVis_.reserve(numberOfLabels);
      for(LabelType l=0;l<numberOfLabels;++l){
         activeVis_.insert(l);
      }
   }

   void removeLabel(const LabelType label){
      OPENGM_CHECK(activeVis_.find(label)!=activeVis_.end());
      activeVis_.remove(label);
   }

   size_t size()const{
      return activeVis_.size();
   }

   LabelType operator[](const size_t labelIndex)const{
      return activeVis_[labelIndex];
   }
   LabelType operator[](const size_t labelIndex){
      return activeVis_[labelIndex];
   }

private:
   SetType activeVis_;
};





template<class GM>
class SparseMessage{
public:
   typedef typename GM::ValueType ValueType;
   typedef typename GM::LabelType LabelType;
   std::RandomAccess<LabelType,ValueType> MapType;

   ValueType operator()(const LabelType l)const{
      OPENGM_CHECK(isInit_)
      OPENGM_CHECK(isInit_==false || values_.find(l)!=values_.end());
      return isInit_ ?  values_[l] : uninitValue_ ;
   }

   const ValueType & valueFromIndex(const LabelType labelIndex)const{
      OPENGM_CHECK(false,"do that");
   }
   ValueType & valueFromIndex(const LabelType labelIndex){
      OPENGM_CHECK(false,"do that");
   }

   bool isInit()const{
      return isInit_
   }

   size_t size()const{
      return values_.size();
   }
private:
   MapType  values_;
   bool isInit_;
   ValueType uninitValue_;
};






template<class FACTOR,class VAR_TO_FAC_MSG_CONT,class FAC_TO_VAR_MSG>
void fac_to_var_msg(
   const IndexType viVar,
   const IndexType fi,
   FAC_TO_VAR_MSG  
   FACTOR & f,
   labelManager lm,
   VAR_TO_FAC_MSG_CONT & varToFacMsgs
){

   if(f.order()==2){
         const static int order = 2;
         const static int nMsgToAcc = 1;
         OPENGM_CHECK(nMsgToAcc==varToFacMsgs.size());
         const IndexType vi[order]    = {factor.variableIndex(1),factor.variableIndex(1)};

         const IndexType otherVis[nMsgToAcc]={ vi[0]==viVar ? vi[1] : vi[0]};
         const IndexType otherVisPos[nMsgToAcc]={ vi[0]==viVar ? 1 : 0};

         LabelType labelBuffer[order] = {0,0};
         
         // trick !!!!!
         // iterate over variable to accumulate first
         // then the minimum for a label , lets say == 1 
         // wil be tidy if all other labels of ohter variable have
         // been iterated

         // !!! THIS IS NOT RELECTED IN THE CODE IN ANY WAY !!!

         for(size_t ll0=0;ll0<labelManager.numberOfActiveLabels(vi[0]) ; ++ll0){
            labelBuffer[0]=labelManager.activeLabel(vi[0],ll0);
            for(size_t ll1=0;ll1<labelManager.numberOfActiveLabels(vi[1]) ; ++ll0){
               labelBuffer[1]=labelManager.activeLabel(vi[1],ll1);

               // configuration is set up 
               const double facVal = factor(labelBuffer);


               // get value from msg
               double msgValue=0;

               // get value from msg for this part. value
               // (here would be a loop over all msg if the order>2)
               msgValue+=varToFacMsgs(otherVis[0],fi).valueForLabel(labelBuffer[otherVisPos[0]]);
               



            }
         }

   }

}







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
