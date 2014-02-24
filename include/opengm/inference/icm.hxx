#pragma once
#ifndef OPENGM_ICM_HXX
#define OPENGM_ICM_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
//#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/datastructures/buffer_vector.hxx"


#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {
  
/// \brief Iterated Conditional Modes Algorithm\n\n
/// J. E. Besag, "On the Statistical Analysis of Dirty Pictures", Journal of the Royal Statistical Society, Series B 48(3):259-302, 1986
/// \ingroup inference 
template<class GM, class ACC>
class ICM : public Inference<GM, ACC>
{
public:
   enum MoveType {
      SINGLE_VARIABLE = 0,
      FACTOR = 1
   };
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef Movemaker<GraphicalModelType> MovemakerType;
   typedef opengm::visitors::VerboseVisitor<ICM<GM,ACC> > VerboseVisitorType;
   typedef opengm::visitors::EmptyVisitor<ICM<GM,ACC> >  EmptyVisitorType;
   typedef opengm::visitors::TimingVisitor<ICM<GM,ACC> > TimingVisitorType;

   class Parameter {
   public:
      Parameter(
         const std::vector<LabelType>& startPoint
      )
      :  moveType_(SINGLE_VARIABLE),
         startPoint_(startPoint) 
         {}

      Parameter(
         MoveType moveType, 
         const std::vector<LabelType>& startPoint 
      )
      :  moveType_(moveType),
         startPoint_(startPoint) 
         {}
      
      Parameter(
         MoveType moveType = SINGLE_VARIABLE
      )
      :  moveType_(moveType),
         startPoint_() 
      {}
      
      MoveType moveType_;
      std::vector<LabelType>  startPoint_;
   };

   ICM(const GraphicalModelType&);
   ICM(const GraphicalModelType&, const Parameter&);
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   void reset();
   template<class VisitorType>
      InferenceTermination infer(VisitorType&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;
   virtual ValueType value()const{return movemaker_.value();}
   size_t currentMoveType() const;

private:
      const GraphicalModelType& gm_;
      MovemakerType movemaker_;
      Parameter param_;
      MoveType currentMoveType_;
       
};

template<class GM, class ACC>
inline size_t
ICM<GM, ACC>::currentMoveType()const{
   return currentMoveType_==SINGLE_VARIABLE?0:1;
}
   
template<class GM, class ACC>
inline
ICM<GM, ACC>::ICM
(
      const GraphicalModelType& gm
)
:  gm_(gm),
   movemaker_(gm),
   param_(Parameter()),
   currentMoveType_(SINGLE_VARIABLE) {
}

template<class GM, class ACC>
ICM<GM, ACC>::ICM
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gm_(gm),
   movemaker_(gm),
   param_(parameter),
   currentMoveType_(SINGLE_VARIABLE)
{
   if(parameter.startPoint_.size() == gm.numberOfVariables()) {
      movemaker_.initialize(parameter.startPoint_.begin() );
   }
   else if(parameter.startPoint_.size() != 0) {
      throw RuntimeError("unsuitable starting point");
   }
}
      
template<class GM, class ACC>
inline void
ICM<GM, ACC>::reset()
{
   if(param_.startPoint_.size() == gm_.numberOfVariables()) {
      movemaker_.initialize(param_.startPoint_.begin() );
   }
   else if(param_.startPoint_.size() != 0) {
      throw RuntimeError("unsuitable starting point");
   }
   else{
      movemaker_.reset();
   }
}
   
template<class GM, class ACC>
inline void 
ICM<GM,ACC>::setStartingPoint
(
   typename std::vector<typename ICM<GM,ACC>::LabelType>::const_iterator begin
) {
   movemaker_.initialize(begin);
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
   bool exitInf=false;
   visitor.begin(*this);
   if(param_.moveType_==SINGLE_VARIABLE ||param_.moveType_==FACTOR) {
      bool updates = true;
      std::vector<bool> isLocalOptimal(gm_.numberOfVariables());
      std::vector<opengm::RandomAccessSet<IndexType> >variableAdjacencyList;
      gm_.variableAdjacencyList(variableAdjacencyList);
      size_t v=0,s=0,n=0;
      while(updates && exitInf==false) {
         updates = false;
         for(v=0; v<gm_.numberOfVariables() && exitInf==false; ++v) {
            if(isLocalOptimal[v]==false) {
               for(s=0; s<gm_.numberOfLabels(v); ++s) {
                  if(s != movemaker_.state(v)) {
                     if(AccumulationType::bop(movemaker_.valueAfterMove(&v, &v+1, &s), movemaker_.value())) {
                        movemaker_.move(&v, &v+1, &s);
                        for(n=0;n<variableAdjacencyList[v].size();++n) {
                           isLocalOptimal[variableAdjacencyList[v][n]]=false;
                        }
                        updates = true;
                        if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){                           exitInf=true;
                           break;
                        }
                     }
                  }
               }
               isLocalOptimal[v]=true;
            }
         }
      }
   }
   if(param_.moveType_==FACTOR) {
      currentMoveType_=FACTOR;
      //visitor(*this, movemaker_.value(),movemaker_.value());
      bool updates = true;
      std::vector<bool> isLocalOptimal(gm_.numberOfFactors(),false);
      //std::vector<opengm::RandomAccessSet<size_t> >variableAdjacencyList;
      opengm::BufferVector<LabelType> stateBuffer;
      stateBuffer.reserve(10);
      //gm_.factorAdjacencyList(variableAdjacencyList);
      size_t f=0,ff=0,v=0;
      while(updates && exitInf==false) {
         updates = false;
         for(f=0; f<gm_.numberOfFactors() && exitInf==false; ++f) {
            if(isLocalOptimal[f]==false && gm_[f].numberOfVariables()>1) {
               stateBuffer.clear();
               stateBuffer.resize(gm_[f].numberOfVariables());
               for(v=0;v<gm_[f].numberOfVariables();++v) {
                  stateBuffer[v]=movemaker_.state(gm_[f].variableIndex(v));
               }
               ValueType oldValue=movemaker_.value();
               ValueType newValue=movemaker_. template moveOptimally<ACC>(gm_[f].variableIndicesBegin(),gm_[f].variableIndicesEnd());   
               if(ACC::bop(newValue,oldValue)) {
                  updates = true ;
                  if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
                     exitInf=true;
                     break;
                  }
                  for(v=0;v<gm_[f].numberOfVariables();++v) {
                     const size_t varIndex=gm_[f].variableIndex(v);
                     if(stateBuffer[v]!=movemaker_.state(varIndex)) {
                        for(ff=0;ff<gm_.numberOfFactors(varIndex);++ff) {
                           isLocalOptimal[gm_.factorOfVariable(varIndex,ff)]=false;
                        }
                     }
                  }
               }
               isLocalOptimal[f]=true;
            }
         }
      }
   }
   visitor.end(*this);
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
   if(N==1) {
      x.resize(gm_.numberOfVariables());
      for(size_t j=0; j<x.size(); ++j) {
         x[j] = movemaker_.state(j);
      }
      return NORMAL;
   }
   else {
      return UNKNOWN;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_ICM_HXX
