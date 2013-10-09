#pragma once
#ifndef OPENGM_PBP_HXX
#define OPENGM_PBP_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/datastructures/dynamic_priority_queue.hxx"

namespace opengm {

   namespace detail_pbp {
      template<class GM>
      class Priority{
      public:
         Priority(const GM & gm)
         :   pQueue_(gm.numberOfVariables())
         {
            for(UInt64Type vi=0;vi<gm.numberOfVariables();++vi){
               pQueue_.insert(vi,gm.numberOfLabels(vi));
            }
         }   

         UInt64Type highestPriorityVi()const{
            OPENGM_CHECK(!pQueue_.isEmpty(),"");
            return pQueue_.minIndex();
         }

         UInt64Type getAndRemoveHighestPriorityVi(){
            OPENGM_CHECK(!pQueue_.isEmpty(),"");
            return pQueue_.deleteMin();
         }


         void removeVi(const UInt64Type vi){
            OPENGM_CHECK(!pQueue_.isEmpty(),"");
            OPENGM_CHECK(pQueue_.contains(vi),"");
            pQueue_.deleteKey(vi);
         }

         void addVi(const UInt64Type vi,const UInt64Type newNLabels){
            //OPENGM_CHECK(!pQueue_.isEmpty(),"");
            OPENGM_CHECK(!pQueue_.contains(vi),"");
            pQueue_.insert(vi,newNLabels);
         }



         void changeConfusingSetSize(const UInt64Type vi,const UInt64Type oldNlabels,const UInt64Type newNLabels){
            OPENGM_CHECK(!pQueue_.isEmpty(),"");
            OPENGM_CHECK(pQueue_.contains(vi),"");
            OPENGM_CHECK_OP(oldNlabels,!=,newNLabels,"");
            if(newNLabels<oldNlabels)
               pQueue_.decreaseKey(vi,newNLabels);
            else
               pQueue_.increaseKey(vi,newNLabels);
         }

      private:

         opengm::MinIndexedPQ pQueue_;
      };
   }
/*

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




template<class GM>
class ActiveLabelSpace{
public:
   typedef GM GraphicalModelType;
   typedef typename GraphicalModelType::LabelType LabelType;
   typedef typename GraphicalModelType::IndexType IndexType;

   LabelType numberOfActiveLabels(const IndexType vi )const{
      return space_[vi].size();
   }

   void removeLabel(const IndexType vi,const LabelType label)const{
      OPENGM_CHECK(space_[vi].find(label)!=space_[vi].end(),"label has already been removed");
   }

   LabelType getLabelAtIndex(const IndexType vi,const IndexType labelIndex){
      OPENGM_CHECK_OP(labelIndex,<,space_[vi].size(),"labelIndex is bigger then allowed");
   }

private:   
   std::vector< RandomAccessSet<LabelType>  > space_;
};

*/
template<class GM>
class MsgBase{
public:
   typedef typename GM::LabelType LabelType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::ValueType ValueType;

   typedef std::map<LabelType,ValueType> MapType;

   MsgBase(const UInt64Type from=0, const UInt64Type to=0,const UInt64Type numberOfLabels=0)
   :  from_(from),
      to_(to),
      numberOfLabels_(numberOfLabels),
      numberOfActiveLabels_(numberOfLabels)
   {
      assertions();
   }

   void setFromTo(const UInt64Type from, const UInt64Type to){
      assertions();
      from_=from;
      to_=to;
      assertions();
   }
   UInt64Type from()const{
      return from_;
   }
   UInt64Type to()const{
      return to_;
   }

   void setNumberOfLabels(const UInt64Type numberOfLabels){
      assertions();
      numberOfLabels_=numberOfLabels;
      numberOfActiveLabels_=numberOfLabels;
      assertions();
   }

   bool hasInformation()const{
      assertions();
      return valueMap_.empty()==false;
   }


   void assertions()const{
      OPENGM_CHECK_OP(numberOfActiveLabels_,<=,numberOfLabels_,"");
      OPENGM_CHECK(numberOfActiveLabels_>=2 || (numberOfLabels_==0) ,"");
      OPENGM_CHECK(numberOfActiveLabels_>=2 || (numberOfLabels_==0 && numberOfActiveLabels_==0) ,"");
      OPENGM_CHECK( valueMap_.size()<=0 || valueMap_.size()==numberOfActiveLabels_,"");
      OPENGM_CHECK( valueMap_.size()!=numberOfLabels_ || valueMap_.size()==0," ");
   }

   template<class LABEL_ITER>
   void pruneLabels
   (
      LABEL_ITER begin,
      LABEL_ITER end
   ){
      this->assertions();
      if(valueMap_.size()==0){

      }
      else{
         // ASSERTIONS
         const size_t nErase=std::distance(begin,end);
         for(size_t i=0;i<nErase;++i){
            OPENGM_CHECK(valueMap_.find(begin[i])!=valueMap_.end(),"some label has been erased");
         }
         while(begin!=end)
            valueMap_.erase(*begin);
      }
      this->assertions();
   }

   size_t mapSize()const{
      return valueMap_;
   }

private:

   UInt64Type from_;
   UInt64Type to_;
   UInt64Type numberOfLabels_;
   UInt64Type numberOfActiveLabels_;

   MapType valueMap_;
};



template<class GM>
class VarToFacMsg : public MsgBase<GM> {
public:
   VarToFacMsg(const UInt64Type from=0, const UInt64Type to=0)
   :  MsgBase<GM>(from,to) {

   }

};

template<class GM>
class FacToVarMsg : public MsgBase<GM> {
public:
   FacToVarMsg(const UInt64Type from=0, const UInt64Type to=0)
   :  MsgBase<GM>(from,to) {

   }
};



template<class GM, class ACC>
class PBP : public Inference<GM, ACC>
{
public:

   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef VerboseVisitor<PBP<GM,ACC> > VerboseVisitorType;
   typedef EmptyVisitor<PBP<GM,ACC> > EmptyVisitorType;
   typedef TimingVisitor<PBP<GM,ACC> > TimingVisitorType;

   typedef FacToVarMsg<GM> FacToVarMsgType;
   typedef VarToFacMsg<GM> VarToFacMsgType;

   class Parameter {
   public:
      Parameter(
         const size_t steps=10
      )
      :  steps_(steps){
      }
      size_t steps_;
   };

   PBP(const GraphicalModelType&, const Parameter& = Parameter());
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

   void forwardPass();
   void backwardPass();
   void getInitialPriorities();

   void updateBeliefAndPriority(const IndexType vi);

   void applyLabelPruning(const IndexType  vi);
   void computeFacToVarMsg(const IndexType fi,const IndexType vi);
   void computeVarToFacMsg(const IndexType vi,const IndexType fi);



   const GraphicalModelType& gm_;
   std::vector<RandomAccessSet<IndexType> > viAdj_;
   Parameter param_;

   // message indexing
   std::vector<UInt64Type> facToVarStar_;
   std::vector<UInt64Type> varToFacStart_;
   std::vector<std::map<UInt64Type,UInt64Type> > facToVarMap_;
   std::vector<std::map<UInt64Type,UInt64Type> > varToFacMap_;


   // message containers
   std::vector<FacToVarMsg<GM> > facToVarMsg_;
   std::vector<VarToFacMsg<GM> > varToFacMsg_;

   // timestamps 
   std::vector<IndexType >   forwardTime_;
   std::vector<LabelType >   confusigSetSize_;

   // 
   detail_pbp::Priority<GM>  queue_;
   std::vector<bool>         isCommited_;

   std::vector < std::set<LabelType > > activeLabels_;
};



template<class GM, class ACC>
PBP<GM, ACC>::PBP
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gm_(gm),
   param_(parameter),
   facToVarStar_(gm.numberOfFactors()),
   varToFacStart_(gm.numberOfVariables()),
   facToVarMap_(gm.numberOfFactors()),
   varToFacMap_(gm.numberOfVariables()),
   facToVarMsg_(),
   varToFacMsg_(),
   forwardTime_(gm.numberOfVariables()),
   confusigSetSize_(gm.numberOfVariables()),
   queue_(gm),
   isCommited_(gm.numberOfVariables(),false),
   activeLabels_(gm.numberOfVariables())
{
   std::cout<<"0\n";
   gm_.variableAdjacencyList(viAdj_);
   // counting the messages
   Int64Type varToFacCounter=0;
   UInt64Type facToVarCounter=0;

   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
      OPENGM_CHECK_OP(gm_.numberOfLabels(vi),>=,2,"wrong gm");
      const IndexType nFac=gm_.numberOfFactors(vi);
      varToFacStart_[vi]=varToFacCounter;
      varToFacCounter+=nFac;

      confusigSetSize_[vi]=gm_.numberOfLabels(vi);

      for(LabelType l=0;l<gm_.numberOfLabels(vi);++l){
         activeLabels_[vi].insert(l);
      }
   }
   std::cout<<"1\n";
   for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
      const IndexType nVar=gm_[fi].numberOfVariables();
      facToVarStar_[fi]=facToVarCounter;
      facToVarCounter+=nVar;
   }
   std::cout<<"2\n";
   // resizing message containers
   varToFacMsg_.resize(varToFacCounter);
   facToVarMsg_.resize(facToVarCounter);
   facToVarCounter=0;
   varToFacCounter=0;
   std::cout<<"3\n";
   // fill message informations
   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
      const IndexType nFac=gm_.numberOfFactors(vi);
      for(IndexType f=0;f<nFac;++f){
         varToFacMsg_[varToFacCounter].setFromTo(vi,gm_.factorOfVariable(vi,f));
         varToFacMsg_[varToFacCounter].setNumberOfLabels(gm_.numberOfLabels(vi));
         varToFacMap_[vi][gm_.factorOfVariable(vi,f)]=varToFacCounter;
         ++varToFacCounter;
      }
   }
   std::cout<<"4\n";
   for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
      const IndexType nVar=gm_[fi].numberOfVariables();
      for(IndexType v=0;v<nVar;++v){
         facToVarMsg_[facToVarCounter].setFromTo(fi,gm_.variableOfFactor(fi,v));
         facToVarMsg_[facToVarCounter].setNumberOfLabels(gm_.numberOfLabels(gm_.variableOfFactor(fi,v)));
         facToVarMap_[fi][gm_.variableOfFactor(fi,v)]=facToVarCounter;
         ++facToVarCounter;
      }
   }
   std::cout<<"5\n";
}


template<class GM, class ACC>
void PBP<GM, ACC>::getInitialPriorities(){

}

template<class GM, class ACC>
void PBP<GM, ACC>::updateBeliefAndPriority(const typename PBP<GM, ACC>::IndexType vi){

}


template<class GM, class ACC>
void PBP<GM, ACC>::applyLabelPruning(const typename PBP<GM, ACC>::IndexType  vi){

   std::vector<LabelType> labelsToPrune;
   // .... 
   // ....
   // ....
   if(!labelsToPrune.empty()){
      // prune all variable to factor msg
      const IndexType nFac = gm_.numberOfFactors(vi);
      for(IndexType f=0;f<nFac;++f){
         const IndexType fi=gm_.factorOfVariable(vi,f);

         varToFacMsg_[varToFacMap_[vi][fi]].pruneLabels(labelsToPrune.begin(),labelsToPrune.end());
         facToVarMsg_[facToVarMap_[fi][vi]].pruneLabels(labelsToPrune.begin(),labelsToPrune.end());
      }
      //beliefs_[vi].pruneLabels(labelsToPrune.begin(),labelsToPrune.end());
   }



}

template<class GM, class ACC>
void PBP<GM, ACC>::computeFacToVarMsg(const typename PBP<GM, ACC>::IndexType fi,const typename PBP<GM, ACC>::IndexType vi){
   FacToVarMsgType & outMsg = facToVarMsg_[facToVarMap_[fi][vi]];
   OPENGM_CHECK_OP(outMsg.from(),==,fi,"");
   OPENGM_CHECK_OP(outMsg.to(),  ==,vi,"");
}

template<class GM, class ACC>
void PBP<GM, ACC>::computeVarToFacMsg(const typename PBP<GM, ACC>::IndexType vi,const typename PBP<GM, ACC>::IndexType fi){

   VarToFacMsgType & outMsg = varToFacMsg_[varToFacMap_[vi][fi]];
   OPENGM_CHECK_OP(outMsg.from(),==,vi,"");
   OPENGM_CHECK_OP(outMsg.to(),  ==,fi,"");


   // Variable to factor update:
   // get  all factor to variable messages
   // where variable is vi and factor is NOT fi
   const IndexType nFac=gm_.numberOfFactors(vi);
   for(IndexType f=0;f<nFac;++f){
      const IndexType otherFi=gm_.factorOfVariable(vi,f);
      if(otherFi!=fi){
         // get the other message
         const FacToVarMsgType & factorToVarMsg = facToVarMsg_[facToVarMap_[otherFi][vi]];
         OPENGM_CHECK_OP(factorToVarMsg.from(),==,otherFi,"");
         OPENGM_CHECK_OP(factorToVarMsg.to(),  ==,vi,"");
      }
   }

}



template<class GM, class ACC>      
void PBP<GM, ACC>::forwardPass(){
   std::cout<<"start forwardPass\n";

   for(IndexType node=0;node<gm_.numberOfVariables();++node){

      // get node with smallest confusing set / highest priority
      const IndexType vi=queue_.getAndRemoveHighestPriorityVi();
      // check that vi IS NOT commited
      OPENGM_CHECK(isCommited_[vi]==false,"internal error");

      // make the node commited
      // and make timstamp
      isCommited_[vi]=true;
      forwardTime_[node]=vi;


      // apply label pruning to this node
      this->applyLabelPruning(vi);

      // send to all uncommited neighbors 
      const IndexType nFac=gm_.numberOfFactors(vi);
      for(IndexType f=0;f<nFac;++f){
         const IndexType fi=gm_.factorOfVariable(vi,f);
         OPENGM_CHECK_OP(gm_[fi].numberOfVariables(),<=,2,"order higher than 2 are not yet implemented");
         if(gm_[fi].numberOfVariables()==1){
            // they are handled only once
         }
         else if(gm_[fi].numberOfVariables()==2) {
            const IndexType otherVi= (gm_[fi].variableIndex(0)==vi ?   gm_[fi].variableIndex(1) : gm_[fi].variableIndex(0));

            // only send messages if other fi IS NOT commited
            if(isCommited_[otherVi]==false){
               // send message  vi -> fi  ; fi -> otherVi
               std::cout<<"send vi("<<vi<<") -> fi("<<fi<<") -> ovi("<<otherVi<<")\n";
               this->computeVarToFacMsg(vi,fi);
               this->computeFacToVarMsg(fi,otherVi);
            }
         }
      }

      // update beliefs for all uncommited neigbours
      for(IndexType nv=0;nv<viAdj_[vi].size();++nv){
         const IndexType otherVi=viAdj_[vi][nv];
         // only send messages if other fi IS NOT commited
         if(isCommited_[otherVi]==false){
            this->updateBeliefAndPriority(otherVi);
         }
      }
   }
}

template<class GM, class ACC>
void PBP<GM, ACC>::backwardPass(){
   std::cout<<"start backwardPass\n";

   for(IndexType node=0;node<gm_.numberOfVariables();++node){
      // get nodes from timestamp
      const IndexType vi=forwardTime_[gm_.numberOfVariables()-1-node];
      // check that vi IS commited
      OPENGM_CHECK(isCommited_[vi]==true,"internal error");

      // make the node commited
      // and make timstamp
      isCommited_[vi]=false;
      // re-add to queue_
      queue_.addVi(vi,confusigSetSize_[vi]);

      const IndexType nFac=gm_.numberOfFactors(vi);
      for(IndexType f=0;f<nFac;++f){
         const IndexType fi=gm_.factorOfVariable(vi,f);
         if(gm_[fi].numberOfVariables()==1){
            // they are handled only once
         }
         else if(gm_[fi].numberOfVariables()==2) {
            const IndexType otherVi= (gm_[fi].variableIndex(0)==vi ?   gm_[fi].variableIndex(1) : gm_[fi].variableIndex(0));
            // only send messages if other fi IS commited
            if(isCommited_[otherVi]==true){
               // send message  vi -> fi  ; fi -> otherVi
               std::cout<<"send vi("<<vi<<") -> fi("<<fi<<") -> ovi("<<otherVi<<")\n";
               this->computeVarToFacMsg(vi,fi);
               this->computeFacToVarMsg(fi,otherVi);
            }
         }
      }
      // update beliefs for all commited neigbours
      for(IndexType nv=0;nv<viAdj_[vi].size();++nv){
         const IndexType otherVi=viAdj_[vi][nv];
         // only send messages if other fi IS  commited
         if(isCommited_[otherVi]==true){
            this->updateBeliefAndPriority(otherVi);
         }
      }
   }
}

template<class GM, class ACC>
template<class VisitorType>
InferenceTermination PBP<GM,ACC>::infer
(
   VisitorType& visitor
)
{
   visitor.begin(*this);
   
   for(size_t s=0;s<param_.steps_;++s){
      std::cout<<"iteration s="<<s<<"\n";
      this->forwardPass();
      this->backwardPass();

   }

   visitor.end(*this);
   return NORMAL;
}


   
template<class GM, class ACC>
inline void 
PBP<GM,ACC>::setStartingPoint
(
   typename std::vector<typename PBP<GM,ACC>::LabelType>::const_iterator begin
) {

}
   
template<class GM, class ACC>
inline std::string
PBP<GM, ACC>::name() const
{
   return "PBP";
}

template<class GM, class ACC>
inline const typename PBP<GM, ACC>::GraphicalModelType&
PBP<GM, ACC>::graphicalModel() const
{
   return gm_;
}
  
template<class GM, class ACC>
inline InferenceTermination
PBP<GM,ACC>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  


template<class GM, class ACC>
inline InferenceTermination
PBP<GM,ACC>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{
   x.resize(gm_.numberOfVariables(),0);
   return NORMAL;
}

} // namespace opengm

#endif // #ifndef OPENGM_PBP_HXX
