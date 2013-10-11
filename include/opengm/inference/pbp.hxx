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

template<class GM>
class MsgBase{
public:
   typedef typename GM::LabelType LabelType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::ValueType ValueType;

   typedef std::map<LabelType,ValueType> MapType;

   typedef typename MapType::const_iterator  ConstMapIter;
   typedef typename MapType::iterator        MapIter;

   MsgBase(const UInt64Type from=0, const UInt64Type to=0,const UInt64Type numberOfLabels=0)
   :  from_(from),
      to_(to),
      numberOfLabels_(numberOfLabels),
      numberOfActiveLabels_(numberOfLabels),
      isInit_(false)
   {
      assertions(__LINE__);
   }

   void setFromTo(const UInt64Type from, const UInt64Type to){
      assertions(__LINE__);
      from_=from;
      to_=to;
      assertions(__LINE__);
   }
   UInt64Type from()const{
      return from_;
   }
   UInt64Type to()const{
      return to_;
   }

   void setNumberOfLabels(const UInt64Type numberOfLabels){
      assertions(__LINE__);
      numberOfLabels_=numberOfLabels;
      numberOfActiveLabels_=numberOfLabels;
      assertions(__LINE__);
   }

   bool hasInformation()const{
      assertions(__LINE__);
      return valueMap_.empty()==false;
   }

   template<class LINE>
   void assertions(LINE line)const{
      
      OPENGM_CHECK_OP(numberOfActiveLabels_,<=,numberOfLabels_,line);
      OPENGM_CHECK(numberOfActiveLabels_>=2 || (numberOfLabels_==0) ,line);
      OPENGM_CHECK(numberOfActiveLabels_>=2 || (numberOfLabels_==0 && numberOfActiveLabels_==0) ,line);
      OPENGM_CHECK( valueMap_.size()>=0 || valueMap_.size()==numberOfActiveLabels_,line);
      OPENGM_CHECK( valueMap_.size()<=numberOfActiveLabels_,line);

   }

   template<class LABEL_ITER>
   void pruneLabels
   (
      LABEL_ITER begin,
      LABEL_ITER end
   ){
      OPENGM_CHECK_OP(numberOfLabels_,!=,0,"");
      OPENGM_CHECK_OP(numberOfActiveLabels_,!=,0,"");
      OPENGM_CHECK_OP(numberOfActiveLabels_,<=,numberOfLabels_,"");

      this->assertions(__LINE__);
      if(valueMap_.size()==0){

      }
      else{
         // ASSERTIONS
         const size_t nErase=std::distance(begin,end);
         for(size_t i=0;i<nErase;++i){
            OPENGM_CHECK(valueMap_.find(begin[i])!=valueMap_.end(),"some label has already been erased");
            valueMap_.erase(begin[i]);
            --numberOfActiveLabels_;
         }
      }
      this->assertions(__LINE__);
   }

   size_t mapSize()const{
      return valueMap_.size();
   }

   const MapType & valueMap()const{
      return valueMap_;
   }

   MapType & valueMap(){
      return valueMap_;
   }

   template<class OTHER>
   void initFrom(const OTHER & msg){
      if(msg.mapSize()==0){
         valueMap_.clear();
      }
      else{
         valueMap_=msg.valueMap();
      }
   }


   template<class OTHER>
   void opMsg(const OTHER & msg){
      if(msg.mapSize()==0){
         // do nothing
      }
      else if(this->mapSize()==0){
         this->initFrom(msg);
      }
      else{
         OPENGM_CHECK_OP(msg.mapSize(),==,this->mapSize(),"different sizes");

         MapIter      beginA = this->valueMap().begin(); 
         ConstMapIter beginB = msg.valueMap().begin();
         

         while(beginA!=this->valueMap_.end()){

            beginA->second+= beginB->second;
            // check label equality
            OPENGM_CHECK_OP(beginA->first,==,beginB->first,"labels do not match");

            ++beginA;
            ++beginB;
         }

      }
   }

   
   void normalize(){
      MapIter     beginA = this->valueMap().begin(); 
      ValueType   minVal = std::numeric_limits<ValueType>::infinity();
      while(beginA!=this->valueMap_.end()){
         minVal=std::min(minVal,beginA->second);
         ++beginA;
      }
      beginA = this->valueMap().begin(); 
      while(beginA!=this->valueMap_.end()){
         beginA->second -= minVal;
         ++beginA;
      }
   }
   

   bool isInit()const{
      return isInit_;
   }

   void setAsInit(){
      isInit_=true;
   }

private:

   UInt64Type from_;
   UInt64Type to_;
   UInt64Type numberOfLabels_;
   UInt64Type numberOfActiveLabels_;

   MapType valueMap_;

   bool isInit_;
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

   typedef typename FacToVarMsgType::MapIter MapIter;
   typedef typename FacToVarMsgType::ConstMapIter ConstMapIter;
   typedef typename FacToVarMsgType::MapType MapType;

   typedef std::set<LabelType > LabelSetType;
   typedef typename LabelSetType::const_iterator ConstLabelSetIter;


   class Parameter {
   public:
      Parameter(
         const size_t      steps                   =10,
         const ValueType   damping                 =0.5,
         const bool        preInitUnarieMessages   =true,
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

   // belief container
   std::vector<VarToFacMsgType> beliefs_;

   // timestamps 
   std::vector<IndexType >   forwardTime_;
   std::vector<LabelType >   confusigSetSize_;

   // 
   detail_pbp::Priority<GM>  queue_;
   std::vector<bool>         isCommited_;

   std::vector < std::set<LabelType > > activeLabels_;

   std::vector<ValueType> valBuffer1_;
   std::vector<ValueType> valBuffer2_;


   //
   std::vector<unsigned char> dirtyPriority_;
};

template<class MAP,class VEC>
void copyToVec(const MAP & map, VEC & vec){
   size_t c=0;
   for(typename MAP::const_iterator i=map.begin();i!=map.end();++i){
      vec[c]=i->second;
      ++c;
   }

}


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
   beliefs_(gm.numberOfVariables()),
   forwardTime_(gm.numberOfVariables()),
   confusigSetSize_(gm.numberOfVariables()),
   queue_(gm),
   isCommited_(gm.numberOfVariables(),false),
   activeLabels_(gm.numberOfVariables()),
   valBuffer1_(),
   valBuffer2_(),
   dirtyPriority_(gm.numberOfVariables(),0)
{
   //std::cout<<"0\n";
   gm_.variableAdjacencyList(viAdj_);
   // counting the messages
   Int64Type varToFacCounter=0;
   UInt64Type facToVarCounter=0;

   LabelType maxLabel=0;

   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
      OPENGM_CHECK_OP(gm_.numberOfLabels(vi),>=,2,"wrong gm");
      const IndexType nFac=gm_.numberOfFactors(vi);
      varToFacStart_[vi]=varToFacCounter;
      varToFacCounter+=nFac;

      confusigSetSize_[vi]=gm_.numberOfLabels(vi);
      maxLabel=std::max(maxLabel,gm_.numberOfLabels(vi));

      beliefs_[vi].setNumberOfLabels(gm_.numberOfLabels(vi));
      for(LabelType l=0;l<gm_.numberOfLabels(vi);++l){
         activeLabels_[vi].insert(l);
      }
   }
   valBuffer1_.resize(maxLabel+1);
   valBuffer2_.resize(maxLabel+1);

   //std::cout<<"1\n";
   for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
      const IndexType nVar=gm_[fi].numberOfVariables();
      facToVarStar_[fi]=facToVarCounter;
      facToVarCounter+=nVar;
   }
   //std::cout<<"2\n";
   // resizing message containers
   varToFacMsg_.resize(varToFacCounter);
   facToVarMsg_.resize(facToVarCounter);
   facToVarCounter=0;
   varToFacCounter=0;
   //std::cout<<"3\n";
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
   //std::cout<<"4\n";
   for(IndexType fi=0;fi<gm_.numberOfFactors();++fi){
      const IndexType nVar=gm_[fi].numberOfVariables();
      //std::cout<<"fi "<<fi<<"   order  "<<nVar<<" \n";
      for(IndexType v=0;v<nVar;++v){
         facToVarMsg_[facToVarCounter].setFromTo(fi,gm_.variableOfFactor(fi,v));
         facToVarMsg_[facToVarCounter].setNumberOfLabels(gm_.numberOfLabels(gm_.variableOfFactor(fi,v)));
         facToVarMap_[fi][gm_.variableOfFactor(fi,v)]=facToVarCounter;
         ++facToVarCounter;
      }
   }
   //std::cout<<"5\n";
}


template<class GM, class ACC>
void PBP<GM, ACC>::getInitialPriorities(){

}

template<class GM, class ACC>
void PBP<GM, ACC>::updateBeliefAndPriority(const typename PBP<GM, ACC>::IndexType vi){
   OPENGM_CHECK(isCommited_[vi]==false,"");


   VarToFacMsgType & belief = beliefs_[vi];
   bool first=true;

   const IndexType nFac = gm_.numberOfFactors(vi);
   for(IndexType f=0;f<nFac;++f){
      const IndexType fi=gm_.factorOfVariable(vi,f);

      const FacToVarMsgType & otherMsg = facToVarMsg_[facToVarMap_[fi][vi]];

      if(first==true){
         belief.initFrom(otherMsg);
         first=false;
      }
      else{
         belief.opMsg(otherMsg);
      }
   }
   //std::cout<<"\n vi == "<<vi<<"\n";
   //for(MapIter it=belief.valueMap().begin()   ;it!=belief.valueMap().end()   ;++it){
   //   std::cout<<" l"<<it->first<<"  "<<it->second<<" \n";
   //} 
}


template<class GM, class ACC>
void PBP<GM, ACC>::applyLabelPruning(const typename PBP<GM, ACC>::IndexType  vi){

   const MapType & belief = beliefs_[vi].valueMap();
   
   if(belief.size()==0){
      //std::cout<<"unit belief "<<vi<<"\n";
   }
   else if (activeLabels_[vi].size()==2){
      //std::cout<<"just 2 left  "<<vi<<"\n";
   }
   else{


      const size_t maxToPrune = activeLabels_[vi].size()  -2 ;

      std::vector<LabelType> labelsToPrune;
      //std::cout<<"belief is initialized "<<vi<<"\n";


      ValueType minValue=std::numeric_limits<ValueType>::infinity();

      for(ConstMapIter iter=belief.begin();iter!=belief.end();++iter){
         //std::cout<<"label "<<iter->first<<" RAW value "<<iter->second<<"\n";
         minValue=std::min(iter->second,minValue);
      }
      //std::cout<<"\nMIN VALUE "<<minValue<<"\n";
      for(ConstMapIter iter=belief.begin();iter!=belief.end();++iter){
         const ValueType relBelief = iter->second-minValue;
         //std::cout<<"label "<<iter->first<<"     value "<<relBelief<<"\n";

         if(relBelief>0.2){
            labelsToPrune.push_back(iter->first);

            //OPENGM_CHECK(activeLabels_[vi].find())
         }
         if(maxToPrune==labelsToPrune.size()){
            break;
         }
      }

      //std::cout<<"\n to prune: "<<labelsToPrune.size()<<"\n";

      OPENGM_CHECK_OP(activeLabels_[vi].size()-labelsToPrune.size(),>=,2,"");

      if(!labelsToPrune.empty()){
         // prune all variable to factor msg
         const IndexType nFac = gm_.numberOfFactors(vi);

         beliefs_[vi].pruneLabels(labelsToPrune.begin(),labelsToPrune.end());


         for(size_t i=0;i<labelsToPrune.size();++i){
            OPENGM_CHECK(activeLabels_[vi].find(labelsToPrune[i])!=activeLabels_[vi].end(),"");
            activeLabels_[vi].erase(labelsToPrune[i]);
            //std::cout<<"pruned label "<<labelsToPrune[i]<<"\n";
         }


         size_t sp = beliefs_[vi].valueMap().size();
         //std::cout<<"\n   B "<<beliefs_[vi].valueMap().size()<<"\n";
         for(IndexType f=0;f<nFac;++f){
            const IndexType fi=gm_.factorOfVariable(vi,f);

            varToFacMsg_[varToFacMap_[vi][fi]].pruneLabels(labelsToPrune.begin(),labelsToPrune.end());
            facToVarMsg_[facToVarMap_[fi][vi]].pruneLabels(labelsToPrune.begin(),labelsToPrune.end());


            //std::cout<<"v->f "<<varToFacMsg_[varToFacMap_[vi][fi]].valueMap().size()<<" "<<varToFacMap_[vi][fi]<<"\n";
            //std::cout<<"f->v "<<facToVarMsg_[facToVarMap_[fi][vi]].valueMap().size()<<" "<<facToVarMap_[fi][vi]<<"\n";
         }
      }

   }



}

template<class GM, class ACC>
void PBP<GM, ACC>::computeFacToVarMsg(const typename PBP<GM, ACC>::IndexType fi,const typename PBP<GM, ACC>::IndexType vi){
   FacToVarMsgType & outMsg = facToVarMsg_[facToVarMap_[fi][vi]];


   //std::cout<<"compute fac tor var msg with index "<<facToVarMap_[fi][vi]<<"\n";
   //std::cout<<"value map size bevore "<<outMsg.valueMap().size()<<"\n";

   
   OPENGM_CHECK_OP(outMsg.from(),==,fi,"");
   OPENGM_CHECK_OP(outMsg.to(),  ==,vi,"");

   OPENGM_CHECK_OP(gm_[fi].numberOfVariables(),<=,2,"");

   if(gm_[fi].numberOfVariables()==1){
      if(outMsg.isInit()==false){
         //std::cout<<"initialize values\n";
      
         for(ConstLabelSetIter iter0 = activeLabels_[vi].begin();iter0!=activeLabels_[vi].end();++iter0){
            const LabelType l=*iter0;
            outMsg.valueMap()[l]=gm_[fi](&l);
         }
         outMsg.setAsInit();
      }

   }
   else{

      const LabelType vis[2] ={  gm_[fi].variableIndex(0),gm_[fi].variableIndex(1)};


      LabelType coordinateBuffer[2]={0,0};



      const IndexType pos0 = ( vi==vis[0] ?  0 : 1 );
      const IndexType pos1 = ( vi==vis[0] ?  1 : 0 );

      const VarToFacMsgType & otherMsg=varToFacMsg_[varToFacMap_[vis[pos1]][fi]];

      size_t lc0=0;
      for(ConstLabelSetIter iter0 = activeLabels_[vis[pos0]].begin();iter0!=activeLabels_[vis[pos0]].end();++iter0){
         const LabelType l0=*iter0;
         coordinateBuffer[pos0]=l0;
         ValueType minVal = std::numeric_limits<ValueType>::infinity();

         if(otherMsg.mapSize()==0){        
            for(ConstLabelSetIter iter1 = activeLabels_[vis[pos1]].begin();iter1!=activeLabels_[vis[pos1]].end();++iter1){
               const LabelType l1=*iter1;
               coordinateBuffer[pos1]=l1;
               minVal = std::min(minVal,gm_[fi](coordinateBuffer));

            }
            // write min value into result message
            valBuffer2_[lc0]=minVal;
         }
         else{
            //copyToVec(otherMsg.valueMap(),valBuffer1_);

            ConstMapIter label2Iter=otherMsg.valueMap().begin();

            OPENGM_CHECK_OP(otherMsg.to(),==,fi,"");
            OPENGM_CHECK_OP(otherMsg.from(),==,vis[pos1],"");

            size_t lc1=0;
            for(ConstLabelSetIter iter1 = activeLabels_[vis[pos1]].begin();iter1!=activeLabels_[vis[pos1]].end();++iter1){
               const LabelType l1=*iter1;
               OPENGM_CHECK_OP(l1,==,label2Iter->first,"");
               coordinateBuffer[pos1]=l1;
               //minVal = std::min(minVal,gm_[fi](coordinateBuffer)+valBuffer1_[lc1]);
               minVal = std::min(minVal,gm_[fi](coordinateBuffer)+label2Iter->second);
               ++lc1;
               ++label2Iter;
            }
            valBuffer2_[lc0]=minVal;
         }
         ++lc0;
      }

      // copy result to out msg
      if(outMsg.mapSize()==0){
         lc0=0;

         //std::cout<<"active label size "<<activeLabels_[vis[pos0]].size()<<" \n";
         for(ConstLabelSetIter iter0 = activeLabels_[vis[pos0]].begin();iter0!=activeLabels_[vis[pos0]].end();++iter0){
            const LabelType l0=*iter0;
            outMsg.valueMap()[l0]=valBuffer2_[lc0];
            ++lc0;
         }
      }
      else{
         lc0=0;
         for(MapIter iter0 = outMsg.valueMap().begin();iter0!=outMsg.valueMap().end();++iter0){
            iter0->second = valBuffer2_[lc0];
            ++lc0;
         }

      }


   }


   if(outMsg.valueMap().size()!=0){
      OPENGM_CHECK_OP(outMsg.valueMap().size(),==,activeLabels_[vi].size(),"");
   }

   //std::cout<<"value map size after "<<outMsg.valueMap().size()<<"\n";
}

template<class GM, class ACC>
void PBP<GM, ACC>::computeVarToFacMsg(const typename PBP<GM, ACC>::IndexType vi,const typename PBP<GM, ACC>::IndexType fi){

   float avLabels = 0;

   for(IndexType vii=0;vii<gm_.numberOfVariables();++vii){
      avLabels+=activeLabels_[vii].size();
   }

   avLabels/=gm_.numberOfVariables();
   std::cout<<"av labels "<<avLabels<<"\n";


   //std::cout<<"computeVarToFacMsg\n";

   VarToFacMsgType & outMsg = varToFacMsg_[varToFacMap_[vi][fi]];

   //std::cout<<"this message index "<<varToFacMap_[vi][fi]<<" var index "<<vi<<"\n";

   OPENGM_CHECK_OP(outMsg.from(),==,vi,"");
   OPENGM_CHECK_OP(outMsg.to(),  ==,fi,"");

   bool first=true;
   // Variable to factor update:
   // get  all factor to variable messages
   // where variable is vi and factor is NOT fi
   const IndexType nFac=gm_.numberOfFactors(vi);
   for(IndexType f=0;f<nFac;++f){
      const IndexType otherFi=gm_.factorOfVariable(vi,f);
      if(otherFi!=fi){
         // get the other message
         //std::cout<<"fac to var msg index "<<facToVarMap_[otherFi][vi]<<" var index "<<vi<<"\n";
         const FacToVarMsgType & factorToVarMsg = facToVarMsg_[facToVarMap_[otherFi][vi]];
         OPENGM_CHECK_OP(factorToVarMsg.from(),==,otherFi,"");
         OPENGM_CHECK_OP(factorToVarMsg.to(),  ==,vi,"");

         if(first){
            outMsg.initFrom(factorToVarMsg);
            first=false;
         }
         else{
            outMsg.opMsg(factorToVarMsg);
         }
      }
   }
   //std::cout<<"computeVarToFacMsg --DONE\n";

}



template<class GM, class ACC>      
void PBP<GM, ACC>::forwardPass(){
   //std::cout<<"start forwardPass\n";


   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
      if(dirtyPriority_[vi]==1){
         updateBeliefAndPriority(vi);
         dirtyPriority_[vi]=0;
      }
   }

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

            this->computeFacToVarMsg(fi,vi);
            this->computeVarToFacMsg(vi,fi);
         }
         else if(gm_[fi].numberOfVariables()==2) {
            const IndexType otherVi= (gm_[fi].variableIndex(0)==vi ?   gm_[fi].variableIndex(1) : gm_[fi].variableIndex(0));

            // only send messages if other fi IS NOT commited
            if(isCommited_[otherVi]==false){
               // send message  vi -> fi  ; fi -> otherVi
               //std::cout<<"send vi("<<vi<<") -> fi("<<fi<<") -> ovi("<<otherVi<<")\n";
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



   //std::cout<<"start backwardPass\n";

   for(IndexType node=0;node<gm_.numberOfVariables();++node){
      // get nodes from timestamp
      const IndexType vi=forwardTime_[gm_.numberOfVariables()-1-node];
      // check that vi IS commited
      OPENGM_CHECK(isCommited_[vi]==true,"internal error");

      // make the node commited
      // and make timstamp
      isCommited_[vi]=false;
      // re-add to queue_
      queue_.addVi(vi,activeLabels_[vi].size());

      const IndexType nFac=gm_.numberOfFactors(vi);
      for(IndexType f=0;f<nFac;++f){
         const IndexType fi=gm_.factorOfVariable(vi,f);
         if(gm_[fi].numberOfVariables()==1){
            this->computeFacToVarMsg(fi,vi);
            this->computeVarToFacMsg(vi,fi);
         }
         else if(gm_[fi].numberOfVariables()==2) {
            const IndexType otherVi= (gm_[fi].variableIndex(0)==vi ?   gm_[fi].variableIndex(1) : gm_[fi].variableIndex(0));
            // only send messages if other fi IS commited
            if(isCommited_[otherVi]==true){
               // send message  vi -> fi  ; fi -> otherVi
               //std::cout<<"send vi("<<vi<<") -> fi("<<fi<<") -> ovi("<<otherVi<<")\n";
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
            dirtyPriority_[otherVi]=1;
            //this->updateBeliefAndPriority(otherVi);
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

      // update priorities from backward pass
      for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
         this->updateBeliefAndPriority(vi);
      }

      // normalize
      //std::cout<<"normalize\n";
      for(IndexType m=0;m<varToFacMsg_.size();++m){
         varToFacMsg_[m].normalize();
      }
      for(IndexType m=0;m<varToFacMsg_.size();++m){
         varToFacMsg_[m].normalize();
      }
      float avLabels = 0;

      for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
         avLabels+=activeLabels_[vi].size();
      }

      avLabels/=gm_.numberOfVariables();
      std::cout<<"av labels "<<avLabels<<"\n";


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
