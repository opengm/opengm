#pragma once
#ifndef OPENGM_CGC_HXX
#define OPENGM_CGC_HXX

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/format.hpp>
#include <boost/unordered_set.hpp>

#include "opengm/opengm.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/datastructures/buffer_vector.hxx"


#include "opengm/inference/cgc/submodel2.hxx"
#include "opengm/inference/cgc/generate_starting_point.hxx"



namespace opengm {
 
namespace detail_gcg{

   /**
    * run connected component labeling of nodes in gm in place
    * given colors in labels.
    * --> dense relabeling
    */
   template<class GM,class LABELS_ITER>
   typename GM::IndexType getCCFromLabels(
      const GM & gm,
      LABELS_ITER  labels
   ){
      typedef typename GM::IndexType IndexType;
      typedef typename GM::LabelType LabelType;

      // merge with UFD
      opengm::Partition<IndexType> ufd(gm.numberOfVariables());
      for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
         const LabelType label=labels[vi];
         const IndexType numFacVar = static_cast<IndexType>(gm.numberOfFactors(vi));
         for(IndexType f=0;f<numFacVar;++f){
            const IndexType fi        = gm.factorOfVariable(vi,f);
            const IndexType numVarFac = gm[fi].numberOfVariables();
            for(size_t v=0;v<numVarFac;++v){
               const IndexType vi2=gm[fi].variableIndex(v);
               const LabelType label2=labels[vi2];
               if(vi!=vi2 && label==label2){
                  ufd.merge(vi,vi2);
               }
            }
         }
      }
      std::map<IndexType,IndexType> repLabeling;
      ufd.representativeLabeling(repLabeling);
      const size_t numberOfCCs=ufd.numberOfSets();

      for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
         IndexType findVi=ufd.find(vi);
         IndexType denseRelabling=repLabeling[findVi];
         labels[vi]=denseRelabling;
      }
      return numberOfCCs;
   }

   /**
    * toFind: colors of interest
    * container: where to search in (a node coloring) 
    * position: index into container for an anchor, has length of toFind
    *           (undefined if not found)
    * found: length of toFind, whether this color was found
    */
   template<class CT,class C,class FP,class F>
   void findFirst(
      const CT & toFind,
      const C & container,
      FP & position,
      F & found
   ){
      typedef typename CT::value_type ToFindType;
      typedef typename FP::value_type ResultTypePosition;
      // fill map with positions of values to find 
      typedef std::map<ToFindType,size_t> MapType;
      typedef typename MapType::const_iterator MapIter;
      MapType toFindPosition;
      for(size_t i=0;i<toFind.size();++i){
         toFindPosition.insert(std::pair<ToFindType,size_t>(toFind[i],i));
         found[i]=false;
      }

      // find values
      size_t numFound=0;
      for(size_t i=0;i<container.size();++i){
         const ToFindType value = container[i];
         MapIter findVal=toFindPosition.find(value);

         if( findVal!=toFindPosition.end()){
            const size_t posInToFind = findVal->second;
            if(found[posInToFind]==false){
               position[posInToFind]=static_cast<ResultTypePosition>(i);
               found[posInToFind]=true;
               numFound+=1;
            }
            if(numFound==toFind.size()){
               break;
            }
         }
      }
   } 
}


/// \brief Experimental Multicut 
///
template<class GM, class ACC>
class CGC : public Inference<GM, ACC>
{
public:

   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<CGC<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<CGC<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<CGC<GM,ACC> >  TimingVisitorType;

   typedef std::pair<int,ValueType> IVPairType;
   typedef PottsFunction<ValueType,IndexType,LabelType> PfType;

   typedef  GraphicalModel<ValueType, Adder, PfType , typename GM::SpaceType> PottsGmType;

   class Parameter {
   public:
      Parameter(
         const bool planar           = true,
         const size_t maxIterations  = 1,
         const bool useBookkeeping   = true,
         const double threshold      = 0.0, 
         const bool startFromThreshold = true,
         const bool doCutMove = true,
         const bool doGlueCutMove = true
      ):
         planar_(planar),
         maxIterations_(maxIterations),
         useBookkeeping_(useBookkeeping),
         threshold_(threshold),
         startFromThreshold_(startFromThreshold),
         doCutMove_(doCutMove),
         doGlueCutMove_(doGlueCutMove_)
      {}
      
      bool planar_;
      size_t maxIterations_;
      bool useBookkeeping_;
      double threshold_;
      bool startFromThreshold_;
      bool doCutMove_;
      bool doGlueCutMove_;



   };

   CGC(const GraphicalModelType&, const Parameter&  param = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   void reset();
   
   ValueType bound() const {
      return bound_+energyOffset_;
   }
   ValueType value() const {
      return value_+energyOffset_;
   }
   ValueType calcBound(){ return 0; }

   InferenceTermination infer();
   template<class VisitorType>
   InferenceTermination infer(VisitorType&);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);

   ValueType evalPrimal() const;
   ValueType evalPrimal2(const std::vector<LabelType>&) const;


   ~CGC(){
      delete submodel_;
   }


   private:
   bool inRecursive2Coloring()const{
      return inRecursive2Coloring_;
   }
   bool inGreedy2Coloring()const{
      return inGreedy2Coloring_;
   }
   
   void findActiveFactors(std::vector<IndexType> activeFactors){
      activeFactors.clear();
      for(IndexType fi=0;fi<numDualVar_;++fi){
         if(argDual_[fi]!=0)
            activeFactors.push_back(fi);
      }
   }
      
   LabelType setStartingPointFromArgPrimal(const bool fillQ);

   void primalToDual();
   ValueType evalDual()const;

   template<class VISITOR>
   void recursive2Coloring(VISITOR & visitor);

   template<class VISITOR>
   void greedy2ColoringPlanar(VISITOR & visitor);


   const GraphicalModelType& gmRaw_;

   PottsGmType gm_;

   Parameter param_;

   std::vector<ValueType> lambdas_;

   SubmodelCGC<PottsGmType> * submodel_;

   // redundant data for easy readability
   IndexType numVar_;
   IndexType numDualVar_;

   // current value and naive bound
   ValueType value_;
   ValueType bound_;

   // current primal and dual arg
   // and the current max Color in arg Primal
   std::vector<LabelType> argPrimal_;
   std::vector<LabelType> argDual_;
   IndexType maxColor_;

   //  deque for recursive 2 coloring
   std::deque<IndexType> toSplit_; 

   // current state of the alg.
   bool inRecursive2Coloring_;
   bool inGreedy2Coloring_;
   
   ValueType energyOffset_;


   std::vector<unsigned char> dirtyFactors_;

   std::string log_;

   bool timeout_;
};


   
template<class GM, class ACC>
inline
CGC<GM, ACC>::CGC
(
      const GraphicalModelType& gm,
      const Parameter& parameter
)
:  gmRaw_(gm),
   gm_(gm.space()),
   param_(parameter),
   //lambdas_(gm.numberOfFactors()),
   //submodel_(gm,3,1),
   numVar_(gm.numberOfVariables()),
   //numDualVar_(gm.numberOfFactors()),
   value_(0),
   bound_(0),
   argPrimal_(gm.numberOfVariables(),0),
   //argDual_(gm.numberOfFactors(),0),
   toSplit_(),
   inRecursive2Coloring_(false),
   inGreedy2Coloring_(false),
   energyOffset_(0),
   timeout_(false)
   //dirtyFactors_(gm_.numberOfFactors(),1))
{
   //////////////////////////////////////
   // find all double edges
   ///////////////////////////////////////// 

   typedef  std::map<UInt64Type,ValueType>  MapType;
   MapType factorMap;
   
   LabelType lAA[]={0,0};
   LabelType lAB[]={0,1};

   for(IndexType fi=0;fi<gm.numberOfFactors();++fi){

      const ValueType o = gm[fi].operator()(lAA);
      const ValueType l = gm[fi].operator()(lAB)-o;
      energyOffset_ += o;

      const UInt64Type key = gm[fi].variableIndex(0)*gm.numberOfVariables() + gm[fi].variableIndex(1);

      if(factorMap.find(key)==factorMap.end() ){
         // factor is not yet added
         factorMap[key]=l;
      }
      else{
         factorMap[key]+=l;
      }

   }

   // iterate over map to add all non-double edge factors to gm_
   for(typename MapType::const_iterator iter=factorMap.begin(); iter!=factorMap.end(); ++iter){
      const UInt64Type key   = iter->first;
      const ValueType lambda = iter->second;

      const UInt64Type v0 = key/gm.numberOfVariables();
      const UInt64Type v1 = key - v0*gm.numberOfVariables();
      const UInt64Type vis[2]={v0,v1};

      PfType f(gm.numberOfLabels(v0),gm.numberOfLabels(v1),0.0,lambda);
      gm_.addFactor( gm_.addFunction(f) ,vis,vis+2);
   }

   numDualVar_=gm_.numberOfFactors();
   argDual_.resize(numDualVar_);
   dirtyFactors_.resize(numDualVar_);
   lambdas_.resize(numDualVar_);

   // gm_ is set up
   //lambdas_(gm.numberOfFactors()),
   //submodel_ = new SubmodelCGC<PottsGmType>(gm_,3,1,false);
   submodel_ = new SubmodelCGC<PottsGmType>(gm_,0,0,false);

   // set up lambdas 
   for(IndexType f=0;f<numDualVar_;++f){
      
      OPENGM_CHECK(gm_[f].isPotts(), "all factors need to be potts factors");
      OPENGM_CHECK_OP(gm_[f].numberOfVariables(),==,2, "all factors need to 2. order");
      
      const ValueType o = gm_[f].operator()(lAA);
      energyOffset_ += o;
      
      const ValueType lambda=gm_[f].operator()(lAB) - o; 
      if(lambda<0.0){
         bound_ +=lambda;
      }
      lambdas_[f]=lambda;
   }
}




template<class GM, class ACC>
template<class VisitorType>
InferenceTermination CGC<GM,ACC>::infer
(
   VisitorType& visitor
)
{
   timeout_ = false;
   //std::cout << boost::format("CGC: infer for %d primary, %d dual variables\n") % gm_.numberOfVariables() % gm_.numberOfFactors();
   visitor.begin(*this);
   if(param_.startFromThreshold_)
      startFromThreshold(gm_,lambdas_,argPrimal_, 0);
   for(IndexType f=0;f<numDualVar_;++f){
      const IndexType v1=gm_[f].variableIndex(0);
      const IndexType v2=gm_[f].variableIndex(1);
   }

   ValueType valA = 0.0;
   ValueType valB = 0.0;
   for(size_t i=0;i<param_.maxIterations_;++i){
      if(!timeout_ && param_.doCutMove_ && ( value_<valA || i==0)){
         //std::cout<<"rec 2 coloring\n";
         this->recursive2Coloring(visitor);
         valA=value_;
      }
      if(!timeout_ && param_.doGlueCutMove_ && (value_<valB || i==0)){
         //std::cout<<"greedy 2 coloring\n";
         this->greedy2ColoringPlanar(visitor);
         valB=value_;
      }
      if(timeout_)
         break;
   }
   visitor.end(*this);
   return NORMAL;
}



template<class GM, class ACC>
template<class VISITOR>
void 
CGC<GM, ACC>::recursive2Coloring(VISITOR & visitor){
   // set mode
   inRecursive2Coloring_=true;
   inGreedy2Coloring_=false;

   // set starting point will set up all invariants
   const LabelType numCCsStart=this->setStartingPointFromArgPrimal(true);
   


   // while there are subsets to cut in deque
   while(!toSplit_.empty()){

      
      // get an example variable of an cc and 
      // the "color" of all variables which are in cc
      const LabelType exampleVariableInCC = toSplit_.front();
      toSplit_.pop_front();
      const LabelType ccColor = argPrimal_[exampleVariableInCC];

      // infer cc  / all variables which have ccColor 
      // the result of inference is writte in self.argPrimal via call by reference 
      IVPairType res = submodel_->inferSubset(argPrimal_,ccColor,exampleVariableInCC,maxColor_+1,toSplit_,param_.planar_, false /*debug*/);
      const int         numCCArg       = res.first;
      const ValueType value2Coloring   = res.second;

      // the 2 coloring on the cc can split the cc in "numCCArg" connected comps
      // and if numCCArg is 1 this means cc can't be splitted any more
      // otherwise we need to add an exampe var for each result cc to the deque
      if(numCCArg>1){
         // increment the maximum color which is in arg Primal
         maxColor_ += numCCArg+1;
         // update current best value
         value_    += value2Coloring;
         if(visitor(*this)!=0){
            timeout_ = true;
            break;
         }
      }
   }   


   // set mode
   inRecursive2Coloring_=false;
   inGreedy2Coloring_=false;
   // set starting point will set up all invariants
   const LabelType numCCsEnd=this->setStartingPointFromArgPrimal(false);
}


template<class GM, class ACC>
template<class VISITOR>
void 
CGC<GM, ACC>::greedy2ColoringPlanar(VISITOR & visitor){
   // set mode
   inRecursive2Coloring_=false;
   inGreedy2Coloring_=true;



   //while there are some improvements
   bool changes=true;

   //std::vector<bool> dirtyFactors(gm_.numberOfFactors(),true);


   //std::fill(dirtyFactors_.begin(),dirtyFactors_.end(),true );

   for(IndexType fi=0;fi<dirtyFactors_.size();++fi){
      if(dirtyFactors_[fi]!=2)
         dirtyFactors_[fi]=1;
   }


   while(changes && timeout_==false){
      changes=false;

      // set starting point will set up all invariants
      const LabelType numCCsStart=this->setStartingPointFromArgPrimal(false);
      if (numCCsStart==1){
         break;
      }

      // find one factor between each cc 
      typedef opengm::UInt64Type KeyType;
      typedef std::map< opengm::UInt64Type , IndexType > MapType;
      typedef typename MapType::const_iterator MapIter;
      MapType factorCCs;
      for(IndexType fi=0;fi<numDualVar_;++fi){
         const LabelType c0 = argPrimal_[ gm_[fi].variableIndex(0) ];
         const LabelType c1 = argPrimal_[ gm_[fi].variableIndex(1) ];
         const KeyType  cA = static_cast<KeyType>(std::min(c0,c1));
         const KeyType  cB = static_cast<KeyType>(std::max(c0,c1));
         
         const KeyType key  = cA  + cB*static_cast<KeyType>(maxColor_+1);
         factorCCs[key]=fi;
      }

      // get 2 adj. connect comp , merge them and try to
      // reoptimize them with colorings
      for(MapIter iter=factorCCs.begin();iter!=factorCCs.end();++iter){
         const IndexType fi = iter->second;
         if(param_.useBookkeeping_==false ||  dirtyFactors_[fi] == 1 ){

            //std::cout<<" fi dirty ? "<<bool(dirtyFactors[fi])<<" \n";
            const LabelType c0 = argPrimal_[ gm_[fi].variableIndex(0) ];
            const LabelType c1 = argPrimal_[ gm_[fi].variableIndex(1) ];

            // infer by merging and resplitting
            if(c0!=c1){
               //std::cout<<"infer 2 subsets \n\n";
               IVPairType res = submodel_->infer2Subsets(
                  argPrimal_,c0,c1,
                  gm_[fi].variableIndex(0),gm_[fi].variableIndex(1),
                  maxColor_+1,
                  param_.planar_
               );
               const int numCCArg               = res.first;
               const ValueType value2Coloring = res.second;
               /*
               if(numCCArg==-1):
                   print " one var problem"
               
                   print " no improvement"
               elif(numCCArg==-3):
                   print " OPT CUT"
               */
               if(numCCArg==0){
                  //std::cout<<"zeros ccs\n";
                  OPENGM_CHECK(false,"internal error");
               }
               // no improvment 
               else if(numCCArg==-2){
                  //std::cout<<" NO improvement\n\n\n";
                  if(param_.useBookkeeping_)
                     submodel_->updateDirtyness(dirtyFactors_,false);
                    //submodel_->cleanInsideAndBorder();
               }
               else if(numCCArg>=1){
                  //OPENGM_CHECK_OP(value2Coloring,<=,0.0,"internal error");
                  //if(numCCArg==1){
                     //OPENGM_CHECK_OP(argPrimal_[gm_[fi].variableIndex(0)],==,argPrimal_[gm_[fi].variableIndex(1)], "internal error");
                  //}
                  changes=true;
                  maxColor_+=numCCArg+1;
                  value_+=value2Coloring;
                  if(param_.useBookkeeping_){
                     submodel_->updateDirtyness(dirtyFactors_,true);
                     //submodel_->cleanInsideAndBorder();
                  }
                 
                  if(visitor(*this)!=0){
                     timeout_ = true;
                     break;
                  }
               }
               submodel_->cleanInsideAndBorder();
            } // if still active
         } // if dirty
         else{
         }
      } // for all factors
   } // while changes...

   inRecursive2Coloring_=false;
   inGreedy2Coloring_=false;
   // set starting point will set up all invariants
   const LabelType numCCsEnd=this->setStartingPointFromArgPrimal(false);
}
     


template<class GM, class ACC>
inline void
CGC<GM, ACC>::reset(){

}
   
template<class GM, class ACC>
inline void 
CGC<GM,ACC>::setStartingPoint
(
   typename std::vector<typename CGC<GM,ACC>::LabelType>::const_iterator begin
) {
   std::copy(begin,begin+numVar_,argPrimal_.begin());
   const LabelType numCC=this->setStartingPointFromArgPrimal(true);
}

template<class GM, class ACC>
inline typename CGC<GM,ACC>::LabelType 
CGC<GM,ACC>::setStartingPointFromArgPrimal(const bool fillQ){

   // get a connected componet labeling from starting point 
   IndexType numCC = detail_gcg::getCCFromLabels(gm_,argPrimal_.begin());


   //this has set  returns the following:
   //   #  argPrimal_[primal variable index / vi] = "color" in [0, numCC]
   this->primalToDual();
   value_ = evalPrimal();
   maxColor_ = numCC-1;

   if(fillQ){
      // fill deque with example variables for each connected componet
      std::vector<LabelType> toFind(numCC);
      std::vector<bool>      found(numCC,false);
      std::vector<IndexType> foundPosition(numCC);
      for(LabelType c=0;c<numCC;++c){
         toFind[c]=c;
      }
      detail_gcg::findFirst(toFind,argPrimal_,foundPosition,found);
      toSplit_.clear();
      for(IndexType c=0;c<numCC;++c){
         toSplit_.push_back(foundPosition[c]);
      }
   }
   return numCC;
}
   
template<class GM, class ACC>
inline std::string
CGC<GM, ACC>::name() const
{
   return "CGC";
}

template<class GM, class ACC>
inline const typename CGC<GM, ACC>::GraphicalModelType&
CGC<GM, ACC>::graphicalModel() const
{
   return gmRaw_;
}
  
template<class GM, class ACC>
inline InferenceTermination
CGC<GM,ACC>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  


template<class GM, class ACC>
inline InferenceTermination
CGC<GM,ACC>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{
   if(N==1) {
      x.resize(gm_.numberOfVariables());
      for(size_t j=0; j<x.size(); ++j) {
         x[j] = argPrimal_[j];
      }
      return NORMAL;
   }
   else {
      return UNKNOWN;
   }
}


template<class GM, class ACC>
inline void
CGC<GM, ACC>::primalToDual(){
   for(IndexType f=0;f<numDualVar_;++f){
      const IndexType v1=gm_[f].variableIndex(0);
      const IndexType v2=gm_[f].variableIndex(1);
      argDual_[f] = argPrimal_[v1]==argPrimal_[v2] ? 0 :1 ;
   }
}


template<class GM, class ACC>
inline typename CGC<GM, ACC>::ValueType
CGC<GM, ACC>::evalPrimal2(
   const std::vector<typename CGC<GM, ACC>::LabelType> & argP
)const{
   ValueType value = 0.0;
   for(IndexType f=0;f<numDualVar_;++f){
      const IndexType v1=gm_[f].variableIndex(0);
      const IndexType v2=gm_[f].variableIndex(1);
      if(argP[v1]!=argP[v2])
         value+=lambdas_[f];
   }
   return value;
}


template<class GM, class ACC>
inline typename CGC<GM, ACC>::ValueType
CGC<GM, ACC>::evalPrimal()const{
   ValueType value = 0.0;
   for(IndexType f=0;f<numDualVar_;++f){
      const IndexType v1=gm_[f].variableIndex(0);
      const IndexType v2=gm_[f].variableIndex(1);
      if(argPrimal_[v1]!=argPrimal_[v2])
         value+=lambdas_[f];
   }
   return value;
}


template<class GM, class ACC>
inline typename CGC<GM, ACC>::ValueType
CGC<GM, ACC>::evalDual()const{
   ValueType value = 0.0;
   for(IndexType f=0;f<numDualVar_;++f){
      if(argDual_[f]!=0)
         value+=lambdas_[f];
   }
   return value;
}



} // namespace opengm

#endif // #ifndef OPENGM_CGC_HXX

// kate: space-indent on; indent-width 3; replace-tabs on; indent-mode cstyle; remove-trailing-space; replace-trailing-spaces-save; 
