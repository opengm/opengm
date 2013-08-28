#pragma once
#ifndef OPENGM_LOC_HXX
#define OPENGM_LOC_HXX

#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <queue>
#include <deque>
#include "opengm/opengm.hxx"
#include "opengm/utilities/random.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/external/ad3.hxx"
#include "opengm/inference/visitors/visitor.hxx"

#include "opengm/inference/auxiliary/submodel/submodel_builder.hxx"


namespace opengm {
/// \ingroup inference
/// LOC Algorithm\n\n
/// K. Jung, P. Kohli and D. Shah, "Local Rules for Global MAP: When Do They Work?", NIPS 2009
///
/// In this implementation, the user needs to set the parameter of the 
/// truncated geometric distribution by hand. Depending on the size of
/// the subgraph, either A* or exhaustive search is used for MAP 
/// estimation on the subgraph 
/// \ingroup inference 
template<class GM, class ACC>
class LOC : public Inference<GM, ACC> {
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef Movemaker<GraphicalModelType> MovemakerType;
   typedef VerboseVisitor<LOC<GM, ACC> > VerboseVisitorType;
   typedef TimingVisitor<LOC<GM, ACC> > TimingVisitorType;
   typedef EmptyVisitor<LOC<GM, ACC> > EmptyVisitorType;


   typedef SubmodelOptimizer<GM,ACC> SubOptimizer;

   class Parameter {
   public:
      /// constuctor
      /// \param phi parameter of the truncated geometric distribution is used to select a certain subgraph radius with a certain probability
      /// \param maxRadius maximum radius for the subgraphes which are optimized within opengm:::LOC
      /// \param maxIteration maximum number of iterations (in one iteration on subgraph gets) optimized
      /// \param ad3Threshold if the subgraph size is bigger than ad3Threshold opengm::external::Ad3Inf is used to optimize the subgraphes
      /// \param stopAfterNBadIterations stop after n iterations without improvement
      Parameter
      (
         const std::string solver="ad3",
         const double phi = 0.3,
         const size_t maxRadius  = 50,
         const double pFastHeuristic = 0.9,
         const size_t maxIterations = 100000,
         const size_t stopAfterNBadIterations=10000,
         const size_t maxSubgraphSize = 0 
      )
      :  solver_(solver),
         phi_(phi),
         maxRadius_(maxRadius),
         pFastHeuristic_(pFastHeuristic),
         maxIterations_(maxIterations),
         stopAfterNBadIterations_(stopAfterNBadIterations),
         maxSubgraphSize_(maxSubgraphSize)
      {

      }
      // subsolver used for submodel ("ad3" or "astar" so far)
      std::string solver_;
      /// phi of the truncated geometric distribution is used to select a certain subgraph radius with a certain probability
      double phi_;
      /// maximum subgraph radius
      size_t maxRadius_;
      /// prob. of f
      double pFastHeuristic_;
      /// maximum number of iterations
      size_t maxIterations_;

      // stop after n iterations without improvement
      size_t stopAfterNBadIterations_;

      // max allowed subgraph size (0  means any is allowed)
      size_t maxSubgraphSize_;
   };

   LOC(const GraphicalModelType&, const Parameter& param = Parameter());
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   InferenceTermination infer();
   void reset();
   template<class VisitorType>
      InferenceTermination infer(VisitorType&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   ValueType value() const;


   template<class VI_ITER>
   void setBorderDirty(VI_ITER begin,VI_ITER end){
      const IndexType nVis=std::distance(begin,end);
      OPENGM_CHECK_OP(subOptimizer_.submodelSize(),==,nVis,"");
      for(IndexType v=0;v<nVis;++v){
         const IndexType vi=begin[v];
         const IndexType nNVar = viAdjacency_[vi].size();
         for(IndexType vo=0;vo<nNVar;++vo){
            const IndexType vio=viAdjacency_[vi][vo];
            if( subOptimizer_.inSubmodel(vio)==false){
               cleanRegion_[vio]=false;
            }
         }
      }
   }

   template<class VI_ITER>
   void setInsideClean(VI_ITER begin,VI_ITER end){
      const IndexType nVis=std::distance(begin,end);
      OPENGM_CHECK_OP(subOptimizer_.submodelSize(),==,nVis,"");
      for(IndexType v=0;v<nVis;++v){
         const IndexType vi=begin[v];
         cleanRegion_[vi]=true;
      }
   }


   template<class VI_ITER>
   bool hasDirtyInsideVariables(VI_ITER begin,VI_ITER end){
      const IndexType nVis=std::distance(begin,end);
      OPENGM_CHECK_OP(subOptimizer_.submodelSize(),==,nVis,"");

      for(IndexType v=0;v<nVis;++v){
         const IndexType vi=begin[v];
         if(cleanRegion_[vi]==false){
            return true;
         }
      }
      return false;
   }



private:
   void getSubgraphVis(const size_t, const size_t, std::vector<size_t>&);
   void getSubgraphTreeVis(const size_t, const size_t, std::vector<size_t>&);
   void inline initializeProbabilities(std::vector<double>&);
   const GraphicalModelType& gm_;
   MovemakerType movemaker_;
   Parameter param_;
   std::vector< RandomAccessSet<IndexType> > viAdjacency_;
   std::vector<bool> usedVi_;

   // submodel
   SubOptimizer subOptimizer_;

   // clean region
   std::vector<bool> cleanRegion_;


};

template<class GM, class ACC>
LOC<GM, ACC>::LOC
(
   const GraphicalModelType& gm,
   const Parameter& parameter
)
:  gm_(gm),
   movemaker_(gm),
   param_(parameter),
   viAdjacency_(gm.numberOfVariables()),
   usedVi_(gm.numberOfVariables(), false),
   subOptimizer_(gm),
   cleanRegion_(gm.numberOfVariables(),false)
{

   // compute variable adjacency
   gm.variableAdjacencyList(viAdjacency_);
   if(this->param_.maxIterations_==0)
      param_.maxIterations_ = gm_.numberOfVariables() * 
         log(double(gm_.numberOfVariables()))*log(double(gm_.numberOfVariables()));
}

template<class GM, class ACC>
void
LOC<GM, ACC>::reset()
{
   movemaker_.reset();
   std::fill(usedVi_.begin(),usedVi_.end(),false);
   // compute variable adjacency is not nessesary
   // since reset assumes that the structure of
   // the graphical model has not changed
   if(this->param_.maxIterations_==0)
      param_.maxIterations_ = gm_.numberOfVariables() * 
         log(double(gm_.numberOfVariables()))*log(double(gm_.numberOfVariables()));
}
   
template<class GM, class ACC>
inline void 
LOC<GM,ACC>::setStartingPoint
(
   typename std::vector<typename LOC<GM,ACC>::LabelType>::const_iterator begin
) {
   try{
      movemaker_.initialize(begin);
   }
   catch(...) {
      throw RuntimeError("unsuitable starting point");
   }
}
   
template<class GM, class ACC>
inline typename LOC<GM, ACC>::ValueType
LOC<GM, ACC>::value()const
{
   return this->movemaker_.value();
}

template<class GM, class ACC>
void inline
LOC<GM, ACC>::initializeProbabilities
(
   std::vector<double>& prob
)
{
   const double phi = param_.phi_;
   if(param_.maxRadius_ < 2) {
      param_.maxRadius_ = 2;
   }
   size_t maxRadius = param_.maxRadius_;
   prob.resize(param_.maxRadius_);
   for(size_t i=0;i<param_.maxRadius_-1;++i) {
      prob[i] = phi * pow((1.0-phi), static_cast<double>(i));
   }
   prob[maxRadius-1]= pow((1.0-phi), static_cast<double>(param_.maxRadius_));
}

template<class GM, class ACC>
inline std::string
LOC<GM, ACC>::name() const {
   return "LOC";
}

template<class GM, class ACC>
inline const typename LOC<GM, ACC>::GraphicalModelType&
LOC<GM, ACC>::graphicalModel() const {
   return gm_;
}

template<class GM, class ACC>
void LOC<GM, ACC>::getSubgraphVis
(
   const size_t startVi,
   const size_t radius,
   std::vector<size_t>& vis
) {
   std::fill(usedVi_.begin(),usedVi_.end(),false);
   vis.clear();
   vis.push_back(startVi);
   usedVi_[startVi]=true;
   std::queue<size_t> viQueue;
   viQueue.push(startVi);
   size_t r=0;
   size_t sgSize=0;
   const size_t maxSgSize = (param_.maxSubgraphSize_==0? gm_.numberOfVariables() :param_.maxSubgraphSize_);
   while(viQueue.size()!=0 && r<radius &&  sgSize<=maxSgSize) {
      size_t cvi=viQueue.front();
      viQueue.pop();
      // for each neigbour of cvi
      for(size_t vni=0;vni<viAdjacency_[cvi].size();++vni) {
         // if neighbour has not been visited
         const size_t vn=viAdjacency_[cvi][vni];
         if(usedVi_[vn]==false) {

            // set as visited
            usedVi_[vn]=true;
            // insert into queue
            viQueue.push(vn);
            // insert into the subgraph vis
            vis.push_back(vn);
            if(vis.size()>=maxSgSize-1){
               break;
            }
         }
      }
      ++r;
   }
}


template<class GM, class ACC>
void LOC<GM, ACC>::getSubgraphTreeVis
(
   const size_t startVi,
   const size_t radius,
   std::vector<size_t>& vis
) {

   //std::cout<<"build tree\n";
   std::fill(usedVi_.begin(),usedVi_.end(),false);
   vis.clear();
   vis.push_back(startVi);
   usedVi_[startVi]=true;
   std::deque<IndexType> viQueue;
   viQueue.push_back(startVi);
   size_t r=0;
   size_t sgSize=0;
   const size_t maxSgSize = (param_.maxSubgraphSize_==0? gm_.numberOfVariables() :param_.maxSubgraphSize_);


   std::vector<IndexType> rr(gm_.numberOfVariables(),0);

   while(viQueue.size()!=0 && /*r<radius &&*/  sgSize<=maxSgSize) {
      IndexType cvi=viQueue.front();
      viQueue.pop_front();

      size_t includeInTree=0;
      // for each neigbour of cvi
      for(size_t vni=0;vni<viAdjacency_[cvi].size();++vni) {
         const IndexType vn=viAdjacency_[cvi][vni];
         if(usedVi_[vn]==true) {
            includeInTree+=1;
         }
      }
      //std::cout<<"icn in tree "<<includeInTree<<"\n";

      //if (usedVi_[cvi]==false && includeInTree<=1){
      if (includeInTree<=1){
         //std::cout<<"in 1....\n";
         // insert into the subgraph vis
         if(usedVi_[cvi]==false){
            vis.push_back(cvi);
            ++sgSize;
             // set as visited
            usedVi_[cvi]=true;
            if(vis.size()>=maxSgSize){
               //std::cout<<"max size exit\n";
               break;
            }
         }

         FastSequence<IndexType,4> adjVis(viAdjacency_[cvi].size());
         for(size_t vni=0;vni<viAdjacency_[cvi].size();++vni) {
            const size_t vn=viAdjacency_[cvi][vni];
            adjVis[vni]=vn;
         }
         std::random_shuffle(adjVis.begin(),adjVis.end());
         
         // for each neigbour of cvi
         for(size_t vni=0;vni<viAdjacency_[cvi].size();++vni) {
            //std::cout<<"hello\n";
            // if neighbour has not been visited
            const size_t vn=adjVis[vni];
            //std::cout<<"in 2....\n";
            if(usedVi_[vn]==false) {
               //std::cout<<"in 3....\n";
               // insert into queue
               rr[vn]=rr[cvi]+1;
               if(rr[vn]<radius)
                  viQueue.push_back(vn);
            }
         }
         ++r;
      }
   }
}

template<class GM, class ACC>
inline InferenceTermination
LOC<GM, ACC>::infer() {
   EmptyVisitorType v;
   return infer(v);
}

template<class GM, class ACC>
template<class VisitorType>
InferenceTermination 
LOC<GM, ACC>::infer
(
   VisitorType& visitor
) {

   const UInt64Type autoStop = param_.stopAfterNBadIterations_==0 ? gm_.numberOfVariables() : param_.stopAfterNBadIterations_;

   visitor.begin(*this,this->value(),this->bound());
   // create random generators
   opengm::RandomUniform<size_t> randomVariable(0, gm_.numberOfVariables());
   opengm::RandomUniform<double> random01(0.0, 1.0);

   std::vector<double> prob;
   this->initializeProbabilities(prob);
   opengm::RandomDiscreteWeighted<size_t, double> randomRadius(prob.begin(), prob.end());
   std::vector<size_t> subgGraphVi;
   // all iterations, usualy n*log(n)

   ValueType e1 = movemaker_.value(),e2;
   size_t badIter=0;

   for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
      subOptimizer_.setLabel(vi,movemaker_.state(vi));
   }

   for(size_t i=0;i<param_.maxIterations_;++i) {
      if(badIter>=autoStop){
         break;
      }

      // select random variable
      size_t viStart = randomVariable();
      // select random radius
      size_t radius=randomRadius()+1;
      //std::cout<<"radius "<<radius<<"\n";
      // grow subgraph from beginning from viStart with r=Radius
      if(param_.solver_==std::string("dp"))
         this->getSubgraphTreeVis(viStart, radius, subgGraphVi);
      else{
         this->getSubgraphVis(viStart, radius, subgGraphVi);
      }
      // find the optimal configuration for all variables in subgGraphVi


      std::sort(subgGraphVi.begin(), subgGraphVi.end());
      subOptimizer_.setVariableIndices(subgGraphVi.begin(), subgGraphVi.end());

      const bool dirtyVarsInside = hasDirtyInsideVariables(subgGraphVi.begin(), subgGraphVi.end());
      if(dirtyVarsInside==false){

         const double rn = random01();

         if(rn<param_.pFastHeuristic_){
            ++badIter;
            subOptimizer_.unsetVariableIndices();
            visitor(*this,this->value(),this->bound(),radius);
            continue;
         }
      }
      if(subgGraphVi.size()>2){
         //std::cout<<"with ad3\n";
         //std::cout<<"subvissize "<<subgGraphVi.size()<<"\n";
         //std::cout<<"infer submodel\n";
         std::vector<LabelType> states;
         const bool changes = subOptimizer_.inferSubmodelOptimal(states,param_.solver_);
         //std::cout<<"improvement  "<<changes<<"\n";
         if(changes){
            this->setBorderDirty(subgGraphVi.begin(), subgGraphVi.end());
         }
         movemaker_.move(subgGraphVi.begin(), subgGraphVi.end(), states.begin());
         
      }
      else{
         subOptimizer_.unsetVariableIndices();
         continue;
         //movemaker_.template moveOptimally<AccumulationType>(subgGraphVi.begin(), subgGraphVi.end());
      }

      this->setInsideClean(subgGraphVi.begin(), subgGraphVi.end());

      // clean the subOptimizer and set the labels
      subOptimizer_.unsetVariableIndices();
      for(IndexType v=0;v<subgGraphVi.size();++v){
         subOptimizer_.setLabel(subgGraphVi[v],movemaker_.state(subgGraphVi[v]));
      }


      visitor(*this,this->value(),this->bound(),radius);

      e2 = movemaker_.value();

      if(ACC::bop(e2,e1)){
         badIter=0;
         e1=e2;
      }
      else{
         //std::cout<<"badIters "<<badIter<<"\n";
         badIter+=1;
      }



   }
   visitor.end(*this,this->value(),this->bound());
   return NORMAL;
}

template<class GM, class ACC>
inline InferenceTermination
LOC<GM, ACC>::arg
(
   std::vector<LabelType>& x,
   const size_t N
) const {
   if(N == 1) {
      x.resize(gm_.numberOfVariables());
      for(size_t j = 0; j < x.size(); ++j) {
         x[j] = movemaker_.state(j);
      }
      return NORMAL;
   }
   else 
      return UNKNOWN;
}

} // namespace opengm

#endif // #ifndef OPENGM_LOC_HXX

