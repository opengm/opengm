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

#include <cmath>
#include <algorithm>

#include <sstream>

#include "opengm/inference/auxiliary/submodel/submodel_builder.hxx"


// internal 
#include "opengm/inference/dynamicprogramming.hxx"
#include "opengm/inference/astar.hxx"
#include "opengm/inference/lazyflipper.hxx"
#include <opengm/inference/messagepassing/messagepassing.hxx>

// external (autoinc)
#include "opengm/inference/external/ad3.hxx"
// external (inclued by with)
#ifdef WITH_CPLEX
#include "opengm/inference/lpcplex.hxx"
#endif

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
   typedef typename SubOptimizer::SubGmType SubGmType;

   // subsolvers 
   
   typedef opengm::DynamicProgramming<SubGmType,AccumulationType> DpSubInf;
   typedef opengm::AStar<SubGmType,AccumulationType> AStarSubInf;
   typedef opengm::LazyFlipper<SubGmType,AccumulationType> LfSubInf;
   typedef opengm::BeliefPropagationUpdateRules<SubGmType,AccumulationType> UpdateRulesTypeBp;
   typedef opengm::TrbpUpdateRules<SubGmType,AccumulationType> UpdateRulesTypeTrbp;
   typedef opengm::MessagePassing<SubGmType, AccumulationType,UpdateRulesTypeBp  , opengm::MaxDistance> BpSubInf;
   typedef opengm::MessagePassing<SubGmType, AccumulationType,UpdateRulesTypeTrbp, opengm::MaxDistance> TrBpSubInf;

   // external (autoincluded)
   typedef opengm::external::AD3Inf<SubGmType,AccumulationType> Ad3SubInf;
   #ifdef WITH_CPLEX
   typedef opengm::LPCplex<SubGmType,AccumulationType> LpCplexSubInf;
   #endif


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
   std::vector<bool> checkedVi_;
   std::vector<UInt64Type> distance_;


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
   checkedVi_(gm.numberOfVariables(), false),
   subOptimizer_(gm),
   cleanRegion_(gm.numberOfVariables(),false),
   distance_(gm.numberOfVariables())
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
   if(param_.maxRadius_ < 1) {
      param_.maxRadius_ = 1;
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

   std::fill(distance_.begin(),distance_.begin()+vis.size(),0);

   const size_t maxSgSize = (param_.maxSubgraphSize_==0? gm_.numberOfVariables() :param_.maxSubgraphSize_);
   while(viQueue.size()!=0  &&  vis.size()<=maxSgSize) {
      size_t cvi=viQueue.front();
      viQueue.pop();
      // for each neigbour of cvi
      for(size_t vni=0;vni<viAdjacency_[cvi].size();++vni) {
         // if neighbour has not been visited
         const size_t vn=viAdjacency_[cvi][vni];
         if(usedVi_[vn]==false) {
            // set as visited
            usedVi_[vn]=true;
            // insert into the subgraph vis
            distance_[vn]=distance_[cvi]+1;
            if(distance_[vn]<=radius){
               if(vis.size()<maxSgSize){
                  vis.push_back(vn);
                  viQueue.push(vn);
               }
               else{
                  break;
               }
            }
         }
      }
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
   std::fill(checkedVi_.begin(),checkedVi_.end(),false);
   vis.clear();
   vis.push_back(startVi);
   usedVi_[startVi]=true;
   checkedVi_[startVi]=true;
   std::deque<IndexType> viQueue;
   viQueue.push_back(startVi);

   bool first=true;
   const size_t maxSgSize = (param_.maxSubgraphSize_==0? gm_.numberOfVariables() :param_.maxSubgraphSize_);


   std::fill(distance_.begin(),distance_.begin()+vis.size(),0);

   while(viQueue.size()!=0 && /*r<radius &&*/  vis.size()<=maxSgSize) {
      IndexType cvi=viQueue.front();

      OPENGM_CHECK(usedVi_[cvi]==false || vis.size()==1,"");
      

      //std::cout<<"cvi "<<cvi<<" size "<<viQueue.size()<<" vis size "<<vis.size()<<"\n";
      viQueue.pop_front();

      if(checkedVi_[cvi]==true && first ==false){
         continue;
      }
      first=false;

      size_t includeInTree=0;
      // for each neigbour of cvi
      for(size_t vni=0;vni<viAdjacency_[cvi].size();++vni) {
         const IndexType vn=viAdjacency_[cvi][vni];
         if(usedVi_[vn]==true) {
            ++includeInTree;
         }
      }
      //std::cout<<"inlcuded in tree "<<includeInTree<<"\n";
      OPENGM_CHECK_OP(includeInTree,<=,vis.size(),"");
      //OPENGM_CHECK_OP(includeInTree,<=,2,"");
      checkedVi_[cvi]=true;
      //std::cout<<"icn in tree "<<includeInTree<<"\n";
      OPENGM_CHECK(includeInTree>0 || (vis.size()==1 && includeInTree==0),"");
      //if (usedVi_[cvi]==false && includeInTree<=1){
      if (includeInTree<=1){
         //std::cout<<"in 1....\n";
         // insert into the subgraph vis
         if(usedVi_[cvi]==false){
            vis.push_back(cvi);
             // set as visited
            usedVi_[cvi]=true;

            if(vis.size()>=maxSgSize){
               //std::cout<<"max size exit\n";
            }
         }

         std::vector<IndexType> adjVis(viAdjacency_[cvi].size());
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
            if(usedVi_[vn]==false && checkedVi_[vn]==false) {
               //std::cout<<"in 3....\n";
               // insert into queue

               distance_[vn]=distance_[cvi]+1;
               if(distance_[vn]<=radius)
                  viQueue.push_back(vn);
            }
         }
      }
      else{
         //usedVi_[cvi]=true;
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
         //std::cout<<"r "<<radius<<" ";
         //std::cout<<"nVar "<<subgGraphVi.size()<<"\n";
         //std::cout<<"infer submodel\n";
         std::vector<LabelType> states;

         bool changes=true;
         bool worseValueMove=false;


         // OPTIMAL OR MONOTON MOVERS
         if(param_.solver_==std::string("ad3")){
            changes = subOptimizer_. template inferSubmodelInplace<Ad3SubInf>(typename Ad3SubInf::Parameter(Ad3SubInf::AD3_ILP) ,states);
         }
         else if (param_.solver_==std::string("dp")){
            changes = subOptimizer_. template inferSubmodel<DpSubInf>(typename DpSubInf::Parameter() ,states);
         }
         else if (param_.solver_==std::string("astar")){
            
         }
         else if (param_.solver_==std::string("cplex")){
            #ifdef WITH_CPLEX
               typedef opengm::LPCplex<SubGmType,AccumulationType> LpCplexSubInf;
               typename LpCplexSubInf::Parameter subParam;
               subParam.integerConstraint_=true;
               changes = subOptimizer_. template inferSubmodel<LpCplexSubInf>(subParam ,states); 
            #else  
               throw RuntimeError("solver cplex needs flag WITH_CPLEX defined bevore the #include of LOC sovler");
            #endif  
         }
         // MONOTON MOVERS
         else if(param_.solver_[0]=='l' && param_.solver_[1]=='f'){
            std::stringstream ss;
            for(size_t i=2;i<param_.solver_.size();++i){
               ss<<param_.solver_[i];
            }
            size_t maxSgSize;
            ss>>maxSgSize;
            changes = subOptimizer_. template inferSubmodel<LfSubInf>(typename LfSubInf::Parameter(maxSgSize) ,states,true,true);  
         }
         // SOLVERS WICH MIGHT NOT BE OPTIMAL
         else{
            const ValueType valueBevoreMove = movemaker_.value();

            // (MAYBE) NON-OPT SOLVERS
            if (param_.solver_==std::string("bp"))
               changes = subOptimizer_. template inferSubmodel<BpSubInf>(typename BpSubInf::Parameter() ,states,false,false);  
            else if (param_.solver_==std::string("trbp"))
               changes = subOptimizer_. template inferSubmodel<TrBpSubInf>(typename TrBpSubInf::Parameter() ,states,false,false);  
            else
               throw RuntimeError("wrong solver");

            // CHECK IF STATE CHANGED AT ALL
            if(changes){
               const ValueType valueAfterMove = movemaker_.valueAfterMove(subgGraphVi.begin(), subgGraphVi.end(), states.begin());
               if(ACC::bop(valueAfterMove,valueBevoreMove)){
                  changes=true;
                  //std::cout<<param_.solver_<<" improvement d="<<std::abs(valueBevoreMove-valueAfterMove)<<"\n";
               }
               else{
                  changes=false;
                  worseValueMove=true;
               }
            }
            if(changes==false){
               //std::cout<<param_.solver_<<"no improvement\n";
            } 
         }
         // inference is done
         if(changes){
            this->setBorderDirty(subgGraphVi.begin(), subgGraphVi.end());
         }
         if(worseValueMove==false)
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


      visitor(*this,this->value(),this->bound());

      e2 = movemaker_.value();


      OPENGM_CHECK(ACC::bop(e2,e1) || ACC::bop(e1,e2)==false,"bad move");

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

