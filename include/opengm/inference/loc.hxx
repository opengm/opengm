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

#include "opengm/opengm.hxx"
#include "opengm/utilities/random.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/astar.hxx"
#include "opengm/inference/visitors/visitor.hxx"

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

   class Parameter {
   public:
      /// constuctor
      /// \param phi parameter of the truncated geometric distribution is used to select a certain subgraph radius with a certain probability
      /// \param maxRadius maximum radius for the subgraphes which are optimized within opengm:::LOC
      /// \param maxIteration maximum number of iterations (in one iteration on subgraph gets) optimized
      /// \param aStarThreshold if the subgraph size is bigger than aStarThreshold opengm::Astar is used to optimize the subgraphes
      /// \param startPoint_ starting point for the inference
      Parameter
      (
         double phi = 0.5,
         size_t maxRadius = 10,
         size_t maxIterations = 0,
         size_t aStarThreshold = 10,
         const std::vector<LabelType>& startPoint = std::vector<LabelType>()
      )
      :  phi_(phi),
         maxRadius_(maxRadius),
         maxIterations_(maxIterations),
         aStarThreshold_(aStarThreshold),
         startPoint_(startPoint)
      {}
      /// phi of the truncated geometric distribution is used to select a certain subgraph radius with a certain probability
      double phi_;
      /// maximum subgraph radius
      size_t maxRadius_;
      /// maximum number of iterations
      size_t maxIterations_;
      /// subgraph size threshold to switch from brute-force to a*star search
      size_t aStarThreshold_;
      /// starting point for warm started inference
      std::vector<LabelType> startPoint_;
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

private:
   void getSubgraphVis(const size_t, const size_t, std::vector<size_t>&);
   void inline initializeProbabilities(std::vector<double>&);
   const GraphicalModelType& gm_;
   MovemakerType movemaker_;
   Parameter param_;
   std::vector<std::vector<size_t> > viAdjacency_;
   std::vector<bool> usedVi_;
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
   usedVi_(gm.numberOfVariables(), false)
{
   if(parameter.startPoint_.size() == gm.numberOfVariables())
      movemaker_.initialize(parameter.startPoint_.begin());
   else if(parameter.startPoint_.size() != 0)
      throw RuntimeError("Unsuitable starting point.");
   // compute variable adjacency
   for(size_t f=0;f<gm_.numberOfFactors();++f) {
      if(gm_[f].numberOfVariables()>1) {
         //connect all vi from factor f with each other
         for(size_t va=0;va<gm_[f].numberOfVariables();++va) {
         for(size_t vb=0;vb<gm_[f].numberOfVariables();++vb)
            if(va!=vb) { //connect
               viAdjacency_[ gm_[f].variableIndex(va)].push_back(gm_[f].variableIndex(vb));
               viAdjacency_[ gm_[f].variableIndex(vb)].push_back(gm_[f].variableIndex(va));
            }
         }
      }
   }
   if(this->param_.maxIterations_==0)
      param_.maxIterations_ = gm_.numberOfVariables() * 
         log(double(gm_.numberOfVariables()))*log(double(gm_.numberOfVariables()));
}

template<class GM, class ACC>
void
LOC<GM, ACC>::reset()
{
   if(param_.startPoint_.size() == gm_.numberOfVariables()) {
      movemaker_.initialize(param_.startPoint_.begin());
   }
   else if(param_.startPoint_.size() != 0) {
      throw RuntimeError("Unsuitable starting point.");
   }
   else{
      movemaker_.reset();
   }
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
   while(viQueue.size()!=0 && r<radius) {
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
         }
      }
      ++r;
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
   visitor.begin(*this,this->value(),this->bound());
   // create random generators
   opengm::RandomUniform<size_t> randomVariable(0, gm_.numberOfVariables());
   std::vector<double> prob;
   this->initializeProbabilities(prob);
   opengm::RandomDiscreteWeighted<size_t, double> randomRadius(prob.begin(), prob.end());
   std::vector<size_t> subgGraphVi;
   // all iterations, usualy n*log(n)
   for(size_t i=0;i<param_.maxIterations_;++i) {
      // select random variable
      size_t viStart = randomVariable();
      // select random radius
      size_t radius=randomRadius()+1;
      // grow subgraph from beginning from viStart with r=Radius
      this->getSubgraphVis(viStart, radius, subgGraphVi);
      // find the optimal configuration for all variables in subgGraphVi
      if(subgGraphVi.size()>param_.aStarThreshold_) {
          std::sort(subgGraphVi.begin(), subgGraphVi.end());
         typedef typename MovemakerType::SubGmType SubGmType;
         typedef opengm::AStar<SubGmType, ACC> SubGmInferenceType;
         typedef typename SubGmInferenceType::Parameter SubGmInferenceParameterType;
         SubGmInferenceParameterType para;
         para.heuristic_ = para.STANDARDHEURISTIC;
         std::vector<LabelType> states(std::distance(subgGraphVi.begin(), subgGraphVi.end()));
         movemaker_. template proposeMoveAccordingToInference< 
            SubGmInferenceType, 
            SubGmInferenceParameterType,
            typename std::vector<size_t>::const_iterator,
            typename std::vector<LabelType>::iterator 
         > (para, subgGraphVi.begin(), subgGraphVi.end(), states);
         movemaker_.move(subgGraphVi.begin(), subgGraphVi.end(), states.begin());

        
         //movemaker_.template moveAstarOptimally<AccumulationType>(subgGraphVi.begin(), subgGraphVi.end());
      }
      else
         movemaker_.template moveOptimally<AccumulationType>(subgGraphVi.begin(), subgGraphVi.end());
      visitor(*this,this->value(),this->bound(),radius);
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

