#pragma once
#ifndef OPENGM_LAZYFLIPPER_HXX
#define OPENGM_LAZYFLIPPER_HXX

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <stdexcept>
#include <list>

#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/utilities/tribool.hxx"

namespace opengm {

/// \cond HIDDEN_SYMBOLS

template<class T>
class Tagging {
public:
   typedef T ValueType;
   typedef std::vector<ValueType> tag_container_type;
   typedef std::vector<size_t> index_container_type;
   typedef index_container_type::const_iterator const_iterator;

   Tagging(const size_t = 0);
   void append(const size_t);
   ValueType tag(const size_t) const;
   void tag(const size_t, const typename Tagging<T>::ValueType);
   void untag(); // untag all
   const_iterator begin();
   const_iterator begin() const;
   const_iterator end();
   const_iterator end() const;

private:
   tag_container_type tags_;
   index_container_type indices_;
};

// A simple undirected graph
class Adjacency {
public:
   typedef std::set<size_t>::const_iterator const_iterator;

   Adjacency(const size_t = 0);
   void resize(const size_t);
   void connect(const size_t, const size_t);
   bool connected(const size_t, const size_t) const;
   const_iterator neighborsBegin(const size_t);
   const_iterator neighborsBegin(const size_t) const;
   const_iterator neighborsEnd(const size_t);
   const_iterator neighborsEnd(const size_t) const;

private:
   std::vector<std::set<size_t> > neighbors_;
};

// Forest with Level Order Traversal.
//
// - no manipulation after construction.
// - level Successors must be set manually
// - implementation nor const correct
//
template<class T>
class Forest {
public:
   typedef T Value;
   typedef size_t NodeIndex;
   typedef size_t Level;

   static const NodeIndex NONODE = -1;

   Forest();
   size_t size();
   size_t levels();
   NodeIndex levelAnchor(const Level&);
   NodeIndex push_back(const Value&, NodeIndex);
   size_t testInvariant();
   std::string asString();
   Value& value(NodeIndex);
   Level level(NodeIndex);
   NodeIndex parent(NodeIndex);
   NodeIndex levelOrderSuccessor(NodeIndex);
   size_t numberOfChildren(NodeIndex);
   NodeIndex child(NodeIndex, const size_t);
   void setLevelOrderSuccessor(NodeIndex, NodeIndex);

private:
   struct Node {
      Node(const Value& value)
         : value_(value), parent_(NONODE),
         children_(std::vector<NodeIndex>()),
         level_(0), levelOrderSuccessor_(NONODE)
      {}
      Value value_;
      NodeIndex parent_;
      std::vector<NodeIndex> children_;
      Level level_;
      NodeIndex levelOrderSuccessor_;
   };
   std::vector<Node> nodes_;
   std::vector<NodeIndex> levelAnchors_;
};

/// \endcond

/// \brief A generalization of ICM\n\n
/// B. Andres, J. H. Kappes, U. Koethe and Hamprecht F. A., The Lazy Flipper: MAP Inference in Higher-Order Graphical Models by Depth-limited Exhaustive Search, Technical Report, 2010, http://arxiv.org/abs/1009.4102
///
/// \ingroup inference 
template<class GM, class ACC = Minimizer>
class LazyFlipper : public Inference<GM, ACC> {
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef Forest<IndexType> SubgraphForest;
   typedef size_t SubgraphForestNode;
   static const SubgraphForestNode NONODE = SubgraphForest::NONODE;
   typedef visitors::VerboseVisitor<LazyFlipper<GM, ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<LazyFlipper<GM, ACC> > EmptyVisitorType;
   typedef visitors::TimingVisitor<LazyFlipper<GM, ACC> > TimingVisitorType;

   struct Parameter
   {
      template<class StateIterator>
      Parameter(
         const size_t maxSubgraphSize,
         StateIterator stateBegin,
         StateIterator stateEnd,
         const Tribool inferMultilabel = Tribool::Maybe
      )
      :  maxSubgraphSize_(maxSubgraphSize),
         startingPoint_(stateBegin, stateEnd),
         inferMultilabel_(inferMultilabel)
      {}

      Parameter(
         const size_t maxSubgraphSize = 2,
         const Tribool inferMultilabel = Tribool::Maybe
      )
      :  maxSubgraphSize_(maxSubgraphSize),
         startingPoint_(),
         inferMultilabel_(inferMultilabel)
      {}

      size_t maxSubgraphSize_;
      std::vector<LabelType> startingPoint_;
      Tribool inferMultilabel_;
   };

   LazyFlipper(const GraphicalModelType&, const size_t = 2, const Tribool useMultilabelInference = Tribool::Maybe);
   LazyFlipper(const GraphicalModelType& gm, typename LazyFlipper::Parameter param);
   template<class StateIterator>
      LazyFlipper(const GraphicalModelType&, const size_t, StateIterator, const Tribool useMultilabelInference = Tribool::Maybe);
   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   const size_t maxSubgraphSize() const;
   ValueType value() const;
   void setMaxSubgraphSize(const size_t);
   void reset();
   InferenceTermination infer();
   template<class VisitorType>
      InferenceTermination infer(VisitorType&);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1)const;

private:
   InferenceTermination inferBinaryLabel();
   template<class VisitorType>
      InferenceTermination inferBinaryLabel(VisitorType&);
   template<class VisitorType>
      InferenceTermination inferMultiLabel(VisitorType&); 
   InferenceTermination inferMultiLabel(); 

   SubgraphForestNode appendVariableToPath(SubgraphForestNode);
   SubgraphForestNode generateFirstPathOfLength(const size_t);
   SubgraphForestNode generateNextPathOfSameLength(SubgraphForestNode);
   void activateInfluencedVariables(SubgraphForestNode, const size_t);
   void deactivateAllVariables(const size_t);
   SubgraphForestNode firstActivePath(const size_t);
   SubgraphForestNode nextActivePath(SubgraphForestNode, const size_t);
   ValueType energyAfterFlip(SubgraphForestNode);
   void flip(SubgraphForestNode);
   const bool flipMultiLabel(SubgraphForestNode); // ???

   const GraphicalModelType& gm_;
   Adjacency variableAdjacency_;
   Movemaker<GraphicalModelType> movemaker_;
   Tagging<bool> activation_[2];
   SubgraphForest subgraphForest_;
   size_t maxSubgraphSize_;
   Tribool useMultilabelInference_;
};

// implementation of Tagging

template<class T>
inline Tagging<T>::Tagging(
   const size_t size
)
:  tags_(tag_container_type(size)),
   indices_(index_container_type())
{}

template<class T>
inline void Tagging<T>::append(
   const size_t number
)
{
   tags_.resize(tags_.size() + number);
}

// runtime complexity: constant
template<class T>
inline typename Tagging<T>::ValueType
Tagging<T>::tag(
   const size_t index
) const
{
   OPENGM_ASSERT(index < tags_.size());
   return tags_[index];
}

// runtime complexity: constant
template<class T>
inline void
Tagging<T>::tag(
   const size_t index,
   const typename Tagging<T>::ValueType tag
)
{
   OPENGM_ASSERT(index < tags_.size());
   OPENGM_ASSERT(tag != T()); // no implicit un-tagging
   if(tags_[index] == T()) { // so far un-tagged
      indices_.push_back(index);
   }
   tags_[index] = tag;
}

// untag all
// runtime complexity: linear in indices_.size()
// note the performance gain over linearity in tags_.size()
template<class T>
inline void
Tagging<T>::untag()
{
   for(const_iterator it = indices_.begin(); it != indices_.end(); ++it) {
      tags_[*it] = T();
   }
   indices_.clear();
}

template<class T>
inline typename Tagging<T>::const_iterator
Tagging<T>::begin() const
{
   return indices_.begin();
}

template<class T>
inline typename Tagging<T>::const_iterator
Tagging<T>::end() const
{
   return indices_.end();
}

template<class T>
inline typename Tagging<T>::const_iterator
Tagging<T>::begin()
{
   return indices_.begin();
}

template<class T>
inline typename Tagging<T>::const_iterator
Tagging<T>::end()
{
   return indices_.end();
}

// implementation of Adjacency
inline
Adjacency::Adjacency(
   const size_t size
)
:  neighbors_(std::vector<std::set<size_t> >(size))
{}

inline void
Adjacency::resize(
   const size_t size
)
{
   neighbors_.resize(size);
}

inline void
Adjacency::connect
(
   const size_t j,
   const size_t k
)
{
   neighbors_[j].insert(k);
   neighbors_[k].insert(j);
}

inline bool
Adjacency::connected(
   const size_t j,
   const size_t k
) const
{
   if(neighbors_[j].size() < neighbors_[k].size()) {
      if(neighbors_[j].find(k) == neighbors_[j].end()) {
         return false;
      }
      else {
         return true;
      }
   }
   else {
      if(neighbors_[k].find(j) == neighbors_[k].end()) {
         return false;
      }
      else {
         return true;
      }
   }
}

inline Adjacency::const_iterator
Adjacency::neighborsBegin(
   const size_t index
)
{
   return neighbors_[index].begin();
}

inline Adjacency::const_iterator
Adjacency::neighborsBegin(
   const size_t index
) const
{
   return neighbors_[index].begin();
}

inline Adjacency::const_iterator
Adjacency::neighborsEnd(
   const size_t index
)
{
   return neighbors_[index].end();
}

inline Adjacency::const_iterator
Adjacency::neighborsEnd(
   const size_t index
) const
{
   return neighbors_[index].end();
}

// implementation

template<class T>
inline Forest<T>::Forest()
:  nodes_(std::vector<typename Forest<T>::Node>()),
   levelAnchors_(std::vector<typename Forest<T>::NodeIndex>())
{}

template<class T>
inline size_t
Forest<T>::levels()
{
   return levelAnchors_.size();
}

template<class T>
inline size_t
Forest<T>::size()
{
   return nodes_.size();
}

template<class T>
inline typename Forest<T>::NodeIndex
Forest<T>::levelAnchor(
   const typename Forest<T>::Level& level
)
{
   OPENGM_ASSERT(level < levels());
   return levelAnchors_[level];
}

template<class T>
inline typename Forest<T>::Value&
Forest<T>::value(
   typename Forest<T>::NodeIndex n
)
{
   OPENGM_ASSERT(n < nodes_.size());
   return nodes_[n].value_;
}

template<class T>
inline typename Forest<T>::Level
Forest<T>::level(
   typename Forest<T>::NodeIndex n
)
{
   OPENGM_ASSERT(n < nodes_.size());
   return nodes_[n].level_;
}

template<class T>
inline typename Forest<T>::NodeIndex
Forest<T>::parent(
   typename Forest<T>::NodeIndex n
)
{
   OPENGM_ASSERT(n < nodes_.size());
   return nodes_[n].parent_;
}

template<class T>
inline typename Forest<T>::NodeIndex
Forest<T>::levelOrderSuccessor(
   typename Forest<T>::NodeIndex n
)
{
   OPENGM_ASSERT(n < nodes_.size());
   return nodes_[n].levelOrderSuccessor_;
}

template<class T>
inline size_t
Forest<T>::numberOfChildren(
   typename Forest<T>::NodeIndex n
)
{
   OPENGM_ASSERT(n < nodes_.size());
   return nodes_[n].children_.size();
}

template<class T>
inline typename Forest<T>::NodeIndex
Forest<T>::child(
   typename Forest<T>::NodeIndex n,
   const size_t j
)
{
   OPENGM_ASSERT((n<nodes_.size() && j<nodes_[n].children_.size()));
   return nodes_[n].children_[j];
}

template<class T>
typename Forest<T>::NodeIndex
Forest<T>::push_back(
   const Value& value,
   typename Forest<T>::NodeIndex parentNodeIndex
)
{
   OPENGM_ASSERT((parentNodeIndex == NONODE || parentNodeIndex < nodes_.size()));
   // lock here in parallel code
   NodeIndex nodeIndex = nodes_.size();
   {
      Node node(value);
      nodes_.push_back(node);
      // unlock here in parallel code
      OPENGM_ASSERT(nodes_.size() == nodeIndex + 1);  // could fail in parallel code
   }
   if(parentNodeIndex != NONODE) {
      nodes_[nodeIndex].parent_ = parentNodeIndex;
      nodes_[parentNodeIndex].children_.push_back(nodeIndex);
      nodes_[nodeIndex].level_ = nodes_[parentNodeIndex].level_ + 1;
   }
   if(nodes_[nodeIndex].level_ >= levelAnchors_.size()) {
      OPENGM_ASSERT(levelAnchors_.size() == nodes_[nodeIndex].level_);
      levelAnchors_.push_back(nodeIndex);
   }
   return nodeIndex;
}

// returns the number of root nodes
template<class T>
size_t
Forest<T>::testInvariant()
{
   if(nodes_.size() == 0) {
      // tree is empty
      OPENGM_ASSERT(levelAnchors_.size() == 0);
      return 0;
   }
   else {
      // tree is not empty
      OPENGM_ASSERT( levelAnchors_.size() != 0);
      size_t numberOfRoots = 0;
      size_t nodesVisited = 0;
      Level level = 0;
      NodeIndex p = levelAnchors_[0];
      while(p != NONODE) {
         ++nodesVisited;
         OPENGM_ASSERT(this->level(p) == level);
         if(level == 0) {
            // p is a root node index
            OPENGM_ASSERT(parent(p) == NONODE);
            ++numberOfRoots;
         }
         else {
            // p is not a root node index
            OPENGM_ASSERT(parent(p) != NONODE);
            // test if p is among the children of its parent:
            bool foundP = false;
            for(size_t j=0; j<nodes_[parent(p)].children_.size(); ++j) {
               if(nodes_[parent(p)].children_[j] == p) {
                  foundP = true;
                  break;
               }
            }
            OPENGM_ASSERT(foundP)
         }
         // continue traversal in level-order
         if(levelOrderSuccessor(p) != NONODE) {
            p = levelOrderSuccessor(p);
         }
         else {
            if(level+1 < levelAnchors_.size()) {
               // tree has more levels
               ++level;
               p = levelAnchors_[level];
            }
            else {
               // tree has no more levels
               break;
            }
         }
      }
      OPENGM_ASSERT(nodesVisited == nodes_.size());
      OPENGM_ASSERT(levels() == level + 1);
      return numberOfRoots;
   }
}

template<class T>
std::string
Forest<T>::asString()
{
   std::ostringstream out(std::ostringstream::out);
   for(size_t level=0; level<levels(); ++level) {
      NodeIndex p = levelAnchor(level);
      while(p != NONODE) {
         // print all variable indices on the path to the root
         NodeIndex q = p;
         while(q != NONODE) {
            // out << value(q) << ' ';
            out << value(q)+1 << ' '; // ??? replace by previous line!!!
            q = parent(q);
         }
         out << std::endl;
         // proceed
         p = levelOrderSuccessor(p);
      }
   }
   return out.str();
}

template<class T>
inline void
Forest<T>::setLevelOrderSuccessor(
   typename Forest<T>::NodeIndex nodeIndex,
   typename Forest<T>::NodeIndex successorNodeIndex
)
{
   OPENGM_ASSERT((nodeIndex < nodes_.size() && successorNodeIndex < nodes_.size()));
   nodes_[nodeIndex].levelOrderSuccessor_ = successorNodeIndex;
}

// implementation of LazyFlipper

template<class GM, class ACC>
inline
LazyFlipper<GM, ACC>::LazyFlipper(
   const GraphicalModelType& gm,
   const size_t maxSubgraphSize,
   const Tribool useMultilabelInference
)
:  gm_(gm),
   variableAdjacency_(Adjacency(gm.numberOfVariables())),
   movemaker_(Movemaker<GM>(gm)),
   subgraphForest_(SubgraphForest()),
   maxSubgraphSize_(maxSubgraphSize),
   useMultilabelInference_(useMultilabelInference)
{
   if(gm_.numberOfVariables() == 0) {
      throw RuntimeError("The graphical model has no variables.");
   }
   setMaxSubgraphSize(maxSubgraphSize);
   // initialize activation_
   activation_[0].append(gm_.numberOfVariables());
   activation_[1].append(gm_.numberOfVariables());
   // initialize variableAdjacency_
   for(size_t j=0; j<gm_.numberOfFactors(); ++j) {
      const FactorType& factor = gm_[j];
      for(size_t m=0; m<factor.numberOfVariables(); ++m) {
         for(size_t n=m+1; n<factor.numberOfVariables(); ++n) {
            variableAdjacency_.connect(factor.variableIndex(m), factor.variableIndex(n));
         }
      }
   }
}

template<class GM, class ACC>
inline
LazyFlipper<GM, ACC>::LazyFlipper(
   const GraphicalModelType& gm,
   typename LazyFlipper::Parameter param
)
:  gm_(gm),
   variableAdjacency_(Adjacency(gm.numberOfVariables())),
   movemaker_(Movemaker<GM>(gm)),
   subgraphForest_(SubgraphForest()),
   maxSubgraphSize_(param.maxSubgraphSize_),
   useMultilabelInference_(param.inferMultilabel_)
{
   if(gm_.numberOfVariables() == 0) {
      throw RuntimeError("The graphical model has no variables.");
   }
   setMaxSubgraphSize(param.maxSubgraphSize_);
   // initialize activation_
   activation_[0].append(gm_.numberOfVariables());
   activation_[1].append(gm_.numberOfVariables());
   // initialize variableAdjacency_
   for(size_t j=0; j<gm_.numberOfFactors(); ++j) {
      const FactorType& factor = gm_[j];
      for(size_t m=0; m<factor.numberOfVariables(); ++m) {
         for(size_t n=m+1; n<factor.numberOfVariables(); ++n) {
            variableAdjacency_.connect(factor.variableIndex(m), factor.variableIndex(n));
         }
      }
   }
   if(param.startingPoint_.size() == gm_.numberOfVariables()) {
      movemaker_.initialize(param.startingPoint_.begin());
   }
}

template<class GM, class ACC>
inline void
LazyFlipper<GM, ACC>::reset()
{}

/// \todo next version: get rid of redundancy with other constructor
template<class GM, class ACC>
template<class StateIterator>
inline
LazyFlipper<GM, ACC>::LazyFlipper(
   const GraphicalModelType& gm,
   const size_t maxSubgraphSize,
   StateIterator it,
   const Tribool useMultilabelInference
)
:  gm_(gm),
   variableAdjacency_(Adjacency(gm_.numberOfVariables())),
   movemaker_(Movemaker<GM>(gm, it)),
   subgraphForest_(SubgraphForest()),
   maxSubgraphSize_(2),
   useMultilabelInference_(useMultilabelInference)
{
   if(gm_.numberOfVariables() == 0) {
      throw RuntimeError("The graphical model has no variables.");
   }
   setMaxSubgraphSize(maxSubgraphSize);
   // initialize activation_
   activation_[0].append(gm_.numberOfVariables());
   activation_[1].append(gm_.numberOfVariables());
   // initialize variableAdjacency_
   for(size_t j=0; j<gm_.numberOfFactors(); ++j) {
      const FactorType& factor = gm_[j];
      for(size_t m=0; m<factor.numberOfVariables(); ++m) {
         for(size_t n=m+1; n<factor.numberOfVariables(); ++n) {
            variableAdjacency_.connect(factor.variableIndex(m), factor.variableIndex(n));
         }
      }
   }
}

template<class GM, class ACC>
inline void
LazyFlipper<GM, ACC>::setStartingPoint(
   typename std::vector<typename LazyFlipper<GM, ACC>::LabelType>::const_iterator begin
) {
   movemaker_.initialize(begin);
}

template<class GM, class ACC>
inline std::string
LazyFlipper<GM, ACC>::name() const
{
   return "LazyFlipper";
}

template<class GM, class ACC>
inline const typename LazyFlipper<GM, ACC>::GraphicalModelType&
LazyFlipper<GM, ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC>
inline const size_t
LazyFlipper<GM, ACC>::maxSubgraphSize() const
{
   return maxSubgraphSize_;
}

template<class GM, class ACC>
inline void
LazyFlipper<GM, ACC>::setMaxSubgraphSize(
   const size_t maxSubgraphSize
)
{
   if(maxSubgraphSize < 1) {
      throw RuntimeError("Maximum subgraph size < 1.");
   }
   else {
      maxSubgraphSize_ = maxSubgraphSize;
   }
}

/// \brief start the algorithm
template<class GM, class ACC>
template<class VisitorType>
inline InferenceTermination
LazyFlipper<GM, ACC>::infer(
   VisitorType& visitor
)
{
   bool multiLabel;
   if(this->useMultilabelInference_ == true) {
      multiLabel = true;
   }
   else if(this->useMultilabelInference_ == false) {
      multiLabel = false;
   }
   else {
      multiLabel = false;
      for(size_t i=0; i<gm_.numberOfVariables(); ++i) {
         if(gm_.numberOfLabels(i) != 2) {
            multiLabel = true;
            break;
         }
      }
   }

   if(multiLabel) {
      return this->inferMultiLabel(visitor);
   }
   else {
      return this->inferBinaryLabel(visitor);
   }
}

/// \brief start the algorithm
template<class GM, class ACC>
inline InferenceTermination
LazyFlipper<GM, ACC>::infer()
{
   EmptyVisitorType visitor;
   return this->infer(visitor);
}

template<class GM, class ACC>
template<class VisitorType>
InferenceTermination
LazyFlipper<GM, ACC>::inferBinaryLabel(
   VisitorType& visitor
) 
{
   bool continueInf = true;
   size_t length = 1;
   //const ValueType bound = this->bound();
   //visitor.begin(*this, movemaker_.value(), bound, length, subgraphForest_.size());
   visitor.begin(*this);
   while(continueInf) {
      //visitor(*this, movemaker_.value(), bound, length, subgraphForest_.size());
      if(visitor(*this)!=0){
         continueInf=false;
         break;
      }
      SubgraphForestNode p = generateFirstPathOfLength(length);
      if(p == NONODE) {
         break;
      }
      else {
         while(p != NONODE) {
            if(AccumulationType::bop(energyAfterFlip(p), movemaker_.value())) {
               flip(p);
               activateInfluencedVariables(p, 0);
               //visitor(*this, movemaker_.value(), bound, length, subgraphForest_.size());
               if(visitor(*this)!=0){
                  continueInf=false;
                  break;
               }
            }
            p = generateNextPathOfSameLength(p);
         }
         size_t currentActivationList = 0;
         size_t nextActivationList = 1;
         while(continueInf) {
            SubgraphForestNode p2 = firstActivePath(currentActivationList);
            if(p2 == NONODE) {
               break;
            }
            else {
               while(p2 != NONODE) {
                  if(AccumulationType::bop(energyAfterFlip(p2), movemaker_.value())) {
                     flip(p2);
                     activateInfluencedVariables(p2, nextActivationList);
                     //visitor(*this, movemaker_.value(), bound, length, subgraphForest_.size());
                     if(visitor(*this)!=0){
                        continueInf=false;
                        break;
                     }
                  }
                  p2 = nextActivePath(p2, currentActivationList);
               }
               deactivateAllVariables(currentActivationList);
               nextActivationList = 1 - nextActivationList;
               currentActivationList = 1 - currentActivationList;
            }
         }
      }
      if(length == maxSubgraphSize_) {
         break;
      }
      else {
         ++length;
      }
   }
   // assertion testing
   if(!NO_DEBUG) {
      subgraphForest_.testInvariant();
   }
   //visitor.end(*this, movemaker_.value(), bound, length, subgraphForest_.size());
   visitor.end(*this);
   // diagnose
   // std::cout << subgraphForest_.asString();
   return NORMAL;
}

template<class GM, class ACC>
inline InferenceTermination
LazyFlipper<GM, ACC>::inferBinaryLabel()
{
   EmptyVisitorType v;
   return infer(v);
}

template<class GM, class ACC>
template<class VisitorType>
InferenceTermination
LazyFlipper<GM, ACC>::inferMultiLabel(
   VisitorType& visitor
)
{
   bool continueInf = true;
   size_t length = 1;
   //const ValueType bound = this->bound();
   //visitor.begin(*this, movemaker_.value(), bound, length, subgraphForest_.size());
   visitor.begin(*this);
   while(continueInf) {
      //visitor(*this, movemaker_.value(), bound, length, subgraphForest_.size());
      if(visitor(*this)!=0){
         continueInf = false;
         break;
      }
      SubgraphForestNode p = generateFirstPathOfLength(length);
      if(p == NONODE) {
         break;
      }
      else {
         while(p != NONODE) {
            bool flipped = flipMultiLabel(p);
            if(flipped) {
               activateInfluencedVariables(p, 0);
               //visitor(*this, movemaker_.value(), bound, length, subgraphForest_.size());
               if(visitor(*this)!=0){
                  continueInf = false;
                  break;
               }
            }
            p = generateNextPathOfSameLength(p);
         }
         size_t currentActivationList = 0;
         size_t nextActivationList = 1;
         while(continueInf) {
            SubgraphForestNode p2 = firstActivePath(currentActivationList);
            if(p2 == NONODE) {
               break;
            }
            else {
               while(p2 != NONODE) {
                  bool flipped = flipMultiLabel(p2);
                  if(flipped) {
                     activateInfluencedVariables(p2, nextActivationList);
                     //visitor(*this, movemaker_.value(), bound, length, subgraphForest_.size());
                     if(visitor(*this)!=0){
                        continueInf = false;
                        break;
                     }
                  }
                  p2 = nextActivePath(p2, currentActivationList);
               }
               deactivateAllVariables(currentActivationList);
               nextActivationList = 1 - nextActivationList;
               currentActivationList = 1 - currentActivationList;
            }
         }
      }
      if(length == maxSubgraphSize_) {
         break;
      }
      else {
         ++length;
      }
   }
   // assertion testing
   if(!NO_DEBUG) {
      subgraphForest_.testInvariant();
   }
   // diagnose
   // std::cout << subgraphForest_.asString();
   //visitor.end(*this, movemaker_.value(), bound, length, subgraphForest_.size());
   visitor.end(*this);
   return NORMAL;
}

template<class GM, class ACC>
inline InferenceTermination
LazyFlipper<GM, ACC>::inferMultiLabel()
{
   EmptyVisitorType visitor;
   return this->inferMultiLabel(visitor);
}

template<class GM, class ACC>
inline InferenceTermination
LazyFlipper<GM, ACC>::arg(
   std::vector<LabelType>& arg,
   const size_t n
) const
{
   if(n > 1) {
      return UNKNOWN;
   }
   else {
      arg.resize(gm_.numberOfVariables());
      for(size_t j=0; j<gm_.numberOfVariables(); ++j) {
         arg[j] = movemaker_.state(j);
      }
      return NORMAL;
   }
}

template<class GM, class ACC>
inline typename LazyFlipper<GM, ACC>::ValueType
LazyFlipper<GM, ACC>::value() const
{
   return movemaker_.value();
}

// Append the next possible variable to a node in the subgraph tree.
// The null pointer is returned if no variable can be appended.
template<class GM, class ACC>
typename LazyFlipper<GM, ACC>::SubgraphForestNode
LazyFlipper<GM, ACC>::appendVariableToPath(
   typename LazyFlipper<GM, ACC>::SubgraphForestNode p // input
)
{
   // collect variable indices on path
   std::vector<size_t> variableIndicesOnPath(subgraphForest_.level(p) + 1);
   {
      SubgraphForestNode p2 = p;
      for(size_t j=0; j<=subgraphForest_.level(p); ++j) {
         OPENGM_ASSERT(p2 != NONODE);
         variableIndicesOnPath[subgraphForest_.level(p) - j] = subgraphForest_.value(p2);
         p2 = subgraphForest_.parent(p2);
      }
      OPENGM_ASSERT(p2 == NONODE);
   }
   // find the mininum and maximum variable index on the path
   size_t minVI = variableIndicesOnPath[0];
   size_t maxVI = variableIndicesOnPath[0];
   for(size_t j=1; j<variableIndicesOnPath.size(); ++j) {
      if(variableIndicesOnPath[j] > maxVI) {
         maxVI = variableIndicesOnPath[j];
      }
   }
   // find the maximum variable index among the children of p.
   // the to be appended variable must have a greater index.
   if(subgraphForest_.numberOfChildren(p) > 0) {
      size_t maxChildIndex = subgraphForest_.numberOfChildren(p) - 1;
      minVI = subgraphForest_.value(subgraphForest_.child(p, maxChildIndex));
   }
   // build set of candidate variable indices for appending
   std::set<size_t> candidateVariableIndices;
   {
      SubgraphForestNode q = p;
      while(q != NONODE) {
         for(Adjacency::const_iterator it = variableAdjacency_.neighborsBegin(subgraphForest_.value(q));
            it != variableAdjacency_.neighborsEnd(subgraphForest_.value(q)); ++it) {
               candidateVariableIndices.insert(*it);
         }
         q = subgraphForest_.parent(q);
      }
   }
   // append candidate if possible
   for(std::set<size_t>::const_iterator it = candidateVariableIndices.begin();
      it != candidateVariableIndices.end(); ++it) {
         // for all variables adjacenct to the one at node p
         if(*it > minVI && std::find(variableIndicesOnPath.begin(), variableIndicesOnPath.end(), *it) == variableIndicesOnPath.end()) {
            // the variable index *it is not smaller than the lower bound AND
            // greater than the minimum variable index on the path AND
            // is not itself on the path (??? consider tagging instead of
            // searching in the previous if-condition)
            if(*it > maxVI) {
               // *it is greater than the largest variable index on the path
               return subgraphForest_.push_back(*it, p); // append to path
            }
            else {
               // *it is not the greatest variable index on the path.
               for(size_t j=1; j<variableIndicesOnPath.size(); ++j) {
                  if(variableAdjacency_.connected(variableIndicesOnPath[j-1], *it)) {
                     // *it could have been added as a child of
                     // variableIndicesOnPath[j-1]
                     for(size_t k=j; k<variableIndicesOnPath.size(); ++k) {
                        if(*it < variableIndicesOnPath[k]) {
                           // adding *it as a child of variableIndicesOnPath[j-1]
                           // would have made the path cheaper
                           goto doNotAppend; // escape loop over j
                        }
                     }
                  }
               }
               // *it could not have been introduced cheaper
               // append to path:
               return subgraphForest_.push_back(*it, p);
doNotAppend:;
            }
         }
   }
   // no neighbor of p could be appended
   return NONODE;
}

template<class GM, class ACC>
typename LazyFlipper<GM, ACC>::SubgraphForestNode
LazyFlipper<GM, ACC>::generateFirstPathOfLength(
   const size_t length
)
{
   OPENGM_ASSERT(length > 0);
   if(length > gm_.numberOfVariables()) {
      return NONODE;
   }
   else {
      if(length == 1) {
         SubgraphForestNode p = subgraphForest_.push_back(0, NONODE);
         // variable index = 0, parent = NONODE
         return p;
      }
      else {
         SubgraphForestNode p = subgraphForest_.levelAnchor(length-2);
         while(p != NONODE) {
            SubgraphForestNode p2 = appendVariableToPath(p);
            if(p2 != NONODE) { // append succeeded
               return p2;
            }
            else { // append failed
               p = subgraphForest_.levelOrderSuccessor(p);
            }
         }
         return NONODE;
      }
   }
}

template<class GM, class ACC>
typename LazyFlipper<GM, ACC>::SubgraphForestNode
LazyFlipper<GM, ACC>::generateNextPathOfSameLength(
   SubgraphForestNode predecessor
)
{
   if(subgraphForest_.level(predecessor) == 0) {
      if(subgraphForest_.value(predecessor) + 1 < gm_.numberOfVariables()) {
         SubgraphForestNode newNode =
            subgraphForest_.push_back(subgraphForest_.value(predecessor) + 1, NONODE);
         subgraphForest_.setLevelOrderSuccessor(predecessor, newNode);
         return newNode;
      }
      else {
         // no more variables
         return NONODE;
      }
   }
   else {
      for(SubgraphForestNode parent = subgraphForest_.parent(predecessor);
         parent != NONODE; parent = subgraphForest_.levelOrderSuccessor(parent) ) {
            SubgraphForestNode newNode = appendVariableToPath(parent);
            if(newNode != NONODE) {
               // a variable has been appended
               subgraphForest_.setLevelOrderSuccessor(predecessor, newNode);
               return newNode;
            }
      }
      return NONODE;
   }
}

template<class GM, class ACC>
void
LazyFlipper<GM, ACC>::activateInfluencedVariables(
   SubgraphForestNode p,
   const size_t activationListIndex
)
{
   OPENGM_ASSERT(activationListIndex < 2);
   while(p != NONODE) {
      activation_[activationListIndex].tag(subgraphForest_.value(p), true);
      for(Adjacency::const_iterator it = variableAdjacency_.neighborsBegin(subgraphForest_.value(p));
         it != variableAdjacency_.neighborsEnd(subgraphForest_.value(p)); ++it) {
            activation_[activationListIndex].tag(*it, true);
      }
      p = subgraphForest_.parent(p);
   }
}

template<class GM, class ACC>
inline void
LazyFlipper<GM, ACC>::deactivateAllVariables(
   const size_t activationListIndex
)
{
   OPENGM_ASSERT(activationListIndex < 2);
   activation_[activationListIndex].untag();
}

template<class GM, class ACC>
typename LazyFlipper<GM, ACC>::SubgraphForestNode
LazyFlipper<GM, ACC>::firstActivePath(
   const size_t activationListIndex
)
{
   if(subgraphForest_.levels() == 0) {
      return NONODE;
   }
   else {
      // ??? improve code: no search, store reference
      SubgraphForestNode p = subgraphForest_.levelAnchor(0);
      while(p != NONODE) {
         if(activation_[activationListIndex].tag(subgraphForest_.value(p))) {
            return p;
         }
         p = subgraphForest_.levelOrderSuccessor(p);
      }
      return NONODE;
   }
}

// \todo next version: improve code: searching over all paths and all 
// variables of each path for active variables is certainly not the ideal 
// way
template<class GM, class ACC>
typename LazyFlipper<GM, ACC>::SubgraphForestNode
LazyFlipper<GM, ACC>::nextActivePath(
   SubgraphForestNode predecessor,
   const size_t activationListIndex
)
{
   for(;;) {
      if(subgraphForest_.levelOrderSuccessor(predecessor) == NONODE) {
         if(subgraphForest_.level(predecessor) + 1 < subgraphForest_.levels()) {
            // there are more levels in the tree
            predecessor = subgraphForest_.levelAnchor(subgraphForest_.level(predecessor) + 1);
         }
         else {
            // there are no more levels in the tree
            return NONODE;
         }
      }
      else {
         // predecessor is not the last node on its level
         predecessor = subgraphForest_.levelOrderSuccessor(predecessor);
      }
      SubgraphForestNode p = predecessor;
      while(p != NONODE) {
         // search along path for active variables:
         if(activation_[activationListIndex].tag(subgraphForest_.value(p))) {
            return predecessor;
         }
         p = subgraphForest_.parent(p);
      }
   }
}

template<class GM, class ACC>
inline typename LazyFlipper<GM, ACC>::ValueType
LazyFlipper<GM, ACC>::energyAfterFlip(
   SubgraphForestNode node
)
{
   size_t numberOfFlippedVariables = subgraphForest_.level(node) + 1;
   std::vector<size_t> flippedVariableIndices(numberOfFlippedVariables);
   std::vector<LabelType> flippedVariableStates(numberOfFlippedVariables);
   for(size_t j=0; j<numberOfFlippedVariables; ++j) {
      OPENGM_ASSERT(node != NONODE);
      flippedVariableIndices[j] = subgraphForest_.value(node);
      // binary flip:
      flippedVariableStates[j] = 1 - movemaker_.state(subgraphForest_.value(node));
      node = subgraphForest_.parent(node);
   }
   OPENGM_ASSERT(node == NONODE);
   return movemaker_.valueAfterMove(flippedVariableIndices.begin(),
      flippedVariableIndices.end(), flippedVariableStates.begin());

}

template<class GM, class ACC>
inline void
LazyFlipper<GM, ACC>::flip(
   SubgraphForestNode node
)
{
   size_t numberOfFlippedVariables = subgraphForest_.level(node) + 1;
   std::vector<size_t> flippedVariableIndices(numberOfFlippedVariables);
   std::vector<LabelType> flippedVariableStates(numberOfFlippedVariables);
   for(size_t j=0; j<numberOfFlippedVariables; ++j) {
      OPENGM_ASSERT(node != NONODE)
         flippedVariableIndices[j] = subgraphForest_.value(node);
      // binary flip:
      flippedVariableStates[j] = 1 - movemaker_.state(subgraphForest_.value(node));
      node = subgraphForest_.parent(node);
   }
   OPENGM_ASSERT(node == NONODE);
   movemaker_.move(flippedVariableIndices.begin(),
      flippedVariableIndices.end(), flippedVariableStates.begin());
}

template<class GM, class ACC>
inline const bool
LazyFlipper<GM, ACC>::flipMultiLabel(
   SubgraphForestNode node
)
{
   size_t numberOfVariables = subgraphForest_.level(node) + 1;
   std::vector<size_t> variableIndices(numberOfVariables);
   for(size_t j=0; j<numberOfVariables; ++j) {
      OPENGM_ASSERT(node != NONODE);
      variableIndices[j] = subgraphForest_.value(node);
      node = subgraphForest_.parent(node);
   }
   OPENGM_ASSERT(node == NONODE);
   ValueType energy = movemaker_.value();
   movemaker_.template moveOptimallyWithAllLabelsChanging<AccumulationType>(variableIndices.begin(), variableIndices.end());
   if(AccumulationType::bop(movemaker_.value(), energy)) {
      return true;
   }
   else {
      return false;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_LAZYFLIPPER_HXX
