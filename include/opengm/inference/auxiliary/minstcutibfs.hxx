#pragma once
#ifndef OPENGM_MINSTCUTKOLMOGOROV_HXX
#define OPENGM_MINSTCUTKOLMOGOROV_HXX

#include <queue> 
#include <cassert>

#include "MaxFlow-v3.01/graph.h"

namespace opengm {
  namespace external {

    /// \brief V. Kolmogorov's solver for the min st-cut framework GraphCut
    template<class NType, class VType>
    class MinSTCutKolmogorov{
    public:
      // Type-Definitions
      typedef NType node_type;
      typedef VType ValueType; 
      typedef maxflowLib::Graph<ValueType,ValueType,ValueType> graph_type;
     
      // Methods
      MinSTCutKolmogorov();
      MinSTCutKolmogorov(size_t numberOfNodes, size_t numberOfEdges);
      void addEdge(node_type,node_type,ValueType);
      void calculateCut(std::vector<bool>&);
      
    private: 
      // Members
      graph_type* graph_;
      size_t      numberOfNodes_;
      size_t      numberOfEdges_;
      static const NType S = 0;
      static const NType T = 1;
    };
    
    //*********************
    //** Implementation  **
    //*********************
    
    template<class NType, class VType>
    MinSTCutKolmogorov<NType,VType>::MinSTCutKolmogorov() {
      numberOfNodes_ = 2;
      numberOfEdges_ = 0;
      graph_         = NULL;
    };

    template<class NType, class VType>
    MinSTCutKolmogorov<NType,VType>::MinSTCutKolmogorov(size_t numberOfNodes, size_t numberOfEdges) {
      numberOfNodes_ = numberOfNodes;
      numberOfEdges_ = numberOfEdges;
      graph_         = new graph_type(numberOfNodes_-2,numberOfEdges_); 
      for(size_t i=0; i<numberOfNodes_-2;++i)
	graph_->add_node(); 
    };
 

    template<class NType, class VType>
    void MinSTCutKolmogorov<NType,VType>::addEdge(node_type n1, node_type n2, ValueType cost) {
      assert(n1 < numberOfNodes_);
      assert(n2 < numberOfNodes_);
      assert(cost >= 0);
      if(n1==S) {
	graph_->add_tweights( n2-2,   /* capacities */  cost, 0);
      }else if(n2==T) {
	graph_->add_tweights( n1-2,   /* capacities */  0, cost);
      }else{
	graph_->add_edge( n1-2, n2-2,    /* capacities */  cost, 0 );
      }
    };
   
    template<class NType, class VType>
    void MinSTCutKolmogorov<NType,VType>::calculateCut(std::vector<bool>& segmentation) {  
      int flow = graph_->maxflow(); 
      segmentation.resize(numberOfNodes_);
      for(size_t i=2; i<numberOfNodes_; ++i) {
	if (graph_->what_segment(i-2) == graph_type::SOURCE)
	  segmentation[i]=false;
	else
	  segmentation[i]=true;
      }  
      return;
    };

  } //namespace external
} // namespace opengm

#endif // #ifndef OPENGM_MINSTCUTBOOST_HXX

