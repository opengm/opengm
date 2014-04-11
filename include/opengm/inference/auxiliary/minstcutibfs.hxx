#pragma once
#ifndef OPENGM_MINSTCUTIBFS_HXX
#define OPENGM_MINSTCUTIBFS_HXX

#include <queue> 
#include <cassert>
#include "ibfs.h"

namespace opengm {
  namespace external {

    /// \brief IBFS solver for the min st-cut framework GraphCut
    template<class NType, class VType>
    class MinSTCutIBFS{
    public:
      // Type-Definitions
      typedef NType node_type;
      typedef VType ValueType;
      //typedef IBFSGraph<ValueType,ValueType,ValueType> graph_type;
      typedef IBFSGraph graph_type;
     
      // Methods
      MinSTCutIBFS();
      MinSTCutIBFS(size_t numberOfNodes, size_t numberOfEdges);
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
    MinSTCutIBFS<NType,VType>::MinSTCutIBFS() {
      numberOfNodes_ = 2;
      numberOfEdges_ = 0;
      graph_         = NULL;
    };

    template<class NType, class VType>
    MinSTCutIBFS<NType,VType>::MinSTCutIBFS(size_t numberOfNodes, size_t numberOfEdges) {
      numberOfNodes_ = numberOfNodes;
      numberOfEdges_ = numberOfEdges;
      //graph_         = new graph_type(numberOfNodes_-2,numberOfEdges_); 
      graph_         = new graph_type();
      graph_->initSize((int)(numberOfNodes_-2),(int)(numberOfEdges_ - 2*(numberOfNodes_-2))); 
      //for(size_t i=0; i<numberOfNodes_-2;++i)
      //  graph_->add_node(); 
    };
 

    template<class NType, class VType>
    void MinSTCutIBFS<NType,VType>::addEdge(node_type n1, node_type n2, ValueType cost) {
      assert(n1 < numberOfNodes_);
      assert(n2 < numberOfNodes_);
      assert(cost >= 0);
      if(n1==S) {
	//graph_->add_tweights( n2-2,   /* capacities */  cost, 0);
	graph_->addNode( (int)(n2-2),   /* capacities */  (int)(cost*10000), 0);
      }else if(n2==T) {
	//graph_->add_tweights( n1-2,   /* capacities */  0, cost);
	graph_->addNode( (int)(n1-2),   /* capacities */  0, (int)(cost*10000));
      }else{
	//graph_->add_edge( n1-2, n2-2,    /* capacities */  cost, 0 );
	graph_->addEdge( (int)(n1-2), (int)(n2-2),    /* capacities */  (int)(cost*10000), 0 );
	  
      }
    };
   
    template<class NType, class VType>
    void MinSTCutIBFS<NType,VType>::calculateCut(std::vector<bool>& segmentation) {  
      //int flow = graph_->maxflow();
      graph_->initGraph();
      graph_->computeMaxFlow();
      segmentation.resize(numberOfNodes_);
      for(size_t i=2; i<numberOfNodes_; ++i) {
	//if (graph_->what_segment(i-2) == graph_type::SOURCE) 
	if(graph_->isNodeOnSrcSide(i-2) )
	  segmentation[i]=false;
	else
	  segmentation[i]=true;
      }  
      return;
    };

  } //namespace external
} // namespace opengm

#endif // #ifndef OPENGM_MINSTCUTIBFS_HXX

