#pragma once
#ifndef OPENGM_PLANAR_MAXCUT_HXX
#define OPENGM_PLANAR_MAXCUT_HXX


#include "planar_maxcut_graph.hxx"

namespace opengm {

/// OpenGM wrappers around third party algorithms
namespace external {

/// \cond HIDDEN_SYMBOLS
   
/// \brief MAXCUT for planar graphs
class PlanarMaxCut {
public:
   typedef size_t node_type;
   typedef double value_type;

   PlanarMaxCut();
   ~PlanarMaxCut();
   PlanarMaxCut(size_t numberOfNodes, size_t numberOfEdges);
   void addEdge(node_type,node_type,value_type);
   template <class VEC> void calculateCut(VEC&);
      
private:
   typename pmc::Graph graph_; 
   typename pmc::Node** NodesPtr; // This should be moved to pmc::Graph
 
   size_t      numberOfNodes_;
   size_t      numberOfEdges_; // usused so far
};

    
   inline PlanarMaxCut::PlanarMaxCut() {
      numberOfNodes_ = 0;
      numberOfEdges_ = 0;
   }

   inline PlanarMaxCut::PlanarMaxCut(size_t numberOfNodes, size_t numberOfEdges) { 
      numberOfNodes_ = numberOfNodes;
      numberOfEdges_ = numberOfEdges;

      NodesPtr = new pmc::Node*[numberOfNodes];
      for(size_t variableId=0; variableId < numberOfNodes; ++variableId)
      {
         pmc::Node* v = graph_.add_node(variableId, 0.0);
         NodesPtr[variableId] = v;
      }
   }
    
   inline PlanarMaxCut::~PlanarMaxCut() 
   {
      if( NodesPtr != NULL)
         delete[] NodesPtr;
   }  
    
   inline void PlanarMaxCut::addEdge(node_type n1, node_type n2, value_type cost) {
      assert(n1 < numberOfNodes_);
      assert(n2 < numberOfNodes_);
      pmc::Node* v_i = NodesPtr[n1];
      pmc::Node* v_j = NodesPtr[n2];  
      pmc::Edge* e_ij = graph_.find_edge(v_i,v_j);
   
      if(e_ij==NULL){
         graph_.add_edge(v_i,v_j,cost);
      }else{
         e_ij->weight += cost;
      }
   }
   
   template <class VEC>
   void PlanarMaxCut::calculateCut(VEC& segmentation) { 

      graph_.planarize(); // Planarize graph
      graph_.construct_dual(); // Construct dual graph
      graph_.mcpm(); // Perform perfect matching / max cut
      graph_.assign_labels(); // Derive labels from cut set
      graph_.read_labels(segmentation);
      //segmentation = graph_.read_labels();
      return;
   }

/// \endcond 

} // namespace external
} // namespace opengm

#endif // #ifndef OPENGM_PLANAR_MAXCUT_HXX
