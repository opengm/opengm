#pragma once
#ifndef OPENGM_PLANAR_MAXCUT_HXX
#define OPENGM_PLANAR_MAXCUT_HXX
#include "planar_graph.hxx"

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
   void calculateCut();
   template <class VEC> void getCut(VEC&);
   template <class VEC> void getLabeling(VEC&);
      
private:
   planargraph::PlanarGraph graph_; 
 
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

      for(size_t variableId=0; variableId < numberOfNodes; ++variableId)
      {
         graph_.add_node();
      }
   }
    
   inline PlanarMaxCut::~PlanarMaxCut() 
   {
   }  
    
   inline void PlanarMaxCut::addEdge(node_type n1, node_type n2, value_type cost) {
      assert(n1 < numberOfNodes_);
      assert(n2 < numberOfNodes_); 
      long int e = graph_.find_edge(n1,n2);
      if(e == -1){
         graph_.add_edge(n1, n2, cost);
      } else {
         graph_.add_edge_weight(e, cost);
      }
   }
   
   void PlanarMaxCut::calculateCut() { 

      graph_.planarize();      // Planarize graph
      graph_.construct_dual(); // Construct dual graph
      graph_.calculate_maxcut();          // Perform perfect matching / max cut  

   }
   
   template <class VEC>
   void PlanarMaxCut::getCut(VEC& cut) { 
      
      // todo: add temptated interface in planargraph
      std::vector<bool> temp = graph_.get_cut();
      if(cut.size()<temp.size())
         cut.resize(temp.size());
      for(size_t i=0; i<temp.size(); ++i)
         cut[i]=temp[i];
      return;
   }

   template <class VEC>
   void PlanarMaxCut::getLabeling(VEC& segmentation) { 
      
      // todo: add temptated interface in planargraph
      std::vector<int> temp;
      graph_.get_labeling(temp);
      if(segmentation.size()<temp.size())
         segmentation.resize(temp.size());
      for(size_t i=0; i<temp.size(); ++i)
         segmentation[i]=temp[i];
      return;
   }

/// \endcond 

} // namespace external
} // namespace opengm

#endif // #ifndef OPENGM_PLANAR_MAXCUT_HXX
