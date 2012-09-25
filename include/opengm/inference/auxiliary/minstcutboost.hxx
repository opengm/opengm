#pragma once
#ifndef OPENGM_MINSTCUTBOOST_HXX
#define OPENGM_MINSTCUTBOOST_HXX

#ifndef BOOST_DISABLE_ASSERTS
#define BOOST_DISABLE_ASSERTS
#define USED_BOOST_DISABLE_ASSERTS
#endif  

#include <queue>
#include <cassert>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>

namespace opengm {

   enum BoostMaxFlowAlgorithm {
      PUSH_RELABEL, EDMONDS_KARP, KOLMOGOROV
   };

   /// \brief Boost solvers for the min st-cut framework GraphCut
   template<class NType, class VType, BoostMaxFlowAlgorithm mfalg>
   class MinSTCutBoost {
   public:
      // Type-Definitions
      typedef NType node_type;
      typedef VType ValueType;
      typedef boost::vecS OutEdgeList;
      typedef boost::vecS VertexList;
      typedef boost::adjacency_list_traits<OutEdgeList, VertexList, boost::directedS> graph_traits;
      typedef graph_traits::edge_descriptor edge_descriptor;
      typedef graph_traits::vertex_descriptor vertex_descriptor;

      /// \cond HIDDEN_SYMBOLS
      struct Edge {
         Edge() : capacity(ValueType()), residual(ValueType()), reverse(edge_descriptor()) 
            {}
         ValueType capacity;
         ValueType residual;
         edge_descriptor reverse;
      };
      /// \endcond

      typedef boost::adjacency_list<OutEdgeList, VertexList, boost::directedS, size_t, Edge> graph_type;
      typedef typename boost::graph_traits<graph_type>::edge_iterator edge_iterator;
      typedef typename boost::graph_traits<graph_type>::out_edge_iterator out_edge_iterator;

      // Methods
      MinSTCutBoost();
      MinSTCutBoost(size_t numberOfNodes, size_t numberOfEdges);
      void addEdge(node_type, node_type, ValueType);
      void calculateCut(std::vector<bool>&);

   private:
      // Members
      graph_type graph_;
      size_t numberOfNodes_;
      size_t numberOfEdges_;
      static const NType S = 0;
      static const NType T = 1;
   };

   //*********************
   //** Implementation  **
   //*********************

   template<class NType, class VType, BoostMaxFlowAlgorithm mfalg>
   MinSTCutBoost<NType, VType, mfalg>::MinSTCutBoost() {
      numberOfNodes_ = 2;
      numberOfEdges_ = 0;
   }

   template<class NType, class VType, BoostMaxFlowAlgorithm mfalg>
   MinSTCutBoost<NType, VType, mfalg>::MinSTCutBoost(size_t numberOfNodes, size_t numberOfEdges) {
      numberOfNodes_ = numberOfNodes;
      numberOfEdges_ = numberOfEdges;
      graph_ = graph_type(numberOfNodes_);
      //std::cout << "#nodes : " << numberOfNodes_ << std::endl;
   }

   template<class NType, class VType, BoostMaxFlowAlgorithm mfalg>
   void MinSTCutBoost<NType, VType, mfalg>::addEdge(node_type n1, node_type n2, ValueType cost) {
      assert(n1 < numberOfNodes_);
      assert(n2 < numberOfNodes_);
      assert(cost >= 0);
      std::pair<edge_descriptor, bool> e = add_edge(n1, n2, graph_);
      std::pair<edge_descriptor, bool> er = add_edge(n2, n1, graph_);
      graph_[e.first].capacity += cost;
      graph_[e.first].reverse = er.first;
      graph_[er.first].reverse = e.first;
      //std::cout << n1 << "->" << n2 << " : " << cost << std::endl;
   }

   template<class NType, class VType, BoostMaxFlowAlgorithm mfalg>
   void MinSTCutBoost<NType, VType, mfalg>::calculateCut(std::vector<bool>& segmentation) {
      if (mfalg == KOLMOGOROV) {//Kolmogorov
         std::vector<boost::default_color_type> color(num_vertices(graph_));
         std::vector<edge_descriptor> pred(num_vertices(graph_));
         std::vector<vertex_descriptor> dist(num_vertices(graph_));
         boykov_kolmogorov_max_flow(graph_,
            get(&Edge::capacity, graph_),
            get(&Edge::residual, graph_),
            get(&Edge::reverse, graph_),
            &pred[0],
            &color[0],
            &dist[0],
            get(boost::vertex_index, graph_),
            S, T
            );
         // find (s,t)-cut set
         segmentation.resize(num_vertices(graph_));
         for (size_t j = 2; j < num_vertices(graph_); ++j) {
            if (color[j] == boost::black_color || color[j] == boost::gray_color) {
               segmentation[j] = false;
            } else if (color[j] == boost::white_color) {
               segmentation[j] = true;
            }
         }
      } 
      else if (mfalg == PUSH_RELABEL) {// PushRelable

         push_relabel_max_flow(graph_, S, T,
            get(&Edge::capacity, graph_),
            get(&Edge::residual, graph_),
            get(&Edge::reverse, graph_),
            get(boost::vertex_index_t(), graph_)
            );
         // find (s,t)-cut set 
         segmentation.resize(num_vertices(graph_), true);
         segmentation[S] = false; // source
         segmentation[T] = false; // sink
         typedef typename boost::property_map<graph_type, boost::vertex_index_t>::type VertexIndexMap;
         VertexIndexMap vertexIndexMap = get(boost::vertex_index, graph_);
         std::queue<vertex_descriptor> q;
         q.push(*(vertices(graph_).first)); // source
         while (!q.empty()) {
            out_edge_iterator current, end;
            tie(current, end) = out_edges(q.front(), graph_);
            q.pop();
            while (current != end) {
               if (graph_[*current].residual > 0) {
                  vertex_descriptor v = target(*current, graph_);
                  if (vertexIndexMap[v] > 1 && segmentation[vertexIndexMap[v]] == true) {
                     segmentation[vertexIndexMap[v]] = false;
                     q.push(v);
                  }
               }
               ++current;
            }
         }
      } 
      else if (mfalg == EDMONDS_KARP) {//EdmondsKarp
         std::vector<boost::default_color_type> color(num_vertices(graph_));
         std::vector<edge_descriptor> pred(num_vertices(graph_));
         edmonds_karp_max_flow(graph_, S, T,
            get(&Edge::capacity, graph_),
            get(&Edge::residual, graph_),
            get(&Edge::reverse, graph_),
            &color[0], &pred[0]
            );
         // find (s,t)-cut set
         segmentation.resize(num_vertices(graph_));
         for (size_t j = 2; j < num_vertices(graph_); ++j) {
            if (color[j] == boost::black_color) {
               segmentation[j] = false;
            } else if (color[j] == boost::white_color) {
               segmentation[j] = true;
            } else {
               throw std::runtime_error("At least one vertex is labeled neither black nor white.");
            }
         }
      } 
      else {//UNKNOWN MaxFlowalgorithm
         throw std::runtime_error("Unknown MaxFlow-algorithm in MinSTCutBoost.hxx");
      }
      return;
   }

} // namespace opengm

#ifdef USED_BOOST_DISABLE_ASSERTS
#undef BOOST_DISABLE_ASSERTS
#undef USED_BOOST_DISABLE_ASSERTS
#endif 


#endif // #ifndef OPENGM_MINSTCUTBOOST_HXX
