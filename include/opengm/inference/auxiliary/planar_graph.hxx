#pragma once
#ifndef OPENGM_PLANAR_GRAPH_HXX
#define OPENGM_PLANAR_GRAPH_HXX

#include <iostream>
#include <vector>
#include <stack>
#include <list>
#include "opengm/opengm.hxx"

//TODO: Fix include path
#include <planarity.src-patched/graph.h>
#include <planarity.src-patched/listcoll.h>
#include <planarity.src-patched/stack.h>
#include <planarity.src-patched/appconst.h>

#include <blossom5.src-patched/PerfectMatching.h>
#include <blossom5.src-patched/PMimplementation.h>
#include <blossom5.src-patched/MinCost/MinCost.h>

namespace opengm {
   namespace external {
      namespace planargraph {

         typedef double DataType;

//////////////////////////////////////////////////////
// PlanarGraph components
//////////////////////////////////////////////////////

         struct Node
         {
            Node() : weight(0.0), adj(0) {};

            // Node weight
            // Unused (0.0) for dual edges
            // do zrobienia: make node class a template class and only use weight in reweighted_planar_max_cut
            DataType weight;
    
            // List of indices of the dual edges
            std::list<size_t> adj;
         };


         struct Edge
         {
            Edge()
               : tail(0), head(0), weight(0.0), left_face(-1), right_face(-1) {};

            // Indices of tail and head node
            size_t tail;
            size_t head;
    
            // Edge weight
            DataType weight;

            // Indices of left and right face (as seen from head to tail).
            // -1 if unassigned
            int left_face;
            int right_face;
         };


         struct Face
         {
            Face() : edges(0), dual_nodes(0) {};

            // List of edges surrounding the face. Is in sorted order (s.t. the
            // edges form an orbit) after calling planarize()
            std::list<size_t> edges;
    
            // List of the dual nodes_ (forming a clique) belonging to  this face
            std::list<size_t> dual_nodes;
         };


//////////////////////////////////////////////////////
// PlanarGraph class definition
//////////////////////////////////////////////////////

         class PlanarGraph
         {
         public:
            PlanarGraph();
            PlanarGraph(size_t n, bool debug);
            ~PlanarGraph();

            size_t num_nodes() const { return nodes_.size(); };
            size_t num_edges() const { return edges.size(); };
            size_t num_faces() const { return faces.size(); };

    
            size_t add_node();
            long int find_edge(size_t u, size_t v) const;
            size_t add_edge(size_t u, size_t v, DataType w);
            void add_edge_weight(size_t e, DataType w);
    
            void print();

            void planarize();
            void construct_dual();
    
            void get_labeling(std::vector<int> & x) const;

            void calculate_maxcut();
            std::vector<bool> get_cut() const; 
            std::vector<int> get_labeling_from_cut(const std::vector<bool>& cut) const;
            double cost_of_cut(const std::vector<int>& x) const;
            double cost_of_cut() const;

         protected:
            long int get_dest(size_t v, size_t e) const;
            long int get_following_edge(size_t v, size_t e) const;
    
            void clear_faces();    

            size_t compute_dual_num_edges() const;

            std::vector<Node> nodes_;
            std::vector<Edge> edges;
            std::vector<Face> faces;
    
            //std::unique_ptr<PerfectMatching> Dual_;
            PerfectMatching* Dual_;

            bool debug_;
    
         };


//////////////////////////////////////////////////////
// Basic functionality
//////////////////////////////////////////////////////


         //

         PlanarGraph::PlanarGraph()
            : nodes_(0), edges (0), faces(0), Dual_(NULL), debug_(false)
              //  : nodes_(0), edges (0), faces(0), Dual_(nullptr), debug_(debug)
         {
            nodes_.reserve(0);
            edges.reserve(0);
         }

         PlanarGraph::PlanarGraph(size_t n, bool debug = false)
            : nodes_(0), edges (0), faces(0), Dual_(NULL), debug_(debug)
              //  : nodes_(0), edges (0), faces(0), Dual_(nullptr), debug_(debug)
         {
            nodes_.reserve(n);
            edges.reserve(3*n);
         }

         PlanarGraph::~PlanarGraph(){
            if(Dual_ != NULL)
               delete Dual_;
         }

         // Adds a new node with weight w to the graph
         size_t PlanarGraph::add_node()
         {
            // Create new node object (in place) and set its weight
            size_t v = nodes_.size();
            nodes_.resize(v+1);

            // Return index of the new node
            return v;
         }


         // Adds new edge with weight w between nodes_ u and v
         size_t PlanarGraph::add_edge(size_t u, size_t v, DataType w)
         {
            // Check that u and v are valid node indices
            OPENGM_ASSERT(u >= 0 && u<nodes_.size());
            OPENGM_ASSERT(v >= 0 && v<nodes_.size());
	
            // Create new edge object (in place) and set properties
            size_t e = edges.size();
            edges.resize(e+1);
            edges[e].tail = u;
            edges[e].head = v;
            edges[e].weight = w;

            // Add edge index to both nodes_' adjacency lists
            nodes_[u].adj.push_back(e);
            nodes_[v].adj.push_back(e);

            // Return index of the new edge
            return e;
         }


         // Adds weight w to the existing edge e
         void PlanarGraph::add_edge_weight(size_t e, DataType w)
         {
            // Check the e is a valid edge index
            OPENGM_ASSERT(e >= 0 && e < edges.size());
	
            edges[e].weight = w;
         }


         // Returns index of the edge connecting nodes_ u and v
         // -1 if such an edge does not exist
         long int PlanarGraph::find_edge(size_t u, size_t v) const
         {
            // Search u's adjacency list for an edge connecting it to v.
            for(size_t e=0; e<nodes_[u].adj.size(); ++e)
            {
               if( edges[e].tail==v || edges[e].head==v )
               {
                  return e;
               }
            }

            // Return -1 if the loop did not find one.
            return -1;
         }


         // Returns index of the destination node of Edge e as seen from node v
         // -1 if e is not incident on v
         long int PlanarGraph::get_dest(size_t v, size_t e) const
         {
            if(v == edges[e].tail)
               return edges[e].head;
            else if (v == edges[e].head)
               return edges[e].tail;
            else
               return -1;
         }
	

         // Simple command line output of the graph for debugging
         void PlanarGraph::print()
         {
            if(debug_)
               std::cout << "PlanarGraph with " << num_nodes() << " nodes_, "
                         << num_edges() << " edges and " << num_faces() << " faces.\n";

            for(size_t u = 0; u<nodes_.size(); ++u)
            {
               // Print current node's id and weight
               if(debug_)
                  std::cout << u << "\t[" << nodes_[u].weight << "]:\t";

               // For all edges in current node's adjacency list
               for(std::list<size_t>::iterator it = nodes_[u].adj.begin();
                   it != nodes_[u].adj.end(); ++it)
               {
                  // Get the destination of the current edge
                  // Print destination id and weight of the edge
                  size_t v = get_dest(u, *it);
                  if(debug_)
                     std::cout << v << " (" << edges[*it].weight << "), ";
               }
               if(debug_)
                  std::cout << "\n";
            }
         }


//////////////////////////////////////////////////////
// Construction of planar embedding
//////////////////////////////////////////////////////

         // Returns the index of the edge that succeeds e in v's adjacency list
         // -1 if e is not incident on v
         long int PlanarGraph::get_following_edge(size_t v, size_t e) const
         {
            // Iterate to e in v's adjacency list
            std::list<size_t>::const_iterator it = nodes_[v].adj.begin();
            while((*it != e) && (it != nodes_[v].adj.end()))
               ++it;

            if(it==nodes_[v].adj.end()) // e is not in v's adj list
            {
               return -1;
            }
            else  // e is in v's adj list
            {
               ++it; // Make one more step
               if(it==nodes_[v].adj.end()) // e is the last element in v's adj list
                  return nodes_[v].adj.front();
               else
                  return *it;
            }
         }


         void PlanarGraph::clear_faces()
         {
            // Pop all elements from the graph's list of faces
            while(!faces.empty())
            {
               faces.pop_back();
            }

            // Set the face indices of all edges to -1
            for(std::vector<Edge>::iterator it = edges.begin(); it != edges.end(); ++it)
            {
               it->left_face = -1;
               it->right_face = -1;
            }
         }

         size_t PlanarGraph::compute_dual_num_edges() const
         {
            // number of dual edges if the number of original edges ( = cross edges ) + clique edges in each face
            size_t dual_num_edges = num_edges();
            for(size_t f=0; f<faces.size(); ++f) {
               const size_t face_size = faces[f].edges.size();
               dual_num_edges += (face_size * (face_size - 1))/2;
            }
            return dual_num_edges;
         }


         // Planarizes the graph, i.e.
         // - Sorts the adjacency lists of all nodes_ to form a rotation system
         // - Constructs faces and sets face/edge relations (see definitions of
         //   struct Edge and struct Face)
         void PlanarGraph::planarize()
         {
            // ToDo
            // Assert that the graph is biconnected
            // Reserve space for faces (how much?)

            //// Copy graph in planarity code graph data structure.
            graphP g = gp_New();
            gp_InitGraph(g, num_nodes());
            for(std::vector<Edge>::iterator it=edges.begin(); it!=edges.end(); ++it)
            {
               gp_AddEdge(g, it->tail, 0, it->head, 0);
            }

            // Invoke code that sorts the adjacency lists
            if (gp_Embed(g, EMBEDFLAGS_PLANAR) == OK) {
               gp_SortVertices(g);
            } else {
               throw("PlanarGraph not planar\n");
            }

            //// Repopulate edges in the embedding order
            for (size_t i = 0; i < g->N; ++i)
            {
               size_t u = i;
		
               size_t j = g->G[i].link[1];
               while (j >= g->N)
               {
                  OPENGM_ASSERT(i != g->G[j].v); // What does this OPENGM_ASSERT do?
                  OPENGM_ASSERT(g->G[j].v < g->N);
            
                  size_t v = g->G[j].v;

                  // Find the edge connecting u and v
                  std::list<size_t>::iterator it = nodes_[u].adj.begin();
                  while(edges[*it].tail != v && edges[*it].head != v && it != nodes_[u].adj.end())
                     ++it;
                  size_t e = *it;
                  OPENGM_ASSERT(it != nodes_[i].adj.end());

                  // Remove the edge from its current position, and insert at the back
                  nodes_[u].adj.erase(it);
                  nodes_[u].adj.push_back(e);

                  j = g->G[j].link[1];
               }
            }

            //// Clear faces
            clear_faces();


            //// Construct faces
            // do zrobienia: code for following the orbit starting from left and right face is mostly duplication: clean up!
            for(size_t e = 0; e < edges.size(); ++e) // Loop over all edges
            {
               // Check if the right face of e has already been dealt with.
               // If not, construct it!
               
               //enum class face_type {left,right};
               typedef int face_type;
               const face_type face_type_left  = 1;
               const face_type face_type_right = 2;

        
               if(edges[e].right_face == -1)
               {
                  // Create new face object
                  const size_t f = faces.size();
                  faces.resize(f+1);

                  // Assign e <-> f
                  faces[f].edges.push_back(e);
                  edges[e].right_face = f;

                  // Follow the orbit in FORWARD direction (i.e. starting with e's head)
                  size_t v = edges[e].head; // Next node
                  size_t ee = e;
                  //size_t ee = get_following_edge(v, e); // Next edge
                  face_type ee_face;
                  do {
                     // Get next node and edge
                     ee = get_following_edge(v, ee);
                     v = get_dest(v, ee);

                     // Set f as face of ee, left or right depends on the formal direction of ee
                     if(v==edges[ee].tail) {
                        edges[ee].left_face = f;
                        ee_face = face_type_left;
                     }
                     if(v==edges[ee].head) {
                        edges[ee].right_face = f;
                        ee_face = face_type_right;
                     }
                     faces[f].edges.push_back(ee); // a face can have the same edge in the left and right. do zeobienia: this must be reflected in the faces data structure as well.


                  } while(! (ee == e && ee_face == face_type_right) ); // If ee==e and we are on the same side again, we went the full circle
               }

               // Check if the left face of e has already been dealt with.
               // If not, construct it!
               // to do: remove duplicate code by switching left for right and vice versa
               if(edges[e].left_face == -1)
               {
                  // Create new face object
                  const size_t f = faces.size();
                  faces.resize(f+1);

                  // Assign e <-> f
                  faces[f].edges.push_back(e);
                  edges[e].left_face = f;

                  // Follow the orbit in BACKWARD direction (i.e. starting with e's tail)
                  size_t v = edges[e].tail; // Next node
                  size_t ee = e;
                  //size_t ee = get_following_edge(v, e); // Next edge
                  face_type ee_face;
                  do {
                     // Get next node and edge
                     ee = get_following_edge(v, ee);
                     v = get_dest(v, ee);

                     // Set f as face of ee, left or right depends on the formal direction of ee
                     if(v==edges[ee].tail) {
                        edges[ee].left_face = f;
                        ee_face = face_type_left;
                     }
                     if(v==edges[ee].head) {
                        edges[ee].right_face = f;
                        ee_face = face_type_right;
                     }
                     faces[f].edges.push_back(ee); // a face can have the same edge in the left and right. do zeobienia: this must be reflected in the faces data structure as well.


                  } while(! (ee == e && ee_face == face_type_left) ); // If ee==e and we are on the same side again, we went the full circle
               }
            }
    
            //// Checks and clean-up
            // Check: Do all edges have a left and a right face?
            for(std::vector<Edge>::iterator it=edges.begin(); it!=edges.end(); ++it)
            {
               OPENGM_ASSERT((*it).left_face != -1);
               OPENGM_ASSERT((*it).right_face != -1);
            }

            // Check if genus = 0, i.e graph is planar
            OPENGM_ASSERT(num_nodes()-num_edges()+num_faces() == 2);

            // Delete planarity code graph
            gp_Free(&g);
         }


//////////////////////////////////////////////////////
// Construction of dual graph
//////////////////////////////////////////////////////

         // Constructs the expanded dual of the graph. The primal graph needs to
         // be planarized before.
         void PlanarGraph::construct_dual()
         {    
            // Allocate dual graph in PerfectMatching data structure
            // Todo: Reasonable number of max dual edges
            Dual_ = new PerfectMatching(2*num_edges(), compute_dual_num_edges());
            // Dual_ = std::unique_ptr<PerfectMatching>(new PerfectMatching(2*num_edges(), compute_dual_num_edges()));
            PerfectMatching::Options Dual_options;
            Dual_options.verbose = false;
            Dual_->options = Dual_options;
    
            // insert all cross edges corresponding to the original edges
            // note: cross edges directly correspond to the original edges
            size_t counter = 0;
            for(size_t e = 0; e < edges.size(); ++e)
            {		
               // For the current edge of G, add dual nodes_ for its two faces
               const size_t u = counter;
               const size_t v = counter + 1;
               counter += 2;

               // Add the dual cross edge of e, connecting u and v
               // Weight is the negative of e's weight
               Dual_->AddEdge(u, v, edges[e].weight);
            }

            // insert clique edges connecting all dual nodes inside a face
            counter = 0;
            for(size_t e = 0; e < edges.size(); ++e)
            {		
               size_t u = counter;
               size_t v = counter + 1;
               counter += 2;

               // "Integrate" u into the left face of e
               size_t f = edges[e].left_face;
               for(std::list<size_t>::iterator it = faces[f].dual_nodes.begin(); it != faces[f].dual_nodes.end(); ++it)
               {
                  Dual_->AddEdge(u, *it, 0.0);
               }
               faces[f].dual_nodes.push_back(u);

               // "Integrate" v into the right face
               f = edges[e].right_face;
               for(std::list<size_t>::iterator it = faces[f].dual_nodes.begin(); it != faces[f].dual_nodes.end(); ++it)
               {
                  Dual_->AddEdge(v, *it, 0.0);
               }
               faces[f].dual_nodes.push_back(v);
            }
         }

         double PlanarGraph::cost_of_cut() const
         {
            double cost = 0.0;
            for(size_t e = 0; e < num_edges(); ++e)
            {
               if(Dual_->GetSolution(e) == 0)
                  cost += edges[e].weight;
            }

            return cost;
         }
	
         double PlanarGraph::cost_of_cut(const std::vector<int>& x) const
         {
            // do zrobienia: check if x is a cut
            double cost = 0.0;
            for(size_t e = 0; e < num_edges(); ++e)
            {
               if(x[e] == 1)
                  cost += edges[e].weight;
            }

            return cost;
         }

         void PlanarGraph::calculate_maxcut()
         {
            //OPENGM_ASSERT(Dual_ != nullptr); 
            OPENGM_ASSERT(Dual_ != NULL);
            Dual_->Solve();
         }

         std::vector<bool> PlanarGraph::get_cut() const
         {
            std::vector<bool> cut(num_edges(),false);
            for(size_t e = 0; e < num_edges(); ++e)
            {
               if(Dual_->GetSolution(e) == 0) {
                  cut[e] = true;
               } else if(Dual_->GetSolution(e) == 1) {
                  cut[e] = false;
               } else {
                  throw std::logic_error("Perfect matching solver did not succeed");
               }
            }
            return cut;	
         }

         // Reads the labeling defined by a given cut into the vector x. Does
         // not output the labels for the unary nodes_. A cut has to be given
         std::vector<int> PlanarGraph::get_labeling_from_cut(const std::vector<bool>& cut) const
         {
            OPENGM_ASSERT(cut.size() == edges.size());
            // Make labeling size num_nodes(), i.e. including unary nodes_. Set all
            // labels to -1 (meaning unassigned)
            std::vector<int> labeling(num_nodes());
            for(size_t v = 0; v < num_nodes(); ++v)
               labeling[v] = -1;

            std::stack<size_t> s;
            size_t visited_nodes=0;
            for(size_t startnode=0; startnode< num_nodes(); ++startnode){
               if(visited_nodes==num_nodes())
                  break;
               if( labeling[startnode]!= -1)
                  continue;
               labeling[startnode] = 0;
               s.push(startnode);
               
               while(!s.empty()) // As long as stack is not empty
               {
                  // Take top element from stack
                  size_t u = s.top();
                  s.pop();
                  ++visited_nodes;
               
                  // Go through all incident edges
                  for(std::list<size_t>::const_iterator it = nodes_[u].adj.begin(); it != nodes_[u].adj.end(); ++it)
                  {
                     size_t e = *it; // Edge and...
                     size_t v = get_dest(u, e); // its destination (i.e. the neighbor)
                  
                     // If the neighbor has not yet been seen, put it on the
                     // stack and assign the respective label
                     if(labeling[v] == -1)
                     {
                        s.push(v);
                     
                        if(cut[e])
                           labeling[v] = (labeling[u] + 1) % 2; // mapping 0->1 and 1->0
                        else
                           labeling[v] = labeling[u];
                     }
                  
                     // Check for inconsistent cut when encountering a node again
                     if(labeling[v] != -1) // node already seen
                     {
                        if(cut[e]) {
                           OPENGM_ASSERT(labeling[v] + labeling[u] == 1);
                        } else {
                           OPENGM_ASSERT(labeling[v] == labeling[u]);
                        }
                     }
                  }
               }
            }
            
            for(size_t v=0; v<labeling.size(); ++v) {
               OPENGM_ASSERT(labeling[v] == 0 || labeling[v] == 1);
            }

            return labeling;
         }


         void PlanarGraph::get_labeling(std::vector<int> & x) const
         {
            std::vector<bool> cut = get_cut();
            x = get_labeling_from_cut(cut);
         }

      } //namespace planargraph
   } // namespace external
} // namespace opengm

#endif // OPENGM_PLANAR_GRAPH_HXX
