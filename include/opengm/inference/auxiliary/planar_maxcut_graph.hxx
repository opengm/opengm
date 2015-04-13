#pragma once
#ifndef OPENGM_PLANAR_MAXCUT_GRAPH_HXX
#define OPENGM_PLANAR_MAXCUT_GRAPH_HXX


#include <queue> 
#include <cassert>
#include <iostream>
#include <list>
#include <stack>
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
   namespace external{
      namespace pmc{

         typedef double DataType;
         typedef size_t IDType;


//////////////////////////////////////////////////////
// Graph components
//////////////////////////////////////////////////////

         struct Node;
         struct Edge;
         struct Face;
         struct DualNode;
         struct DualEdge;


         struct Node
         {
            Node(IDType id_) : id(id_), weight(0.0), adj(0), label(-1) {};
            Node(IDType id_, DataType weight_) : id(id_), weight(weight_), adj(0), label(-1) {};

            IDType id;
            DataType weight;
            std::list<Edge*> adj; // List of adjacent edges
            int label;
         };

         struct Face
         {
            Face() : edges(0), dual_nodes(0) {};

            std::list<Edge*> edges; // List of edges surrounding the face
            std::list<DualNode*> dual_nodes; // Clique of dual nodes for this face
         };

         struct Edge
         {
            Edge(Node* tail_, Node* head_, DataType weight_) : tail(tail_), head(head_), weight(weight_), left_face(NULL), right_face(NULL), in_cut(false) {};

            Node* tail;
            Node* head;
            DataType weight;

            Face* left_face; // pointers to the left and right face as seen from
            Face* right_face; // formal head to tail

            bool in_cut; // true iff edge is in the cut set
         };

         struct DualNode
         {
            DualNode(IDType id_) : id(id_), adj(0) {};

            IDType id;
            std::list<DualEdge*> adj; // List of adjacent dual edges
         };

         struct DualEdge
         {
            DualEdge(DualNode* tail_, DualNode* head_, DataType weight_, Edge* original_cross_edge_) :
               tail(tail_), head(head_), weight(weight_), original_cross_edge(original_cross_edge_), in_matching(false) {};

            DualNode* tail;
            DualNode* head;
            DataType weight;

            Edge* original_cross_edge; // Pointer to the original edge crossed by this dual edge, NULL if its a face clique edge

            bool in_matching; // true iff dual edge is in the matching
         };


         Node* get_dest(Node* v, Edge* e)
         // Returns a pointer to the destination node of Edge e as seen from node v. NULL if e is not incident on v.
         {
            if(v == e->tail)
               return e->head;
            else if (v == e->head)
               return e->tail;
            else
               return NULL;
         }

         DualNode* get_dest(DualNode* v, DualEdge* e)
         // Returns a pointer to the destination node of Edge e as seen from node v. NULL if e is not incident on v.
         {
            if(v == e->tail)
               return e->head;
            else if (v == e->head)
               return e->tail;
            else
               return NULL;
         }

         Edge* get_following_edge(Edge* e, Node* v)
         // Returns edge that succeeds e in v's adjacency list (NULL if e is not incident on v).
         {
            std::list<Edge*>::iterator it = v->adj.begin();
            while( (*it!=e) && (it!=v->adj.end()) ) ++it;

            if(it==v->adj.end()) // e is not in v's adj list
            {
               return NULL;
            }
            else // e is in v's adj list
            {
               ++it; // Make one more step
               if(it==v->adj.end()) // e is the last element in v's adj list
                  return v->adj.front();
               else
                  return *(it);
            }
         }


//////////////////////////////////////////////////////
// Graph class definition
//////////////////////////////////////////////////////

         class Graph
         {
         public:
            Graph() : nodes(0), edges(0), faces(0), dual_nodes(0), dual_edges(0) {};
            ~Graph() {};

            size_t num_nodes() const { return nodes.size(); };
            size_t num_edges() const { return edges.size(); };
            size_t num_faces() const { return faces.size(); };
            size_t num_dual_nodes() const { return dual_nodes.size(); };
            size_t num_dual_edges() const { return dual_edges.size(); };

            void print();

            Node* add_node(IDType id_, DataType weight_);
            Edge* add_edge(Node* tail_, Node* head_, DataType weight_);

            DualNode* add_dual_node(IDType id_);
            DualEdge* add_dual_edge(DualNode* tail_, DualNode* head_, DataType weight_, Edge* original_cross_edge_);

            Edge* find_edge(Node* v1, Node* v2);

            void planarize();

            void construct_dual();

            void mcpm();
            void assign_labels();
    
            template<class VEC> void read_labels(VEC& sol) const;
            std::vector<int> read_labels();

         private:
            std::list<Node*> nodes;
            std::list<Edge*> edges;
            std::list<Face*> faces;

            std::list<DualNode*> dual_nodes;
            std::list<DualEdge*> dual_edges;
         };



//////////////////////////////////////////////////////
// Basic functionality
//////////////////////////////////////////////////////

         inline Node* Graph::add_node(IDType id_, DataType weight_)
         {
            // Create new node and add to graph's list of nodes
            Node* v = new Node(id_, weight_);
            nodes.push_back(v);

            // Return pointer to the new node
            return v;
         }

         inline Edge* Graph::add_edge(Node* tail_, Node* head_, DataType weight_)
         {
            // Create new edge and add to graph's list of edges
            Edge* e = new Edge(tail_, head_, weight_);
            edges.push_back(e);

            // Add edge to both nodes' adjacency lists
            tail_->adj.push_back(e);
            head_->adj.push_back(e);

            // Return pointer to the new edge
            return e;
         }

         inline DualNode* Graph::add_dual_node(IDType id_)
         {
            // Create new dual node and add to graph's list of dual nodes
            DualNode* v = new DualNode(id_);
            dual_nodes.push_back(v);

            // Return pointer to the new dual node
            return v;
         }

         inline DualEdge* Graph::add_dual_edge(DualNode* tail_, DualNode* head_, DataType weight_, Edge* original_cross_edge_)
         {
            // Create new dual edge and add to graph's list of dual edges
            DualEdge* e = new DualEdge(tail_, head_, weight_, original_cross_edge_);
            dual_edges.push_back(e);

            // Add dual edge to both dual nodes' adjacency lists
            tail_->adj.push_back(e);
            head_->adj.push_back(e);

            // Return pointer to the new edge
            return e;
         }

         inline Edge* Graph::find_edge(Node* v1, Node* v2)
         // Returns edge connecting nodes v1 and v2 (NULL if it does not exist).
         {
            // Search v1's adjacency list for an edge connecting it to v2. Return that edge.
            for(std::list<Edge*>::iterator it=v1->adj.begin(); it!=v1->adj.end(); ++it)
            {
               if( (*it)->tail==v2 || (*it)->head==v2 )
               {
                  return *it;
               }
            }

            // Return NULL if the loop did not find one.
            return NULL;
         }


         inline void Graph::print()
         // Simple output of graph for debugging
         {
            std::cout << "Graph with " << num_nodes() << " nodes and "
                      << num_edges() << " edges. It has " << num_faces() << " faces.\n";

            // Iterate through the nodes of the graph
            for(std::list<Node*>::iterator it = nodes.begin(); it != nodes.end(); ++it)
            {
               // Print current node's id and weight
               std::cout << (*it)->id << "\t[weight "<< (*it)->weight << ";\tlabel "
                         << (*it)->label << "]:\t";

               // For all edges in current node's adjacency list
               for(std::list<Edge*>::iterator jt = (*it)->adj.begin(); jt != (*it)->adj.end(); ++jt)
               {
                  // Get the destination of the current edge as seen from current node.
                  // Print destination id and weight of the edge
                  Node* v = get_dest(*it, *jt);
                  std::cout << v->id << " (" << (*jt)->weight << "), ";
               }
               std::cout << "\n";
            }
         }


//////////////////////////////////////////////////////
// Construction of planar embedding
//////////////////////////////////////////////////////

         inline void Graph::planarize()
         // Planarizes the graph. Sorts the adjacency lists of all nodes,
         // constructs faces and assigns the edges their faces
         {
            // Important:
            // The nodes need to have ids from 0 to num_nodes()-1
            // The graph needs to be biconnected
            // ToDo: Check for those conditions!

            //// Keep pointers to nodes in a vector
            std::vector<Node*> nodes_ptr (num_nodes());
            for(std::list<Node*>::iterator it=nodes.begin(); it!=nodes.end(); ++it)
            {
               nodes_ptr[(*it)->id] = *it;
            }

            //// Intiliaze graph in planarity code graph data structure.
            graphP g = gp_New();
            gp_InitGraph(g, num_nodes());
            for(std::list<Edge*>::iterator it=edges.begin(); it!=edges.end(); ++it)
            {
               Node* u = (*it)->tail;
               Node* v = (*it)->head;

               gp_AddEdge(g, u->id, 0, v->id, 0);
            }

            //// Invoke code that finds a planar embedding, i.e. sorts the adjacency lists
            if (gp_Embed(g, EMBEDFLAGS_PLANAR) == OK)
               gp_SortVertices(g);
            else
               std::cout << "Graph not planar\n"; // ToDo: Runtime error einfÃ¼gen!

            //// Repopulate edges in the embedding order
            for (size_t i = 0; i < g->N; ++i)
            {
               Node* u = nodes_ptr[i];

               size_t j = g->G[i].link[1];
               while (j >= g->N)
               {
                  OPENGM_ASSERT(i != g->G[j].v); // ToDo: Was machen asserts?
                  OPENGM_ASSERT(g->G[j].v < g->N);

                  // Find the node and the connecting edge
                  Node* v = nodes_ptr[g->G[j].v];
                  std::list<Edge*>::iterator it = u->adj.begin();
                  while( (*it)->tail!=v && (*it)->head!=v && it!=u->adj.end())
                     ++it;
                  Edge* e = *it;
                  OPENGM_ASSERT(it != u->adj.end());

                  // Remove the edge from its current position, and insert at the back
                  u->adj.erase(it);
                  u->adj.push_back(e);

                  j = g->G[j].link[1];
               }
            }

            //// Clear faces
            // Pop all elements from the graph's list of faces and delete them
            while(!faces.empty())
            {
               delete faces.back();
               faces.pop_back();
            }

            // Set the face pointer of all edges to NULL
            for(std::list<Edge*>::iterator it=edges.begin(); it!=edges.end(); ++it)
            {
               (*it)->left_face = NULL;
               (*it)->right_face = NULL;
            }

            //// Construct faces
            for(std::list<Edge*>::iterator it=edges.begin(); it!=edges.end(); ++it) // Loop over all edges
            {
               Edge* e = (*it); // Current edge

               // Check if the left face of e has already been dealt with.
               // If not, construct it!
               Face* f = e->left_face;
               if(f==NULL)
               {
                  f = new Face(); // Create new face object
                  faces.push_back(f); // Add it to the graph's list of faces
                  e->left_face = f; // Set it as left face of current edge
                  f->edges.push_back(e); // Add e to f's list of edges

                  // Follow the orbit in FORWARD direction (i.e. starting with e's head)
                  Node* v = e->head;
                  Edge* ee = get_following_edge(e, v);
                  while(ee!=e) // If ee==e, we went the full circle
                  {
                     // Set f as face of ee, left or right depends on the formal direction of ee
                     if(v==ee->tail)
                        ee->left_face = f;
                     if(v==ee->head)
                        ee->right_face = f;
                     f->edges.push_back(ee); // add e to f's list of edges
                     v = get_dest(v, ee);
                     ee = get_following_edge(ee,v);
                  }
               }

               // Check if the right_face of e has already been dealt with.
               // If not, construct it!
               f = e->right_face;
               if (f==NULL)
               {
                  f = new Face(); // Create new face object
                  faces.push_back(f); // Add it to the graph's list of faces
                  e->right_face = f; // Set it as left face of current edge e
                  f->edges.push_back(e); // Add e to f's list of edges

                  // Follow the orbit in BACKWARD direction (i.e. starting with e's tail)
                  Node* v = e->tail;
                  Edge* ee = get_following_edge(e, v);
                  while(ee!=e) // If ee==e, we went the full circle
                  {
                     // Set f as face of ee, left or right depends on the formal direction of ee
                     if(v==ee->tail)
                        ee->left_face = f;
                     if(v==ee->head)
                        ee->right_face = f;
                     f->edges.push_back(ee); // add e to f's list of edges
                     v = get_dest(v, ee);
                     ee = get_following_edge(ee,v);
                  }
               }
            }

            //// Check: Do all edges have a left and a right face?
            for(std::list<Edge*>::iterator it=edges.begin(); it!=edges.end(); ++it)
            {
               OPENGM_ASSERT((*it)->left_face != NULL); // ToDo: Was machen asserts?
               OPENGM_ASSERT((*it)->right_face != NULL);
            }

            //// Check if genus = 0, i.e graph is planar
            OPENGM_ASSERT(num_nodes()-num_edges()+num_faces() == 2);
            if(num_nodes()-num_edges()+num_faces() != 2)
               std::cout << "Genus not equal to zero\n"; // ToDO: Runtime error einfÃ¼gen!

            //// Delete planarity code graph
            gp_Free(&g);
         }


//////////////////////////////////////////////////////
// Construction of planar embedding
//////////////////////////////////////////////////////

         inline void Graph::construct_dual()
         // Constructs the expanded dual of the graph
         {
            // Important:
            // G needs to be planarized in the sense that faces have to
            // be constructed and edges have to have faces assigned correclty
            // (it is not necessary that the adjacency lists are sorted)

            size_t cnt_dual_nodes = 0;

            // Loop over all edges
            for(std::list<Edge*>::iterator it=edges.begin(); it!=edges.end(); ++it)
            {
               // For the current edge of G, add two dual nodes, one for each face
               DualNode* u = add_dual_node(cnt_dual_nodes);
               DualNode* v = add_dual_node(cnt_dual_nodes + 1);
               cnt_dual_nodes += 2;

               // "Integrate" u into the left face: Connect it to all dual nodes already in that face and add
               // it to the face's list of dual nodes
               Face* lf = (*it)->left_face;
               for(std::list<DualNode*>::iterator jt=lf->dual_nodes.begin(); jt!=lf->dual_nodes.end(); ++jt)
               {
                  add_dual_edge(u, *jt, 0.0, NULL);
               }
               lf->dual_nodes.push_back(u);

               // "Integrate" u into the left face: Connect it to all dual nodes already in that face and add
               // it to the face's list of dual nodes
               Face* rf = (*it)->right_face;
               for(std::list<DualNode*>::iterator jt=rf->dual_nodes.begin(); jt!=rf->dual_nodes.end(); ++jt)
               {
                  add_dual_edge(v, *jt, 0.0, NULL);
               }
               rf->dual_nodes.push_back(v);

               // Connect the two nodes by a dual edge with weight=negative of the crossed edge's weight
               add_dual_edge(u, v, (-1.)*(*it)->weight, *it);
            }
         }


//////////////////////////////////////////////////////
// Max-Cut via a perfect matching
//////////////////////////////////////////////////////

         inline void Graph::mcpm()
         // Perform perfect matching in the dual graph
         {
            // Important:
            // Dual graph has to be constructed first
            // Dual nodes need to have id's from 0 to num_nodes()-1

            //// Read dual graph into Blossom V code
            // Note: Blossom V AddEdge assigns automatically edgeids 0,...,num_dual_edges()-1
            PerfectMatching PM(num_dual_nodes(), num_dual_edges());
            PerfectMatching::Options options;
            options.verbose = false;
            for(std::list<DualEdge*>::iterator it=dual_edges.begin(); it!=dual_edges.end(); ++it)
            {
               DualEdge* e = *it;
               PM.AddEdge(e->tail->id, e->head->id, e->weight);
            }

            //// Invoke perfect matching solver

            PM.options = options;
            PM.Solve();

            //// Read out solution, one dual edge at a time
            size_t i=0;
            for(std::list<DualEdge*>::iterator it=dual_edges.begin(); it!=dual_edges.end(); ++it)
            {
               DualEdge* e = *it;

               if(PM.GetSolution(i)==1) // Check solution from blossom v code
               {
                  // If dual edge is in the matching, tell it. If it crosses an original edge,
                  // tell the original edge that it is in the cut.
                  e->in_matching = true;
                  if(e->original_cross_edge != NULL)
                  {
                     e->original_cross_edge->in_cut = true;
                  }
               }
               else
               {
                  // If dual edge is NOT in the matching, tell it. If it crosses an original edge,
                  // tell the original edge that it is NOT in the cut
                  e->in_matching = false;
                  if(e->original_cross_edge != NULL)
                  {
                     e->original_cross_edge->in_cut = false;
                  }
               }

               ++i;
            }
         }

         inline void Graph::assign_labels()
         // Given a cut (i.e. edges have bool in_cut assigned), assign a labeling to the nodes
         {
            // Important:
            // A cut has to be given first, i.e. edges need to have the boolean in_cut set correctly

            //// Set all labels to -1 (meaning unassigned)
            for(std::list<Node*>::iterator it = nodes.begin(); it!=nodes.end(); ++it)
            {
               (*it)->label = -1;
            }

            // For a start, put an arbitrary node on a stack and label it arbitrarily
            std::stack<Node*> s;
            Node* u = nodes.front();
            u->label = 0;
            s.push(u);


            while(!s.empty()) // As long as stack is not empty
            {
               // Take top element from stack
               u = s.top();
               s.pop();

               // Go through all incident edges
               for(std::list<Edge*>::iterator it = u->adj.begin(); it!=u->adj.end(); ++it)
               {
                  Edge* e = *it; // Edge and...
                  Node* v = get_dest(u, e); // its destination (i.e. the neighbor)

                  // If the neighbor has not yet been seen, assign the respective label
                  // and put it on the stack.
                  if(v->label==-1)
                  {
                     s.push(v);

                     if(e->in_cut)
                        v->label = !(u->label);
                     else
                        v->label = u->label;
                  }
               }
            }
         }

         template<class VEC>
         void Graph::read_labels(VEC& sol) const
         {
            sol.resize(num_nodes(), -1);
            for(std::list<Node*>::const_iterator it = nodes.begin(); it!=nodes.end(); ++it){
               sol[(*it)->id] = (*it)->label;
            }
            return;
         }


         std::vector<int> Graph::read_labels()
         {
            // Important: Nodes need to have id's from 0 to num_nodes()-1, corresponding to the
            // openGM variable id

            std::vector<int> sol(num_nodes(), -1);

            for(std::list<Node*>::iterator it = nodes.begin(); it!=nodes.end(); ++it)
            {
               sol[(*it)->id] = (*it)->label;
            }

            return sol;
         }

      }
   }
} 

#endif // #ifndef OPENGM_PLANAR_MAXCUT_GRAPH_HXX
