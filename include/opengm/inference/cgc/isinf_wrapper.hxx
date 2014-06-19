#ifndef OPENGM_HMC_ISINF_WRAPPER_HXX
#define OPENGM_HMC_ISINF_WRAPPER_HXX

#include <planarity/graph.h>
#include <blossom.hpp>


#undef OPENGM_CHECK_OP
#define OPENGM_CHECK_OP(A,OP,B,TXT) 


#undef OPENGM_CHECK
#define OPENGM_CHECK(B,TXT) 


namespace Isinf{

    ///////////////////////////////////////
    // mexGraph is a class derived from the Graph class in graph.hpp
    // This class contains some functions not present in Graph, useful
    // for the mex wrappers.
    ///////////////////////////////////////
    class MyBlossomGraph : public BlossomGraph {
        public:
        MyBlossomGraph(){}
        //MyBlossomGraph(EmbeddingType xvalues,int n_nodes,int max_degree); // build graph from embedding matrix

        void resizeNodes(const int numNodes){
            this->nodes.resize(numNodes);
        }
        void resizeEdges(const int numEdges){
            this->edges.resize(numEdges);
        }

        template<class COST>
        void my_mmc_nodestate(const COST & cost,size_t size)
        {
            //OPENGM_CHECK_OP(cost.size(),<=,nedges, " ");
            undone();
            size_t i;

            // sort edges by cost
            std::vector<std::pair<double,size_t> > idx;
            for (i = 0; i < size ; ++i)
                idx.push_back(pair<double,size_t>(-cost[i], i));
            for (; i < nedges; ++i)
                idx.push_back(pair<double,size_t>(0.0, i));
            std::sort(idx.begin(), idx.end());

            // build spanning tree (Kruskal's algorithm)
            Disunion du(nodes.size());
            for (i = 0; i < nedges; ++i)
            {
                Edgelet *e = edges[idx[i].second];
                if (!du.join(e->node, e->other->node))
                {
                    e->done = true;
                    e->other->done = true;
                }
                // else assert(e->index < cost.size());
            }

            // set node states along spanning tree
            bfs(nodes.size() - 1, agree);
        }
        
        void addEdgeAndNode(const int nodeIndex,const int edgeIndex){
            Edgelet *  e = new Edgelet;
            //printf("> node edgeIndex=%d,edge j=%d\n",edgeIndex,j);

            e->index = edgeIndex;
            if (edgeIndex >= edges.size()){
            OPENGM_CHECK(false, " ");
                edges.resize(edgeIndex+1);
            }
            if (edges[edgeIndex]) {
                e->other = edges[edgeIndex];
                assert(!edges[edgeIndex]->other);
                edges[edgeIndex]->other = e;
                ++nedges;
            }
            else {
                edges[edgeIndex] = e;
            }

            e->node = nodeIndex;
            if (nodeIndex >= nodes.size()) {
            OPENGM_CHECK(false, " ");
                nodes.resize(nodeIndex+1);
            }
            if (nodes[nodeIndex].edges) {
                e->prev = nodes[nodeIndex].edges->prev;
                e->prev->next = e;
                e->next = nodes[nodeIndex].edges;
                nodes[nodeIndex].edges->prev = e;
            }
            else {
                nodes[nodeIndex].edges = e->prev = e->next = e;
            }
        }
    }; //class MyBlossomGraph
} //namespace IsInf

#ifdef MIN
#undef MIN
#endif


#ifdef MAX
#undef MAX
#endif

#endif /* OPENGM_HMC_ISINF_WRAPPER_HXX */
