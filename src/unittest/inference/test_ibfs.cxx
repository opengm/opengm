#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>
#include <iostream>


#ifdef WITH_MAXFLOW_IBFS
#  include <opengm/inference/auxiliary/minstcutibfs.hxx>
#endif

int main() {
#ifdef WITH_MAXFLOW_IBFS
   typedef opengm::external::MinSTCutIBFS<int, int> MinStCutType;

   MinStCutType g(5,2+3*2);

   g.addEdge(0,2,1000); 
   g.addEdge(0,3,1); 
   g.addEdge(0,4,1);

   g.addEdge(2,1,5); 
   g.addEdge(3,1,5); 
   g.addEdge(4,1,5);

   g.addEdge(2,3,1);
   g.addEdge(3,4,1);


   std::vector<bool> x(5);
     g.calculateCut(x);

   for(size_t i=0; i<x.size(); ++i)
      std::cout <<x[i]<< " ";
   std::cout << std::endl;

#endif
   return 0;
}
