#include <vector>
#include <iostream>
#include <stdlib.h>
#include <limits>

//#ifdef WITH_PLANARMAXCUT 
#include <opengm/inference/auxiliary/planar_maxcut.hxx>
//#endif



int main()
{
   std::cout << "Planar MaxCut Test ... "<<std::endl;


   {
      opengm::external::PlanarMaxCut pmc(4,12);
      std::vector<int> l;
      
      pmc.addEdge(0,1,-2.0); 
      pmc.addEdge(0,2, 2.0);
      pmc.addEdge(1,3, 2.0);
      pmc.addEdge(2,3,-2.0);
      pmc.calculateCut(l);
      
      OPENGM_ASSERT(l[0]==l[1]);
      OPENGM_ASSERT(l[2]==l[3]);
      OPENGM_ASSERT(l[0]!=l[2]);
      OPENGM_ASSERT(l[1]!=l[3]);

      for(size_t i=0; i<4; ++i)
         std::cout <<l[i]<<", ";
      std::cout<<std::endl;
   }  

   {
      opengm::external::PlanarMaxCut pmc(5,12);
      std::vector<int> l;
      
      pmc.addEdge(0,1, -2.0); 
      pmc.addEdge(0,2, -2.0);
      pmc.addEdge(1,3, -2.0);
      pmc.addEdge(2,3, -2.0);
      pmc.addEdge(3,4, -2.0);
      pmc.calculateCut(l);
      
      OPENGM_ASSERT(l[0]==l[1]);
      OPENGM_ASSERT(l[1]==l[2]);
      OPENGM_ASSERT(l[2]==l[3]);
      OPENGM_ASSERT(l[3]==l[4]);

      for(size_t i=0; i<5; ++i)
         std::cout <<l[i]<<", ";
      std::cout<<std::endl;
   }
   {
      opengm::external::PlanarMaxCut pmc(5,12);
      std::vector<int> l;
      
      pmc.addEdge(0,1, -2.0); 
      pmc.addEdge(0,2, -2.0);
      pmc.addEdge(1,3, -2.0);
      pmc.addEdge(2,3, -2.0);
      pmc.addEdge(3,4, 2.0);
      pmc.calculateCut(l);
      
      OPENGM_ASSERT(l[0]==l[1]);
      OPENGM_ASSERT(l[1]==l[2]);
      OPENGM_ASSERT(l[2]==l[3]);
      OPENGM_ASSERT(l[3]!=l[4]);

      for(size_t i=0; i<5; ++i)
         std::cout <<l[i]<<", ";
      std::cout<<std::endl;
   }


   std::cout << " OK!" << std::endl;
 
}
