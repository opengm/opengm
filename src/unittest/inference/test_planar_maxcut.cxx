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
      std::vector<bool> cut;
      
      pmc.addEdge(0,1,-2.0); 
      pmc.addEdge(0,2, 2.0);
      pmc.addEdge(1,3, 2.0);
      pmc.addEdge(2,3,-2.0);
      pmc.calculateCut();
      pmc.getLabeling(l);
      pmc.getCut(cut);  

      for(size_t i=0; i<4; ++i)
         std::cout <<l[i]<<", ";
      std::cout<<std::endl;
      
      OPENGM_ASSERT(l[0]==l[1]);
      OPENGM_ASSERT(l[2]==l[3]);
      OPENGM_ASSERT(l[0]!=l[2]);
      OPENGM_ASSERT(l[1]!=l[3]); 

      OPENGM_ASSERT(cut[0]==0);
      OPENGM_ASSERT(cut[1]==1);
      OPENGM_ASSERT(cut[2]==1);
      OPENGM_ASSERT(cut[3]==0);
     

   }  
   {
      opengm::external::PlanarMaxCut pmc(5,12);
      std::vector<int> l;
      std::vector<bool> cut;
      
      pmc.addEdge(0,1, -2.0); 
      pmc.addEdge(0,2, -2.0);
      pmc.addEdge(1,3, -2.0);
      pmc.addEdge(2,3, -2.0);
      pmc.addEdge(3,4, -2.0);
      pmc.calculateCut();
      pmc.getLabeling(l);
      pmc.getCut(cut); 

      for(size_t i=0; i<cut.size(); ++i)
         std::cout <<cut[i]<<", ";
      std::cout<<std::endl;    
 
      for(size_t i=0; i<5; ++i)
         std::cout <<l[i]<<", ";
      std::cout<<std::endl;      

      OPENGM_ASSERT(l[0]==l[1]);
      OPENGM_ASSERT(l[1]==l[2]);
      OPENGM_ASSERT(l[2]==l[3]);
      OPENGM_ASSERT(l[3]==l[4]);

      OPENGM_ASSERT(cut[0]==0);
      OPENGM_ASSERT(cut[1]==0);
      OPENGM_ASSERT(cut[2]==0);
      OPENGM_ASSERT(cut[3]==0);
      OPENGM_ASSERT(cut[4]==0);

   }
   {
      opengm::external::PlanarMaxCut pmc(5,12);
      std::vector<int> l;
      std::vector<bool> cut;
      
      pmc.addEdge(0,1, -2.0); 
      pmc.addEdge(0,2, -2.0);
      pmc.addEdge(1,3, -2.0);
      pmc.addEdge(2,3, -2.0);
      pmc.addEdge(3,4, 2.0);
      pmc.calculateCut();
      pmc.getLabeling(l);
      pmc.getCut(cut); 

      for(size_t i=0; i<5; ++i)
         std::cout <<l[i]<<", ";
      std::cout<<std::endl;
      
      OPENGM_ASSERT(l[0]==l[1]);
      OPENGM_ASSERT(l[1]==l[2]);
      OPENGM_ASSERT(l[2]==l[3]);
      OPENGM_ASSERT(l[3]!=l[4]);

      OPENGM_ASSERT(cut[0]==0);
      OPENGM_ASSERT(cut[1]==0);
      OPENGM_ASSERT(cut[2]==0);
      OPENGM_ASSERT(cut[3]==0);
      OPENGM_ASSERT(cut[4]==1);

   }
   {
      opengm::external::PlanarMaxCut pmc(4,12);
      std::vector<int> l;
      std::vector<bool> cut;
      
      pmc.addEdge(0,1, 2.0); 
      pmc.addEdge(1,2, 2.0);
      pmc.addEdge(2,3, 2.0);
      pmc.addEdge(0,3, 2.0);
      pmc.calculateCut();
      pmc.getLabeling(l); 
      pmc.getCut(cut); 

      for(size_t i=0; i<4; ++i)
         std::cout <<l[i]<<", ";
      std::cout<<std::endl;
      
      OPENGM_ASSERT(l[0]!=l[1]);
      OPENGM_ASSERT(l[1]!=l[2]);
      OPENGM_ASSERT(l[2]!=l[3]);

      OPENGM_ASSERT(cut[0]==1);
      OPENGM_ASSERT(cut[1]==1);
      OPENGM_ASSERT(cut[2]==1);
      OPENGM_ASSERT(cut[3]==1);
   }



   std::cout << " OK!" << std::endl;
 
}
