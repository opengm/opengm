#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include "opengm/unittests/test.hxx"
#include "opengm/utilities/memoryusage.hxx"

int main(int argc, char** argv) {
   opengm::MemoryUsage mem;
   std::cout << "Initial physical memory usage : "<<mem.usedPhysicalMem() << " kB"<<std::endl;
   std::cout << "Initial virtual memory usage : "<<mem.usedVirtualMem()  << " kB"<<std::endl;

   std::cout << "--> allocate  doubles (80.000k)" <<std::endl;
   double *a = new double[10000*1024];
   for(size_t i=0; i<10000*1024; ++i) *(a+i) = i;

   std::cout << "New physical memory usage : "<<mem.usedPhysicalMem() << " kB" <<std::endl;
   std::cout << "New virtual memory usage : "<<mem.usedVirtualMem()  << " kB"<<std::endl;
   OPENGM_TEST(mem.usedPhysicalMem()>80000);
   OPENGM_TEST(mem.usedVirtualMem()>80000);
   OPENGM_TEST(mem.usedPhysicalMem()<100000);
   OPENGM_TEST(mem.usedVirtualMem()<100000);


   std::cout << "--> allocate  doubles (80.000k)" <<std::endl;
   double *b = new double[10000*1024]; 
   for(size_t i=0; i<10000*1024; ++i) *(b+i) = i;
 
   std::cout << "New physical memory usage : "<<mem.usedPhysicalMem()  << " kB"<<std::endl;
   std::cout << "New virtual memory usage : "<<mem.usedVirtualMem()  << " kB"<<std::endl;
   OPENGM_TEST(mem.usedPhysicalMem()>160000);
   OPENGM_TEST(mem.usedVirtualMem()>160000);
   OPENGM_TEST(mem.usedPhysicalMem()<180000);
   OPENGM_TEST(mem.usedVirtualMem()<180000);
 
   std::cout << "<-- free first doubles" <<std::endl;
   delete a;
  
   std::cout << "New physical memory usage : "<<mem.usedPhysicalMem() << " kB" <<std::endl;
   std::cout << "New virtual memory usage : "<<mem.usedVirtualMem() << " kB" <<std::endl;

   std::cout << "<-- free second doubles" <<std::endl;
   delete b;  

   std::cout << "New physical memory usage : "<<mem.usedPhysicalMem() << " kB" <<std::endl;
   std::cout << "New virtual memory usage : "<<mem.usedVirtualMem()  << " kB"<<std::endl; 
   
   std::cout << "Max physical memory usage : "<<mem.usedPhysicalMemMax() << " kB" <<std::endl;
   std::cout << "Max virtual memory usage : "<<mem.usedVirtualMemMax() << " kB" <<std::endl; 


   std::cout << "System memory : "<<mem.systemMem()/1024/1024 << " GB" <<std::endl;
   std::cout << "used system memory : "<<mem.usedSystemMem()/1024/1024 << " GB" <<std::endl;
   std::cout << "free system memory : "<<mem.freeSystemMem()/1024/1024 << " GB" <<std::endl;
   
    return (EXIT_SUCCESS);
}


