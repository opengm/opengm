#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <opengm/utilities/meminfo.hxx>
#include <opengm/unittests/test.hxx>

int main(int argc, char** argv) {
   sys::MemoryInfo memInfo;
   std::cout << "Initial physical memory usage : "<<memInfo.usedPhysicalMem() << " kB"<<std::endl;
   std::cout << "Initial virtual memory usage : "<<memInfo.usedVirtualMem()  << " kB"<<std::endl;

   std::cout << "--> allocate  doubles (80.000k)" <<std::endl;
   double *a = new double[10000*1024];
   for(size_t i=0; i<10000*1024; ++i) a[i] = i;

   std::cout << "New physical memory usage : "<<memInfo.usedPhysicalMem() << " kB" <<std::endl;
   std::cout << "New virtual memory usage : "<<memInfo.usedVirtualMem()  << " kB"<<std::endl;
   OPENGM_TEST(memInfo.usedPhysicalMem()>80000);
   OPENGM_TEST(memInfo.usedVirtualMem()>80000);
   OPENGM_TEST(memInfo.usedPhysicalMem()<100000);
//   assert(memInfo.usedVirtualMem()<100000);



   std::cout << "--> allocate  doubles (80.000k)" <<std::endl;
   double *b = new double[10000*1024]; 
   for(size_t i=0; i<10000*1024; ++i) *(b+i) = i;
 
   std::cout << "New physical memory usage : "<<memInfo.usedPhysicalMem()  << " kB"<<std::endl;
   std::cout << "New virtual memory usage : "<<memInfo.usedVirtualMem()  << " kB"<<std::endl;
   OPENGM_TEST(memInfo.usedPhysicalMem()>160000);
   OPENGM_TEST(memInfo.usedVirtualMem()>160000);
   OPENGM_TEST(memInfo.usedPhysicalMem()<180000);
//   assert(memInfo.usedVirtualMem()<180000);
 
   std::cout << "<-- free first doubles" <<std::endl; 
   std::cout << "Ones use the mem, otherwise clang optimize it away ;-) " << a[10000*1024-1]<<std::endl;
   delete a;
  
   std::cout << "New physical memory usage : "<<memInfo.usedPhysicalMem() << " kB" <<std::endl;
   std::cout << "New virtual memory usage : "<<memInfo.usedVirtualMem() << " kB" <<std::endl;

   std::cout << "<-- free second doubles" <<std::endl;
   std::cout << "Ones use the mem, otherwise clang optimize it away ;-) " << b[10000*1024-1]<<std::endl;
   delete b;  

   std::cout << "New physical memory usage : "<<memInfo.usedPhysicalMem() << " kB" <<std::endl;
   std::cout << "New virtual memory usage : "<<memInfo.usedVirtualMem()  << " kB"<<std::endl; 
   
   std::cout << "Max physical memory usage : "<<memInfo.usedPhysicalMemMax() << " kB" <<std::endl;
   std::cout << "Max virtual memory usage : "<<memInfo.usedVirtualMemMax() << " kB" <<std::endl; 


   std::cout << "System memory      : "<<memInfo.systemMem()/1024/1024 << " GB" <<std::endl;
   std::cout << "Used system memory : "<<memInfo.usedSystemMem()/1024/1024 << " GB" <<std::endl;
   std::cout << "Free system memory : "<<memInfo.freeSystemMem()/1024/1024 << " GB" <<std::endl;
   
   return (EXIT_SUCCESS);
}


