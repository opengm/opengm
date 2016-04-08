#include <stdexcept>
#include <iostream>
#include "opengm/utilities/partitions.hxx"
#include <opengm/unittests/test.hxx>
#include <stdint.h>


void testPartition()
{
   opengm::Partitions<uint64_t,uint64_t> P;
   P.resize(3); 
   std::cout <<"Test BellNumber: ..."<<std::flush;
   OPENGM_TEST(P.BellNumber(2)==2);
   std::cout <<"  OK!"<<std::endl;

   std::cout <<"Test getPartition: ..."<<std::flush;
   OPENGM_TEST(P.getPartition(0)==0);
   OPENGM_TEST(P.getPartition(1)==1);
   OPENGM_TEST(P.getPartition(2)==2);
   OPENGM_TEST(P.getPartition(3)==4);
   OPENGM_TEST(P.getPartition(4)==7); 
   //OPENGM_TEST(P.getPartition(5)>8);

   std::vector<uint64_t> ltest(3);
   for(uint64_t i=0; i<5; ++i){
      //const size_t el = P.getPartition(i);
      P.getPartition(i,ltest);
      //std::cout << P.label2Index(ltest) <<" ("<< el <<") "<<ltest[0]<<ltest[1]<<ltest[2]<<std::endl;
      OPENGM_TEST(P.label2Index(ltest)==i);
   }
   std::cout <<"  OK!"<<std::endl;



   std::cout <<"Test label2Index: ..."<<std::flush;
   P.resize(3);
   std::vector<uint64_t> l(3,0);
   OPENGM_TEST(P.label2Index(l)==4);
   l[0]=1;
   OPENGM_TEST(P.label2Index(l)==3);
   l[1]=2;
   OPENGM_TEST(P.label2Index(l)==0); 

   for(uint64_t b=4;b<12;++b){
      P.resize(b);
      std::vector<uint64_t> l(b,0);
      OPENGM_TEST(P.label2Index(l)==P.BellNumber(b)-1);
   }
   std::cout <<"  OK!"<<std::endl;
   
}



int main(int argc, char** argv) {
   testPartition();
   return (EXIT_SUCCESS);
}


