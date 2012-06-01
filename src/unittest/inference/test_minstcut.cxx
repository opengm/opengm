#include <vector>
#include <iostream>
#include <stdlib.h>
#include <limits>

#ifdef WITH_MAXFLOW 
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif
#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif

template<class ALG>
void test1(size_t id)
{
   srand(id);
   size_t numberOfNodes = 20;
   size_t numberOfEdges = 2 * numberOfNodes + 500;
   ALG alg(numberOfNodes + 2, numberOfEdges);

   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, rand() % 1000);
      alg.addEdge(i + 2, 1, rand() % 1000);
   }
   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, rand() % 1000);
      alg.addEdge(i + 2, 1, rand() % 1000);
   }
   for(size_t i = 0; i < numberOfEdges - 2 * numberOfNodes; ++i) {
      size_t node1 = rand() % numberOfNodes;
      size_t node2 = node1;
      while(node1 == node2) {
         node2 = rand() % numberOfNodes;
      }
      alg.addEdge(node1 + 2, node2 + 2, rand() % 1000);
   }
   std::vector<bool> cut;
   alg.calculateCut(cut);
}

template<class ALG>
void test2(size_t id)
{
   srand(id);
   size_t numberOfNodes = 20;
   size_t numberOfEdges = 2 * numberOfNodes + 500;
   ALG alg(numberOfNodes + 2, numberOfEdges);

   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, rand() % 1000);
      alg.addEdge(i + 2, 1, rand() % 1000);
   }
   for(size_t i = 0; i < numberOfEdges - 2 * numberOfNodes; ++i) {
      size_t node1 = rand() % numberOfNodes;
      size_t node2 = node1;
      while(node1 == node2) {
         node2 = rand() % numberOfNodes;
      }
      alg.addEdge(node1 + 2, node2 + 2, rand() % 1000);
   }
   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, rand() % 1000);
      alg.addEdge(i + 2, 1, rand() % 1000);
   }
   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, rand() % 1000);
      alg.addEdge(i + 2, 1, rand() % 1000);
   }
   std::vector<bool> cut;
   alg.calculateCut(cut);
}

template<class ALG>
void test3(size_t id)
{
   srand(id);
   size_t numberOfNodes = 20;
   size_t numberOfEdges = 2 * numberOfNodes + 1000;
   ALG alg(numberOfNodes + 2, numberOfEdges);

   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, (rand() % 100000) * 0.01);
      alg.addEdge(i + 2, 1, (rand() % 100000) * 0.01);
   }
   for(size_t i = 0; i < numberOfEdges - 2 * numberOfNodes; ++i) {
      size_t node1 = rand() % numberOfNodes;
      size_t node2 = node1;
      while(node1 == node2) {
         node2 = rand() % numberOfNodes;
      }
      alg.addEdge(node1 + 2, node2 + 2, (rand() % 100000) * 0.01);
   }
   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, (rand() % 100000) * 0.01);
      alg.addEdge(i + 2, 1, (rand() % 100000) * 0.01);
   }
   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, (rand() % 100000) * 0.01);
      alg.addEdge(i + 2, 1, (rand() % 100000) * 0.01);
   }
   for(size_t i = 0; i < 20; ++i) {
      alg.addEdge(0, i + 2, (rand() % 100000) * 0.01);
      alg.addEdge(i + 2, 1, (rand() % 100000) * 0.01);
   }
   std::vector<bool> cut;
   alg.calculateCut(cut);
}


template<class ALG>
void test4()
{
   size_t numberOfNodes = 10;
   size_t numberOfEdges = 10;
   ALG alg(numberOfNodes + 2, numberOfEdges);

   for(size_t i = 0; i < numberOfNodes; ++i) {
      alg.addEdge(0, i + 2, std::numeric_limits<typename ALG::ValueType>::infinity());
   }
   std::vector<bool> cut;
   alg.calculateCut(cut);
}

template<class ALG>
void test(size_t numTests)
{
   for(size_t id = 0; id < numTests; ++id)
      test1<ALG>(id);  
   std::cout << "*" << std::flush;

   for(size_t id = 0; id < numTests; ++id)
      test2<ALG>(id); 
   std::cout << "*" << std::flush;

   for(size_t id = 0; id < numTests; ++id)
      test3<ALG>(id);
   std::cout << "*" << std::flush;
   test4<ALG>();
   std::cout << "*" << std::flush;
}

int main()
{
   std::cout << "MinStCut Test ... "<<std::endl;
#ifdef WITH_MAXFLOW
   {
      std::cout << "  * Test Kolomogorov ... " << std::flush;
      typedef opengm::external::MinSTCutKolmogorov<size_t, float> ALG;
      test<ALG>(5);
      std::cout << " OK!" << std::endl;
   }
#endif 
#ifdef WITH_BOOST 
   {
      std::cout << "  * Test BOOST-Push Relabel ... " << std::flush;
      typedef opengm::MinSTCutBoost<size_t, float, opengm::PUSH_RELABEL> ALG;
      test<ALG>(5);
      std::cout << " OK!" << std::endl;
   }
  
   {
      std::cout << "  * Test BOOST-Edmonds Karp ... " << std::flush;
      typedef opengm::MinSTCutBoost<size_t, float, opengm::EDMONDS_KARP> ALG;
      test<ALG>(5);
      std::cout << " OK!" << std::endl;
   }
   {
      std::cout << "  * Test BOOST-Kolmogorov ... " << std::flush; 
      typedef opengm::MinSTCutBoost<size_t, float, opengm::KOLMOGOROV> ALG;
      test<ALG>(5);
      std::cout << " OK!" << std::endl;
   }
#endif
   std::cout<< "done!"<<std::endl;
}
