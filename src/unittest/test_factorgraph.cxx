#include <vector>
#include <algorithm>

#include "opengm/unittests/test.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/adder.hxx"

typedef opengm::GraphicalModel<int, opengm::Adder> GraphicalModel;
typedef opengm::ExplicitFunction<int> Function;
typedef GraphicalModel::FunctionIdentifier FID;
typedef GraphicalModel::IndexType IndexType;
typedef opengm::FactorGraph<GraphicalModel,IndexType> FactorGraph;


void testNumberOfFactors() {
   size_t numbersOfStates[] = {2, 2, 2, 2};
   Function f1(numbersOfStates, numbersOfStates + 2);
   GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
   FID fid1 = gm.addFunction(f1);
   size_t vi[] = {0,1,2,3,0,3};
   gm.addFactor(fid1, vi+0, vi+2);
   gm.addFactor(fid1, vi+1, vi+3);
   gm.addFactor(fid1, vi+2, vi+4);
   gm.addFactor(fid1, vi+4, vi+6);
   OPENGM_ASSERT(gm.numberOfFactors(0)==2);
}

void testIsAcyclic()
{
   size_t numbersOfStates[] = {2, 2, 2, 2, 2};
   Function f1(numbersOfStates, numbersOfStates + 1);
   Function f2(numbersOfStates, numbersOfStates + 2);
   Function f3(numbersOfStates, numbersOfStates + 3);
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      OPENGM_TEST(gm.isAcyclic());
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      OPENGM_TEST(gm.isAcyclic());
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      OPENGM_TEST(!gm.isAcyclic());
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid2 = gm.addFunction(f2);
      FID fid3 = gm.addFunction(f3);
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 1, 2}; gm.addFactor(fid3, vi, vi + 3); }
      OPENGM_TEST(!gm.isAcyclic());
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid3 = gm.addFunction(f3);
      { size_t vi[] = {0, 1, 2}; gm.addFactor(fid3, vi, vi + 3); }
      { size_t vi[] = {1, 2, 3}; gm.addFactor(fid3, vi, vi + 3); }
      OPENGM_TEST(!gm.isAcyclic());
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      OPENGM_TEST(!gm.isAcyclic());
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 3}; gm.addFactor(fid2, vi, vi + 2); }
      OPENGM_TEST(!gm.isAcyclic());
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 5));
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 4}; gm.addFactor(fid2, vi, vi + 2); }
      OPENGM_TEST(gm.isAcyclic());
   }
}

bool eachComponentHasOneRepresentative(const marray::Vector<size_t>& representatives,
      const std::vector<std::vector<size_t> > components) {
   if(components.size() != representatives.size()) {
      return false;
   }
   marray::Vector<size_t> countRepresentativseOfEachComponent(components.size(), 0);
   for(size_t i = 0; i < representatives.size(); i++) {
      for(size_t j = 0; j < components.size(); j++) {
         if(std::find(components[j].begin(), components[j].end(), representatives[i]) !=  components[j].end()) {
            countRepresentativseOfEachComponent[j]++;
            break;
         }
      }
   }
   for(size_t i = 0; i < countRepresentativseOfEachComponent.size(); i++) {
      if(countRepresentativseOfEachComponent[i] != 1) {
         return false;
      }
   }
   return true;
}

void testIsConnected()
{
   size_t numbersOfStates[] = {2, 2, 2, 2, 2};
   Function f1(numbersOfStates, numbersOfStates + 1);
   Function f2(numbersOfStates, numbersOfStates + 2);
   Function f3(numbersOfStates, numbersOfStates + 3);
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 1));
      FID fid1 = gm.addFunction(f1);
      //FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      std::vector<std::vector<size_t> > components(1);
      size_t componentIDs0[] = {0};
      components[0] = std::vector<size_t>(componentIDs0, componentIDs0 + 1);
      marray::Vector<size_t> representatives;
      OPENGM_TEST(gm.isConnected(representatives));
      OPENGM_TEST(eachComponentHasOneRepresentative(representatives, components));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      //FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {1}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {2}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {3}; gm.addFactor(fid1, vi, vi + 1); }
      std::vector<std::vector<size_t> > components(4);
      size_t componentIDs0[] = {0};
      size_t componentIDs1[] = {1};
      size_t componentIDs2[] = {2};
      size_t componentIDs3[] = {3};
      components[0] = std::vector<size_t>(componentIDs0, componentIDs0 + 1);
      components[1] = std::vector<size_t>(componentIDs1, componentIDs1 + 1);
      components[2] = std::vector<size_t>(componentIDs2, componentIDs2 + 1);
      components[3] = std::vector<size_t>(componentIDs3, componentIDs3 + 1);
      marray::Vector<size_t> representatives;
      OPENGM_TEST(!gm.isConnected(representatives));
      OPENGM_TEST(eachComponentHasOneRepresentative(representatives, components));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {3}; gm.addFactor(fid1, vi, vi + 1); }
      std::vector<std::vector<size_t> > components(3);
      size_t componentIDs0[] = {0};
      size_t componentIDs1[] = {1, 2};
      size_t componentIDs2[] = {3};
      components[0] = std::vector<size_t>(componentIDs0, componentIDs0 + 1);
      components[1] = std::vector<size_t>(componentIDs1, componentIDs1 + 2);
      components[2] = std::vector<size_t>(componentIDs2, componentIDs2 + 1);
      marray::Vector<size_t> representatives;
      OPENGM_TEST(!gm.isConnected(representatives));
      OPENGM_TEST(eachComponentHasOneRepresentative(representatives, components));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {3}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 3}; gm.addFactor(fid2, vi, vi + 2); }
      std::vector<std::vector<size_t> > components(2);
      size_t componentIDs0[] = {0, 3};
      size_t componentIDs1[] = {1, 2};
      components[0] = std::vector<size_t>(componentIDs0, componentIDs0 + 2);
      components[1] = std::vector<size_t>(componentIDs1, componentIDs1 + 2);
      marray::Vector<size_t> representatives;
      OPENGM_TEST(!gm.isConnected(representatives));
      OPENGM_TEST(eachComponentHasOneRepresentative(representatives, components));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3}; gm.addFactor(fid1, vi, vi + 1); }
      std::vector<std::vector<size_t> > components(1);
      size_t componentIDs0[] = {0, 1, 2, 3};
      components[0] = std::vector<size_t>(componentIDs0, componentIDs0 + 1);
      marray::Vector<size_t> representatives;
      OPENGM_TEST(gm.isConnected(representatives));
      OPENGM_TEST(eachComponentHasOneRepresentative(representatives, components));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3}; gm.addFactor(fid1, vi, vi + 1); }
      std::vector<std::vector<size_t> > components(1);
      size_t componentIDs0[] = {0, 1, 2, 3};
      components[0] = std::vector<size_t>(componentIDs0, componentIDs0 + 1);
      marray::Vector<size_t> representatives;
      OPENGM_TEST(gm.isConnected(representatives));
      OPENGM_TEST(eachComponentHasOneRepresentative(representatives, components));
   }
   {
      GraphicalModel gm;
      marray::Vector<size_t> representatives;
      OPENGM_TEST(!gm.isConnected(representatives));
   }
}

void testIsChain()
{
   size_t numbersOfStates[] = {2, 2, 2, 2, 2};
   Function f1(numbersOfStates, numbersOfStates + 1);
   Function f2(numbersOfStates, numbersOfStates + 2);
   Function f3(numbersOfStates, numbersOfStates + 3);
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 1));
      FID fid1 = gm.addFunction(f1);
      //FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      size_t expectedchainIDs[] = {0};
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(gm.isChain(chainIDs));
      OPENGM_TEST(chainIDs.size() == 1);
      OPENGM_TEST_EQUAL_SEQUENCE(chainIDs.begin(), chainIDs.end(), expectedchainIDs);
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(!gm.isChain(chainIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      FID fid3 = gm.addFunction(f3);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 1, 2}; gm.addFactor(fid3, vi, vi + 3); }
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(!gm.isChain(chainIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      size_t expectedchainIDs[] = {0, 1, 2, 3};
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(gm.isChain(chainIDs));
      OPENGM_TEST(chainIDs.size() == 4);
      OPENGM_TEST_EQUAL_SEQUENCE(chainIDs.begin(), chainIDs.end(), expectedchainIDs);
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      //FID fid3 = gm.addFunction(f3);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {1}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {3}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      size_t expectedchainIDs[] = {0, 1, 2, 3};
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(gm.isChain(chainIDs));
      OPENGM_TEST(chainIDs.size() == 4);
      OPENGM_TEST_EQUAL_SEQUENCE(chainIDs.begin(), chainIDs.end(), expectedchainIDs);
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {1, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      size_t expectedchainIDs[] = {1, 3, 0, 2};
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(gm.isChain(chainIDs));
      OPENGM_TEST(chainIDs.size() == 4);
      OPENGM_TEST_EQUAL_SEQUENCE(chainIDs.begin(), chainIDs.end(), expectedchainIDs);
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 3}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(!gm.isChain(chainIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(!gm.isChain(chainIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 5));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 4}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(!gm.isChain(chainIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 5));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 4}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Vector<size_t> chainIDs;
      OPENGM_TEST(!gm.isChain(chainIDs));
   }
}

inline marray::Matrix<size_t> rotate(const marray::Matrix<size_t>& A) {
   marray::Matrix<size_t> rotated(A.shape(1), A.shape(0));
   for(size_t i = 0; i < A.shape(0); i++) {
      for(size_t j = 0; j < A.shape(1); j++) {
         rotated(j, (A.shape(1) - 1) - i) = A(i,j);
      }
   }
   return rotated;
}

bool equalMatrices(const marray::Matrix<size_t>& A, const marray::Matrix<size_t>& B) {
   if(!(A.shape(0) == B.shape(0) && A.shape(1) == B.shape(1))) {
      return false;
   }
   for(size_t i = 0; i < A.shape(0); i++) {
      for(size_t j = 0; j < A.shape(1); j++) {
         if(A(i,j) != B(i,j)) {
            return false;
         }
      }
   }
   return true;
}

bool equalMatricesExceptRotation(const marray::Matrix<size_t>& A, const marray::Matrix<size_t>& B) {
   if(!((A.shape(0) == B.shape(0) && A.shape(1) == B.shape(1)) || (A.shape(0) == B.shape(1) && A.shape(1) == B.shape(0)))) {
      // dimension mismatch
      return false;
   }

   marray::Matrix<size_t> BRotated(B);
   marray::View<size_t> view(B);
   view.transpose();
   marray::Matrix<size_t> BTransposedRotated(view);
   for(size_t i = 0; i < 4; i++) {
      // try all four rotations
      if(equalMatrices(A, BRotated)) {
         return true;
      } else if(equalMatrices(A, BTransposedRotated)) {
         return true;
      }
      BRotated = rotate(BRotated);
      marray::Matrix<size_t> temp;
      temp = rotate(BTransposedRotated);
      BTransposedRotated = temp;
   }
   return false;
}

void testIsGrid()
{
   size_t numbersOfStates[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
   Function f1(numbersOfStates, numbersOfStates + 1);
   Function f2(numbersOfStates, numbersOfStates + 2);
   Function f3(numbersOfStates, numbersOfStates + 3);
   {
      GraphicalModel gm;
      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(!gm.isGrid(gridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 1));
      FID fid1 = gm.addFunction(f1);
      //FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(gm.isGrid(gridIDs));
      marray::Matrix<size_t> expectedGridIDs(1, 1);
      expectedGridIDs(0, 0) = 0;
      OPENGM_TEST(equalMatricesExceptRotation(gridIDs, expectedGridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(gm.isGrid(gridIDs));
      marray::Matrix<size_t> expectedGridIDs(1, 4);
      expectedGridIDs(0, 0) = 0; expectedGridIDs(0, 1) = 1; expectedGridIDs(0, 2) = 2; expectedGridIDs(0, 3) = 3;
      OPENGM_TEST(equalMatricesExceptRotation(gridIDs, expectedGridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 4));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(gm.isGrid(gridIDs));
      marray::Matrix<size_t> expectedGridIDs(2, 2);
      expectedGridIDs(0, 0) = 0; expectedGridIDs(0, 1) = 1;
      expectedGridIDs(1, 0) = 2; expectedGridIDs(1, 1) = 3;
      OPENGM_TEST(equalMatricesExceptRotation(gridIDs, expectedGridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 6));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 5}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(gm.isGrid(gridIDs));
      marray::Matrix<size_t> expectedGridIDs(3, 2);
      expectedGridIDs(0, 0) = 0; expectedGridIDs(0, 1) = 1;
      expectedGridIDs(1, 0) = 2; expectedGridIDs(1, 1) = 3;
      expectedGridIDs(2, 0) = 4; expectedGridIDs(2, 1) = 5;
      OPENGM_TEST(equalMatricesExceptRotation(gridIDs, expectedGridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 6));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 5}; gm.addFactor(fid2, vi, vi + 2); }
      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(!gm.isGrid(gridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 9));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 8}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {7, 8}; gm.addFactor(fid2, vi, vi + 2); }

      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(gm.isGrid(gridIDs));
      marray::Matrix<size_t> expectedGridIDs(3, 3);
      expectedGridIDs(0, 0) = 0; expectedGridIDs(0, 1) = 1; expectedGridIDs(0, 2) = 2;
      expectedGridIDs(1, 0) = 3; expectedGridIDs(1, 1) = 4; expectedGridIDs(1, 2) = 5;
      expectedGridIDs(2, 0) = 6; expectedGridIDs(2, 1) = 7; expectedGridIDs(2, 2) = 8;
      OPENGM_TEST(equalMatricesExceptRotation(gridIDs, expectedGridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 16));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 8}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {7, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 12}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {11, 15}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {12, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {13, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {14, 15}; gm.addFactor(fid2, vi, vi + 2); }

      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(gm.isGrid(gridIDs));
      marray::Matrix<size_t> expectedGridIDs(4, 4);
      expectedGridIDs(0, 0) =  0; expectedGridIDs(0, 1) =  1; expectedGridIDs(0, 2) =  2; expectedGridIDs(0, 3) =  3;
      expectedGridIDs(1, 0) =  4; expectedGridIDs(1, 1) =  5; expectedGridIDs(1, 2) =  6; expectedGridIDs(1, 3) =  7;
      expectedGridIDs(2, 0) =  8; expectedGridIDs(2, 1) =  9; expectedGridIDs(2, 2) = 10; expectedGridIDs(2, 3) = 11;
      expectedGridIDs(3, 0) = 12; expectedGridIDs(3, 1) = 13; expectedGridIDs(3, 2) = 14; expectedGridIDs(3, 3) = 15;
      OPENGM_TEST(equalMatricesExceptRotation(gridIDs, expectedGridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 16));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 8}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {7, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 12}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {11, 15}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {12, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {13, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {14, 15}; gm.addFactor(fid2, vi, vi + 2); }

      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(!gm.isGrid(gridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 16));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 8}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {7, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 12}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {11, 15}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {12, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {13, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {14, 15}; gm.addFactor(fid2, vi, vi + 2); }

      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(!gm.isGrid(gridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 16));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 8}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {7, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 12}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {11, 15}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {12, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {13, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {14, 15}; gm.addFactor(fid2, vi, vi + 2); }

      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(!gm.isGrid(gridIDs));
   }
   {
      GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates, numbersOfStates + 16));
      FID fid1 = gm.addFunction(f1);
      FID fid2 = gm.addFunction(f2);
      { size_t vi[] = {0}; gm.addFactor(fid1, vi, vi + 1); }
      { size_t vi[] = {0, 1}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {0, 4}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 2}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {1, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 3}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {2, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {3, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 5}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {4, 8}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 6}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {5, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 7}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {6, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {7, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 9}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {8, 12}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 10}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {9, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 11}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {10, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {11, 15}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {12, 13}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {13, 14}; gm.addFactor(fid2, vi, vi + 2); }
      { size_t vi[] = {14, 15}; gm.addFactor(fid2, vi, vi + 2); }

      marray::Matrix<size_t> gridIDs;
      OPENGM_TEST(!gm.isGrid(gridIDs));
   }
}

int main() {
   // build graphical model for testing
   std::vector<size_t> numbersOfStates(4, 2);
   GraphicalModel gm(opengm::DiscreteSpace<size_t,size_t>(numbersOfStates.begin(), numbersOfStates.end()));

   // add dummy functions
   size_t shape[] = {2, 2, 2, 2};
   Function firstOrderFunction(shape, shape + 1);
   Function secondOrderFunction(shape, shape + 2);
   Function fourthOrderFunction(shape, shape + 4);
   FID f1 = gm.addFunction(firstOrderFunction);
   FID f2 = gm.addFunction(secondOrderFunction);
   FID f4 = gm.addFunction(fourthOrderFunction);

   // add factors
   for(size_t j=0; j<4; ++j) {
      size_t n = gm.addFactor(f1, &j, &j+1);
      OPENGM_TEST(n == j);
   }
   OPENGM_TEST(gm.numberOfFactors(1) == 1);
   {
      size_t vi[] = {0, 1};
      size_t n = gm.addFactor(f2, vi, vi+2);
      OPENGM_TEST(n == 4);
   }
   OPENGM_TEST(gm.numberOfFactors(1) == 2);
   {
      size_t vi[] = {0, 2};
      size_t n = gm.addFactor(f2, vi, vi+2);
      OPENGM_TEST(n == 5);
   }
   {
      size_t vi[] = {1, 3};
      size_t n = gm.addFactor(f2, vi, vi+2);
      OPENGM_TEST(n == 6);
   }
   OPENGM_TEST(gm.numberOfFactors(1) == 3);
   {
      size_t vi[] = {2, 3};
      size_t n = gm.addFactor(f2, vi, vi+2);
      OPENGM_TEST(n == 7);
   }
   {
      size_t vi[] = {0, 1, 2, 3};
      size_t n = gm.addFactor(f4, vi, vi+4);
      OPENGM_TEST(n == 8);
   }
   OPENGM_TEST(gm.numberOfFactors(1) == 4);
   FactorGraph& fg = gm; // cast to parent

   OPENGM_TEST(fg.numberOfVariables() == 4);

   OPENGM_TEST(fg.numberOfVariables(0) == 1);
   OPENGM_TEST(fg.numberOfVariables(1) == 1);
   OPENGM_TEST(fg.numberOfVariables(2) == 1);
   OPENGM_TEST(fg.numberOfVariables(3) == 1);
   OPENGM_TEST(fg.numberOfVariables(4) == 2);
   OPENGM_TEST(fg.numberOfVariables(5) == 2);
   OPENGM_TEST(fg.numberOfVariables(6) == 2);
   OPENGM_TEST(fg.numberOfVariables(7) == 2);
   OPENGM_TEST(fg.numberOfVariables(8) == 4);

   OPENGM_TEST(fg.numberOfFactors() == 9);
   OPENGM_TEST(gm.numberOfFactors(0) == 4);
   OPENGM_TEST(fg.numberOfFactors(0) == 4);
   OPENGM_TEST(gm.numberOfFactors(1) == 4);
   OPENGM_TEST(fg.numberOfFactors(1) == 4);
   OPENGM_TEST(gm.numberOfFactors(2) == 4);
   OPENGM_TEST(fg.numberOfFactors(2) == 4);
   OPENGM_TEST(gm.numberOfFactors(3) == 4);
   OPENGM_TEST(fg.numberOfFactors(3) == 4);

   OPENGM_TEST(fg.variableOfFactor(0, 0) == 0);
   OPENGM_TEST(fg.variableOfFactor(1, 0) == 1);
   OPENGM_TEST(fg.variableOfFactor(2, 0) == 2);
   OPENGM_TEST(fg.variableOfFactor(3, 0) == 3);
   OPENGM_TEST(fg.variableOfFactor(4, 0) == 0);
   OPENGM_TEST(fg.variableOfFactor(4, 1) == 1);
   OPENGM_TEST(fg.variableOfFactor(5, 0) == 0);
   OPENGM_TEST(fg.variableOfFactor(5, 1) == 2);
   OPENGM_TEST(fg.variableOfFactor(6, 0) == 1);
   OPENGM_TEST(fg.variableOfFactor(6, 1) == 3);
   OPENGM_TEST(fg.variableOfFactor(7, 0) == 2);
   OPENGM_TEST(fg.variableOfFactor(7, 1) == 3);
   OPENGM_TEST(fg.variableOfFactor(8, 0) == 0);
   OPENGM_TEST(fg.variableOfFactor(8, 1) == 1);
   OPENGM_TEST(fg.variableOfFactor(8, 2) == 2);
   OPENGM_TEST(fg.variableOfFactor(8, 3) == 3);

   OPENGM_TEST(fg.factorOfVariable(0, 0) == 0);
   OPENGM_TEST(fg.factorOfVariable(0, 1) == 4);
   OPENGM_TEST(fg.factorOfVariable(0, 2) == 5);
   OPENGM_TEST(fg.factorOfVariable(0, 3) == 8);
   OPENGM_TEST(fg.factorOfVariable(1, 0) == 1);
   OPENGM_TEST(fg.factorOfVariable(1, 1) == 4);
   OPENGM_TEST(fg.factorOfVariable(1, 2) == 6);
   OPENGM_TEST(fg.factorOfVariable(1, 3) == 8);
   OPENGM_TEST(fg.factorOfVariable(2, 0) == 2);
   OPENGM_TEST(fg.factorOfVariable(2, 1) == 5);
   OPENGM_TEST(fg.factorOfVariable(2, 2) == 7);
   OPENGM_TEST(fg.factorOfVariable(2, 3) == 8);
   OPENGM_TEST(fg.factorOfVariable(3, 0) == 3);
   OPENGM_TEST(fg.factorOfVariable(3, 1) == 6);
   OPENGM_TEST(fg.factorOfVariable(3, 2) == 7);
   OPENGM_TEST(fg.factorOfVariable(3, 3) == 8);

   // test
   // - FactorGraph::ConstVariableIterator
   // - FactorGraph::variablesOfFactorBegin(factor)
   // - FactorGraph::variablesOfFactorEnd(factor)
   for(size_t factor=0; factor<fg.numberOfFactors(); ++factor) {
      OPENGM_TEST(std::distance(fg.variablesOfFactorBegin(factor),
                                fg.variablesOfFactorEnd(factor))
                  == fg.numberOfVariables(factor));
      FactorGraph::ConstVariableIterator it = fg.variablesOfFactorBegin(factor);
      for(size_t j=0; j<fg.numberOfVariables(factor); ++j, ++it) {
         OPENGM_TEST(*it == fg.variableOfFactor(factor, j));
      }
      OPENGM_TEST(it == fg.variablesOfFactorEnd(factor));
   }

   // test
   // - FactorGraph::ConstFactorIterator
   // - FactorGraph::factorsOfVariableBegin
   // - FactorGraph::factorsOfVariableEnd
   for(size_t variable=0; variable<fg.numberOfVariables(); ++variable) {
      OPENGM_TEST(std::distance(fg.factorsOfVariableBegin(variable),
                                fg.factorsOfVariableEnd(variable))
                  == fg.numberOfFactors(variable));
      FactorGraph::ConstFactorIterator it = fg.factorsOfVariableBegin(variable);
      for(size_t j=0; j<fg.numberOfFactors(variable); ++j, ++it) {
         OPENGM_TEST(*it == fg.factorOfVariable(variable, j));
      }
      OPENGM_TEST(it == fg.factorsOfVariableEnd(variable));
   }

   // test FactorGraph::factorVariableConnection
   OPENGM_TEST(fg.factorVariableConnection(0, 0));
   OPENGM_TEST(!fg.factorVariableConnection(1, 0));
   OPENGM_TEST(!fg.factorVariableConnection(2, 0));
   OPENGM_TEST(!fg.factorVariableConnection(3, 0));
   OPENGM_TEST(fg.factorVariableConnection(4, 0));
   OPENGM_TEST(fg.factorVariableConnection(5, 0));
   OPENGM_TEST(!fg.factorVariableConnection(6, 0));
   OPENGM_TEST(!fg.factorVariableConnection(7, 0));
   OPENGM_TEST(fg.factorVariableConnection(8, 0));

   OPENGM_TEST(!fg.factorVariableConnection(0, 1));
   OPENGM_TEST(fg.factorVariableConnection(1, 1));
   OPENGM_TEST(!fg.factorVariableConnection(2, 1));
   OPENGM_TEST(!fg.factorVariableConnection(3, 1));
   OPENGM_TEST(fg.factorVariableConnection(4, 1));
   OPENGM_TEST(!fg.factorVariableConnection(5, 1));
   OPENGM_TEST(fg.factorVariableConnection(6, 1));
   OPENGM_TEST(!fg.factorVariableConnection(7, 1));
   OPENGM_TEST(fg.factorVariableConnection(8, 1));

   OPENGM_TEST(!fg.factorVariableConnection(0, 2));
   OPENGM_TEST(!fg.factorVariableConnection(1, 2));
   OPENGM_TEST(fg.factorVariableConnection(2, 2));
   OPENGM_TEST(!fg.factorVariableConnection(3, 2));
   OPENGM_TEST(!fg.factorVariableConnection(4, 2));
   OPENGM_TEST(fg.factorVariableConnection(5, 2));
   OPENGM_TEST(!fg.factorVariableConnection(6, 2));
   OPENGM_TEST(fg.factorVariableConnection(7, 2));
   OPENGM_TEST(fg.factorVariableConnection(8, 2));

   OPENGM_TEST(!fg.factorVariableConnection(0, 3));
   OPENGM_TEST(!fg.factorVariableConnection(1, 3));
   OPENGM_TEST(!fg.factorVariableConnection(2, 3));
   OPENGM_TEST(fg.factorVariableConnection(3, 3));
   OPENGM_TEST(!fg.factorVariableConnection(4, 3));
   OPENGM_TEST(!fg.factorVariableConnection(5, 3));
   OPENGM_TEST(fg.factorVariableConnection(6, 3));
   OPENGM_TEST(fg.factorVariableConnection(7, 3));
   OPENGM_TEST(fg.factorVariableConnection(8, 3));

   // test FactorGraph::variableFactorConnection
   OPENGM_TEST(fg.variableFactorConnection(0, 0));
   OPENGM_TEST(!fg.variableFactorConnection(0, 1));
   OPENGM_TEST(!fg.variableFactorConnection(0, 2));
   OPENGM_TEST(!fg.variableFactorConnection(0, 3));
   OPENGM_TEST(fg.variableFactorConnection(0, 4));
   OPENGM_TEST(fg.variableFactorConnection(0, 5));
   OPENGM_TEST(!fg.variableFactorConnection(0, 6));
   OPENGM_TEST(!fg.variableFactorConnection(0, 7));
   OPENGM_TEST(fg.variableFactorConnection(0, 8));

   OPENGM_TEST(!fg.variableFactorConnection(1, 0));
   OPENGM_TEST(fg.variableFactorConnection(1, 1));
   OPENGM_TEST(!fg.variableFactorConnection(1, 2));
   OPENGM_TEST(!fg.variableFactorConnection(1, 3));
   OPENGM_TEST(fg.variableFactorConnection(1, 4));
   OPENGM_TEST(!fg.variableFactorConnection(1, 5));
   OPENGM_TEST(fg.variableFactorConnection(1, 6));
   OPENGM_TEST(!fg.variableFactorConnection(1, 7));
   OPENGM_TEST(fg.variableFactorConnection(1, 8));

   OPENGM_TEST(!fg.variableFactorConnection(2, 0));
   OPENGM_TEST(!fg.variableFactorConnection(2, 1));
   OPENGM_TEST(fg.variableFactorConnection(2, 2));
   OPENGM_TEST(!fg.variableFactorConnection(2, 3));
   OPENGM_TEST(!fg.variableFactorConnection(2, 4));
   OPENGM_TEST(fg.variableFactorConnection(2, 5));
   OPENGM_TEST(!fg.variableFactorConnection(2, 6));
   OPENGM_TEST(fg.variableFactorConnection(2, 7));
   OPENGM_TEST(fg.variableFactorConnection(2, 8));

   OPENGM_TEST(!fg.variableFactorConnection(3, 0));
   OPENGM_TEST(!fg.variableFactorConnection(3, 1));
   OPENGM_TEST(!fg.variableFactorConnection(3, 2));
   OPENGM_TEST(fg.variableFactorConnection(3, 3));
   OPENGM_TEST(!fg.variableFactorConnection(3, 4));
   OPENGM_TEST(!fg.variableFactorConnection(3, 5));
   OPENGM_TEST(fg.variableFactorConnection(3, 6));
   OPENGM_TEST(fg.variableFactorConnection(3, 7));
   OPENGM_TEST(fg.variableFactorConnection(3, 8));

   // test FactorGraph::factorFactorConnection
   OPENGM_TEST(!fg.factorFactorConnection(0, 0));
   OPENGM_TEST(!fg.factorFactorConnection(0, 1));
   OPENGM_TEST(!fg.factorFactorConnection(0, 2));
   OPENGM_TEST(!fg.factorFactorConnection(0, 3));
   OPENGM_TEST(fg.factorFactorConnection(0, 4));
   OPENGM_TEST(fg.factorFactorConnection(0, 5));
   OPENGM_TEST(!fg.factorFactorConnection(0, 6));
   OPENGM_TEST(!fg.factorFactorConnection(0, 7));
   OPENGM_TEST(fg.factorFactorConnection(0, 8));

   OPENGM_TEST(!fg.factorFactorConnection(1, 0));
   OPENGM_TEST(!fg.factorFactorConnection(1, 1));
   OPENGM_TEST(!fg.factorFactorConnection(1, 2));
   OPENGM_TEST(!fg.factorFactorConnection(1, 3));
   OPENGM_TEST(fg.factorFactorConnection(1, 4));
   OPENGM_TEST(!fg.factorFactorConnection(1, 5));
   OPENGM_TEST(fg.factorFactorConnection(1, 6));
   OPENGM_TEST(!fg.factorFactorConnection(1, 7));
   OPENGM_TEST(fg.factorFactorConnection(1, 8));

   OPENGM_TEST(!fg.factorFactorConnection(2, 0));
   OPENGM_TEST(!fg.factorFactorConnection(2, 1));
   OPENGM_TEST(!fg.factorFactorConnection(2, 2));
   OPENGM_TEST(!fg.factorFactorConnection(2, 3));
   OPENGM_TEST(!fg.factorFactorConnection(2, 4));
   OPENGM_TEST(fg.factorFactorConnection(2, 5));
   OPENGM_TEST(!fg.factorFactorConnection(2, 6));
   OPENGM_TEST(fg.factorFactorConnection(2, 7));
   OPENGM_TEST(fg.factorFactorConnection(2, 8));

   // test FactorGraph::variableVariableConnection
   OPENGM_TEST(!fg.variableVariableConnection(0, 0));
   OPENGM_TEST(fg.variableVariableConnection(0, 1));
   OPENGM_TEST(fg.variableVariableConnection(0, 2));
   OPENGM_TEST(fg.variableVariableConnection(0, 3));

   OPENGM_TEST(fg.variableVariableConnection(1, 0));
   OPENGM_TEST(!fg.variableVariableConnection(1, 1));
   OPENGM_TEST(fg.variableVariableConnection(1, 2));
   OPENGM_TEST(fg.variableVariableConnection(1, 3));

   OPENGM_TEST(fg.variableVariableConnection(2, 0));
   OPENGM_TEST(fg.variableVariableConnection(2, 1));
   OPENGM_TEST(!fg.variableVariableConnection(2, 2));
   OPENGM_TEST(fg.variableVariableConnection(2, 3));

   OPENGM_TEST(fg.variableVariableConnection(3, 0));
   OPENGM_TEST(fg.variableVariableConnection(3, 1));
   OPENGM_TEST(fg.variableVariableConnection(3, 2));
   OPENGM_TEST(!fg.variableVariableConnection(3, 3));

   // test 
   // - FactorGraph::variableAdjacencyMatrix 
   // - FactorGraph::variableVariableConnection
   marray::Matrix<bool> matrix;
   fg.variableAdjacencyMatrix(matrix);
   OPENGM_TEST(matrix.shape(0) == fg.numberOfVariables());
   OPENGM_TEST(matrix.shape(1) == fg.numberOfVariables());
   for(size_t j=0; j<fg.numberOfVariables(); ++j) {
      for(size_t k=0; k<fg.numberOfVariables(); ++k) {
         OPENGM_TEST(matrix(j, k) == fg.variableVariableConnection(j, k));
         OPENGM_TEST(matrix(j, k) == matrix(k, j));
      }
   }

   // test
   // - FactorGraph::variableAdjacencyLists
   // - FactorGraph::variableVariableConnection
   {
      std::vector<std::set<size_t> > list;
      fg.variableAdjacencyList(list);
      OPENGM_TEST(list.size() == fg.numberOfVariables());
      for(size_t j=0; j<fg.numberOfVariables(); ++j) {
         std::vector<bool> connected(fg.numberOfVariables());
         for(std::set<size_t>::const_iterator it = list[j].begin(); it != list[j].end(); ++it) {
            connected[*it] = true;
         }
         for(size_t k=0; k<fg.numberOfVariables(); ++k) {
            OPENGM_TEST(connected[k] == fg.variableVariableConnection(j, k));
         }
      }
   }

   // test
   // - FactorGraph::variableAdjacencyLists
   // - FactorGraph::variableVariableConnection
   {
      std::vector<opengm::RandomAccessSet<size_t> > list;
      fg.variableAdjacencyList(list);
      OPENGM_TEST(list.size() == fg.numberOfVariables());
      for(size_t j=0; j<fg.numberOfVariables(); ++j) {
         std::vector<bool> connected(fg.numberOfVariables());
         for(opengm::RandomAccessSet<size_t>::const_iterator it = list[j].begin(); it != list[j].end(); ++it) {
            connected[*it] = true;
         }
         for(size_t k=0; k<fg.numberOfVariables(); ++k) {
            OPENGM_TEST(connected[k] == fg.variableVariableConnection(j, k));
         }
      }
   }

   testIsAcyclic();
   testNumberOfFactors(); 
   
   testIsChain();
   testIsConnected();
   testIsGrid();

   return 0;
}

