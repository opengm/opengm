#include <vector>

#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel_explicit_storage.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/inference/bruteforce.hxx>

int main() {
    std::cout << "Function Storage Test...  " << std::endl;
    {

     typedef opengm::meta::TypeListGenerator
         <
         opengm::ExplicitFunction<int>,
         opengm::PottsFunction<int>,
         opengm::PottsNFunction<int>
         >::type FunctionTypeList;
      typedef opengm::GraphicalModel<int, opengm::Minimizer, FunctionTypeList, opengm::DiscreteSpace< > > GmTypeA;
      typedef opengm::GraphicalModel<int, opengm::Minimizer, opengm::ExplicitFunction<int>, opengm::DiscreteSpace< > > GmTypeB;
      typedef GmTypeA::FunctionIdentifier FIA;
      typedef GmTypeB::FunctionIdentifier FIB;

      typedef opengm::ExplicitFunction<int> EF;

      size_t nos[] = {2, 2, 3};
      GmTypeA gmA(opengm::DiscreteSpace<  > (nos, nos + 3));
      GmTypeB gmB(opengm::DiscreteSpace<  > (nos, nos + 3));

      opengm::PottsNFunction<int> fp1(nos + 1, nos + 3, 0, 1);
      opengm::PottsFunction<int> fp2(2, 2, 0, 1);

      EF fe1(nos + 1, nos + 3, 1);
      fe1(0, 0) = 0;
      fe1(1, 1) = 0;

      EF fe2(nos, nos + 2, 1);
      fe2(0, 0) = 0;
      fe2(1, 1) = 0;

      FIA ia = gmA.addFunction(fp1);
      FIA ib = gmA.addFunction(fp2);

      FIB iae = gmB.addFunction(fe1);
      FIB ibe = gmB.addFunction(fe2);
      FIB iaes = gmB.addSharedFunction(fe1);
      FIB ibes = gmB.addSharedFunction(fe2);
      OPENGM_ASSERT(iae==iaes);
      OPENGM_ASSERT(ibe==ibes);
      size_t vi[] = {0, 1, 2};
      gmA.addFactor(ia, vi + 1, vi + 3);
      gmA.addFactor(ib, vi, vi + 2);

      gmB.addFactor(iae, vi + 1, vi + 3);
      gmB.addFactor(ibe, vi, vi + 2);
      
      gmB.addFactor(iaes, vi + 1, vi + 3);
      gmB.addFactor(ibes, vi, vi + 2);
      
      
      
      std::vector<int> denseStorage(1000);
      opengm::ExplicitStorage<GmTypeA> storageA(gmA);
      opengm::ExplicitStorage<GmTypeB> storageB(gmB);
      typedef GmTypeA::FactorType::ShapeIteratorType ShapeIteratorTypeA;
      typedef GmTypeB::FactorType::ShapeIteratorType ShapeIteratorTypeB;
      
      typedef opengm::ConvertToExplicit<GmTypeA>::ExplicitGraphicalModelType GmTypeAE;
      GmTypeAE gmAE;
      opengm::ConvertToExplicit<GmTypeA>::convert(gmA,gmAE);
      
      
      for(size_t factor=0;factor<gmA.numberOfFactors();++factor){
          opengm::ShapeWalker< ShapeIteratorTypeA > walker(gmA[factor].shapeBegin(),gmA[factor].numberOfVariables());
          int const *  ptr=storageA[gmA[factor]];
          gmA[factor].copyValues(denseStorage.begin());
          for (size_t i = 0; i < gmA[factor].size(); ++i) {
              OPENGM_TEST_EQUAL(gmA[factor](walker.coordinateTuple().begin()),gmAE[factor](walker.coordinateTuple().begin()));
              OPENGM_TEST_EQUAL(gmA[factor](walker.coordinateTuple().begin()),ptr[i]);
              OPENGM_TEST_EQUAL(denseStorage[i],ptr[i]);
               ++walker;
          }
      }
      for(size_t factor=0;factor<gmB.numberOfFactors();++factor){
          opengm::ShapeWalker< ShapeIteratorTypeB > walker(gmB[factor].shapeBegin(),gmB[factor].numberOfVariables());
          int const *  ptr=storageB[gmB[factor]];
          gmB[factor].copyValues(denseStorage.begin());
          for (size_t i = 0; i < gmB[factor].size(); ++i) {
              OPENGM_TEST_EQUAL(gmB[factor](walker.coordinateTuple().begin()),ptr[i]);
              OPENGM_TEST_EQUAL(denseStorage[i],ptr[i]);
               ++walker;
          }
      }
      
    }
    std::cout << "done.." << std::endl;
    return 0;
}
