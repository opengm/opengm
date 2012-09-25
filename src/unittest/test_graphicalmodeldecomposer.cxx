#include <stdlib.h>

#include <opengm/operations/adder.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/decomposition/graphicalmodeldecomposer.hxx>
#include <opengm/graphicalmodel/modelgenerators/syntheticmodelgenerator.hxx>
#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/functions/modelviewfunction.hxx>
#include <opengm/functions/constant.hxx>

template<class GM, class SUBGM>
void getSubModel(
   const GM& gm, 
   const opengm::GraphicalModelDecomposition& decomposition, 
   const size_t subModelId,
   SUBGM& subGm
   )   
{ 
   typedef typename GM::ValueType                             ValueType; 
   typedef typename GM::OperatorType                          OpType;
   typedef GM                                                 GraphicalModelType;
   typedef opengm::ModelViewFunction<GraphicalModelType, marray::Marray<ValueType> > ViewFunctionType;
   typedef opengm::ConstantFunction<ValueType>                ConstantFunctionType;
   typedef typename SUBGM::FunctionIdentifier                 FunctionIdentifierType;

   typedef opengm::GraphicalModelDecomposition                DecompositionType;
   typedef typename DecompositionType::SubVariable            SubVariableType;
   typedef typename DecompositionType::SubVariableListType    SubVariableListType;
   typedef typename DecompositionType::SubFactor              SubFactorType;
   typedef typename DecompositionType::SubFactorListType      SubFactorListType; 
   typedef typename DecompositionType::EmptySubFactor         EmptySubFactorType;
   typedef typename DecompositionType::EmptySubFactorListType EmptySubFactorListType;

   const std::vector<SubVariableListType> subVariableList =  decomposition.getVariableLists();
   const std::vector<SubFactorListType>   subFactorList   =  decomposition.getFactorLists();
   const std::map<std::vector<size_t>,EmptySubFactorListType> emptySubFactorLists = decomposition.getEmptyFactorLists();
     
   std::vector<size_t> numStates(decomposition.numberOfSubVariables(subModelId),0);
   SubVariableListType::const_iterator it;
   for(size_t varId=0; varId<gm.numberOfVariables(); ++varId) {
      for(it = subVariableList[varId].begin(); it!=subVariableList[varId].end();++it) {
         if((*it).subModelId_ == subModelId) {
            numStates[(*it).subVariableId_] = gm.numberOfLabels(varId);
         }
      }
   }
   subGm = SUBGM(opengm::DiscreteSpace<size_t,size_t>(numStates.begin(), numStates.end() ));

   // Add Factors
   SubFactorListType::const_iterator it2;
   for(size_t factorId=0; factorId<gm.numberOfFactors(); ++factorId) {
      for(it2 = subFactorList[factorId].begin(); it2!=subFactorList[factorId].end();++it2) {
         if((*it2).subModelId_ == subModelId) {
            //addFactor
            ViewFunctionType function(gm,factorId, 1.0/subFactorList[factorId].size()); 
            FunctionIdentifierType funcId = subGm.addFunction(function); 
            subGm.addFactor(funcId,(*it2).subIndices_.begin(),(*it2).subIndices_.end());
         }
      }
   }
   // Add EmptyFactors 
   EmptySubFactorListType::const_iterator it3;
   std::map<std::vector<size_t>,EmptySubFactorListType>::const_iterator it4;
   for(it4=emptySubFactorLists.begin() ; it4 != emptySubFactorLists.end(); it4++ ) {
      //size_t i=0;
      std::vector<size_t> shape((*it4).first.size()); 
      for(size_t i=0; i<(*it4).first.size(); ++i) {
         shape[i] = gm.numberOfLabels((*it4).first[i]);
      } 
      for(it3 = (*it4).second.begin(); it3!=(*it4).second.end();++it3) { 
         if(subModelId == (*it3).subModelId_) { 
            ConstantFunctionType function(shape.begin(),shape.end(),0);             
            FunctionIdentifierType funcId = subGm.addFunction(function); 
            subGm.addFactor(funcId,(*it3).subIndices_.begin(),(*it3).subIndices_.end());
         }
      } 
   } 
   return;
}
//////////////////////////////////////////

template<class DECOMPOSITION, class GM>
void test(const DECOMPOSITION& decomposition, const GM& gm, const bool acyclicSubs=false)
{
   typedef typename GM::ValueType                                       ValueType; 
   typedef typename GM::OperatorType                                    OpType;
   typedef GM                                                           GraphicalModelType;
   typedef opengm::ModelViewFunction<GM, marray::Marray<ValueType> >    ViewFunctionType;
   typedef opengm::ConstantFunction<ValueType>                          ConstantFunctionType;
   typedef typename opengm::meta::TypeListGenerator<ViewFunctionType, ConstantFunctionType>::type FunctionListType;
   typedef opengm::GraphicalModel<ValueType,OpType,FunctionListType>    SubGmType;

   assert(decomposition.isValid(gm));
   std::vector<SubGmType> subGm(decomposition.numberOfSubModels());
   for(size_t subModelId=0; subModelId<decomposition.numberOfSubModels(); ++subModelId) {
      getSubModel(gm,decomposition,subModelId, subGm[subModelId]); 
      if(acyclicSubs) {
         assert(subGm[subModelId].isAcyclic());
      } 
      assert(decomposition.numberOfSubVariables(subModelId)==subGm[subModelId].numberOfVariables());
      assert(decomposition.numberOfSubFactors(subModelId)==subGm[subModelId].numberOfFactors());
   }

   // Check Factors
   typedef typename DECOMPOSITION::SubFactorListType      SubFactorListType; 
   
   const std::vector<SubFactorListType> subFactorList = decomposition.getFactorLists();
   typename SubFactorListType::const_iterator it2;
   std::vector<size_t> subFactorCount(decomposition.numberOfSubModels(),0);

   for(size_t factorId=0; factorId < gm.numberOfFactors(); ++factorId) {
      typename GM::IndependentFactorType f = gm[factorId];
      for(it2 = subFactorList[factorId].begin(); it2!=subFactorList[factorId].end();++it2) {
         const size_t subModelId = (*it2).subModelId_;
         if( gm[factorId].numberOfVariables()==1 ) {
            for(size_t i=0; i<gm[factorId].numberOfLabels(0); ++i) {
               f(&i) -= subGm[subModelId][subFactorCount[subModelId]](&i);
            }
         }
         if( gm[factorId].numberOfVariables()==2 ) {
            size_t conf[2];
            for(conf[0]=0; conf[0]<gm[factorId].numberOfLabels(0); ++conf[0]) {
            for(conf[1]=0; conf[1]<gm[factorId].numberOfLabels(1); ++conf[1]) {
               f(conf) -= subGm[subModelId][subFactorCount[subModelId]](conf);
            }
            }
         }
         ++subFactorCount[subModelId];
      }
      if( gm[factorId].numberOfVariables()==1 ) {
         for(size_t i=0; i<gm[factorId].numberOfLabels(0); ++i) {
            OPENGM_ASSERT(fabs(f(&i))<0.00001);
         }
      }
      if( gm[factorId].numberOfVariables()==2 ) {
         size_t conf[2];
         for(conf[0]=0; conf[0]<gm[factorId].numberOfLabels(0); ++conf[0]) {
            for(conf[1]=0; conf[1]<gm[factorId].numberOfLabels(1); ++conf[1]) {
               OPENGM_ASSERT(fabs(f(conf))<0.00001);
            }
         }
      }
      
   }
   // Check EmptyFactors 
   typedef typename DECOMPOSITION::EmptySubFactorListType  EmptySubFactorListType; 
    
   const std::map<std::vector<size_t>,EmptySubFactorListType> emptySubFactorLists = decomposition.getEmptyFactorLists();
   typename EmptySubFactorListType::const_iterator it3;
   typename std::map<std::vector<size_t>,EmptySubFactorListType>::const_iterator it4;
   for(it4=emptySubFactorLists.begin() ; it4 != emptySubFactorLists.end(); it4++ ) {
      //size_t i=0;
      std::vector<size_t> shape((*it4).first.size()); 
      for(size_t i=0; i<(*it4).first.size(); ++i) {
         shape[i] = gm.numberOfLabels((*it4).first[i]);
      } 
      marray::Marray<ValueType> f(shape.begin(),shape.end(),0);

      for(it3 = (*it4).second.begin(); it3!=(*it4).second.end();++it3) { 
         const size_t subModelId = (*it3).subModelId_;
         if(shape.size()==1 ) {
            for(size_t i=0; i<shape[0]; ++i) {
               f(&i) -= subGm[subModelId][subFactorCount[subModelId]](&i);
            }
         }
         if(shape.size()==2 ) {
            size_t conf[2];
            for(conf[0]=0; conf[0]<shape[0]; ++conf[0]) {
               for(conf[1]=0; conf[1]<shape[1]; ++conf[1]) {
                  f(conf) -= subGm[subModelId][subFactorCount[subModelId]](conf);
               }
            }
         }
         ++subFactorCount[subModelId];
      }
      
      if(shape.size()==1 ) {
         for(size_t i=0; i<shape[0]; ++i) {
            OPENGM_ASSERT(fabs(f(&i))<0.00001);
         }
      }
      if(shape.size()==2 ) {
         size_t conf[2];
         for(conf[0]=0; conf[0]<shape[0]; ++conf[0]) {
            for(conf[1]=0; conf[1]<shape[1]; ++conf[1]) {
               OPENGM_ASSERT(fabs(f(conf))<0.00001);
            }
         }
      }
      
   } 
}

//////////////////////////////////////////



template <class GM, class SUBGM >
struct GraphicalModelDecomposerTest
{
   typedef GM                                         GraphicalModelType;
   typedef SUBGM                                      SubGmType;
   typedef opengm::SyntheticModelGenerator2<GraphicalModelType>   ModelGeneratorType;
   typedef opengm::GraphicalModelDecomposer<GraphicalModelType>   DecomposerType;
   typedef opengm::GraphicalModelDecomposition        DecompositionType;

   ModelGeneratorType modelGenerator_;

   void testGrid() {
      std::cout << "    -- 2nd order grid with unary factors..."<< std::flush;
      const size_t height = 4;
      const size_t width  = 4;
      //const size_t numVar = height * width;
      const size_t numStates = 3;
      typename ModelGeneratorType::Parameter para;

      //*************************************************************************
      GraphicalModelType gm  = modelGenerator_.buildGrid(0,height, width, numStates, para);
      GraphicalModelType gmL = modelGenerator_.buildGrid(0,40,40, numStates, para);
      {
         DecomposerType decomposer;
         DecompositionType decomposition = decomposer.decomposeIntoTree(gm);
         assert(decomposition.numberOfSubModels()==1);
         assert(decomposition.numberOfSubVariables(0)==16+9);
         assert(decomposition.numberOfSubFactors(0)==16+24);
         test(decomposition,gm,true);
         decomposition.complete();
         test(decomposition,gm,true);
         assert(decomposition.numberOfSubModels()==1);
         assert(decomposition.numberOfSubVariables(0)==16+9);
         assert(decomposition.numberOfSubFactors(0)==16+24+9);
         std::cout << "*" <<std::flush;
      } 
      {
         DecomposerType decomposer;
         DecompositionType decomposition = decomposer.decomposeIntoTree(gmL);
         test(decomposition,gmL,true);
         decomposition.complete();
         test(decomposition,gmL,true);
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoSpanningTrees(gm);
         test(decomposition,gm,true);
         assert(decomposition.numberOfSubModels()==2);
         for(size_t i=0;i<2;++i) {
            assert(decomposition.numberOfSubVariables(i)==16);
            assert(decomposition.numberOfSubFactors(i)==16+12+3);
         }
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoSpanningTrees(gmL);
         test(decomposition,gmL,true);
         assert(decomposition.numberOfSubModels()==2);
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoKFans(gm,2);
         test(decomposition,gm,true);
         assert(decomposition.numberOfSubModels()==8);
         for(size_t i=0;i<8;++i) {
            assert(decomposition.numberOfSubVariables(i)==16);
            assert(decomposition.numberOfSubFactors(i)==16+4 || decomposition.numberOfSubFactors(i)==16+6);
         }
         std::cout << "*" <<std::flush;
      } 
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         std::vector<std::set<size_t> > t(2);
         size_t set0[] = {0,1,2,3,4,5,6,7,8,9,10,11};
         size_t set1[] = {4,5,6,7,8,9,10,11,12,13,14,15};
         t[0] = std::set<size_t>(set0,set0+12);
         t[1] = std::set<size_t>(set1,set1+12);
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoOpenBlocks(gm,t);
         test(decomposition,gm);
         assert(decomposition.numberOfSubModels()==2);
         for(size_t i=0;i<2;++i) {
            assert(decomposition.numberOfSubVariables(i)==12);
            assert(decomposition.numberOfSubFactors(i)==12+8+9);
         }
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         std::vector<std::set<size_t> > t(2);
         size_t set0[] = {0,1,2,3,4,5,6,7};
         size_t set1[] = {8,9,10,11,12,13,14,15};
         t[0] = std::set<size_t>(set0,set0+8);
         t[1] = std::set<size_t>(set1,set1+8);
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoClosedBlocks(gm,t);
         test(decomposition,gm);
         assert(decomposition.numberOfSubModels()==2);
         for(size_t i=0;i<2;++i) {
            assert(decomposition.numberOfSubVariables(i)==12);
            assert(decomposition.numberOfSubFactors(i)==8+8+6);
         }
         std::cout << "*" <<std::flush;
      }


      std::cout << " OK" << std::endl;
      //**********************************************************************
      std::cout << "    -- 2nd order grid without unary factors..."<< std::flush;
      para.functionTypes_[0] = ModelGeneratorType::EMPTY;
      GraphicalModelType gm2 = modelGenerator_.buildGrid(0,height, width, numStates, para);
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoTree(gm2);
       
         test(decomposition,gm2,true);
         assert(decomposition.numberOfSubModels()==1);
         assert(decomposition.numberOfSubVariables(0)==16+9);
         assert(decomposition.numberOfSubFactors(0)==24); 
         decomposition.complete();
         test(decomposition,gm2,true); 
         assert(decomposition.numberOfSubModels()==1);
         assert(decomposition.numberOfSubVariables(0)==16+9);
         assert(decomposition.numberOfSubFactors(0)==24+2*9);
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoSpanningTrees(gm2);
         test(decomposition,gm2,true);
         assert(decomposition.numberOfSubModels()==2);
         for(size_t i=0; i<2;++i) {
            assert(decomposition.numberOfSubVariables(i)==16);
            assert(decomposition.numberOfSubFactors(i)==12+3);
         }
         decomposition.complete();
         test(decomposition,gm2,true);
         assert(decomposition.numberOfSubModels()==2);
         for(size_t i=0; i<2;++i) {
            assert(decomposition.numberOfSubVariables(i)==16);
            assert(decomposition.numberOfSubFactors(i)==16+12+3);
         }
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoKFans(gm2,2);
         test(decomposition,gm2,true);
         assert(decomposition.numberOfSubModels()==8);
         for(size_t i=0;i<8;++i) {
            assert(decomposition.numberOfSubVariables(i)==16);
            assert(decomposition.numberOfSubFactors(i)==4 || decomposition.numberOfSubFactors(i)==6);
         }
         decomposition.complete();
         test(decomposition,gm2,true);
         assert(decomposition.numberOfSubModels()==8);
         for(size_t i=0;i<8;++i) {
            assert(decomposition.numberOfSubVariables(i)==16);
            assert(decomposition.numberOfSubFactors(i)==16+4 || decomposition.numberOfSubFactors(i)==16+6);
         }
         std::cout << "*" <<std::flush;
      }
      std::cout << " OK" << std::endl;
      //***********************************************************************************
   };

   void testFull() {
      const size_t numVar    = 7;
      const size_t numStates = 3;
      typename ModelGeneratorType::Parameter para;

      //**************************************************************************
      std::cout << "    -- 2nd order full-graph with unary factors..."<< std::flush;
      GraphicalModelType gm  = modelGenerator_.buildFull(0,numVar, numStates, para);
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoTree(gm);
         test(decomposition,gm,true);
         decomposition.complete();
         test(decomposition,gm,true);
         assert(decomposition.numberOfSubVariables(0)==7+5+4+3+2+1);
         assert(decomposition.numberOfSubFactors(0)==7+6+5+4+3+2+1+5+4+3+2+1);
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoSpanningTrees(gm);
         test(decomposition,gm,true);
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoKFans(gm,2);
         test(decomposition,gm);
         assert(decomposition.numberOfSubModels()==4);
         for(size_t i=0;i<4;++i) {
            assert(decomposition.numberOfSubVariables(i)==7);
            assert(decomposition.numberOfSubFactors(i)==7+1+2*5);
         }
         std::cout << "*" <<std::flush;
      }
      std::cout << " OK" << std::endl;
      //****************************************************************************
      std::cout << "    -- 2nd order full-graph without unary factors..."<< std::flush;
      para.functionTypes_[0] = ModelGeneratorType::EMPTY;
      GraphicalModelType gm2 = modelGenerator_.buildFull(0,numVar, numStates, para);
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoTree(gm2);
         test(decomposition,gm2,true);
         decomposition.complete();
         test(decomposition,gm2,true);
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoSpanningTrees(gm2);
         test(decomposition,gm2,true);
         assert(decomposition.isValid(gm2));
         std::cout << "*" <<std::flush;
      }
      {
         opengm::GraphicalModelDecomposer<GraphicalModelType> decomposer;
         opengm::GraphicalModelDecomposition decomposition = decomposer.decomposeIntoKFans(gm2,2);
         test(decomposition,gm2);
         assert(decomposition.numberOfSubModels()==4);
         for(size_t i=0;i<4;++i) {
            assert(decomposition.numberOfSubVariables(i)==7);
            assert(decomposition.numberOfSubFactors(i)==1+2*5);
         }
         decomposition.complete();
         test(decomposition,gm2);
         assert(decomposition.numberOfSubModels()==4);
         for(size_t i=0;i<4;++i) {
            assert(decomposition.numberOfSubVariables(i)==7);
            assert(decomposition.numberOfSubFactors(i)==7+1+2*5);
         }
         std::cout << "*" <<std::flush;
      }
      std::cout << " OK" << std::endl;
      //****************************************************************************
   };
   void run() {
      testGrid();
      testFull();
   };
};

int main() {
   std::cout << "Decomposer test...  "<< std::endl;
   {  
      typedef float                                                            ValueType; 
      typedef opengm::Adder                                                    OpType;
      typedef opengm::GraphicalModel<ValueType,OpType>                         GraphicalModelType;
      typedef opengm::ModelViewFunction<GraphicalModelType, marray::Marray<ValueType> >    ViewFunctionType;
      typedef opengm::ConstantFunction<ValueType>                              ConstantFunctionType;
      typedef opengm::meta::TypeListGenerator<ViewFunctionType, ConstantFunctionType>::type FunctionListType;
      typedef opengm::GraphicalModel<ValueType,OpType,FunctionListType>         SubGmType;

      GraphicalModelDecomposerTest<GraphicalModelType, SubGmType> t;
      t.run();
   }
   std::cout << "done.." << std::endl;
   return 0;
}
