#include <vector>

#include <opengm/unittests/test.hxx>
#include <opengm/graphicalmodel/graphicalmodel_manipulator.hxx>
#include <opengm/operations/adder.hxx>


struct ManipulatorTest {
   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType; 
   typedef opengm::ExplicitFunction<ValueType, IndexType, LabelType> ExplicitFunctionType;
   typedef opengm::GraphicalModel<ValueType,opengm::Adder,ExplicitFunctionType,opengm::DiscreteSpace<IndexType,LabelType> > GraphicalModelType; 
   typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;
   
   void test() {
      IndexType numVars = 10;
      std::vector<LabelType> nos(numVars,3);
      GraphicalModelType gm(opengm::DiscreteSpace<IndexType,LabelType>(nos.begin(), nos.end()));
      
      for (IndexType var=0; var<gm.numberOfVariables(); ++var){
         ExplicitFunctionType f(&nos[var], &nos[var]+1);
         for(LabelType i=0; i<gm.numberOfLabels(var); ++i)
            f(i) = i+10*var+20;
         FunctionIdentifier fid = gm.addFunction(f);
         gm.addFactor( fid , &var, (&var)+1 ); 
      }
      for (IndexType var=0; var<gm.numberOfVariables()-1; ++var){
         ExplicitFunctionType f(&nos[var], &nos[var]+2); 
         for(LabelType i=0; i<gm.numberOfLabels(var); ++i)
            for(LabelType j=0; j<gm.numberOfLabels(var+1); ++j)
               f(i,j) = i+10*var+20+j*8; 
         FunctionIdentifier fid = gm.addFunction(f);
         IndexType vars[2]; vars[0]=var; vars[1]=var+1;
         gm.addFactor( fid , vars, vars+2 );  
      }
   
      std::cout << "fix variables ..."<<std::endl;
      opengm::GraphicalModelManipulator<GraphicalModelType> gmm(gm);
      gmm.fixVariable(0,2);
      gmm.fixVariable(3,2); 
      gmm.fixVariable(4,1); 
  
      std::cout << "lock model ..."<<std::endl;
      gmm.lock();

      std::cout << "build model ..."<<std::endl;
      gmm.buildModifiedModel();
      const opengm::GraphicalModelManipulator<GraphicalModelType>::MGM gm2 = gmm.getModifiedModel(); 

      std::vector<LabelType> l1(numVars,1); l1[0]=2;l1[3]=2;l1[4]=1;
      std::vector<LabelType> l2(numVars-3,1);


      OPENGM_TEST_EQUAL_TOLERANCE(gm.evaluate(l1), gm2.evaluate(l2),0.000001); 

      gmm.buildModifiedSubModels();
      OPENGM_TEST_EQUAL(gmm.numberOfSubmodels(), 2);

      const opengm::GraphicalModelManipulator<GraphicalModelType>::MGM gm2a = gmm.getModifiedSubModel(0); 
      const opengm::GraphicalModelManipulator<GraphicalModelType>::MGM gm2b = gmm.getModifiedSubModel(1);
      std::vector<LabelType> l2a(2,1);
      std::vector<LabelType> l2b(5,1);
      ValueType v =  gm2a.evaluate(l2a)+ gm2b.evaluate(l2b);
      OPENGM_TEST_EQUAL_TOLERANCE(gm.evaluate(l1), v, 0.000001);
 
      std::vector<LabelType> l2y;
      gmm.modifiedState2OriginalState(l2,l2y);
      OPENGM_TEST_EQUAL(l2y.size(),l1.size());
      for(IndexType i=0; i<l1.size();++i)
         OPENGM_ASSERT(l1[i]==l2y[i]);


      std::vector<std::vector<LabelType> > ll(2);
      ll[0] = l2a;
      ll[1] = l2b;
      std::vector<LabelType> l2x;
      gmm.modifiedSubStates2OriginalState(ll,l2x);
      OPENGM_TEST_EQUAL(l2x.size(),l1.size());
      for(IndexType i=0; i<l1.size();++i)
         OPENGM_ASSERT(l1[i]==l2x[i]);
      

      return;
   }
  

};


int main() {
   std::cout << "GraphicalModelManipulator test...  " << std::endl;
   
   { 
      ManipulatorTest t;
      t.test();
   }
   
   std::cout << "done.." << std::endl;
   return 0;
}
