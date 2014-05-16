#include <vector>

#include "opengm/functions/learnablefunction.hxx"
#include "opengm/functions/learnable/lpotts.hxx" 
#include <opengm/unittests/test.hxx>

template<class T>
struct LearnableFunctionsTest {
  typedef size_t LabelType;
  typedef size_t IndexType;
  typedef T      ValueType;

  void testLearnableFeatureFunction(){
    std::cout << " * Learnable Feature Function ..." << std::flush; 
    // parameter
    const size_t numparam = 1;
    opengm::Parameters<ValueType,IndexType> param(numparam);
    param.setParameter(0,5.0);
    
    std::vector<LabelType> shape(2,3);
    std::vector<size_t> pIds(1,0);
    std::vector<ValueType> feat(1,1);
    // function
    opengm::LearnableFeatureFunction<ValueType,IndexType,LabelType> lfunc(param,shape,pIds,feat);

    LabelType l[] ={0,0};
    for(l[0]=0;l[0]<shape[0];++l[0]){
      for(l[1]=0;l[1]<shape[1];++l[1]){
	OPENGM_TEST(lfunc(l)==0);
      }
    }
    std::cout << "OK" << std::endl; 
  }

  void testLPotts(){
    std::cout << " * LearnablePotts ..." << std::flush; 
    // parameter
    const size_t numparam = 1;
    opengm::Parameters<ValueType,IndexType> param(numparam);
    param.setParameter(0,5.0);
    
    LabelType numL = 3;
    std::vector<size_t> pIds(1,0);
    std::vector<ValueType> feat(1,1);
    // function
    opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> lfunc(param,numL,pIds,feat);

    LabelType l[] ={0,0};
    for(l[0]=0;l[0]<numL;++l[0]){
      for(l[1]=0;l[1]<numL;++l[1]){
	if(l[0]==l[1]){
	  OPENGM_TEST_EQUAL_TOLERANCE(lfunc(l),0, 0.0001);
	}else{
	  OPENGM_TEST_EQUAL_TOLERANCE(lfunc(l),5.0, 0.0001);
	}
      }
    }
    std::cout << "OK" << std::endl; 
  }

};


int main() {
   std::cout << "Learnable Functions test...  " << std::endl;
   {
      LearnableFunctionsTest<double>t;
      t.testLearnableFeatureFunction();
      t.testLPotts();
   }
   std::cout << "done.." << std::endl;
   return 0;
}
