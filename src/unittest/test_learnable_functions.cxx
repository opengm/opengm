#include <vector>

#include "opengm/functions/learnable/lpotts.hxx" 
#include <opengm/unittests/test.hxx>

template<class T>
struct LearnableFunctionsTest {
  typedef size_t LabelType;
  typedef size_t IndexType;
  typedef T      ValueType;

  void testLPotts(){
    std::cout << " * LearnablePotts ..." << std::endl; 

    std::cout  << "    - test basics ..." <<std::flush;
    // parameter
    const size_t numparam = 1;
    opengm::learning::Weights<ValueType> param(numparam);
    param.setWeight(0,5.0);
    
    LabelType numL = 3;
    std::vector<size_t> pIds(1,0);
    std::vector<ValueType> feat(1,1);
    // function
    opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> f(param,numL,pIds,feat);

    LabelType l[] ={0,0};
    for(l[0]=0;l[0]<numL;++l[0]){
      for(l[1]=0;l[1]<numL;++l[1]){
	if(l[0]==l[1]){
	  OPENGM_TEST_EQUAL_TOLERANCE(f(l),0, 0.0001);
	}else{
	  OPENGM_TEST_EQUAL_TOLERANCE(f(l),5.0, 0.0001);
	}
      }
    }
    std::cout << " OK" << std::endl; 
    std::cout  << "    - test serializations ..." <<std::flush;
    {
       typedef  opengm::functions::learnable::LPotts<ValueType,IndexType,LabelType> FUNCTION;
       const size_t sizeIndices=opengm::FunctionSerialization<FUNCTION>::indexSequenceSize(f);
       const size_t sizeValues=opengm::FunctionSerialization<FUNCTION>::valueSequenceSize(f);
       std::vector<long long unsigned> indices(sizeIndices);
       std::vector<T> values(sizeValues);
      
       opengm::FunctionSerialization<FUNCTION>::serialize(f,indices.begin(),values.begin());
       FUNCTION f2;
       opengm::FunctionSerialization<FUNCTION>::deserialize(indices.begin(),values.begin(),f2);
       f2.setWeights(param);

       OPENGM_TEST(f.dimension()==f2.dimension());
       OPENGM_TEST(f.size() == f2.size());
       std::vector<size_t> shape(f.dimension());
       for(size_t i=0;i<f.dimension();++i) {
          shape[i]=f.shape(i);
          OPENGM_TEST(f.shape(i)==f2.shape(i));
       }
       opengm::ShapeWalker<std::vector<size_t>::const_iterator > walker(shape.begin(),f.dimension());
       for(size_t i=0;i<f.size();++i) {
          OPENGM_TEST(walker.coordinateTuple().size()==f.dimension());
          OPENGM_TEST(f(walker.coordinateTuple().begin())==f2(walker.coordinateTuple().begin()) );
          ++walker;
       }
    }
    std::cout << " OK" << std::endl; 
  }

};


int main() {
   std::cout << "Learnable Functions test...  " << std::endl;
   {
      LearnableFunctionsTest<double>t;
      t.testLPotts();
   }
   std::cout << "done.." << std::endl;
   return 0;
}
