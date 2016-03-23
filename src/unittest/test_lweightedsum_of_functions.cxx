#include <vector>
#include "opengm/functions/learnable/lweightedsum_of_functions.hxx"
#include <opengm/unittests/test.hxx>

template<class T>
struct LWeightedSumOfFunctionsTest {
  typedef size_t LabelType;
  typedef size_t IndexType;
  typedef T      ValueType;

  void testLWeightedSumOfFunctions(){
    std::cout << " * Learnable Weighted Sum of Functions ..." << std::endl;

    std::cout  << "    - test weightGradient ..." <<std::flush;
    std::cout << "\n_____________________\n";
    std::cout << "features:\n";

    const size_t numparam = 1;
    opengm::learning::Weights<ValueType> param(numparam);
    param.setWeight(0,5.0);

    std::vector<size_t> pIds;
    pIds.push_back((size_t)0);

    size_t num_vars = 2;
    size_t max_number_objects = 3;
    std::vector<size_t> shape(num_vars, (max_number_objects + 1));
    marray::Marray<double> energies0(shape.begin(), shape.end(), 1);
    std::vector<marray::Marray<double> > features;
    features.push_back(energies0);

    for(size_t i = 0; i < shape[0]; i++){
        for(size_t j = 0; j < shape[1]; j++){
            features[0](i,j) = (0+i+10*(j+1)+1)*features[0](i,j);
            std::cout << features[0](i,j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "_____________________\n";

    std::vector<size_t> varShape;
    varShape.push_back((size_t)max_number_objects + 1);
    varShape.push_back((size_t)max_number_objects + 1);

    opengm::functions::learnable::LWeightedSumOfFunctions<ValueType,IndexType,LabelType> f(varShape,param,pIds,features);
    std::cout << "weightGradient:\n";

    for(size_t i = 0; i < shape[0]; i++){
        for(size_t j = 0; j < shape[1]; j++){
            std::vector<size_t> coords;
            coords.push_back((size_t)i);
            coords.push_back((size_t)j);
            std::cout << f.weightGradient(0,coords.begin() ) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "_____________________\n";

    for(size_t i = 0; i < shape[0]; i++){
        for(size_t j = 0; j < shape[1]; j++){
            std::vector<size_t> coords;
            coords.push_back((size_t)i);
            coords.push_back((size_t)j);
            std::cout << "feature (" << i << "," << j << ") = " << features[0](i,j) << "   =?=   " << f.weightGradient(0,coords.begin()) << " = " << " weightGradient(" << i << "," << j << ")   ";
            OPENGM_TEST(features[0](i,j)==f.weightGradient(0,coords.begin()));
            std::cout << " OK\n";
        }
        std::cout << "\n";
    }

    std::cout << " OK" << std::endl; 
  }

};


int main() {
   std::cout << "Learnable Weighted Sum of Functions Test...  " << std::endl;
   {
      LWeightedSumOfFunctionsTest<double>t;
      t.testLWeightedSumOfFunctions();
   }
   std::cout << "done.." << std::endl;
   return 0;
}
