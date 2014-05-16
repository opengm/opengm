#include <vector>

#include "opengm/functions/learnablefunction.hxx"
#include <opengm/unittests/test.hxx>

template<class T>
struct LearnableFunctionsTest {

  void run(){
    std::cout << "OK" << std::endl;
  }

};


int main() {
   std::cout << "Learnable Functions test...  " << std::endl;
   {
      LearnableFunctionsTest<int >t;
      t.run();
   }
   std::cout << "done.." << std::endl;
   return 0;
}
