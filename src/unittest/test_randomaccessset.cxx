#include <set>

#include "opengm/unittests/test.hxx"
#include "opengm/datastructures/randomaccessset.hxx"

struct TestRandomAccessSet{
   void run() {

      std::set<size_t> stdset;
      opengm::RandomAccessSet<size_t> myset;

      OPENGM_TEST_EQUAL(0,myset.size());



      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.insert(2);
      myset.insert(2);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.insert(1);
      myset.insert(1);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.insert(3);
      myset.insert(3);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.insert(3);
      myset.insert(3);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.insert(10);
      myset.insert(10);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.insert(10);
      myset.insert(10);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.insert(0);
      myset.insert(0);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.erase(10);
      myset.erase(10);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.erase(2);
      myset.erase(2);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
      stdset.erase(112);
      myset.erase(112);
      OPENGM_TEST_EQUAL(stdset.size(),myset.size());
      OPENGM_TEST_EQUAL(std::distance(stdset.begin(),stdset.end()),std::distance(myset.begin(),myset.end()));
      OPENGM_TEST_EQUAL_SEQUENCE(stdset.begin(),stdset.end(),myset.begin());
   }
};

int main() {
    std::cout << "Test Sorted Vector (random access set)  "<< std::endl;
    {
       TestRandomAccessSet t;
       t.run();
    }
    std::cout << "done.." << std::endl;
    return 0;
}