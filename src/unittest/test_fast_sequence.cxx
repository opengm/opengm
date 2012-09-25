#include <vector>
#include <algorithm>

#include <opengm/unittests/test.hxx>
#include "opengm/datastructures/fast_sequence.hxx"

void testFastSequence()
{
   {
      opengm::FastSequence<size_t,5> fs(4);
      OPENGM_TEST(fs.size()==4);
      OPENGM_TEST(size_t(std::distance(fs.begin(),fs.end()))==size_t(fs.size()));
      for(size_t i=0;i<fs.size();++i) {
         fs[i]=i;
      }
      opengm::FastSequence<size_t,5>::ConstIteratorType iter=fs.begin();
      size_t counter=0;
      while(iter !=fs.end()) {
         OPENGM_TEST(*iter==counter);
         OPENGM_TEST(fs[counter]==counter);
         ++counter;
         ++iter;
      }
   }
   {
      opengm::FastSequence<size_t,5> fs(10);
      OPENGM_TEST(fs.size()==10);
      OPENGM_TEST(size_t(std::distance(fs.begin(),fs.end()))==size_t(fs.size()));
      for(size_t i=0;i<fs.size();++i) {
         fs[i]=i;
      }
      opengm::FastSequence<size_t,5>::ConstIteratorType iter=fs.begin();
      size_t counter=0;
      while(iter !=fs.end()) {
         OPENGM_TEST(*iter==counter);
         OPENGM_TEST(fs[counter]==counter);
         ++counter;
         ++iter;
      }
   }
   {
      opengm::FastSequence<size_t,5> fs;
      OPENGM_TEST(fs.size()==0);
      OPENGM_TEST(size_t(std::distance(fs.begin(),fs.end()))==size_t(fs.size()));
      for(size_t i=0;i<10;++i) {
         fs.push_back(i);
      }
      OPENGM_TEST(fs.size()==10);
      opengm::FastSequence<size_t,5>::ConstIteratorType iter=fs.begin();
      size_t counter=0;
      while(iter !=fs.end()) {
         OPENGM_TEST(*iter==counter);
         OPENGM_TEST(fs[counter]==counter);
         ++counter;
         ++iter;
      }
   }
   {
      opengm::FastSequence<size_t,5> fs;
      OPENGM_TEST(fs.size()==0);
      OPENGM_TEST(size_t(std::distance(fs.begin(),fs.end()))==size_t(fs.size()));
      for(size_t i=0;i<10;++i) {
         fs.push_back(i);
      }
      OPENGM_TEST(fs.size()==10);

      opengm::FastSequence<size_t,5>::ConstIteratorType iter=fs.begin();
      size_t counter=0;
      while(iter !=fs.end()) {
         OPENGM_TEST(*iter==counter);
         OPENGM_TEST(fs[counter]==counter);
         ++counter;
         ++iter;
      }

      fs.resize(20);
      {
         OPENGM_TEST(fs.size()==20)
         for(size_t i=0;i<10;++i) {
            OPENGM_TEST(fs[i]==i);
         }
         for(size_t i=0;i<20;++i) {
            fs[i]=i;
            OPENGM_TEST(fs[i]==i);
         }
      }
      {
         opengm::FastSequence<size_t,5> fs2;
         fs2=fs;
         OPENGM_TEST(fs.size()==20);
         OPENGM_TEST(fs2.size()==20);
         {
            for(size_t i=0;i<20;++i) {
               OPENGM_TEST(fs[i]==i);
               OPENGM_TEST(fs2[i]==i);
            }
         }
      }
      {
         opengm::FastSequence<size_t,5> fs2(fs);
         OPENGM_TEST(fs.size()==20);
         OPENGM_TEST(fs2.size()==20);
         {
            for(size_t i=0;i<20;++i) {
               OPENGM_TEST(fs[i]==i);
               OPENGM_TEST(fs2[i]==i);
            }
         }
      }
   }
}



int main(int argc, char** argv) {
	testFastSequence();
    return (EXIT_SUCCESS);
}


