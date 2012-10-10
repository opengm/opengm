#include <vector>
#include <iostream>

#include "opengm/unittests/test.hxx"
#include "opengm/utilities/accessor_iterator.hxx"

namespace opengm {

struct TestType {
   TestType() : value_(10) {}
   size_t value_;
};

template<class T, bool isConst>
class VectorAccessor {
public:
   typedef T value_type;
   typedef typename meta::If<isConst, const value_type&, value_type&>::type reference;
   typedef typename meta::If<isConst, const value_type*, value_type*>::type pointer;
   typedef typename meta::If<isConst, const std::vector<value_type>&, std::vector<value_type>&>::type vector_reference;
   typedef typename meta::If<isConst, const std::vector<value_type>*, std::vector<value_type>*>::type vector_pointer;

   VectorAccessor(vector_pointer v = 0)
      : vector_(v) {}
   VectorAccessor(vector_reference v)
      : vector_(&v) {}
   size_t size() const
      { return vector_ == 0 ? 0 : vector_->size(); }
   reference operator[](const size_t j)
      { return (*vector_)[j]; }
   const value_type& operator[](const size_t j) const
      { return (*vector_)[j]; }
   template<bool isConstLocal>
      bool operator==(const VectorAccessor<T, isConstLocal>& other) const
         { return vector_ == other.vector_; }

private:
   vector_pointer vector_;
};

struct AccessorIteratorTest {
   template<bool isConst>
   void accessTest() {
      typedef AccessorIterator<VectorAccessor<size_t, isConst>, isConst> Iterator;

      std::vector<size_t> vec(10);
      for(size_t j=0; j<vec.size(); ++j) {
         vec[j] = static_cast<size_t>(j);
      }
      VectorAccessor<size_t, isConst> accessor(vec);

      // constructor
      {
         Iterator it1;
         Iterator it2(accessor);
         Iterator it3(accessor, 0);
      }

      // operator=
      {
         Iterator it1(accessor);
         Iterator it2(accessor);
         it1 = it2;
      }

      // operator* and operator[]
      {
         Iterator it1(accessor);
         for(size_t j=0; j<accessor.size(); ++j) {
            OPENGM_TEST(it1[j] == accessor[j]);

            Iterator it2(accessor, j);
            OPENGM_TEST(*it2 == accessor[j]);
         }
      }

      // operator->
      {
         std::vector<TestType> v(5);
         AccessorIterator<std::vector<TestType> > it(v);
         OPENGM_TEST(it->value_ == 10);
      }

      // comparison operators
      for(size_t j=0; j<accessor.size(); ++j) {
         for(size_t k=0; k<accessor.size(); ++k) {
            Iterator it1(accessor, j);
            Iterator it2(accessor, k);
            if(j == k) {
               OPENGM_TEST(it1 == it2);
               OPENGM_TEST(it1 <= it2);
               OPENGM_TEST(it1 >= it2);
               OPENGM_TEST(!(it1 != it2));
            }
            else {
               OPENGM_TEST(it1 != it2);
               OPENGM_TEST(!(it1 == it2));
               if(j < k) {
                  OPENGM_TEST(it1 < it2);
                  OPENGM_TEST(!(it1 > it2));
                  OPENGM_TEST(it1 <= it2);
                  OPENGM_TEST(!(it1 >= it2));
               }
               else if(j > k) {
                  OPENGM_TEST(it1 > it2);
                  OPENGM_TEST(!(it1 < it2));
                  OPENGM_TEST(it1 >= it2);
                  OPENGM_TEST(!(it1 <= it2));
               }
            }
         }
      }

      // incrementation, decrementation
      {
         Iterator it1(accessor);
         Iterator it2(accessor, accessor.size());
         for(size_t j=0; j<accessor.size(); ++j) {
            OPENGM_TEST(*it1 == accessor[j]);
            ++it1;

            --it2;
            OPENGM_TEST(*it2 == accessor[accessor.size()-j-1]);
         }
      }
      {
         Iterator it1(accessor);
         Iterator it2(accessor, accessor.size());
         for(size_t j=0; j<accessor.size(); ++j) {
            OPENGM_TEST(*it1 == accessor[j]);
            it1++;

            it2--;
            OPENGM_TEST(*it2 == accessor[accessor.size()-j-1]);
         }
      }
      {
         Iterator it1(accessor);
         Iterator it2(accessor);
         OPENGM_TEST(it1++ == it2);
      }
      {
         Iterator it1(accessor, 1);
         Iterator it2(accessor, 1);
         OPENGM_TEST(it1-- == it2);
      }
      {
         Iterator it1(accessor);
         Iterator it2(accessor, accessor.size());
         for(size_t j=0; j<accessor.size(); j += 2) {
            OPENGM_TEST(*it1 == accessor[j]);
            it1 += 2;

            it2 -= 2;
            OPENGM_TEST(*it2 == accessor[accessor.size()-j-2]);
         }
      }

      // operator+
      {
         Iterator it(accessor);
         for(size_t j=0; j<accessor.size()-1; ++j) {
            Iterator it2 = it + j;
            OPENGM_TEST(*it2 == accessor[j]);

            Iterator it3 = j + it;
            OPENGM_TEST(*it3 == accessor[j]);
         }
      }

      // operator-
      {
         Iterator it1(accessor);
         for(size_t j=0; j<accessor.size(); ++j) {
            Iterator it2(accessor, j);
            //gcc 4.6 bugfix
            #if __GNUC__ == 4 && __GNUC_MINOR__ >= 6
            typedef std::ptrdiff_t difference_type;
            #else
            typedef ptrdiff_t difference_type;
            #endif
            OPENGM_TEST(it2 - it1 == j);
            OPENGM_TEST( int(it1 - it2) == int(-static_cast<difference_type>(j)));
         }
      }
   }

   void manipulationTest() {
      typedef AccessorIterator<VectorAccessor<size_t, false>, false> Iterator;

      std::vector<size_t> vec1(10);
      for(size_t j=0; j<vec1.size(); ++j) {
         vec1[j] = static_cast<size_t>(j);
      }
      VectorAccessor<size_t, false> accessor1(vec1);

      std::vector<size_t> vec2(10);
      VectorAccessor<size_t, false> accessor2(vec2);

      Iterator it1(accessor1);
      Iterator it2(accessor2);
      for(size_t j=0; j<vec1.size(); ++j) {
         *it2 = *it1;
         ++it1;
         ++it2;
         OPENGM_TEST(vec2[j] == vec1[j]);
      }
   }

   void run() {
      accessTest<true>();
      accessTest<false>();
      manipulationTest();
   }
};

} // namespace opengm

int main() {
   {
      std::cout << "AccessIterator test... " << std::flush;
      opengm::AccessorIteratorTest t; t.run();
      std::cout << "done." << std::endl;
   }

   return 0;
}

