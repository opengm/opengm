
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <opengm/datastructures/sparsemarray/sparsemarray.hxx>


#define OPERATOR_ARGS_D1  x0
#define OPERATOR_ARGS_D2  x0 ,x1
#define OPERATOR_ARGS_D3  x0 ,x1, x2
#define OPERATOR_ARGS_D4  x0 ,x1, x2, x3
#define OPERATOR_ARGS_D5  x0 ,x1, x2, x3, x4
#define OPERATOR_ARGS_D6  x0 ,x1, x2, x3, x4, x5
#define OPERATOR_ARGS_D7  x0 ,x1, x2, x3, x4, x5, x6
#define OPERATOR_ARGS_D8  x0 ,x1, x2, x3, x4, x5, x6, x7
#define OPERATOR_ARGS_D9  x0 ,x1, x2, x3, x4, x5, x6, x7, x8
#define OPERATOR_ARGS_D10 x0 ,x1, x2, x3, x4, x5, x6, x7, x8, x9

namespace opengm {

   template <class T_Container>
   class SparseMarrayTestSuit {
      typedef opengm::SparseMarray<T_Container> SparseMarrayWrapperType;
      typedef typename SparseMarrayWrapperType::ValueType ValueType;
      typedef typename SparseMarrayWrapperType::param_type param_type;
      typedef typename SparseMarrayWrapperType::key_type key_type;
      typedef typename SparseMarrayWrapperType::coordinate_type coordinate_type;
      typedef typename SparseMarrayWrapperType::reference_type reference_type;
      typedef typename SparseMarrayWrapperType::const_reference_type const_reference_type;
      std::vector<std::string> mErrors;

      std::vector<ValueType> mDataVector;

      SparseMarrayWrapperType & reference(SparseMarrayWrapperType & in)const {
         return in;
      };

      SparseMarrayWrapperType & constReference(SparseMarrayWrapperType & in)const {
         return in;
      };

      void test(bool testResult, std::string testName, std::string reasonFail = std::string("")) {
         if (testResult == true) {
            std::cout << "[o]" << " PASSED: " << testName << std::endl;
         } else {
            std::cout << "[-]" << " FAILED: " << testName << std::endl;
            if (reasonFail.size() != 0) {
               std::cout << "	Reason: " << reasonFail << std::endl;
            }
            throw std::logic_error("test failed.");
         }
      };

   public:

      SparseMarrayTestSuit(void) : mDataVector(1000) {
         for (size_t i = 0; i < 1000; i++) {
            mDataVector[i] = static_cast<ValueType> (i);
         }
      };

      void testCopy() {
         //	    T src;
         //	    U& dest;
         //	    opengm::CopyAssociativeContainer copyAssociativeContainer;
         //	    copyAssociativeContainer.copy(src, dest);
         //		if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testCopy (newsimpletest1) message=error message sample" << std::endl;
         //}
      };

      void testConstructor() {
         std::string testName = "TestConstructor1";
         size_t shape3dArray[] = {10, 20, 30};
         std::vector<size_t> shape3dVector(shape3dArray, shape3dArray + 3);
         coordinate_type shape3dArrayB[] = {10, 20, 30};
         std::vector<coordinate_type> shape3dVectorB(shape3dArrayB, shape3dArrayB + 3);

         SparseMarrayWrapperType a(shape3dArray, shape3dArray + 3, 0);
         SparseMarrayWrapperType av(shape3dVector.begin(), shape3dVector.end(), 0);
         SparseMarrayWrapperType b(shape3dArrayB, shape3dArrayB + 3, 0);
         SparseMarrayWrapperType bv(shape3dVectorB.begin(), shape3dVectorB.end(), 0);
         this->test(
            a.getDefaultValue() == static_cast<ValueType> (0) &&
            a.size(0) == 10 &&
            a.size(1) == 20 &&
            a.size(2) == 30 &&
            a.size() == 10 * 20 * 30 &&
            a.getDimension() == 3,
            testName);
         this->test(
            av.getDefaultValue() == static_cast<ValueType> (0) &&
            av.size(0) == 10 &&
            av.size(1) == 20 &&
            av.size(2) == 30 &&
            av.size() == 10 * 20 * 30 &&
            av.getDimension() == 3,
            testName);
         this->test(
            b.getDefaultValue() == static_cast<ValueType> (0) &&
            b.size(0) == 10 &&
            b.size(1) == 20 &&
            b.size(2) == 30 &&
            b.size() == 10 * 20 * 30 &&
            b.getDimension() == 3,
            testName);
         this->test(
            bv.getDefaultValue() == static_cast<ValueType> (0) &&
            bv.size(0) == 10 &&
            bv.size(1) == 20 &&
            bv.size(2) == 30 &&
            bv.size() == 10 * 20 * 30 &&
            bv.getDimension() == 3,
            testName);

      };

      void testConstructor2() {
         std::string testName = "TestConstrutor2";
         size_t shape3dArray[] = {1, 2, 3};
         ValueType data3dArray[] ={
            static_cast<ValueType> (0), static_cast<ValueType> (2),
            static_cast<ValueType> (4), static_cast<ValueType> (6),
            static_cast<ValueType> (8), static_cast<ValueType> (10)
         };
         SparseMarrayWrapperType a(shape3dArray, shape3dArray + 3, data3dArray, data3dArray + 6, 10);
         this->test(
            a.getDefaultValue() == static_cast<ValueType> (10) &&
            a.size(0) == 1 &&
            a.size(1) == 2 &&
            a.size(2) == 3 &&
            a.size() == 1 * 2 * 3 &&
            a.getDimension() == 3 &&
            a(0, 0, 0) == data3dArray[0] &&
            a(0, 1, 0) == data3dArray[1] &&
            a(0, 0, 1) == data3dArray[2] &&
            a(0, 1, 1) == data3dArray[3] &&
            a(0, 0, 2) == data3dArray[4] &&
            a(0, 1, 2) == data3dArray[5],
            testName);


         std::vector< std::pair< key_type, ValueType > > SparseData;
         std::pair<key_type, ValueType> element(static_cast<key_type> (2), static_cast<ValueType> (2));
         SparseData.push_back(element);
         SparseMarrayWrapperType b(shape3dArray, shape3dArray + 3, SparseData.begin(), SparseData.end(), 10);
         this->test(
            b.getDefaultValue() == static_cast<ValueType> (10) &&
            b.size(0) == 1 &&
            b.size(1) == 2 &&
            b.size(2) == 3 &&
            b.size() == 1 * 2 * 3 &&
            b.getDimension() == 3 &&
            b.sizeOfAssociativeContainer() == 1 &&
            b(0, 0, 1) == static_cast<ValueType> (2),
            testName + std::string(" c"));
      };

      void testConstructor5() {
         std::string testName = "TestConstrutor3";
         SparseMarrayWrapperType a;

         this->test(
            a.size() == 1 &&
            a.getDimension() == 0,
            testName);


      };

      void testConstructor3() {
         std::string testName = "TestCopyConstrutor";
         size_t shape3dArray[] = {1, 2, 3};
         ValueType data3dArray[] ={
            static_cast<ValueType> (0), static_cast<ValueType> (2),
            static_cast<ValueType> (4), static_cast<ValueType> (6),
            static_cast<ValueType> (8), static_cast<ValueType> (10)
         };
         SparseMarrayWrapperType b(shape3dArray, shape3dArray + 3, data3dArray, data3dArray + 6, 10);
         SparseMarrayWrapperType a = b;
         this->test(
            a.getDefaultValue() == static_cast<ValueType> (10) &&
            a.size(0) == 1 &&
            a.size(1) == 2 &&
            a.size(2) == 3 &&
            a.size() == 1 * 2 * 3 &&
            a.getDimension() == 3 &&
            a(0, 0, 0) == data3dArray[0] &&
            a(0, 1, 0) == data3dArray[1] &&
            a(0, 0, 1) == data3dArray[2] &&
            a(0, 1, 1) == data3dArray[3] &&
            a(0, 0, 2) == data3dArray[4] &&
            a(0, 1, 2) == data3dArray[5],
            testName);
      }

      void testConstructor4() {

      }

      void testClear() {
         std::string testName = "ClearTest";
         size_t shape3dArray[] = {1, 2, 3};
         ValueType data3dArray[] ={
            static_cast<ValueType> (0), static_cast<ValueType> (2),
            static_cast<ValueType> (4), static_cast<ValueType> (6),
            static_cast<ValueType> (8), static_cast<ValueType> (10)
         };
         SparseMarrayWrapperType a(shape3dArray, shape3dArray + 3, data3dArray, data3dArray + 6, 10);
         a.clear();
         this->test(
            a.sizeOfAssociativeContainer() == 0 &&
            a.getDimension() == 0 &&
            a.size() == 1,
            testName);
      }

      void testAccess1() {

         size_t n = 1;
         size_t shape[] = {12};
         SparseMarrayWrapperType a(shape, shape + 1, mDataVector.begin(), mDataVector.begin() + 12, 0);

#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D1
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;
         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {

            key = a.coordinateToKey(OPERATOR_ARGS_N);
            if (key != static_cast<key_type> (counter)) {
               passed = false;
               testName = testName + std::string("-CoordinatToKey");
               break;
            }

            coordinate = a.keyToCoordinate(key);
            if
               (
               coordinate.size() != n ||
               coordinate[0] != x0
               ) {
               passed = false;
               testName = testName + std::string("-KeyToCoordinatet");
               break;
            }
            if (a.coordinateToKey(coordinate.begin()) != key) {
               passed = false;
               testName = testName + std::string("-KeyToCoordinatet(Iterator)");
               break;
            }
            if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
               passed = false;
               testName = testName + std::string("-Read-Operator(Iterator)-Const");
               break;
            }
            if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
               passed = false;
               testName = testName + std::string("-Read-Operator(Iterator)");
               break;
            }
            if (this->constReference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
               passed = false;
               testName = testName + std::string("-Read-Operator()-Const");
               break;
            }
            if ((this->constReference(a))[x0] != mDataVector[counter]) {
               passed = false;
               testName = testName + std::string("-Read-Operator[]-Const");
               break;
            }
            if (this->reference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
               passed = false;
               testName = testName + std::string("-Read-Operator()");
               break;
            }

            if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
               passed = false;
               testName = testName + std::string("-ConstReference");
               break;
            }

            a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
            if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
               passed = false;
               testName = testName + std::string("-Write-Operator()");
               break;
            }

            a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
            if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
               passed = false;
               testName = testName + std::string("-Reference");
               break;
            }
            counter++;
         }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N
      }

      void testAccess2() {
         size_t n = 2;
         size_t shape[] = {10, 5};
         SparseMarrayWrapperType a(shape, shape + n, mDataVector.begin(), mDataVector.begin() + 50, 0);

#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D2

         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
            for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
               key = a.coordinateToKey(x0, x1);
               if (key != static_cast<key_type> (counter)) {
                  passed = false;
                  testName = testName + std::string("-CoordinatToKey");
                  break;
               }
               coordinate = a.keyToCoordinate(key);
               if
                  (
                  coordinate.size() != n ||
                  coordinate[1] != x1 ||
                  coordinate[0] != x0
                  ) {
                  passed = false;
                  testName = testName + std::string("-KeyToCoordinatet");
                  break;
               }
               if (a.coordinateToKey(coordinate.begin()) != key) {
                  passed = false;
                  testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                  break;
               }
               if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                  passed = false;
                  testName = testName + std::string("-Read-Operator(Iterator)-Const");
                  break;
               }
               if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                  passed = false;
                  testName = testName + std::string("-Read-Operator(Iterator)");
                  break;
               }
               if ((this->constReference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                  passed = false;
                  testName = testName + std::string("-Read-Operator()-Const");
                  break;
               }
               if (this->reference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                  passed = false;
                  testName = testName + std::string("-Read-Operator()");
                  break;
               }
               if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                  passed = false;
                  testName = testName + std::string("-ConstReference");
                  break;
               }
               a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
               if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                  passed = false;
                  testName = testName + std::string("-Write-Operator()");
                  break;
               }
               a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
               if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                  passed = false;
                  testName = testName + std::string("-Reference");
                  break;
               }
               counter++;
            }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N
      }

      void testAccess3() {
         size_t n = 3;
         size_t shape[] = {2, 10, 5};
         SparseMarrayWrapperType a(shape, shape + n, mDataVector.begin(), mDataVector.begin() + 2 * 10 * 5, 0);
#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D3
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x2 = 0; x2 < shape[2] && passed == true; x2++)
            for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
               for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
                  key = a.coordinateToKey(OPERATOR_ARGS_N);
                  if (key != static_cast<key_type> (counter)) {
                     passed = false;
                     testName = testName + std::string("-CoordinatToKey");
                     break;
                  }
                  coordinate = a.keyToCoordinate(key);
                  if
                     (
                     coordinate.size() != n ||
                     coordinate[2] != x2 ||
                     coordinate[1] != x1 ||
                     coordinate[0] != x0
                     ) {
                     passed = false;
                     testName = testName + std::string("-KeyToCoordinatet");
                     break;
                  }
                  if (a.coordinateToKey(coordinate.begin()) != key) {
                     passed = false;
                     testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                     break;
                  }
                  if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                     passed = false;
                     testName = testName + std::string("-Read-Operator(Iterator)-Const");
                     break;
                  }
                  if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                     passed = false;
                     testName = testName + std::string("-Read-Operator(Iterator)");
                     break;
                  }
                  if ((this->constReference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                     passed = false;
                     testName = testName + std::string("-Read-Operator()-Const");
                     break;
                  }
                  if (this->constReference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                     passed = false;
                     testName = testName + std::string("-Read-Operator()");
                     break;
                  }
                  if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                     passed = false;
                     testName = testName + std::string("-ConstReference");
                     break;
                  }
                  a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
                  if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                     passed = false;
                     testName = testName + std::string("-Write-Operator()");
                     break;
                  }
                  a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
                  if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                     passed = false;
                     testName = testName + std::string("-Reference");
                     break;
                  }
                  counter++;
               }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N
      }

      void testAccess4() {
         size_t n = 4;
         size_t shape[] = {3, 5, 1, 7};
         SparseMarrayWrapperType a(shape, shape + n, mDataVector.begin(), mDataVector.begin() + 3 * 5 * 1 * 7, 0);
#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D4
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x3 = 0; x3 < shape[3] && passed == true; x3++)
            for (size_t x2 = 0; x2 < shape[2] && passed == true; x2++)
               for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
                  for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
                     key = a.coordinateToKey(OPERATOR_ARGS_N);
                     if (key != static_cast<key_type> (counter)) {
                        passed = false;
                        testName = testName + std::string("-CoordinatToKey");
                        break;
                     }
                     coordinate = a.keyToCoordinate(key);
                     if
                        (
                        coordinate.size() != n ||
                        coordinate[3] != x3 ||
                        coordinate[2] != x2 ||
                        coordinate[1] != x1 ||
                        coordinate[0] != x0
                        ) {
                        passed = false;
                        testName = testName + std::string("-KeyToCoordinatet");
                        break;
                     }
                     if (a.coordinateToKey(coordinate.begin()) != key) {
                        passed = false;
                        testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                        break;
                     }
                     if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                        passed = false;
                        testName = testName + std::string("-Read-Operator(Iterator)-Const");
                        break;
                     }
                     if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                        passed = false;
                        testName = testName + std::string("-Read-Operator(Iterator)");
                        break;
                     }
                     if ((this->constReference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                        passed = false;
                        testName = testName + std::string("-Read-Operator()-Const");
                        break;
                     }
                     if (this->reference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                        passed = false;
                        testName = testName + std::string("-Read-Operator()");
                        break;
                     }
                     if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                        passed = false;
                        testName = testName + std::string("-ConstReference");
                        break;
                     }
                     a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
                     if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                        passed = false;
                        testName = testName + std::string("-Write-Operator()");
                        break;
                     }
                     a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
                     if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                        passed = false;
                        testName = testName + std::string("-Reference");
                        break;
                     }
                     counter++;
                  }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N

      }

      void testAccess5() {
         size_t n = 5;
         size_t shape[] = {4, 2, 1, 3, 2};
         SparseMarrayWrapperType a(shape, shape + 5, mDataVector.begin(), mDataVector.begin() + 4 * 2 * 1 * 3 * 2, 0);
#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D5
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x4 = 0; x4 < shape[4] && passed == true; x4++)
            for (size_t x3 = 0; x3 < shape[3] && passed == true; x3++)
               for (size_t x2 = 0; x2 < shape[2] && passed == true; x2++)
                  for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
                     for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
                        key = a.coordinateToKey(OPERATOR_ARGS_N);
                        if (key != static_cast<key_type> (counter)) {
                           passed = false;
                           testName = testName + std::string("-CoordinatToKey");
                           break;
                        }
                        coordinate = a.keyToCoordinate(key);
                        if
                           (
                           coordinate.size() != n ||
                           coordinate[4] != x4 ||
                           coordinate[3] != x3 ||
                           coordinate[2] != x2 ||
                           coordinate[1] != x1 ||
                           coordinate[0] != x0
                           ) {
                           passed = false;
                           testName = testName + std::string("-KeyToCoordinatet");
                           break;
                        }
                        if (a.coordinateToKey(coordinate.begin()) != key) {
                           passed = false;
                           testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                           break;
                        }
                        if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                           passed = false;
                           testName = testName + std::string("-Read-Operator(Iterator)-Const");
                           break;
                        }
                        if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                           passed = false;
                           testName = testName + std::string("-Read-Operator(Iterator)");
                           break;
                        }
    
                        if ((this->reference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                           passed = false;
                           testName = testName + std::string("-Read-Operator()-Const");
                           break;
                        }
                        if (this->constReference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                           passed = false;
                           testName = testName + std::string("-Read-Operator()");
                           break;
                        }
                        if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                           passed = false;
                           testName = testName + std::string("-ConstReference");
                           break;
                        }
                        a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
                        if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                           passed = false;
                           testName = testName + std::string("-Write-Operator()");
                           break;
                        }
                        a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
                        if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                           passed = false;
                           testName = testName + std::string("-Reference");
                           break;
                        }
                        counter++;
                     }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N
      }

      void testAccess6() {
         size_t n = 6;
         size_t shape[] = {6, 3, 1, 2, 2, 2};
         SparseMarrayWrapperType a(shape, shape + n, mDataVector.begin(), mDataVector.begin() + 6 * 3 * 1 * 2 * 2 * 2, 0);
#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D6
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x5 = 0; x5 < shape[5] && passed == true; x5++)
            for (size_t x4 = 0; x4 < shape[4] && passed == true; x4++)
               for (size_t x3 = 0; x3 < shape[3] && passed == true; x3++)
                  for (size_t x2 = 0; x2 < shape[2] && passed == true; x2++)
                     for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
                        for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
                           key = a.coordinateToKey(OPERATOR_ARGS_N);
                           if (key != static_cast<key_type> (counter)) {
                              passed = false;
                              testName = testName + std::string("-CoordinatToKey");
                              break;
                           }
                           coordinate = a.keyToCoordinate(key);
                           if
                              (
                              coordinate.size() != n ||
                              coordinate[5] != x5 ||
                              coordinate[4] != x4 ||
                              coordinate[3] != x3 ||
                              coordinate[2] != x2 ||
                              coordinate[1] != x1 ||
                              coordinate[0] != x0
                              ) {
                              passed = false;
                              testName = testName + std::string("-KeyToCoordinatet");
                              break;
                           }
                           if (a.coordinateToKey(coordinate.begin()) != key) {
                              passed = false;
                              testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                              break;
                           }
                           if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                              passed = false;
                              testName = testName + std::string("-Read-Operator(Iterator)-Const");
                              break;
                           }
                           if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                              passed = false;
                              testName = testName + std::string("-Read-Operator(Iterator)");
                              break;
                           }
                           if ((this->constReference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                              passed = false;
                              testName = testName + std::string("-Read-Operator()-Const");
                              break;
                           }
                           if (this->reference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                              passed = false;
                              testName = testName + std::string("-Read-Operator()");
                              break;
                           }
                           if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                              passed = false;
                              testName = testName + std::string("-ConstReference");
                              break;
                           }
                           a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
                           if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                              passed = false;
                              testName = testName + std::string("-Write-Operator()");
                              break;
                           }
                           a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
                           if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                              passed = false;
                              testName = testName + std::string("-Reference");
                              break;
                           }
                           counter++;
                        }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N



      }

      void testAccess7() {
         size_t n = 7;
         size_t shape[] = {3, 2, 1, 5, 1, 2, 3};
         SparseMarrayWrapperType a(shape, shape + n, mDataVector.begin(), mDataVector.begin() + 3 * 2 * 1 * 5 * 1 * 2 * 3, 0);
#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D7
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x6 = 0; x6 < shape[6] && passed == true; x6++)
            for (size_t x5 = 0; x5 < shape[5] && passed == true; x5++)
               for (size_t x4 = 0; x4 < shape[4] && passed == true; x4++)
                  for (size_t x3 = 0; x3 < shape[3] && passed == true; x3++)
                     for (size_t x2 = 0; x2 < shape[2] && passed == true; x2++)
                        for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
                           for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
                              key = a.coordinateToKey(OPERATOR_ARGS_N);
                              if (key != static_cast<key_type> (counter)) {
                                 passed = false;
                                 testName = testName + std::string("-CoordinatToKey");
                                 break;
                              }
                              coordinate = a.keyToCoordinate(key);
                              if
                                 (
                                 coordinate.size() != n ||
                                 coordinate[6] != x6 ||
                                 coordinate[5] != x5 ||
                                 coordinate[4] != x4 ||
                                 coordinate[3] != x3 ||
                                 coordinate[2] != x2 ||
                                 coordinate[1] != x1 ||
                                 coordinate[0] != x0
                                 ) {
                                 passed = false;
                                 testName = testName + std::string("-KeyToCoordinatet");
                                 break;
                              }
                              if (a.coordinateToKey(coordinate.begin()) != key) {
                                 passed = false;
                                 testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                                 break;
                              }
                              if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                                 passed = false;
                                 testName = testName + std::string("-Read-Operator(Iterator)-Const");
                                 break;
                              }
                              if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                                 passed = false;
                                 testName = testName + std::string("-Read-Operator(Iterator)");
                                 break;
                              }
                              if ((this->constReference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                 passed = false;
                                 testName = testName + std::string("-Read-Operator()-Const");
                                 break;
                              }
                              if (this->reference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                 passed = false;
                                 testName = testName + std::string("-Read-Operator()");
                                 break;
                              }
                              if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                 passed = false;
                                 testName = testName + std::string("-ConstReference");
                                 break;
                              }
                              a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
                              if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                                 passed = false;
                                 testName = testName + std::string("-Write-Operator()");
                                 break;
                              }
                              a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
                              if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                                 passed = false;
                                 testName = testName + std::string("-Reference");
                                 break;
                              }
                              counter++;
                           }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N

      }

      void testAccess8() {
         size_t n = 8;
         size_t shape[] = {1, 1, 10, 2, 2, 1, 1, 1};
         SparseMarrayWrapperType a(shape, shape + n, mDataVector.begin(), mDataVector.begin() + 40, 0);
#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D8
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x7 = 0; x7 < shape[7] && passed == true; x7++)
            for (size_t x6 = 0; x6 < shape[6] && passed == true; x6++)
               for (size_t x5 = 0; x5 < shape[5] && passed == true; x5++)
                  for (size_t x4 = 0; x4 < shape[4] && passed == true; x4++)
                     for (size_t x3 = 0; x3 < shape[3] && passed == true; x3++)
                        for (size_t x2 = 0; x2 < shape[2] && passed == true; x2++)
                           for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
                              for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
                                 key = a.coordinateToKey(OPERATOR_ARGS_N);
                                 if (key != static_cast<key_type> (counter)) {
                                    passed = false;
                                    testName = testName + std::string("-CoordinatToKey");
                                    break;
                                 }
                                 coordinate = a.keyToCoordinate(key);
                                 if
                                    (
                                    coordinate.size() != n ||
                                    coordinate[7] != x7 ||
                                    coordinate[6] != x6 ||
                                    coordinate[5] != x5 ||
                                    coordinate[4] != x4 ||
                                    coordinate[3] != x3 ||
                                    coordinate[2] != x2 ||
                                    coordinate[1] != x1 ||
                                    coordinate[0] != x0
                                    ) {
                                    passed = false;
                                    testName = testName + std::string("-KeyToCoordinatet");
                                    break;
                                 }
                                 if (a.coordinateToKey(coordinate.begin()) != key) {
                                    passed = false;
                                    testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                                    break;
                                 }
                                 if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                                    passed = false;
                                    testName = testName + std::string("-Read-Operator(Iterator)-Const");
                                    break;
                                 }
                                 if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                                    passed = false;
                                    testName = testName + std::string("-Read-Operator(Iterator)");
                                    break;
                                 }
                                 if ((this->constReference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                    passed = false;
                                    testName = testName + std::string("-Read-Operator()-Const");
                                    break;
                                 }
                                 if (this->reference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                    passed = false;
                                    testName = testName + std::string("-Read-Operator()");
                                    break;
                                 }
                                 if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                    passed = false;
                                    testName = testName + std::string("-ConstReference");
                                    break;
                                 }
                                 a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
                                 if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                                    passed = false;
                                    testName = testName + std::string("-Write-Operator()");
                                    break;
                                 }
                                 a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
                                 if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                                    passed = false;
                                    testName = testName + std::string("-Reference");
                                    break;
                                 }
                                 counter++;
                              }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N

      }

      void testAccess9() {
         size_t n = 9;
         size_t shape[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
         SparseMarrayWrapperType a(shape, shape + 9, mDataVector.begin(), mDataVector.begin() + 2 * 2 * 2 * 3 * 3 * 3, 0);
#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D9
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x8 = 0; x8 < shape[8] && passed == true; x8++)
            for (size_t x7 = 0; x7 < shape[7] && passed == true; x7++)
               for (size_t x6 = 0; x6 < shape[6] && passed == true; x6++)
                  for (size_t x5 = 0; x5 < shape[5] && passed == true; x5++)
                     for (size_t x4 = 0; x4 < shape[4] && passed == true; x4++)
                        for (size_t x3 = 0; x3 < shape[3] && passed == true; x3++)
                           for (size_t x2 = 0; x2 < shape[2] && passed == true; x2++)
                              for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
                                 for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
                                    key = a.coordinateToKey(OPERATOR_ARGS_N);
                                    if (key != static_cast<key_type> (counter)) {
                                       passed = false;
                                       testName = testName + std::string("-CoordinatToKey");
                                       break;
                                    }
                                    coordinate = a.keyToCoordinate(key);
                                    if
                                       (
                                       coordinate.size() != n ||
                                       coordinate[8] != x8 ||
                                       coordinate[7] != x7 ||
                                       coordinate[6] != x6 ||
                                       coordinate[5] != x5 ||
                                       coordinate[4] != x4 ||
                                       coordinate[3] != x3 ||
                                       coordinate[2] != x2 ||
                                       coordinate[1] != x1 ||
                                       coordinate[0] != x0
                                       ) {
                                       passed = false;
                                       testName = testName + std::string("-KeyToCoordinatet");
                                       break;
                                    }
                                    if (a.coordinateToKey(coordinate.begin()) != key) {
                                       passed = false;
                                       testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                                       break;
                                    }
                                    if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                                       passed = false;
                                       testName = testName + std::string("-Read-Operator(Iterator)-Const");
                                       break;
                                    }
                                    if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                                       passed = false;
                                       testName = testName + std::string("-Read-Operator(Iterator)");
                                       break;
                                    }
                                    if ((this->constReference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                       passed = false;
                                       testName = testName + std::string("-Read-Operator()-Const");
                                       break;
                                    }
                                    if (this->reference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                       passed = false;
                                       testName = testName + std::string("-Read-Operator()");
                                       break;
                                    }
                                    if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                       passed = false;
                                       testName = testName + std::string("-ConstReference");
                                       break;
                                    }
                                    a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
                                    if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                                       passed = false;
                                       testName = testName + std::string("-Write-Operator()");
                                       break;
                                    }
                                    a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
                                    if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                                       passed = false;
                                       testName = testName + std::string("-Reference");
                                       break;
                                    }
                                    counter++;
                                 }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N

      }

      void testAccess10() {
         size_t n = 10;
         size_t shape[] = {1, 1, 1, 1, 2, 1, 4, 1, 1, 10};
         SparseMarrayWrapperType a(shape, shape + n, mDataVector.begin(), mDataVector.begin() + 80, 0);
#ifdef OPERATOR_ARGS_N
#undef OPERATOR_ARGS_N
#endif
#define OPERATOR_ARGS_N  OPERATOR_ARGS_D10
         std::stringstream ss(std::stringstream::in | std::stringstream::out);
         ss << n << "D";
         std::string dimString = ss.str();
         std::string testName = std::string("TestAccess-") + dimString;

         //testloop
         bool passed = true;
         size_t counter = 0;
         std::vector<coordinate_type> coordinate;
         key_type key;
         for (size_t x9 = 0; x9 < shape[9] && passed == true; x9++)
            for (size_t x8 = 0; x8 < shape[8] && passed == true; x8++)
               for (size_t x7 = 0; x7 < shape[7] && passed == true; x7++)
                  for (size_t x6 = 0; x6 < shape[6] && passed == true; x6++)
                     for (size_t x5 = 0; x5 < shape[5] && passed == true; x5++)
                        for (size_t x4 = 0; x4 < shape[4] && passed == true; x4++)
                           for (size_t x3 = 0; x3 < shape[3] && passed == true; x3++)
                              for (size_t x2 = 0; x2 < shape[2] && passed == true; x2++)
                                 for (size_t x1 = 0; x1 < shape[1] && passed == true; x1++)
                                    for (size_t x0 = 0; x0 < shape[0] && passed == true; x0++) {
                                       key = a.coordinateToKey(OPERATOR_ARGS_N);
                                       if (key != static_cast<key_type> (counter)) {
                                          passed = false;
                                          testName = testName + std::string("-CoordinatToKey");
                                          break;
                                       }
                                       coordinate = a.keyToCoordinate(key);
                                       if
                                          (
                                          coordinate.size() != n ||
                                          coordinate[9] != x9 ||
                                          coordinate[8] != x8 ||
                                          coordinate[7] != x7 ||
                                          coordinate[6] != x6 ||
                                          coordinate[5] != x5 ||
                                          coordinate[4] != x4 ||
                                          coordinate[3] != x3 ||
                                          coordinate[2] != x2 ||
                                          coordinate[1] != x1 ||
                                          coordinate[0] != x0
                                          ) {
                                          passed = false;
                                          testName = testName + std::string("-KeyToCoordinatet");
                                          break;
                                       }
                                       if (a.coordinateToKey(coordinate.begin()) != key) {
                                          passed = false;
                                          testName = testName + std::string("-KeyToCoordinatet(Iterator)");
                                          break;
                                       }
                                       if (this->constReference(a)(coordinate.begin()) != mDataVector[counter]) {
                                          passed = false;
                                          testName = testName + std::string("-Read-Operator(Iterator)-Const");
                                          break;
                                       }
                                       if (this->reference(a)(coordinate.begin()) != mDataVector[counter]) {
                                          passed = false;
                                          testName = testName + std::string("-Read-Operator(Iterator)");
                                          break;
                                       }
                                       if ((this->constReference(a))(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                          passed = false;
                                          testName = testName + std::string("-Read-Operator()-Const");
                                          break;
                                       }
                                       if (this->reference(a)(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                          passed = false;
                                          testName = testName + std::string("-Read-Operator()");
                                          break;
                                       }
                                       if (a.const_reference(OPERATOR_ARGS_N) != mDataVector[counter]) {
                                          passed = false;
                                          testName = testName + std::string("-ConstReference");
                                          break;
                                       }
                                       a(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 1);
                                       if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 1)) {
                                          passed = false;
                                          testName = testName + std::string("-Write-Operator()");
                                          break;
                                       }
                                       a.reference(OPERATOR_ARGS_N) = static_cast<ValueType> (counter + 2);
                                       if ((this->constReference(a))(OPERATOR_ARGS_N) != static_cast<ValueType> (counter + 2)) {
                                          passed = false;
                                          testName = testName + std::string("-Reference");
                                          break;
                                       }
                                       counter++;
                                    }
         this->test(passed, testName);

#undef OPERATOR_ARGS_N
      }

      void testAccessN() {
         std::vector<size_t> highDimShape(100);
         for (size_t i = 0; i < 100; i++) {
            highDimShape[i] = i;
         }
         //                    SparseMarrayWrapperType a(highDimShape.begin(),highDimShape.begin()+1,dataVector.begin(),dataVector.begin()+,0);
      }

      void testGetDefaultValue() {
         std::string testName = "TestGetDefaultValue";
         size_t shape[] = {10};
         SparseMarrayWrapperType a(shape, shape + 1, 10);
         SparseMarrayWrapperType b(shape, shape + 1, mDataVector.begin(), mDataVector.begin() + 10, 10);
         test
            (
            a.getDefaultValue() == static_cast<ValueType> (10) &&
            b.getDefaultValue() == static_cast<ValueType> (10),
            testName
            );
      }

      void testGetShape() {
         std::string testName = "TestGetShape";
         typename SparseMarrayWrapperType::coordinate_tuple aShape;
         size_t shape[] = {10, 20, 30};
         SparseMarrayWrapperType a(shape, shape + 3, 10);
         a.getShape(aShape);
         this->test
            (
            aShape[0] == static_cast<coordinate_type> (shape[0]) &&
            aShape[1] == static_cast<coordinate_type> (shape[1]) &&
            aShape[2] == static_cast<coordinate_type> (shape[2]) &&
            aShape[0] == static_cast<coordinate_type> (a.size(0)) &&
            aShape[1] == static_cast<coordinate_type> (a.size(1)) &&
            aShape[2] == static_cast<coordinate_type> (a.size(2)),
            testName
            );
      }

      void testGetShape2() {
         std::string testName = "TestGetShape2";
         typename SparseMarrayWrapperType::coordinate_tuple aShape;
         size_t shape[] = {10, 20, 30};
         SparseMarrayWrapperType a(shape, shape + 3, 10);
         aShape = a.getShape();
         test
            (
            aShape[0] == static_cast<coordinate_type> (shape[0]) &&
            aShape[1] == static_cast<coordinate_type> (shape[1]) &&
            aShape[2] == static_cast<coordinate_type> (shape[2]) &&
            aShape[0] == static_cast<coordinate_type> (a.size(0)) &&
            aShape[1] == static_cast<coordinate_type> (a.size(1)) &&
            aShape[2] == static_cast<coordinate_type> (a.size(2)),
            testName
            );
      }

      void testInit() {
         std::string testName = "TestInit";
         size_t shape[] = {4, 7, 11};
         SparseMarrayWrapperType a;
         a.init(shape, shape + 3, 1);
         test
            (
            a.size(0) == static_cast<size_t> (4) &&
            a.size(1) == static_cast<size_t> (7) &&
            a.size(2) == static_cast<size_t> (11) &&
            a.getDimension() == 3 &&
            a.getDefaultValue() == static_cast<ValueType> (1),
            testName
            );
      }

      void testKeyToCoordinate() {
         //	    key_type key;
         //	    coordinate_tuple& coordinate;
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    sparseMarrayWrapper.keyToCoordinate(key, coordinate);
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testKeyToCoordinate (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testKeyToCoordinate2() {
         //	    key_type key;
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    coordinate_tuple result = sparseMarrayWrapper.keyToCoordinate(key);
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testKeyToCoordinate2 (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testReference11() {
         //	    CoordinateIter coordinateBegin;
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    SparseMarrayWrapper<T_AssociativeContainer,T_Coordinate,T_ComparePolicy,T_RedundancyPolicy>::reference_type result = sparseMarrayWrapper.reference(coordinateBegin);
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testReference11 (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testReshape() {
         std::string testName = "TestReshape";
         size_t shapea[] = {10, 3, 2};
         size_t shapeb[] = {10 * 3 * 2};
         size_t shapec[] = {10 * 3, 2};
         size_t shaped[] = {2, 5, 3, 2};
         SparseMarrayWrapperType a1(shapea, shapea + 3, mDataVector.begin(), mDataVector.begin() + 60, 1);
         SparseMarrayWrapperType a2(shapea, shapea + 3, mDataVector.begin(), mDataVector.begin() + 60, 1);
         SparseMarrayWrapperType a3(shapea, shapea + 3, mDataVector.begin(), mDataVector.begin() + 60, 1);
         SparseMarrayWrapperType a4(shapeb, shapeb + 1, mDataVector.begin(), mDataVector.begin() + 60, 1);

         SparseMarrayWrapperType b(shapeb, shapeb + 1, mDataVector.begin(), mDataVector.begin() + 60, 1);
         SparseMarrayWrapperType c(shapec, shapec + 2, mDataVector.begin(), mDataVector.begin() + 60, 1);
         SparseMarrayWrapperType d(shaped, shaped + 4, mDataVector.begin(), mDataVector.begin() + 60, 1);

         a1.reshape(shapeb, shapeb + 1);
         a2.reshape(shapec, shapec + 2);
         a3.reshape(shaped, shaped + 4);
         a4.reshape(shaped, shaped + 4);

         bool passed = true;
         //size_t counter=0;
         std::vector<coordinate_type> coordinate;
         //key_type key;

         //a1 b
         for (size_t x0 = 0; x0 < shapeb[0] && passed == true; x0++) {
            if (a1.const_reference(x0) != b.const_reference(x0)) {
               passed = false;
               break;
            }
         }
         //a2 c
         for (size_t x1 = 0; x1 < shapec[1] && passed == true; x1++)
            for (size_t x0 = 0; x0 < shapec[0] && passed == true; x0++) {
               if (a2.const_reference(x0, x1) != c.const_reference(x0, x1)) {
                  passed = false;
                  break;
               }
            }
         //a3 d
         //a4 d
         for (size_t x3 = 0; x3 < shaped[3] && passed == true; x3++)
            for (size_t x2 = 0; x2 < shaped[2] && passed == true; x2++)
               for (size_t x1 = 0; x1 < shaped[1] && passed == true; x1++)
                  for (size_t x0 = 0; x0 < shaped[0] && passed == true; x0++) {
                     if
                        (
                        a3.const_reference(x0, x1, x2, x3) != d.const_reference(x0, x1, x2, x3) ||
                        a4.const_reference(x0, x1, x2, x3) != d.const_reference(x0, x1, x2, x3)
                        ) {
                        passed = false;
                        break;
                     }
                  }

         test
            (
            a1.getDimension() == 1 &&
            a2.getDimension() == 2 &&
            a3.getDimension() == 4 &&
            a4.getDimension() == 4 &&
            passed,
            testName
            );

      }

      void testSetDefaultValue() {
         std::string testName = "TestSetDefaultValue";
         size_t shape[] = {10};
         SparseMarrayWrapperType a(shape, shape + 1, 10);
         SparseMarrayWrapperType b(shape, shape + 1, mDataVector.begin(), mDataVector.begin() + 10, 10);
         a.setDefaultValue(15);
         b.setDefaultValue(15);
         test
            (
            a.getDefaultValue() == static_cast<ValueType> (15) &&
            b.getDefaultValue() == static_cast<ValueType> (15),
            testName
            );
      }

      void testSize() {
         std::string testName = "TestSetSize";
         size_t shape[] = {10, 20, 30};
         SparseMarrayWrapperType a(shape, shape + 3, 10);
         a.setDefaultValue(15);
         test
            (
            a.size(0) == 10 &&
            a.size(1) == 20 &&
            a.size(2) == 30,
            testName
            );
      }

      void testSize2() {
         std::string testName = "TestSetSize2";
         size_t shape[] = {10, 20, 30};
         SparseMarrayWrapperType a(shape, shape + 3, 10);
         a.setDefaultValue(15);
         test
            (
            a.size() == 10 * 20 * 30,
            testName
            );
      }

      void testGetDimension() {
         std::string testName = "TestGetDimension";
         size_t shape[] = {10, 20, 30};
         SparseMarrayWrapperType a(shape, shape + 1, 10);
         SparseMarrayWrapperType b(shape, shape + 2, 10);
         SparseMarrayWrapperType c(shape, shape + 3, 10);
         a.setDefaultValue(15);
         test
            (
            a.getDimension() == 1 &&
            b.getDimension() == 2 &&
            c.getDimension() == 3,
            testName
            );
      }

      void testAssigned_assoziative_begin() {
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    assigned_assoziative_iterator result = sparseMarrayWrapper.assigned_assoziative_begin();
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testAssigned_assoziative_begin (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testAssigned_assoziative_begin2() {
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    const_assigned_assoziative_iterator result = sparseMarrayWrapper.assigned_assoziative_begin();
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testAssigned_assoziative_begin2 (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testAssigned_assoziative_end() {
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    assigned_assoziative_iterator result = sparseMarrayWrapper.assigned_assoziative_end();
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testAssigned_assoziative_end (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testAssigned_assoziative_end2() {
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    const_assigned_assoziative_iterator result = sparseMarrayWrapper.assigned_assoziative_end();
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testAssigned_assoziative_end2 (newsimpletest1) message=error messa/usr/include/c++/4.4/bits/stl_iterator_base_types.h:127: error: ‘long unsigned int’ is not a class, struct, or union typege sample" << std::endl;
         //	    }
      }

      void testBegin() {
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    const_iterator result = sparseMarrayWrapper.begin();
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testBegin (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testBegin2() {
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    iterator result = sparseMarrayWrapper.begin();
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testBegin2 (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testEnd() {
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    const_iterator result = sparseMarrayWrapper.end();
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testEnd (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testEnd2() {
         //	    opengm::SparseMarrayWrapper sparseMarrayWrapper;
         //	    iterator result = sparseMarrayWrapper.end();
         //	    if(true /*check result*/) {
         //		std::cout << "%TEST_FAILED% time=0 testname=testEnd2 (newsimpletest1) message=error message sample" << std::endl;
         //	    }
      }

      void testComparePoliciy() {
      };

      void testCompareValues() {
      };

      void testAssignmentOperator() {
      };

      void testArithmeticOperator() {
      };

      void testCombineFunction() {
      };

      void testForEach() {
      };

   };

   template <class T_Container>
   void testSparseMarrayAllFuctions(void) {
      SparseMarrayTestSuit <T_Container> testSuite;
      testSuite.testConstructor();
      testSuite.testConstructor2();
      testSuite.testConstructor3();
      testSuite.testConstructor4();
      testSuite.testConstructor5();
      testSuite.testClear();
      testSuite.testGetDefaultValue();
      testSuite.testSetDefaultValue();
      testSuite.testGetDimension();
      testSuite.testGetShape();
      testSuite.testGetShape2();
      testSuite.testInit();
      testSuite.testKeyToCoordinate();
      testSuite.testKeyToCoordinate2();
      testSuite.testReshape();
      testSuite.testSize();
      testSuite.testSize2();
      testSuite.testCopy();
      testSuite.testAccess1();
      testSuite.testAccess2();
      testSuite.testAccess3();
      testSuite.testAccess4();
      testSuite.testAccess5();
      testSuite.testAccess6();
      testSuite.testAccess7();
      testSuite.testAccess8();
      testSuite.testAccess9();
      testSuite.testAccess10();
      testSuite.testAccessN();
      testSuite.testAssigned_assoziative_begin();
      testSuite.testAssigned_assoziative_end();
      testSuite.testAssigned_assoziative_begin2();
      testSuite.testAssigned_assoziative_end2();
      testSuite.testBegin();
      testSuite.testEnd();
      testSuite.testBegin2();
      testSuite.testEnd2();
      testSuite.testComparePoliciy();
      testSuite.testCompareValues();
      testSuite.testAssignmentOperator();
      testSuite.testArithmeticOperator();
      testSuite.testCombineFunction();
      testSuite.testForEach();
   }

   void testSparseMarray(void) {
      opengm::testSparseMarrayAllFuctions
      <
      std::map<size_t, float>
      >();

      opengm::testSparseMarrayAllFuctions
      <
      std::map<unsigned short, float>
      >();
   }
}

int main(int argc, char** argv) {
   opengm::testSparseMarray();
   return (EXIT_SUCCESS);
}

#undef OPERATOR_ARGS_D1
#undef OPERATOR_ARGS_D2
#undef OPERATOR_ARGS_D3
#undef OPERATOR_ARGS_D4
#undef OPERATOR_ARGS_D5
#undef OPERATOR_ARGS_D6
#undef OPERATOR_ARGS_D7
#undef OPERATOR_ARGS_D8
#undef OPERATOR_ARGS_D9
#undef OPERATOR_ARGS_D10
