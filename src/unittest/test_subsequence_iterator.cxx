#include <iostream>
#include <vector>
#include <deque>

#include <opengm/unittests/test.hxx>

#include <opengm/utilities/subsequence_iterator.hxx>

template<class SEQUENCE_CONTAINER_TYPE, class SUBSEQUENCE_INDICES_CONTAINER_TYPE>
void testSubsequenceIterator();

int main(int argc, char** argv){
   std::cout << "Subsequence Iterator test... " << std::endl;

   testSubsequenceIterator<std::vector<double>, std::vector<size_t> >();
   testSubsequenceIterator<std::vector<int>, std::vector<size_t> >();
   testSubsequenceIterator<std::vector<double>, std::vector<int> >();
   testSubsequenceIterator<std::vector<int>, std::vector<int> >();

   testSubsequenceIterator<std::deque<double>, std::deque<size_t> >();
   testSubsequenceIterator<std::deque<int>, std::deque<size_t> >();
   testSubsequenceIterator<std::deque<double>, std::deque<int> >();
   testSubsequenceIterator<std::deque<int>, std::deque<int> >();

   testSubsequenceIterator<std::vector<double>, std::deque<size_t> >();
   testSubsequenceIterator<std::vector<int>, std::deque<size_t> >();
   testSubsequenceIterator<std::vector<double>, std::deque<int> >();
   testSubsequenceIterator<std::vector<int>, std::deque<int> >();

   testSubsequenceIterator<std::deque<double>, std::vector<size_t> >();
   testSubsequenceIterator<std::deque<int>, std::vector<size_t> >();
   testSubsequenceIterator<std::deque<double>, std::vector<int> >();
   testSubsequenceIterator<std::deque<int>, std::vector<int> >();

   std::cout << "done..." << std::endl;
   return 0;
}

template<class SEQUENCE_CONTAINER_TYPE, class SUBSEQUENCE_INDICES_CONTAINER_TYPE>
void testSubsequenceIterator() {
   typedef SEQUENCE_CONTAINER_TYPE                                                                             SequenceContainerType;
   typedef SUBSEQUENCE_INDICES_CONTAINER_TYPE                                                                  SubsequenceIndicesContainerType;
   typedef typename SequenceContainerType::const_iterator                                                      SequenceContainerIteratorType;
   typedef typename SubsequenceIndicesContainerType::const_iterator                                            SubsequenceIndicesContainerIteratorType;
   typedef opengm::SubsequenceIterator<SequenceContainerIteratorType, SubsequenceIndicesContainerIteratorType> SubsequenceIteratorType;
   typedef typename SequenceContainerType::value_type                                                          ValueType;
   typedef typename SubsequenceIndicesContainerType::value_type                                                IndexType;
   typedef typename SubsequenceIteratorType::difference_type                                                   DifferenceType;

   const size_t sequenceLength = 10;
   const size_t subsequenceLength = 5;
   SequenceContainerType sequence(sequenceLength);
   for(size_t i = 0; i < sequenceLength; ++i) {
      sequence[i] = static_cast<ValueType>(i);
   }
   SubsequenceIndicesContainerType subsequenceIndices(subsequenceLength);
   for(size_t i = 0; i < subsequenceLength; ++i) {
      subsequenceIndices[i] = static_cast<IndexType>(i) * IndexType(2);
   }

   // construction
   const SubsequenceIteratorType emptySubsequenceIterator;
   const SubsequenceIteratorType subsequenceBegin1(sequence.begin(), subsequenceIndices.begin());
   const SubsequenceIteratorType subsequenceEnd1(sequence.begin(), subsequenceIndices.begin(), subsequenceLength);
   SubsequenceIteratorType subsequenceBegin2(subsequenceBegin1);
   SubsequenceIteratorType subsequenceEnd2;
   SubsequenceIteratorType subsequenceEnd3;

   // assignment
   subsequenceEnd2 = subsequenceEnd1;
   subsequenceEnd3 = subsequenceEnd1;

   // increment
   const SubsequenceIteratorType firstElement1(subsequenceBegin2++);
   const SubsequenceIteratorType thirdElement1(++subsequenceBegin2);
   const SubsequenceIteratorType fourthElement1(subsequenceBegin2 + 1);
   const SubsequenceIteratorType lastElement1(subsequenceBegin2 += 2);

   // decrement
   const SubsequenceIteratorType lastElement2(--subsequenceEnd2);
   const SubsequenceIteratorType lastElement3(subsequenceEnd2--);
   const SubsequenceIteratorType thirdElement2(subsequenceEnd2 - 1);
   const SubsequenceIteratorType secondElement1(subsequenceEnd2 -= 2);

   // operator*
   OPENGM_TEST_EQUAL(*firstElement1, sequence[subsequenceIndices[0]]);
   OPENGM_TEST_EQUAL(&(*firstElement1), &(sequence[subsequenceIndices[0]]));

   OPENGM_TEST_EQUAL(*thirdElement1, sequence[subsequenceIndices[2]]);
   OPENGM_TEST_EQUAL(&(*thirdElement1), &(sequence[subsequenceIndices[2]]));

   OPENGM_TEST_EQUAL(*fourthElement1, sequence[subsequenceIndices[3]]);
   OPENGM_TEST_EQUAL(&(*fourthElement1), &(sequence[subsequenceIndices[3]]));

   OPENGM_TEST_EQUAL(*lastElement1, sequence[subsequenceIndices[4]]);
   OPENGM_TEST_EQUAL(&(*lastElement1), &(sequence[subsequenceIndices[4]]));

   OPENGM_TEST_EQUAL(*lastElement2, sequence[subsequenceIndices[4]]);
   OPENGM_TEST_EQUAL(&(*lastElement2), &(sequence[subsequenceIndices[4]]));

   OPENGM_TEST_EQUAL(*lastElement3, sequence[subsequenceIndices[4]]);
   OPENGM_TEST_EQUAL(&(*lastElement3), &(sequence[subsequenceIndices[4]]));

   OPENGM_TEST_EQUAL(*thirdElement2, sequence[subsequenceIndices[2]]);
   OPENGM_TEST_EQUAL(&(*thirdElement2), &(sequence[subsequenceIndices[2]]));

   OPENGM_TEST_EQUAL(*secondElement1, sequence[subsequenceIndices[1]]);
   OPENGM_TEST_EQUAL(&(*secondElement1), &(sequence[subsequenceIndices[1]]));

   // operator->
   OPENGM_TEST_EQUAL(firstElement1.operator ->(), &(sequence[subsequenceIndices[0]]));
   OPENGM_TEST_EQUAL(thirdElement1.operator ->(), &(sequence[subsequenceIndices[2]]));
   OPENGM_TEST_EQUAL(fourthElement1.operator ->(), &(sequence[subsequenceIndices[3]]));
   OPENGM_TEST_EQUAL(lastElement1.operator ->(), &(sequence[subsequenceIndices[4]]));

   OPENGM_TEST_EQUAL(lastElement2.operator ->(), &(sequence[subsequenceIndices[4]]));
   OPENGM_TEST_EQUAL(lastElement3.operator ->(), &(sequence[subsequenceIndices[4]]));
   OPENGM_TEST_EQUAL(thirdElement2.operator ->(), &(sequence[subsequenceIndices[2]]));
   OPENGM_TEST_EQUAL(secondElement1.operator ->(), &(sequence[subsequenceIndices[1]]));

   // operator[]
   for(size_t i = 0; i < subsequenceLength; ++i) {
      OPENGM_TEST_EQUAL(subsequenceBegin1[i], sequence[subsequenceIndices[i]]);
      OPENGM_TEST_EQUAL(&(subsequenceBegin1[i]), &(sequence[subsequenceIndices[i]]));
   }

   // operator==
   OPENGM_TEST(subsequenceBegin1 == subsequenceBegin1);
   OPENGM_TEST(!(subsequenceBegin1 == subsequenceEnd1));
   OPENGM_TEST(subsequenceBegin1 == firstElement1);
   OPENGM_TEST(!(subsequenceBegin1 == thirdElement1));
   OPENGM_TEST(subsequenceEnd1 == subsequenceEnd3);
   OPENGM_TEST(lastElement1 == lastElement2);
   OPENGM_TEST(lastElement2 == lastElement3);

   // operator==
   OPENGM_TEST(!(subsequenceBegin1 != subsequenceBegin1));
   OPENGM_TEST(subsequenceBegin1 != subsequenceEnd1);
   OPENGM_TEST(!(subsequenceBegin1 != firstElement1));
   OPENGM_TEST(subsequenceBegin1 != thirdElement1);
   OPENGM_TEST(!(subsequenceEnd1 != subsequenceEnd3));
   OPENGM_TEST(!(lastElement1 != lastElement2));
   OPENGM_TEST(!(lastElement2 != lastElement3));

   // operator<
   OPENGM_TEST(subsequenceBegin1 < subsequenceEnd1);
   OPENGM_TEST(!(subsequenceBegin1 < subsequenceBegin1));
   OPENGM_TEST(firstElement1 < secondElement1);
   OPENGM_TEST(firstElement1 < thirdElement1);
   OPENGM_TEST(firstElement1 < fourthElement1);
   OPENGM_TEST(firstElement1 < lastElement1);
   OPENGM_TEST(!(secondElement1 < firstElement1));
   OPENGM_TEST(!(thirdElement1 < firstElement1));
   OPENGM_TEST(!(fourthElement1 < firstElement1));
   OPENGM_TEST(!(lastElement1 < firstElement1));
   OPENGM_TEST(!(lastElement1 < lastElement2));
   OPENGM_TEST(!(lastElement2 < lastElement1));

   // operator>
   OPENGM_TEST(!(subsequenceBegin1 > subsequenceEnd1));
   OPENGM_TEST(!(subsequenceBegin1 > subsequenceBegin1));
   OPENGM_TEST(!(firstElement1 > secondElement1));
   OPENGM_TEST(!(firstElement1 > thirdElement1));
   OPENGM_TEST(!(firstElement1 > fourthElement1));
   OPENGM_TEST(!(firstElement1 > lastElement1));
   OPENGM_TEST(secondElement1 > firstElement1);
   OPENGM_TEST(thirdElement1 > firstElement1);
   OPENGM_TEST(fourthElement1 > firstElement1);
   OPENGM_TEST(lastElement1 > firstElement1);
   OPENGM_TEST(!(lastElement1 > lastElement2));
   OPENGM_TEST(!(lastElement2 > lastElement1));

   // operator<=
   OPENGM_TEST(subsequenceBegin1 <= subsequenceEnd1);
   OPENGM_TEST(subsequenceBegin1 <= subsequenceBegin1);
   OPENGM_TEST(firstElement1 <= secondElement1);
   OPENGM_TEST(firstElement1 <= thirdElement1);
   OPENGM_TEST(firstElement1 <= fourthElement1);
   OPENGM_TEST(firstElement1 <= lastElement1);
   OPENGM_TEST(!(secondElement1 <= firstElement1));
   OPENGM_TEST(!(thirdElement1 <= firstElement1));
   OPENGM_TEST(!(fourthElement1 <= firstElement1));
   OPENGM_TEST(!(lastElement1 <= firstElement1));
   OPENGM_TEST(lastElement1 <= lastElement2);
   OPENGM_TEST(lastElement2 <= lastElement1);

   // operator>=
   OPENGM_TEST(!(subsequenceBegin1 >= subsequenceEnd1));
   OPENGM_TEST(subsequenceBegin1 >= subsequenceBegin1);
   OPENGM_TEST(!(firstElement1 >= secondElement1));
   OPENGM_TEST(!(firstElement1 >= thirdElement1));
   OPENGM_TEST(!(firstElement1 >= fourthElement1));
   OPENGM_TEST(!(firstElement1 >= lastElement1));
   OPENGM_TEST(secondElement1 >= firstElement1);
   OPENGM_TEST(thirdElement1 >= firstElement1);
   OPENGM_TEST(fourthElement1 >= firstElement1);
   OPENGM_TEST(lastElement1 >= firstElement1);
   OPENGM_TEST(lastElement1 >= lastElement2);
   OPENGM_TEST(lastElement2 >= lastElement1);

   // operator-
   OPENGM_TEST_EQUAL(subsequenceBegin1 - subsequenceBegin1, DifferenceType(0));
   OPENGM_TEST_EQUAL(subsequenceBegin1 - subsequenceEnd1, -static_cast<DifferenceType>(subsequenceLength));
   OPENGM_TEST_EQUAL(subsequenceEnd1 - subsequenceBegin1, DifferenceType(subsequenceLength));
   OPENGM_TEST_EQUAL(firstElement1 - secondElement1, DifferenceType(-1));
   OPENGM_TEST_EQUAL(secondElement1 - firstElement1, DifferenceType(1));
}
