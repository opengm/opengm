//#include <vector>
#include <iostream>
#include <limits>
#include <time.h>

#include <opengm/opengm.hxx>
#include <opengm/utilities/random2.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/datastructures/marray/marray.hxx>

/***************************
 * class declaration... *
 ***************************/
template <class RANDOMGENERATOR>
class TestRand {
protected:
   static const size_t numIter = 10;
   static const size_t vecSizeBase = 5000;
   static const size_t vecSizeTestInRange = vecSizeBase ;
   static const size_t vecSizeTestSeed = vecSizeBase;
   static const size_t vecSizeTestUniformDistribution = vecSizeBase * 20;
   static const size_t vecSizeTestHistogram = vecSizeBase * 20;
   static const size_t vecSizeTestNormalDistribution = vecSizeBase;
   static const size_t vecSizeTestMultivariateNormalDistribution = vecSizeBase;
   static const size_t vecSizeTestPermutation = vecSizeBase / 5;
   static const size_t numBins = 10;
   static const size_t expectedBinSizeTestUniformDistribution = vecSizeTestUniformDistribution / numBins;
   static const size_t allowableToleranceTestUniformDistribution = expectedBinSizeTestUniformDistribution / 10;
   //chiSquareValues for different probability values (2 degrees of freedom):
   // probability = 0.001 --> chiSquareValue = 13.82
   // probability = 0.01 --> chiSquareValue = 9.21
   // probability = 0.05 --> chiSquareValue = 5.99
   // probability = 0.1 --> chiSquareValue = 4.60
   // probability = 0.2 --> chiSquareValue = 3.22
   //static const double chiSquareValue = 9.21;
   static const size_t minBase = 100000;
   static const size_t maxBase = 100000;
   static const size_t AllowedSignificantDeviations = (numIter )/3;
   static size_t floatCountSignificantDeviationsUniformDistribution;
   static size_t doubleCountSignificantDeviationsUniformDistribution;
   static size_t intCountSignificantDeviationsUniformDistribution;
   static size_t size_tCountSignificantDeviationsUniformDistribution;
   static size_t floatCountSignificantDeviationsNormalDistribution;
   static size_t doubleCountSignificantDeviationsNormalDistribution;
   static size_t size_tCountDeviationsHistogramDistribution;
   static size_t doubleCountDeviationsHistogramDistribution;

   static float floatMin;
   static float floatMax;
   static double doubleMin;
   static double doubleMax;
   static int intMin;
   static int intMax;
   static size_t size_tMin;
   static size_t size_tMax;

   //random generator
   static size_t seed;
   static RANDOMGENERATOR randomGenerator;

   //result storage
   static std::vector<float> floatVec1;
   static std::vector<double> doubleVec1;
   static std::vector<int> intVec1;
   static std::vector<size_t> size_tVec1;
   static float floatRandValue1;
   static double doubleRandValue1;
   static int intRandValue1;
   static size_t size_tRandValue1;

   template <typename ITER>
   static bool isNormalDistributed(ITER sequenceBegin, ITER sequenceEnd);

   struct LessDereference {
      template <class T>
      bool operator()(const T * lhs, const T * rhs) const { return *lhs < *rhs; }
   };

   template <class VECTOR>
   static std::vector<size_t> computeSortedIndices(VECTOR& vec);
   template <typename T>
   static void inRange(T value, T min, T max);
   template <typename ITER, typename T>
   static void inRangeSequence(ITER sequenceBegin, ITER sequenceEnd, T min, T max);
   template <typename ITER>
   static bool isAlmostEqualSequence(ITER firstBegin, ITER firstEnd, ITER secondBegin);

public:
   static void testCaller(void (*pointerToTest)());
   static void testInRange();
   static void testSeed();
   static void testUniformDistribution();
   static void testHistogram();
   static void testNormalDistribution();
   static void testMultivariateNormalDistribution();
   static void testPermutation();
   static void run();
   static void resetValues(const size_t vecSize);
   static void fillRandomValues();
};

/******************
 * implementation *
 ******************/
int main(int argc, char** argv) {
   std::cout << "starting test random..." << std::endl;
   TestRand<opengm::Random2<opengm::RandomStd> >::run();
   std::cout << "test random finished successfully" << std::endl;
   return 0;
}

template <class RANDOMGENERATOR>
template <typename ITER>
bool TestRand<RANDOMGENERATOR>::isNormalDistributed(const ITER sequenceBegin, const ITER sequenceEnd){
   //distance
   const double distance = (double)std::distance(sequenceBegin, sequenceEnd);

   //mean
   double sampleMean = 0.0;
   for(ITER iter = sequenceBegin; iter != sequenceEnd; iter++) {
      sampleMean += (double)*iter;
   }
   sampleMean /= (distance);
   //std::cout << "sample mean: " << sampleMean << std::endl;

   //variance
   double sampleVariance = 0.0;
   for(ITER iter = sequenceBegin; iter != sequenceEnd; iter++) {
      sampleVariance += ((double)*iter - sampleMean) * ((double)*iter - sampleMean);
   }
   sampleVariance /= (distance - 1);
   //std::cout << "sample variance: " << sampleVariance << std::endl;

   //S
   double S = 0.0;
   for(ITER iter = sequenceBegin; iter != sequenceEnd; iter++) {
      S += ((double)*iter - sampleMean) * ((double)*iter - sampleMean) * ((double)*iter - sampleMean);
   }
   S /= distance;
   S /= sqrt(sampleVariance * sampleVariance * sampleVariance);

   //K
   double K = 0.0;
   for(ITER iter = sequenceBegin; iter != sequenceEnd; iter++) {
      K += ((double)*iter - sampleMean) * ((double)*iter - sampleMean) * ((double)*iter - sampleMean) * ((double)*iter - sampleMean);
   }
   K /= distance;

   K /= sampleVariance * sampleVariance;

   K -= 3.0;

   double JB = (distance / 6.0) * ((S * S) + ((K * K)/4.0));
   //std::cout << "JB: " << JB << std::endl;
   if(JB > 9.21) {
/*      std::cout << "sample mean: " << sampleMean << std::endl;
      std::cout << "sample variance: " << sampleVariance << std::endl;
      std::cout << "S: " << S << std::endl;
      std::cout << "K: " << K << std::endl;
      std::cout << "JB: " << JB << std::endl;*/
      return false;
   } else {
      return true;
   }
}

template <class RANDOMGENERATOR>
template <class VECTOR>
std::vector<size_t> TestRand<RANDOMGENERATOR>::computeSortedIndices(VECTOR& vec) {
   std::vector<typename VECTOR::value_type*> vecPointer;
   vecPointer.reserve(vec.size());
   for (size_t i = 0; i < vec.size(); i++) {
      vecPointer.push_back(&vec[i]);
   }
   std::sort(vecPointer.begin(), vecPointer.end(), LessDereference());

   std::vector<size_t> sortedIndices;
   sortedIndices.reserve(vec.size());
   const typename VECTOR::value_type* const start = &vec[0];
   for (size_t i = 0; i < vec.size(); i++) {
      const typename VECTOR::value_type* p = vecPointer[i];
      sortedIndices.push_back(p - start);
   }
   return sortedIndices;
}

template <class RANDOMGENERATOR>
template <typename T>
void TestRand<RANDOMGENERATOR>::inRange(T value, T min, T max) {
   OPENGM_TEST(value >= min); // value not in range because value >= min doesn't hold
   OPENGM_TEST(value <= max); // value not in range because value <= max doesn't hold
}

template <class RANDOMGENERATOR>
template <typename ITER, typename T>
void TestRand<RANDOMGENERATOR>::inRangeSequence(ITER sequenceBegin, ITER sequenceEnd, T min, T max) {
   for(ITER iter = sequenceBegin; iter != sequenceEnd; iter++) {
      inRange(*iter, min, max);
   }
}

template <class RANDOMGENERATOR>
template <typename ITER>
bool TestRand<RANDOMGENERATOR>::isAlmostEqualSequence(ITER firstBegin, ITER firstEnd, ITER secondBegin) {
   ITER firstIter = firstBegin;
   ITER secondIter = secondBegin;
   while(firstIter != firstEnd) {
      if(*firstIter == *secondIter) {
         firstIter++;
         secondIter++;
      } else if(firstIter != firstBegin && *firstIter == *(secondIter - 1)) {
         firstIter++;
         secondIter++;
      } else if (firstIter + 1 != firstEnd && *firstIter == *(secondIter + 1)) {
         firstIter++;
         secondIter++;
      } else {
         return false;
      }
   }
   return true;
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::testCaller(void (*pointerToTest)()) {
   for(size_t i = 0; i < numIter; i++) {
      (*pointerToTest)();
   }
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::testInRange() {
   resetValues(vecSizeTestInRange);
   fillRandomValues();

   inRange(floatRandValue1, floatMin, floatMax);
   inRange(doubleRandValue1, doubleMin, doubleMax);
   inRange(intRandValue1, intMin, intMax);
   inRange(size_tRandValue1, size_tMin, size_tMax);

   inRangeSequence(floatVec1.begin(), floatVec1.end(), floatMin, floatMax);
   inRangeSequence(doubleVec1.begin(), doubleVec1.end(), doubleMin, doubleMax);
   inRangeSequence(intVec1.begin(), intVec1.end(), intMin, intMax);
   inRangeSequence(size_tVec1.begin(), size_tVec1.end(), size_tMin, size_tMax);
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::testSeed() {
   resetValues(vecSizeTestSeed);

   RANDOMGENERATOR randomGenerator1(seed);
   floatRandValue1 = randomGenerator1.rand(floatMin, floatMax);
   doubleRandValue1 = randomGenerator1.rand(doubleMin, doubleMax);
   intRandValue1 = randomGenerator1.irand(intMin, intMax);
   size_tRandValue1 = randomGenerator1.irand(size_tMin, size_tMax);

   randomGenerator1.rand(floatVec1, floatMin, floatMax);
   randomGenerator1.rand(doubleVec1, doubleMin, doubleMax);
   randomGenerator1.irand(intVec1, intMin, intMax);
   randomGenerator1.irand(size_tVec1, size_tMin, size_tMax);

   RANDOMGENERATOR randomGenerator2(seed);

   float floatRandValue2 = randomGenerator2.rand(floatMin, floatMax);
   double doubleRandValue2 = randomGenerator2.rand(doubleMin, doubleMax);
   int intRandValue2 = randomGenerator2.irand(intMin, intMax);
   size_t size_tRandValue2 = randomGenerator2.irand(size_tMin, size_tMax);
   std::vector<float> floatVec2(vecSizeTestSeed, std::numeric_limits<float>::quiet_NaN());
   randomGenerator2.rand(floatVec2, floatMin, floatMax);
   std::vector<double> doubleVec2(vecSizeTestSeed, std::numeric_limits<double>::quiet_NaN());
   randomGenerator2.rand(doubleVec2, doubleMin, doubleMax);
   std::vector<int> intVec2(vecSizeTestSeed, std::numeric_limits<int>::quiet_NaN());
   randomGenerator2.irand(intVec2, intMin, intMax);
   std::vector<size_t> size_tVec2(vecSizeTestSeed, std::numeric_limits<size_t>::quiet_NaN());
   randomGenerator2.irand(size_tVec2, size_tMin, size_tMax);

   OPENGM_TEST_EQUAL(floatRandValue1, floatRandValue2);
   OPENGM_TEST_EQUAL(doubleRandValue1, doubleRandValue2);
   OPENGM_TEST_EQUAL(intRandValue1, intRandValue2);
   OPENGM_TEST_EQUAL(size_tRandValue1, size_tRandValue2);

   OPENGM_TEST_EQUAL_SEQUENCE(floatVec1.begin(), floatVec1.end(), floatVec2.begin());
   OPENGM_TEST_EQUAL_SEQUENCE(doubleVec1.begin(), doubleVec1.end(), doubleVec2.begin());
   OPENGM_TEST_EQUAL_SEQUENCE(intVec1.begin(), intVec1.end(), intVec2.begin());
   OPENGM_TEST_EQUAL_SEQUENCE(size_tVec1.begin(), size_tVec1.end(), size_tVec2.begin());
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::testUniformDistribution() {
   resetValues(vecSizeTestUniformDistribution);
   fillRandomValues();
   //compute bin size
   double floatBinSize = ((double)(floatMax - floatMin)) / numBins;
   double doubleBinSize = ((double)(doubleMax - doubleMin)) / numBins;
   double intBinSize = ((double)(intMax - intMin)) / numBins;
   double size_tBinSize = ((double)(size_tMax - size_tMin)) / numBins;

   std::vector<size_t> floatBins(numBins, 0);
   std::vector<size_t> doubleBins(numBins, 0);
   std::vector<size_t> intBins(numBins, 0);
   std::vector<size_t> size_tBins(numBins, 0);

   //compute bin height
   for(size_t i = 0; i < numBins; i++) {
      for(size_t j = 0; j < vecSizeTestUniformDistribution; j++) {
         if((floatVec1.at(j) >= (floatMin + (i * floatBinSize))) && (floatVec1.at(j) < (floatMin + ((i + 1) * floatBinSize)))) {
            floatBins.at(i)++;
         }
         if((doubleVec1.at(j) >= (doubleMin + (i * doubleBinSize))) && (doubleVec1.at(j) < (doubleMin + ((i + 1) * doubleBinSize)))) {
            doubleBins.at(i)++;
         }
         if((intVec1.at(j) >= (intMin + (i * intBinSize))) && (intVec1.at(j) < (intMin + ((i + 1) * intBinSize)))) {
            intBins.at(i)++;
         }
         if((size_tVec1.at(j) >= (size_tMin + (i * size_tBinSize))) && (size_tVec1.at(j) < (size_tMin + ((i + 1) * size_tBinSize)))) {
            size_tBins.at(i)++;
         }
      }
   }

   //check if bin size is within allowable Tolerance
   bool floatError = false;
   bool doubleError = false;
   bool intError = false;
   bool size_tError = false;
   for(size_t i = 0; i < numBins; i++) {
      if(floatBinSize != 0.0) {
         if((floatBins[i] < expectedBinSizeTestUniformDistribution - allowableToleranceTestUniformDistribution) || (floatBins[i] > expectedBinSizeTestUniformDistribution + allowableToleranceTestUniformDistribution)) {
            floatError = true;
         }
      }
      if(doubleBinSize != 0.0) {
         if((doubleBins[i] < expectedBinSizeTestUniformDistribution - allowableToleranceTestUniformDistribution) || (doubleBins[i] > expectedBinSizeTestUniformDistribution + allowableToleranceTestUniformDistribution)) {
            doubleError = true;
         }
      }
      if(intBinSize < 1.0) {
         if((intBins[i] < expectedBinSizeTestUniformDistribution - allowableToleranceTestUniformDistribution) || (intBins[i] > expectedBinSizeTestUniformDistribution + allowableToleranceTestUniformDistribution)) {
            intError = true;
         }
      }
      if(size_tBinSize < 1.0) {
         if((size_tBins[i] < expectedBinSizeTestUniformDistribution - allowableToleranceTestUniformDistribution) || (size_tBins[i] > expectedBinSizeTestUniformDistribution + allowableToleranceTestUniformDistribution)) {
            size_tError = true;
         }
      }
   }
   //TODO check boarder (random generator returns min and max)
   if(floatError) {
      floatCountSignificantDeviationsUniformDistribution++;
   }
   if(doubleError) {
      doubleCountSignificantDeviationsUniformDistribution++;
   }
   if(intError) {
      intCountSignificantDeviationsUniformDistribution++;
   }
   if(size_tError) {
      size_tCountSignificantDeviationsUniformDistribution++;
   }
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::testHistogram() {
   resetValues(0);
   std::vector<size_t> size_tHist(numBins, std::numeric_limits<size_t>::quiet_NaN());
   std::vector<double> doubleHist(numBins, std::numeric_limits<double>::quiet_NaN());

   randomGenerator.irand(size_tHist, 0, maxBase);
   randomGenerator.rand(doubleHist, 0.0, (double)maxBase);

   //compute sorted indices
   std::vector<size_t> size_tHistSortedIndices = computeSortedIndices(size_tHist);
   std::vector<size_t> doubleHistSortedIndices = computeSortedIndices(doubleHist);

   //generate random numbers from histogram
   std::vector<size_t> size_tHistResult(vecSizeTestHistogram, std::numeric_limits<size_t>::quiet_NaN());
   std::vector<size_t> doubleHistResult(vecSizeTestHistogram, std::numeric_limits<size_t>::quiet_NaN());
   randomGenerator.template irand<std::vector<size_t>, std::vector<size_t> >(size_tHistResult, size_tHist);
   randomGenerator.irand(doubleHistResult, doubleHist);
   inRangeSequence(size_tHistResult.begin(), size_tHistResult.end(), (size_t)0, numBins - 1);
   inRangeSequence(doubleHistResult.begin(), doubleHistResult.end(), (size_t)0, numBins - 1);

   //count frequentness
   std::vector<size_t> size_tHistFrequentness;
   size_tHistFrequentness.reserve(numBins);
   std::vector<size_t>doubleHistFrequentness;
   doubleHistFrequentness.reserve(numBins);
   for(size_t i = 0; i < numBins; i++) {
      size_tHistFrequentness.push_back(std::count(size_tHistResult.begin(), size_tHistResult.end(), i));
      doubleHistFrequentness.push_back(std::count(doubleHistResult.begin(), doubleHistResult.end(), i));
   }

   //sort frequentness
   std::vector<size_t> size_tSortedHistFrequentness = computeSortedIndices(size_tHistFrequentness);
   std::vector<size_t> doubleSortedHistFrequentness = computeSortedIndices(doubleHistFrequentness);

/*   //print values
   //size_t
   std::cout << "size_t" << std::endl;
   for(size_t i = 0; i < numBins; i++) {
      std::cout << "value: " << size_tHist[size_tHistSortedIndices[i]] << "; count: " << size_tHistFrequentness[size_tHistSortedIndices[i]] << std::endl;
   }
   //double
   std::cout << "double" << std::endl;
   for(double i = 0; i < numBins; i++) {
      std::cout << "value: " << doubleHist[doubleHistSortedIndices[i]] << "; count: " << doubleHistFrequentness[doubleHistSortedIndices[i]] << std::endl;
   }*/

   //check correctness of order
   if(!isAlmostEqualSequence(size_tSortedHistFrequentness.begin(), size_tSortedHistFrequentness.end(), size_tHistSortedIndices.begin())) {
      size_tCountDeviationsHistogramDistribution++;
   }
   if(!isAlmostEqualSequence(doubleSortedHistFrequentness.begin(), doubleSortedHistFrequentness.end(), doubleHistSortedIndices.begin())) {
      doubleCountDeviationsHistogramDistribution++;
   }
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::testNormalDistribution() {
   resetValues(0);
   //Jarque–Bera test
   std::vector<float> floatNormalVec1(vecSizeTestNormalDistribution, std::numeric_limits<float>::quiet_NaN());
   std::vector<double> doubleNormalVec1(vecSizeTestNormalDistribution, std::numeric_limits<double>::quiet_NaN());
   float floatMean = randomGenerator.rand(-(float)minBase, (float)maxBase);
   float floatVar = randomGenerator.rand((float)0.0, (float)maxBase);
   double doubleMean = randomGenerator.rand(-(double)minBase, (double)maxBase);
   double doubleVar = randomGenerator.rand((double)0.0, (double)maxBase);

   randomGenerator.nrand(floatNormalVec1, floatMean, floatVar);
   randomGenerator.nrand(doubleNormalVec1, doubleMean, doubleVar);



   if(!isNormalDistributed((std::vector<float>::const_iterator)floatNormalVec1.begin(), (std::vector<float>::const_iterator)floatNormalVec1.end())) {
      floatCountSignificantDeviationsNormalDistribution++;
      //std::cout << "floatMean: " << floatMean << std::endl;
      //std::cout << "floatVar: " << floatVar << std::endl;
   }

   if(!isNormalDistributed((std::vector<double>::const_iterator)doubleNormalVec1.begin(), (std::vector<double>::const_iterator)doubleNormalVec1.end())) {
      doubleCountSignificantDeviationsNormalDistribution++;
      //std::cout << "doubleMean: " << doubleMean << std::endl;
      //std::cout << "doubleVar: " << doubleVar << std::endl;
   }
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::testMultivariateNormalDistribution() {
   //TODO implement test for multivariate normal distribution
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::testPermutation() {
   //permutation vector
   std::vector<size_t> permutation = randomGenerator.template randPermutationVector<std::vector<size_t> >(vecSizeTestPermutation);
   OPENGM_TEST(permutation.size() == vecSizeTestPermutation);
   for(size_t i = 0; i < vecSizeTestPermutation; i++) {
      OPENGM_TEST(std::find(permutation.begin(), permutation.end(), i) != permutation.end());
   }
   //permutation matrix
   //reduce matrix size for permutation test
   size_t permutationMatrixSize = vecSizeTestPermutation / 100;
   marray::Matrix<double> permutationMatrix = randomGenerator.template randPermutationMatrix<marray::Matrix<double> >(permutationMatrixSize);
   OPENGM_TEST(permutationMatrix.dimension() == 2);
   OPENGM_TEST(permutationMatrix.size() == permutationMatrixSize * permutationMatrixSize);
   OPENGM_TEST(permutationMatrix.shape(0) == permutationMatrixSize);
   OPENGM_TEST(permutationMatrix.shape(1) == permutationMatrixSize);
   for(size_t i = 0; i < permutationMatrixSize; i++) {
      double sumRow = 0.0;
      double sumCol = 0.0;
      for(size_t j = 0; j < permutationMatrixSize; j++) {
         sumRow += permutationMatrix(i, j);
         sumCol += permutationMatrix(j, i);
      }
      OPENGM_TEST(sumRow == 1.0);
      OPENGM_TEST(sumCol == 1.0);
   }
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::run() {
   std::cout << "starting test: testInRange ..." << std::endl;
   testCaller(&testInRange);
   std::cout << "done" << std::endl;
   std::cout << "starting test: testSeed ..." << std::endl;
   testCaller(&testSeed);
   std::cout << "done" << std::endl;
   std::cout << "starting test: testUniformDistribution ..." << std::endl;
   testCaller(&testUniformDistribution);
   std::cout << "floatCountSignificantDeviationsUniformDistribution: " << floatCountSignificantDeviationsUniformDistribution << std::endl;
   std::cout << "doubleCountSignificantDeviationsUniformDistribution: " << doubleCountSignificantDeviationsUniformDistribution << std::endl;
   std::cout << "intCountSignificantDeviationsUniformDistribution: " << intCountSignificantDeviationsUniformDistribution << std::endl;
   std::cout << "size_tCountSignificantDeviationsUniformDistribution: " << size_tCountSignificantDeviationsUniformDistribution << std::endl;
   OPENGM_TEST(floatCountSignificantDeviationsUniformDistribution <= AllowedSignificantDeviations);
   OPENGM_TEST(doubleCountSignificantDeviationsUniformDistribution <= AllowedSignificantDeviations);
   OPENGM_TEST(intCountSignificantDeviationsUniformDistribution <= AllowedSignificantDeviations);
   OPENGM_TEST(size_tCountSignificantDeviationsUniformDistribution <= AllowedSignificantDeviations);
   std::cout << "done" << std::endl;
   std::cout << "starting test: testHistogram ..." << std::endl;
   testCaller(&testHistogram);
   std::cout << "size_tCountDeviationsHistogramDistribution: " << size_tCountDeviationsHistogramDistribution << std::endl;
   std::cout << "doubleCountDeviationsHistogramDistribution: " << doubleCountDeviationsHistogramDistribution << std::endl;
   OPENGM_TEST(size_tCountDeviationsHistogramDistribution <= AllowedSignificantDeviations);
   OPENGM_TEST(doubleCountDeviationsHistogramDistribution <= AllowedSignificantDeviations);
   std::cout << "done" << std::endl;
   std::cout << "starting test: testNormalDistribution ..." << std::endl;
   testCaller(&testNormalDistribution);
   std::cout << "floatCountSignificantDeviationsNormalDistribution: " << floatCountSignificantDeviationsNormalDistribution << std::endl;
   std::cout << "doubleCountSignificantDeviationsNormalDistribution: " << doubleCountSignificantDeviationsNormalDistribution << std::endl;
   OPENGM_TEST(floatCountSignificantDeviationsNormalDistribution <= AllowedSignificantDeviations);
   OPENGM_TEST(doubleCountSignificantDeviationsNormalDistribution <= AllowedSignificantDeviations);
   std::cout << "done" << std::endl;
   std::cout << "starting test: testMultivariateNormalDistribution ..." << std::endl;
   testCaller(&testMultivariateNormalDistribution);
   std::cout << "done" << std::endl;
   std::cout << "starting test: testPermutation ..." << std::endl;
   testCaller(&testPermutation);
   std::cout << "done" << std::endl;
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::resetValues(const size_t vecSize) {
   floatMin = -(float)minBase;
   floatMax = (float)maxBase;
   doubleMin = -(double)minBase;
   doubleMax = (double)maxBase;
   intMin = -(int)minBase;
   intMax = (int)maxBase;
   size_tMin = std::numeric_limits<size_t>::min();
   size_tMax = (size_t)maxBase;

  //generate new random intervals

   floatMin = randomGenerator.rand(floatMin, floatMax);
   inRange(floatMin, -(float)minBase, (float)maxBase);
   floatMax = randomGenerator.rand(floatMin, floatMax);
   inRange(floatMax, floatMin, (float)maxBase);
   doubleMin = randomGenerator.rand(doubleMin, doubleMax);
   inRange(doubleMin, -(double)minBase, (double)maxBase);
   doubleMax = randomGenerator.rand(doubleMin, doubleMax);
   inRange(doubleMax, doubleMin, (double)maxBase);
   intMin = randomGenerator.irand(intMin, intMax);
   inRange(intMin, -(int)minBase, (int)maxBase);
   intMax = randomGenerator.irand(intMin, intMax);
   inRange(intMax, intMin, (int)maxBase);
   size_tMin = randomGenerator.irand(size_tMin, size_tMax);
   inRange(size_tMin, std::numeric_limits<size_t>::min(), (size_t)maxBase);
   size_tMax = randomGenerator.irand(size_tMin, size_tMax);
   inRange(size_tMax, size_tMin, (size_t)maxBase);

   //result storage
   floatVec1.assign(vecSize, std::numeric_limits<float>::quiet_NaN());
   doubleVec1.assign(vecSize, std::numeric_limits<double>::quiet_NaN());
   intVec1.assign(vecSize, std::numeric_limits<int>::quiet_NaN());
   size_tVec1.assign(vecSize, std::numeric_limits<size_t>::quiet_NaN());
   floatRandValue1 = std::numeric_limits<float>::quiet_NaN();
   doubleRandValue1 = std::numeric_limits<double>::quiet_NaN();
   intRandValue1 = std::numeric_limits<int>::quiet_NaN();
   size_tRandValue1 = std::numeric_limits<size_t>::quiet_NaN();
}

template <class RANDOMGENERATOR>
void TestRand<RANDOMGENERATOR>::fillRandomValues() {
   floatRandValue1 = randomGenerator.rand(floatMin, floatMax);
   doubleRandValue1 = randomGenerator.rand(doubleMin, doubleMax);
   intRandValue1 = randomGenerator.irand(intMin, intMax);
   size_tRandValue1 = randomGenerator.irand(size_tMin, size_tMax);

   randomGenerator.rand(floatVec1, floatMin, floatMax);
   randomGenerator.rand(doubleVec1, doubleMin, doubleMax);
   randomGenerator.irand(intVec1, intMin, intMax);
   randomGenerator.irand(size_tVec1, size_tMin, size_tMax);
}

/**************************************
 * static member variables definition *
 **************************************/
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::floatCountSignificantDeviationsUniformDistribution = 0;
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::doubleCountSignificantDeviationsUniformDistribution = 0;
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::intCountSignificantDeviationsUniformDistribution = 0;
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::size_tCountSignificantDeviationsUniformDistribution = 0;
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::floatCountSignificantDeviationsNormalDistribution = 0;
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::doubleCountSignificantDeviationsNormalDistribution = 0;
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::size_tCountDeviationsHistogramDistribution;
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::doubleCountDeviationsHistogramDistribution;
template <class RANDOMGENERATOR> float TestRand<RANDOMGENERATOR>::floatMin = -std::numeric_limits<float>::infinity();
template <class RANDOMGENERATOR> float TestRand<RANDOMGENERATOR>::floatMax = std::numeric_limits<float>::infinity();
template <class RANDOMGENERATOR> double TestRand<RANDOMGENERATOR>::doubleMin = -std::numeric_limits<double>::infinity();
template <class RANDOMGENERATOR> double TestRand<RANDOMGENERATOR>::doubleMax = std::numeric_limits<double>::infinity();
template <class RANDOMGENERATOR> int TestRand<RANDOMGENERATOR>::intMin = std::numeric_limits<int>::min();
template <class RANDOMGENERATOR> int TestRand<RANDOMGENERATOR>::intMax = std::numeric_limits<int>::max();
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::size_tMin = std::numeric_limits<size_t>::min();
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::size_tMax = std::numeric_limits<size_t>::max();
template <class RANDOMGENERATOR> std::vector<float> TestRand<RANDOMGENERATOR>::floatVec1(TestRand<RANDOMGENERATOR>::vecSizeBase, std::numeric_limits<float>::quiet_NaN());
template <class RANDOMGENERATOR> std::vector<double> TestRand<RANDOMGENERATOR>::doubleVec1(TestRand<RANDOMGENERATOR>::vecSizeBase, std::numeric_limits<double>::quiet_NaN());
template <class RANDOMGENERATOR> std::vector<int> TestRand<RANDOMGENERATOR>::intVec1(TestRand<RANDOMGENERATOR>::vecSizeBase, std::numeric_limits<int>::quiet_NaN());
template <class RANDOMGENERATOR> std::vector<size_t> TestRand<RANDOMGENERATOR>::size_tVec1(TestRand<RANDOMGENERATOR>::vecSizeBase, std::numeric_limits<size_t>::quiet_NaN());
template <class RANDOMGENERATOR> float TestRand<RANDOMGENERATOR>::floatRandValue1 = std::numeric_limits<float>::quiet_NaN();
template <class RANDOMGENERATOR> double TestRand<RANDOMGENERATOR>::doubleRandValue1 = std::numeric_limits<double>::quiet_NaN();
template <class RANDOMGENERATOR> int TestRand<RANDOMGENERATOR>::intRandValue1 = std::numeric_limits<int>::quiet_NaN();
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::size_tRandValue1 = std::numeric_limits<size_t>::quiet_NaN();
template <class RANDOMGENERATOR> size_t TestRand<RANDOMGENERATOR>::seed = time(NULL);
template <class RANDOMGENERATOR> RANDOMGENERATOR TestRand<RANDOMGENERATOR>::randomGenerator(TestRand<RANDOMGENERATOR>::seed);
