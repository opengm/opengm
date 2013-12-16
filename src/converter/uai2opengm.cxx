#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>

// remove leading and ending spaces of given input as this might screw up parsing the input
void removeSpaces(std::string& input);

// compute log of input as network types of type MARKOV store conditional probabilities
template <class T>
void logTransform(T& input);

// sort vector in non descending and compute corresponding permutation vector
template<class T, class U>
void sortingPermutation(const T& values, U& permutation);

// check if vector is in non descending order
template <class T>
bool vecIsSorted(const T& vector);

// transfomrs linear index to subscript values
template <class T, class U, class V>
void ind2sub(T& sub, const U ind, const V& sizes);

int main(int argc, const char* argv[] ) {
   if(argc != 3) {
      std::cerr << "Two input arguments required" << std::endl;
      return 1;
   }

   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;
   typedef opengm::Adder OperatorType;
   typedef opengm::Minimizer AccumulatorType;
   typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

   // Set functions for graphical model
   typedef opengm::meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType, IndexType, LabelType>
   >::type FunctionTypeList;

   typedef opengm::GraphicalModel<
      ValueType,
      OperatorType,
      FunctionTypeList,
      SpaceType
   > GmType;

   GmType gm;
   std::string opengmfile = argv[2];
   std::string uaifile    = argv[1];

   // open uai file
   std::ifstream myuaifile(uaifile.c_str());
   if(!myuaifile.is_open()) {
      std::cerr << "Could not open file " << uaifile << std::endl;
      return 1;
   }

   // temp storage for current line of uai file
   std::string currentLine;

   // check if type of network is MARKOV
   std::getline(myuaifile, currentLine);
   removeSpaces(currentLine);
   if(currentLine != "MARKOV") {
      std::cerr << "Unsupported network type: \"" << currentLine <<"\". Only networks of type MARKOV are supported by now." << std::endl;
      return 1;
   }

   // get number of variables
   IndexType numVariables;
   if(myuaifile.eof()) {
      std::cerr << "Bad file format: Preamble is incomplete (Number of variables is missing)." << std::endl;
      return 1;
   } else {
      myuaifile >> numVariables;
   }

   // add variables
   if(myuaifile.eof()) {
         std::cerr << "Bad file format: Preamble is incomplete (Cardinalities of variables is missing)." << std::endl;
         return 1;
   } else {
      for(IndexType i=0; i<numVariables; ++i){
         LabelType currentNumberOfStates;
         myuaifile >>currentNumberOfStates;
         //std::cout <<currentNumberOfStates <<" ";
         gm.addVariable(currentNumberOfStates);
      }
   }

   // get number of factors
   IndexType numFactors;
   if(myuaifile.eof()) {
      std::cerr << "Bad file format: Preamble is incomplete (Number of factors is missing)." << std::endl;
      return 1;
   } else {
      myuaifile >> numFactors;
      //std::cout <<std::endl<< numFactors <<std::endl;
   }

   // get factors;
   std::vector<std::vector<IndexType> > factors(numFactors);
   if(myuaifile.eof()) {
      std::cerr << "Bad file format: Preamble is incomplete (factors are missing)." << std::endl;
      return 1;
   } else {
      for(IndexType f=0;f<numFactors;++f){
         IndexType numFactorVariables;
         myuaifile >> numFactorVariables; 
         factors[f].resize(numFactorVariables);
         for(size_t i=0; i<numFactorVariables; ++i){ 
            myuaifile >>  factors[f][i]; 
         }
      }
   }



   // add functions to gm
   if(myuaifile.eof()) {
      std::cerr << "Bad file format: Function tables are missing." << std::endl;
      return 1;
   } else {
      for(IndexType f=0; f<numFactors; ++f){
         // read number of values
         unsigned long numFunctionValues;
         myuaifile >> numFunctionValues;
        
         // get shape of function
         std::vector<LabelType> currentShape;
         currentShape.reserve(factors[f].size()); 
         for(IndexType i = 0; i < factors[f].size(); i++) {
            IndexType currentVariable = factors[f][i];
            LabelType currentVariableDimension = gm.space().numberOfLabels(currentVariable);
            currentShape.push_back(currentVariableDimension);
         } 
         marray::Marray<ValueType> currentFunctionValues(currentShape.begin(), currentShape.end());
         
         // check for matching size
         OPENGM_ASSERT_OP(currentFunctionValues.size(), ==, numFunctionValues);  
         
         // set function values
         if(myuaifile.eof()) {
            std::cerr << "Bad file format: Function values are missing." << std::endl;
            return 1;
         } else {
            std::vector<LabelType> index;
            for(size_t i=0; i<currentFunctionValues.size(); ++i){
               ValueType val;
               myuaifile >> val;
               logTransform(val);
               ind2sub(index, i, currentShape);
               currentFunctionValues(index.begin()) =val;
            }
         }

         // find variable permutation
         std::vector<size_t> permutation;
         sortingPermutation(factors[f], permutation);
         bool isSorted = vecIsSorted(permutation); 
         std::vector<LabelType> sortedShape;
         sortedShape.resize(currentShape.size()); 
         for(size_t i=0;i<currentShape.size();++i)
            sortedShape[i] = currentShape[permutation[i]];
         

         opengm::ExplicitFunction<ValueType, IndexType, LabelType> currentFunction(sortedShape.begin(), sortedShape.end());
      
         // copy function values to function
         // compute permutation vector, if variables are stored in non descending order
         if(!isSorted) {
            // sort variables and permute function
            std::sort(factors[f].begin(), factors[f].end());
            marray::Marray<ValueType>::base currentFunctionValuesView(currentFunctionValues);
            currentFunctionValuesView.permute(permutation.begin());
            for(size_t i = 0; i < currentFunctionValuesView.size(); i++) {
               currentFunction(i) = currentFunctionValuesView(i);
            }
         } else {
            for(size_t i = 0; i < currentFunctionValues.size(); i++) {
               currentFunction(i) = currentFunctionValues(i);
            }
         }
         
         // add function to gm
         GmType::FunctionIdentifier currentFunctionIndex = gm.addSharedFunction(currentFunction);
         
         // add corresponding factor
         gm.addFactor(currentFunctionIndex, factors[f].begin(), factors[f].end());
      }
   }

   // close uai file
   myuaifile.close();

   // store gm
   opengm::hdf5::save(gm, opengmfile,"gm");

   return 0;
}


void removeSpaces(std::string& input) {
   if(input.length() == 0) {
      return;
   } else {
      // remove leading spaces
      while(input[0] == ' ') {
         input.erase(0, 1);
         if(input.length() == 0) {
            return;
         }
      }
      // remove ending spaces
      while(input[input.length() - 1] == ' ') {
         input.erase(input.length() - 1, 1);
         if(input.length() == 0) {
            return;
         }
      }
   }
}

template <class T>
void logTransform(T& input) {
   OPENGM_ASSERT(input >= 0);
   if(input == 0) {
      input = std::numeric_limits<T>::infinity();
   } else {
      input = -log(input);
   }
   input = std::min(input, static_cast<T>(10000000));
}

template<class T>
struct CmpPairs{
  CmpPairs(const T& vectorIn): vector(vectorIn) {}
  const T& vector;
  bool operator()(int a, int b){ return vector[a] < vector[b]; }
};

template<class T>
CmpPairs<T> CreateCmpPairs(const T& vector) { return CmpPairs<T>(vector); }

template<class T, class U>
void sortingPermutation(const T& values, U& permutation){
   permutation.clear();
   permutation.reserve(values.size());
   for(size_t i = 0; i < values.size(); i++) {
      permutation.push_back(i);
   }
   std::sort(permutation.begin(), permutation.end(), CreateCmpPairs(values));
}

template <class T>
bool vecIsSorted(const T& vector) {
   if(vector.size() <= 1) {
      return true;
   } else {
      for(size_t i = 0; i < vector.size() - 1; i++) {
         if(vector[i] > vector[i + 1]) {
            return false;
         }
      }
      return true;
   }
}

template <class T, class U, class V>
void ind2sub(T& sub, const U ind, const V& sizes) {
   sub.clear();
   sub.reserve(sizes.size());
   U remainingInd = ind;
   for(size_t i = 0; i < sizes.size() - 1; i++) {
      size_t numSubentries = 1;
      for(size_t j = i + 1; j < sizes.size(); j++) {
         numSubentries *= sizes[j];
      }
      size_t currentIndex = remainingInd / numSubentries;
      remainingInd -= currentIndex * numSubentries;
      sub.push_back(currentIndex);
   }
   sub.push_back(remainingInd);
}
