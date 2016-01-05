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
   std::string opengmfile   = argv[2];
   std::string matchingfile = argv[1];
   ValueType   infty        = 1000000;

   // open matching file
   std::ifstream mymatchingfile(matchingfile.c_str());
   if(!mymatchingfile.is_open()) {
      std::cerr << "Could not open file " << matchingfile << std::endl;
      return 1;
   }

   // temp storage for current line of uai file
   std::string currentLine; 
   char command;
   size_t pointsLeft;
   size_t pointsRight;
   size_t A;
   size_t E; 

   mymatchingfile >> command;
   if(command!='p'){
      std::cerr << "Missing Model info (p)!" << std::endl;
      return 1;
   }
   mymatchingfile >> pointsLeft;
   mymatchingfile >> pointsRight;
   mymatchingfile >> A;
   mymatchingfile >> E;
   std::vector<std::pair<size_t, size_t> > assigments(A);

//   std::vector<std::vector<ValueType> >    unaries( pointsLeft, std::vector<ValueType>(pointsRight));
   std::vector<opengm::ExplicitFunction<ValueType, IndexType, LabelType> >    unaries( pointsLeft, opengm::ExplicitFunction<ValueType, IndexType, LabelType>(&pointsRight,&pointsRight+1));

   std::vector<size_t> shape(2,pointsRight);
   std::vector<size_t> labeling(2,0);
   opengm::ExplicitFunction<ValueType, IndexType, LabelType> tempPair(shape.begin(), shape.end());
   for(size_t i = 0; i < shape[0]; i++) {
      labeling[0] = labeling[1] = i;
      tempPair(labeling.begin()) = infty;
   }
   std::vector<std::vector<opengm::ExplicitFunction<ValueType, IndexType, LabelType> > > pairs(pointsLeft,std::vector<opengm::ExplicitFunction<ValueType, IndexType, LabelType> >(pointsLeft,tempPair));

   std::cout << "Problem has "<<pointsLeft<<" and "<<pointsRight<<" points to match. ("<<A<<","<<E<<")"<<std::endl;

   size_t countA =0; 
   size_t countE =0;
   while(!mymatchingfile.eof()) {
      mymatchingfile >> command;
      if(command =='a'){++countA;
         size_t aID, pL, pR;
         ValueType v;
         mymatchingfile >> aID;
         mymatchingfile >> pL;  
         mymatchingfile >> pR;
         assigments[aID] = std::pair<size_t,size_t>(pL,pR);
         mymatchingfile >> v;
         unaries[pL](&pR) = v;
      }
      if(command =='e'){++countE;
         size_t aID1, aID2;
         double v;
         mymatchingfile >> aID1;
         mymatchingfile >> aID2;
         mymatchingfile >> v;
         if(assigments[aID1].first < assigments[aID2].first){
            labeling[0] = assigments[aID1].second;
            labeling[1] = assigments[aID2].second;
            pairs[assigments[aID1].first][assigments[aID2].first](labeling.begin()) += v;
         }
         else{
            //std::cout <<"ERROR  " << assigments[aID1].first<<" < "<<assigments[aID2].first<<std::endl; 
            labeling[1] = assigments[aID1].second;
            labeling[0] = assigments[aID2].second;
            pairs[assigments[aID2].first][assigments[aID1].first](labeling.begin()) += v;
         }
      }
   }
   if (A!=countA)
      std::cout << "Wrong number of Assignments "<<  A <<" != " << countA <<std::endl;
   if (E!=countE)
      std::cout << "Wrong number of Edges "<<  E <<" != " << countE <<std::endl;
   
 
   for(IndexType i=0; i<pointsLeft; ++i){
      gm.addVariable(pointsRight);
   }
   // ADD UNARIES
   for(IndexType var=0; var<gm.numberOfVariables(); ++var){
      GmType::FunctionIdentifier currentFunctionIndex = gm.addFunction(unaries[var]);
      gm.addFactor(currentFunctionIndex, &var, &var+1);
   }

   // ADD PAIRS
   IndexType vars[2] = {0,0};
   for(vars[0]=0; vars[0]<gm.numberOfVariables(); ++vars[0]){
      for(vars[1]=vars[0]+1; vars[1]<gm.numberOfVariables(); ++vars[1]){
         GmType::FunctionIdentifier currentFunctionIndex = gm.addFunction(pairs[vars[0]][vars[1]]);
         gm.addFactor(currentFunctionIndex, vars, vars+2);
      }
   }

   // close matching file
   mymatchingfile.close();

   // store gm
   opengm::hdf5::save(gm, opengmfile,"gm");

   return 0;
}
