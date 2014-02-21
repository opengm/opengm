#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/graphicalmodel_hdf5.hxx"
#include "opengm/datastructures/marray/marray_hdf5.hxx"
#include "opengm/operations/adder.hxx"
#include <opengm/utilities/metaprogramming.hxx>
#include "opengm/functions/potts.hxx"

typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType;
typedef opengm::GraphicalModel<
   ValueType,
   opengm::Adder,
   opengm::meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType,IndexType,LabelType>,
      opengm::PottsFunction<ValueType,IndexType,LabelType>
      >::type,
   opengm::DiscreteSpace<IndexType, LabelType>
   > GraphicalModelType;

typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>      ExplicitFunctionType;
typedef opengm::PottsFunction<ValueType,IndexType,LabelType>         PottsFunctionType;
typedef GraphicalModelType::FunctionIdentifier   FunctionIdentifier;




int
main(int argc, const char* argv[] ) {
   if(argc!=3){
      std::cout << "Syntax: INFILE OUTFILE" << std::endl; 
      return 0;
   } 

   GraphicalModelType gm; 
   std::string infile  = argv[1]; 
   std::string outfile = argv[2];

   opengm::hdf5::load(gm, infile,  "gm");
 
   LabelType l00[] = {0,0};
   LabelType l01[] = {0,1};

   //check model and calculate constant
   ValueType constValue = 0;
   for(IndexType f=0; f<gm.numberOfFactors();++f){
      if(gm[f].numberOfVariables() == 0){
         LabelType l0 =0;
         constValue += gm[f](&l0);
      }
      else if(gm[f].numberOfVariables() == 2){
         constValue += gm[f](l00);
      }
      else{
         std::cout << "ERROR: Unsupported factor order !!!" << std::endl;
         return 1;
      }    
   }
   
   std::ofstream outputFile;
   outputFile.open(outfile.c_str());
   outputFile << "  "<< gm.numberOfVariables() << " " << gm.numberOfFactors() << " "<< std::setprecision(20) << constValue << std::endl;
   for(IndexType f=0; f<gm.numberOfFactors();++f){
      IndexType vars[] = {0,0};
      vars[0] = gm[f].variableIndex(0);
      vars[1] = gm[f].variableIndex(1);
      ValueType v00 = gm[f](l00);
      ValueType v01 = gm[f](l01);
      outputFile <<  vars[0] << " " << vars[1] << " "<< std::setprecision(20) << v01-v00 << std::endl;
   }
   outputFile.close();

   return 0;
}
