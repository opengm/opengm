#include <string>
#include <vector>
#include <iostream>

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
   if(argc!=4){
      std::cout << "Syntax: INFILE OUTFILE NUMLABELS" << std::endl; 
      return 0;
   } 

   GraphicalModelType gm1; 
   std::string infile  = argv[1]; 
   std::string outfile = argv[2];
   size_t L            = atoi(argv[3]);

   opengm::hdf5::load(gm1, infile,  "gm");

  
   std::vector<LabelType> numbersOfLabels(gm1.numberOfVariables(),L);
   GraphicalModelType gm2(opengm::DiscreteSpace<IndexType, LabelType >(numbersOfLabels.begin(), numbersOfLabels.end()) );
   LabelType nos[2] = {L, L};

   ExplicitFunctionType f1(nos, nos+1, 0);
   ExplicitFunctionType f2(nos, nos+1, 0);
   for(LabelType l=1; l<L; ++l){
      f1(l) = 1000000000;
   } 
   FunctionIdentifier fid1 = gm2.addFunction(f1);  
   FunctionIdentifier fid2 = gm2.addFunction(f2);  

   for(IndexType i=0; i<gm1.numberOfVariables(); ++i){
      if(i==0)
         gm2.addFactor(fid1, &i, &i+1);
      else 
         gm2.addFactor(fid2, &i, &i+1);
   }

   LabelType l00[] = {0,0};
   LabelType l01[] = {0,1};
   for(IndexType f=0; f<gm1.numberOfFactors();++f){
      IndexType vars[] = {0,0};
      vars[0] = gm1[f].variableIndex(0);
      vars[1] = gm1[f].variableIndex(1);
      ValueType v00 = gm1[f](l00);
      ValueType v01 = gm1[f](l01);
      PottsFunctionType pottsFunction(L,L,v00,v01);
      FunctionIdentifier fid = gm2.addFunction(pottsFunction); 
      gm2.addFactor(fid, vars, vars+2);  
   }

   opengm::hdf5::save(gm2, outfile, "gm"); 

   return 0;
}
