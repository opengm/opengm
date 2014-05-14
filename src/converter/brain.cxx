#include <iostream>
#include <fstream>
#include <strstream>

#include <opengm/unittests/test.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/utilities/metaprogramming.hxx>

typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType;
typedef opengm::GraphicalModel<
   ValueType,
   opengm::Adder,
   opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType,IndexType,LabelType>,opengm::PottsFunction<ValueType,IndexType,LabelType> >::type,
   opengm::DiscreteSpace<IndexType, LabelType>
   > GraphicalModelType;

typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
typedef opengm::PottsFunction<ValueType>    PottsFunctionType;
typedef GraphicalModelType::FunctionIdentifier          FunctionIdentifier;


int main(int argc, char *argv[])
{  
   if(argc==1){
      std::cout<<std::endl << "usage: brain [infile] [outfile] [dim1 dim2 dim3] [pottsweight]" <<std::endl << std::endl;
      return 0;
   }

   const size_t    dim1 = atoi(argv[3]);
   const size_t    dim2 = atoi(argv[4]);
   const size_t    dim3 = atoi(argv[5]);
   const ValueType w    = atof(argv[6]);  
   const LabelType numberOfLabels = 5;
   const IndexType numberOfVariables = dim1*dim2*dim3;

// std::cout << w << std::endl;
   
   std::vector<LabelType> numbersOfLabels(numberOfVariables,numberOfLabels);
   GraphicalModelType gm(opengm::DiscreteSpace<IndexType, LabelType >(numbersOfLabels.begin(), numbersOfLabels.end()) );
   LabelType nos[2] = {numberOfLabels, numberOfLabels};
   std::vector<FunctionIdentifier> fid(257);

   for(size_t i=0; i<=255; ++i){
      ExplicitFunctionType f(nos, nos+1, 0);
      f(0) = fabs(i-4.0);
      f(1) = fabs(i-45.0);//40
      f(2) = fabs(i-105.0);//96
      f(3) = fabs(i-150.0);//135
      f(4) = fabs(i-204.0);
      fid[i] = gm.addFunction(f);
   }
   PottsFunctionType func(nos[0], nos[1], 0, w);
   fid[256] = gm.addSharedFunction(func);

   unsigned char value;
   IndexType varId[2];
   std::ifstream file(argv[1],std::ios::in | std::ios::binary);
   for(size_t d3=0; d3<dim3; ++d3){
      for(size_t d2=0; d2<dim2; ++d2){
         for(size_t d1=0; d1<dim1; ++d1){
            file.read(reinterpret_cast<char *>(&value),1); 
// std::cout <<(size_t) value << "  ";
            varId[0] = d3*dim2*dim1 + d2*dim1 + d1;
            gm.addFactor(fid[(size_t) value], varId, varId+1);
            if(d1+1<dim1){
               varId[1] = varId[0]+1;
               gm.addFactor(fid[256], varId, varId+2); 
            } 
            if(d2+1<dim2){
               varId[1] = varId[0]+dim1;
               gm.addFactor(fid[256], varId, varId+2); 
            } 
            if(d3+1<dim3){
               varId[1] = varId[0]+dim1*dim2;
               gm.addFactor(fid[256], varId, varId+2); 
            }
         }
      }
   }

   opengm::hdf5::save(gm, argv[2], "gm");
   file.close();

   return 0;
}
