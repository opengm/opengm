#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/utilities/metaprogramming.hxx>

typedef double ValueType;
typedef size_t IndexType;
typedef unsigned char LabelType;
typedef opengm::GraphicalModel<
   ValueType,
   opengm::Adder,
   opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType> >::type,
   opengm::DiscreteSpace<IndexType, LabelType>
   > GraphicalModelType;

typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   /* INPUT */
   /* 1.) unaray*/
   /* 2.) pairs*/
   /* 3.) pairfunction*/
   /* 4.) output-filename */
 
   if (nrhs != 4) {
      mexErrMsgTxt("Incorrect number of inputs.");
   }
  
 
   char* filename; 
   filename  = (char*)mxCalloc(mxGetN(prhs[3])+1,sizeof(char));
   mxGetString(prhs[3], filename, mxGetN(prhs[3])+1);

   double* unary             = (double*) mxGetData(prhs[0]);
   double* pairwise_node     = (double*) mxGetData(prhs[1]); 
  
   size_t numberOfVariables = mxGetN(prhs[0]);
   size_t numberOfFactors   = mxGetN(prhs[1]);
 
   /* Output Information*/
   printf("Number of variables:           %d\n",numberOfVariables);
   printf("Number of pairwise factors:    %d\n",numberOfFactors);
   printf("Output file:                   %s\n",filename);

   std::vector<LabelType> numStates(numberOfVariables,numberOfVariables);
   GraphicalModelType gm(opengm::DiscreteSpace<IndexType, LabelType >(numStates.begin(), numStates.end()) );

   //Add Unaries
   std::cout << "add Unaries"<<std::endl;
   for(size_t var=0; var<numberOfVariables; ++var) {
      LabelType shape[] = {gm.numberOfLabels(var)};
      ExplicitFunctionType func(shape,shape+1); 
      for(size_t i=0; i<shape[0]; ++i) {
         func(i) = unary[i+var*numberOfVariables];
      }
      FunctionIdentifier fid = gm.addSharedFunction(func);
      gm.addFactor(fid, &var, &var+1);
   }

   
   //ADD Pairwise 
   std::cout << "add pairwise"<<std::endl;
   for(size_t f=0;f<numberOfFactors;++f) {
      IndexType vars[2];
      LabelType shape[]={numberOfVariables,numberOfVariables};
      vars[0] = pairwise_node[0+2*f];
      vars[1] = pairwise_node[1+2*f];
      ExplicitFunctionType func(shape,shape+2);
      mxArray* data = mxGetCell(prhs[2], f);
      double* pdata     = (double*) mxGetData(data); 
      for(size_t i1=0; i1<shape[0]; ++i1) { 
         for(size_t i2=0; i2<shape[1]; ++i2) {
            if(i1==i2)
               func(i1,i2) = 10000000000;
            else
               func(i1,i2) = pdata[i1+i2*shape[0]];
         }
      }
      FunctionIdentifier fid = gm.addSharedFunction(func);
      gm.addFactor(fid, vars, vars+2);
   }

   opengm::hdf5::save(gm, filename, "gm");

   mxFree(filename);
}
