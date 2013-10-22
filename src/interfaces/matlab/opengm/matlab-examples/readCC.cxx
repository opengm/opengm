// Written by Jörg Hendrik Kappes
// code to generate matching instances

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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   /* INPUT */
   /* 1.) filename*/
 
   if (nrhs != 1) {
      mexErrMsgTxt("Incorrect number of inputs.");
   }
  
 
   char* filename; 
   filename  = (char*)mxCalloc(mxGetN(prhs[0])+1,sizeof(char));
   mxGetString(prhs[0], filename, mxGetN(prhs[0])+1);
   
   GraphicalModelType gm; 
   opengm::hdf5::load(gm, filename,  "gm");
   size_t numVar = gm.numberOfVariables(); 
   for(IndexType f=0; f<gm.numberOfFactors();++f){
      if(gm[f].numberOfVariables()!=2)
         std::cout << "ERROR : Factor has not 2 variables!"<<std::endl;
   }

   // plhs[0] = mxCreateDoubleMatrix(numVar, numVar, mxREAL); 
   plhs[0] = mxCreateSparse(numVar,numVar,2*gm.numberOfFactors(),mxREAL);
   plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL); 

   //double* W = mxGetPr(plhs[0]);
   double* C = mxGetPr(plhs[1]);
   double*  w = mxGetPr(plhs[0]);
   mwIndex* ir = mxGetIr(plhs[0]);
   mwIndex* ij = mxGetJc(plhs[0]);
   ij[0]=0;

   LabelType l00[] = {0,0};
   LabelType l01[] = {0,1};

   
   for(IndexType var=0; var<gm.numberOfVariables();++var){
      std::set<size_t> nb;
      for(GraphicalModelType::ConstFactorIterator fit=gm.factorsOfVariableBegin(var); fit !=gm.factorsOfVariableEnd(var); ++fit){
         if(gm[*fit].variableIndex(0)==var)
            nb.insert(gm[*fit].variableIndex(1));
         if(gm[*fit].variableIndex(1)==var)
            nb.insert(gm[*fit].variableIndex(0));
      }

      ij[var+1]=ij[var]+nb.size();
      for(size_t ind=ij[var]; ind<ij[var+1];++ind)
         w[ind]=0;

      size_t c=ij[var];
      for(std::set<size_t>::iterator it=nb.begin(); it !=nb.end(); ++it){
         ir[c]=*it;
         ++c;
      }

      for(GraphicalModelType::ConstFactorIterator fit=gm.factorsOfVariableBegin(var); fit !=gm.factorsOfVariableEnd(var); ++fit){
         size_t var2;
         size_t ind;
         if(gm[*fit].variableIndex(0)==var){
            var2 = gm[*fit].variableIndex(1);
            for(ind=ij[var]; ind<ij[var+1];++ind)
               if(ir[ind]==var2)
                  break;
         }
         if(gm[*fit].variableIndex(1)==var){
             var2 = gm[*fit].variableIndex(0);
            for(ind=ij[var]; ind<ij[var+1];++ind)           
               if(ir[ind]==var2)
                  break;
         }
         const ValueType v00 = gm[*fit](l00);
         const ValueType v01 = gm[*fit](l01);
         C[0] += v00;
         w[ind] += v01-v00;
      }
   }
   C[0]/=2.0;
   /*  

   C[0]=0;
   for(size_t i=0; i<numVar;++i)
      for(size_t j=0; j<numVar;++j) 
         W[i+j*numVar]=0;
    
   for(IndexType f=0; f<gm.numberOfFactors();++f){
      IndexType vars[] = {0,0};
      vars[0] = gm[f].variableIndex(0);
      vars[1] = gm[f].variableIndex(1);
      ValueType v00 = gm[f](l00);
      ValueType v01 = gm[f](l01);
      
      C[0] += v00;
      W[vars[0]+vars[1]*numVar] += v01-v00;
      W[vars[1]+vars[0]*numVar] += v01-v00;
   }
   */
   mxFree(filename);
}
