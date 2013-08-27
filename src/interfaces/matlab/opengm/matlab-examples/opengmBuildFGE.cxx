// Written by Jörg Hendrik Kappes
// Transforms grante models into opengm-models


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
   /* 1.) FGE*/
   /* 3.) output-filename */
 
   if (nrhs != 2) {
      mexErrMsgTxt("Incorrect number of inputs.");
   }
   /* Check data type of input argument */
   if (!(mxIsStruct(prhs[0]))) {
      mexErrMsgTxt("Input array 1 must be of type struct.");
   }
  

  
   char* filename; 
   filename  = (char*)mxCalloc(mxGetN(prhs[1])+1,sizeof(char));
   mxGetString(prhs[1], filename, mxGetN(prhs[1])+1);

   //mxArray* fg = mxGetField(prhs[0], 0,"fg");
   mxArray* card = mxGetField(prhs[0], 0,"card"); 
   mxArray* factors = mxGetField(prhs[0], 0,"factors"); 
   size_t numberOfVariables = mxGetNumberOfElements(card);
   size_t numberOfFactors = mxGetNumberOfElements(factors);
 

 
   /* Output Information*/
   printf("Number of variables:  %d\n",numberOfVariables);
   printf("Number of factors:    %d\n",numberOfFactors);
   printf("Output file:          %s\n",filename);

   std::vector<LabelType> numStates(numberOfVariables,0);
   double *pStates = (double*) mxGetData(card);
   for(size_t i=0; i<numberOfVariables;++i) {
      numStates[i] = (size_t)pStates[i];
   }
   GraphicalModelType gm(opengm::DiscreteSpace<IndexType, LabelType >(numStates.begin(), numStates.end()) );

 
   std::map<std::vector<ValueType> , FunctionIdentifier> functionMap;

   //Search Unaries
   std::cout << "Search Unaries"<<std::endl;
   std::vector<std::vector<size_t> > ufac(numberOfVariables);
   for(size_t f=0;f<numberOfFactors;++f) {
      const mxArray* vars = mxGetField(factors, f,"vars");
      const double * pVars = (double*) mxGetData(vars); 
      const size_t numVars =  mxGetNumberOfElements(vars);
      if(numVars == 1) ufac[size_t(pVars[0])-1].push_back(f);
   }
   //Add Unaries
   std::cout << "add Unaries"<<std::endl;
   for(size_t var=0; var<numberOfVariables; ++var) {
      if(ufac[var].size()==0) continue;
      LabelType shape[] = {gm.numberOfLabels(var)};
      ExplicitFunctionType func(shape,shape+1); 
      for(size_t i=0; i<shape[0]; ++i) {
         func(i) = 0;
      }
      for(size_t i=0; i<ufac[var].size();++i) {
         const size_t f = ufac[var][i];
         const mxArray* data = mxGetField(factors, f,"data");
         const double* pData = (double*) mxGetData(data); 
         for(size_t i=0; i<shape[0]; ++i) {
            func(i) += pData[i];
         }
      } 
      FunctionIdentifier fid = gm.addSharedFunction(func);
      gm.addFactor(fid, &var, &var+1);
   }

   
   //ADD NoneUnaries  
   std::cout << "add noneUnaries"<<std::endl;
   for(size_t f=0;f<numberOfFactors;++f) {
      const mxArray* vars = mxGetField(factors, f,"vars");
      const double * pVars = (double*) mxGetData(vars); 
      const mxArray* data = mxGetField(factors, f,"data");
      const double* pData = (double*) mxGetData(data); 
      //const mwSize* dimData = mxGetDimensions(data);
      const size_t  numData = mxGetNumberOfElements(data);
      const size_t numVars =  mxGetNumberOfElements(vars);

      if(numVars == 1)
         continue;

      std::vector<IndexType> varIDs(numVars);
      std::vector<LabelType> shape(numVars);
      std::vector<ValueType> values(numData);
      if(shape.size()==1) {
         varIDs[0] = (size_t)pVars[0]-1;
         shape[0]  = numStates[varIDs[0]];
         for(size_t i=0; i<shape[0]; ++i) {
            values[i] = pData[i];
         }
      }
      else if(shape.size()==2) {
         if(pVars[0]>pVars[1]) {//reorder
            varIDs[0] = (size_t)pVars[1]-1;
            varIDs[1] = (size_t)pVars[0]-1;
            shape[0]  = numStates[varIDs[0]];
            shape[1]  = numStates[varIDs[1]];
            for(size_t i0=0; i0<shape[0]; ++i0) {
               for(size_t i1=0; i1<shape[1]; ++i1) {
                  values[i0+i1*shape[0]] = pData[i1+i0*shape[1]];
               }
            } 
         }
         else{
            varIDs[0] = (size_t)pVars[0]-1;
            varIDs[1] = (size_t)pVars[1]-1;
            shape[0]  = numStates[varIDs[0]];
            shape[1]  = numStates[varIDs[1]];
            for(size_t i0=0; i0<shape[0]; ++i0) {
               for(size_t i1=0; i1<shape[1]; ++i1) {
                  values[i0+i1*shape[0]] = pData[i0+i1*shape[0]];
               }
            } 
         }
      }
      else{
         std::cout << "Can only handle first and second order functions!"<<std::endl;
      }

      
      std::map< std::vector<ValueType> , FunctionIdentifier>::iterator t = functionMap.find(values);
      FunctionIdentifier fid;
      if(t == functionMap.end()) {
         ExplicitFunctionType f(shape.begin(),shape.end());
         if(shape.size()==1) {
            for(size_t i=0; i<shape[0]; ++i) {
               f(i) = values[i];
            }
         }
         if(shape.size()==2) {
            for(size_t i0=0; i0<shape[0]; ++i0) {
               for(size_t i1=0; i1<shape[1]; ++i1) {
                  f(i0,i1) = values[i0+i1*shape[0]];
               }
            }
         }
         else{
/*
            opengm::ShapeWalker< std::vector<size_t>::iterator > walker(shape.begin(),shape.size());
            for(size_t i=0; i<numData; ++i) {
               std::cout <<i <<walker.coordinateTuple()[0]<<std::endl;
               f(walker.coordinateTuple().begin()) = values[i];
               ++walker;
            }
*/
         }
         fid = gm.addFunction(f);
         functionMap[values] = fid;
      }
      else{
         fid = t->second;
      }
      gm.addFactor(fid, varIDs.begin(), varIDs.end());
   }

   opengm::hdf5::save(gm, filename, "gm");

   mxFree(filename);
}
