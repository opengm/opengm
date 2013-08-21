// Code written by Jan Lellmann and Joerg Hendrik Kappes
// Can be used to build grid structure discrete models with the same regularizer
// also implements different versions of total variation discretizations

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
   opengm::meta::TypeListGenerator<opengm::ExplicitFunction<ValueType> >::type,
   opengm::DiscreteSpace<IndexType, LabelType>
   > GraphicalModelType;

typedef opengm::ExplicitFunction<ValueType> ExplicitFunctionType;
typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;


template<typename G, typename F> void addfac1(G& g, F& f, const mwSize* dims, size_t i1, size_t j1) {
   IndexType vars[1];
   vars[0] = i1 + j1 * (size_t)(dims[0]);
   g.addFactor(f, vars, vars+1);
}

template<typename G, typename F> void addfac2(G& g, F& f, const mwSize* dims, size_t i1, size_t j1, size_t i2, size_t j2) {
   IndexType vars[2];
   vars[0] = i1 + j1 * (size_t)(dims[0]);
   vars[1] = i2 + j2 * (size_t)(dims[0]);
   g.addFactor(f, vars, vars+2);
}

template<typename G, typename F> void addfac3(G& g, F& f, const mwSize* dims, size_t i1, size_t j1, size_t i2, size_t j2, size_t i3, size_t j3) {
   IndexType vars[3];
   vars[0] = i1 + j1 * (size_t)(dims[0]);
   vars[1] = i2 + j2 * (size_t)(dims[0]);
   vars[2] = i3 + j3 * (size_t)(dims[0]);
   g.addFactor(f, vars, vars+3);
}

template<typename G, typename F> void addfac4(G& g, F& f, const mwSize* dims, size_t i1, size_t j1, size_t i2, size_t j2, size_t i3, size_t j3, size_t i4, size_t j4) {
   IndexType vars[4];
   vars[0] = i1 + j1 * (size_t)(dims[0]);
   vars[1] = i2 + j2 * (size_t)(dims[0]);
   vars[2] = i3 + j3 * (size_t)(dims[0]);
   vars[3] = i4 + j4 * (size_t)(dims[0]);
   g.addFactor(f, vars, vars+4);
}

bool indexvalid(const mwSize* dims, size_t i, size_t j) {
   return ((i >= 0) && (j >= 0) && (i < (size_t)(dims[0])) && (j < (size_t)(dims[1])));
}

template<typename G, typename F> void addfac2ifvalid(G& g, F& f, const mwSize* dims, size_t i1, size_t j1, size_t i2, size_t j2) {
   if (indexvalid(dims,i1,j1) && indexvalid(dims,i2,j2)) {
      addfac2(g, f, dims, i1, j1, i2, j2);
   }
}

void mxCheck(bool cond, const char* msg) {
   if (!cond) mexErrMsgTxt(msg);
}

template<class FUNCTION, class GM>
FunctionIdentifier addFunction
(
   std::map<std::vector<typename GM::ValueType>,FunctionIdentifier>& functionIdMap, 
   GM& gm,
   FUNCTION& f, 
   std::vector<typename GM::ValueType>& v,
   bool compression
   )
{
   if(compression && functionIdMap.find(v) == functionIdMap.end()) {
      FunctionIdentifier fid = gm.addFunction(f);
      functionIdMap[v] = fid;
      return fid;
   }
   else{
      return gm.addFunction(f);
   }
}
      

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   /* INPUT */
   /* 1.) model-type*/
   /* 2.) data-matrix nDim1 x ... x nDimN x nLabels */
   /* 3.) output-filename */ 
   /* 4.) regularizer parameter [optional] */


   if (nrhs < 3) {
      mexErrMsgTxt("Incorrect number of inputs.");
   }
   /* Check data type of input argument */
   if (!(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Input array 2 must be of type double.");
   }
   /* Check data dimension */
   if(mxGetNumberOfDimensions(prhs[1]) !=3) {
      mexErrMsgTxt("Input array 2 must have 3 dimensions - 2 spatial and the last one for the labels.\n Higher order grids are not supported yet.");
   }

   char* modelType;
   char* filename;
   mwSize numberOfDims;
   mwSize numberOfVars;
   mwSize numberOfLabels;
   const mwSize *dims;
   bool compression = false;
   std::map<std::vector<ValueType>,FunctionIdentifier> functionIdMap;
   if(mxIsChar(prhs[nrhs-1])) {
      char* temp = (char*)mxCalloc(mxGetN(prhs[nrhs-1])+1,sizeof(char)); 
      mxGetString(prhs[nrhs-1], modelType, mxGetN(prhs[nrhs-1])+1);
      if(strcmp(temp,"compress") == 0) {
         compression = true; 
         printf("Enable sharing functions! \n",modelType);
      }
      else{
         printf("Disable sharing functions! \n",modelType);
      }
   }
   modelType      = (char*)mxCalloc(mxGetN(prhs[0])+1,sizeof(char));
   filename       = (char*)mxCalloc(mxGetN(prhs[2])+1,sizeof(char));
   mxGetString(prhs[0], modelType, mxGetN(prhs[0])+1);
   mxGetString(prhs[2], filename, mxGetN(prhs[2])+1);
   numberOfDims   = mxGetNumberOfDimensions(prhs[1])-1;
   dims       = mxGetDimensions(prhs[1]);

   /* Output Information*/
   printf("Model type:           %s\n",modelType);
   printf("Number of dimensions: %d",numberOfDims);
   printf(" ( ");
   for(size_t i=0; i<numberOfDims; ++i) {
      printf("%d",dims[i]);
      if(i<numberOfDims-1)
         printf(", ");
   }
   printf(" )\n");
   numberOfLabels = dims[numberOfDims];
   printf("Number of labels:     %d\n",numberOfLabels);
   numberOfVars=1;
   for(size_t i=0; i<numberOfDims; ++i) {
      numberOfVars *= dims[i];
   }
   printf("Number of variables:  %d\n",numberOfVars);
   printf("Output file:          %s\n",filename);


   // Build model
   std::vector<LabelType> numOfLabels((IndexType)(numberOfVars),(LabelType)(numberOfLabels));
   GraphicalModelType gm(opengm::DiscreteSpace<IndexType, LabelType >(numOfLabels.begin(), numOfLabels.end()) );

   LabelType nos = (LabelType)(numberOfLabels);
   ExplicitFunctionType f(&nos, &nos+1, 0);

   for(size_t d0=0; d0<(size_t)(dims[0]); ++d0) {
      for(size_t d1=0; d1<(size_t)(dims[1]); ++d1) {
         size_t var = d0 + d1 * (size_t)(dims[0]);
         std::vector<ValueType> v(numberOfLabels);
         for(LabelType l=0; l<numberOfLabels; ++l) {
            f(l) = (ValueType)mxGetPr(prhs[1])[l*numberOfVars+var];
            v[l] = (ValueType)mxGetPr(prhs[1])[l*numberOfVars+var];
         } 
         FunctionIdentifier fid = addFunction(functionIdMap,gm,f,v,compression);
         //FunctionIdentifier fid = gm.addFunction(f);
         addfac1(gm, fid, dims, d0, d1);
      }
   }
   functionIdMap.clear();

   bool reg_valid = true;
   if(strcmp(modelType,"none") == 0) {
      //no regularizer
   }else if(strcmp(modelType,"N4") == 0) {
      printf("Regularizer: 4-neighborhood\n");

      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input array 3 must have 2 dimensions.");
      mxCheck(mxGetDimensions(prhs[3])[0] == numberOfLabels, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == numberOfLabels, "Input array 3 has wrong size.");

      LabelType nos[]= {numberOfLabels, numberOfLabels};
      ExplicitFunctionType f2(nos, nos+2, 0);

      for(LabelType l0=0; l0<nos[0]; ++l0) {
         for(LabelType l1=0; l1<nos[1]; ++l1) {
            f2(l0,l1) = (ValueType)mxGetPr(prhs[3])[l0+l1*nos[0]];
         }
      }
      FunctionIdentifier fid2 = gm.addFunction(f2);
      for(size_t d0=0; d0<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1<(size_t)(dims[1]); ++d1) {
            addfac2ifvalid(gm, fid2, dims, d0,d1, d0+1,d1);
            addfac2ifvalid(gm, fid2, dims, d0,d1, d0,d1+1);
         }
      }
   } else if(strcmp(modelType,"N8")==0) {
      printf("Regularizer: 8-neighborhood\n");

      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input array 3 must have 2 dimensions.");
      mxCheck(mxGetDimensions(prhs[3])[0] == numberOfLabels, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == numberOfLabels, "Input array 3 has wrong size.");

      LabelType nos[]= {numberOfLabels, numberOfLabels};
      ExplicitFunctionType f2(nos, nos+2, 0);
      for(LabelType l0=0; l0<nos[0]; ++l0) {
         for(LabelType l1=0; l1<nos[1]; ++l1) {
            f2(l0,l1) = (ValueType)mxGetPr(prhs[3])[l0+l1*nos[0]];
         }
      }
      ExplicitFunctionType f2diag(nos, nos+2, 0);
      for(LabelType l0=0; l0<nos[0]; ++l0) {
         for(LabelType l1=0; l1<nos[1]; ++l1) {
            f2diag(l0,l1) = ((ValueType)mxGetPr(prhs[3])[l0+l1*nos[0]]) / sqrt(2.0); // scale diagonal potentials
         }
      }

      FunctionIdentifier fid2 = gm.addFunction(f2);
      FunctionIdentifier fid2diag = gm.addFunction(f2diag);
      for(size_t d0=0; d0<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1<(size_t)(dims[1]); ++d1) {
            addfac2ifvalid(gm, fid2, dims, d0,d1, d0+1,d1);
            addfac2ifvalid(gm, fid2, dims, d0,d1, d0,d1+1);
            addfac2ifvalid(gm, fid2diag, dims, d0,d1, d0+1,d1+1);
            addfac2ifvalid(gm, fid2diag, dims, d0+1,d1, d0,d1+1);
         }
      }
   } else if(strcmp(modelType,"FD-L2")==0) {
      // In the two-class case (numberOfLabels), this corresponds to a 6-neighborhood
      // with 2-potentials with horizontal/vertical (a-b,a-c) weights sqrt(2)/2.0 and
      // diagonal (b-c) weight (2.0-sqrt(2.0))/2.0.
      printf("Regularizer: Forward differences L2 norm; 3-potentials\n");

      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input 3 must be scalar.");
      mxCheck(mxGetDimensions(prhs[3])[0] == 1, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == 1, "Input array 3 has wrong size.");
      double regweight = mxGetScalar(prhs[3]);
      printf("Regularizer weight: %.5f\n", regweight);	  
	  
      LabelType nos2[] = {numberOfLabels, numberOfLabels};
      LabelType nos3[] = {numberOfLabels, numberOfLabels, numberOfLabels};

      // layout:  a  b
      //          c
      ExplicitFunctionType f3(nos3, nos3+3, 0);
      for(LabelType a=0; a<nos3[0]; ++a) {
         for(LabelType b=0; b<nos3[1]; ++b) {
            for(LabelType c=0; c<nos3[2]; ++c) {
               double t;
               if ((a == b) && (b == c))
                  t = 0;
               else if (((a == b) && (a != c)) || ((a == c) && (a != b)))
                  t = 1.0;
               else if ((a != b) & (a != c))
                  t = sqrt(2.0);
               else assert(false); // Logic error
               f3(a,b,c) = t * regweight;
            }
         }
      }
      FunctionIdentifier fid3 = gm.addFunction(f3);

      ExplicitFunctionType f2(nos2, nos2+2, 0);
      for(LabelType a=0; a<nos2[0]; ++a) {
         for(LabelType b=0; b<nos2[1]; ++b) {
            f2(a,b) = ((a != b) ? 1.0 : 0.0) * regweight;
         }
      }
      FunctionIdentifier fid2 = gm.addFunction(f2);
	 
      for(size_t d0=0; d0<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1<(size_t)(dims[1]); ++d1) {
            if (indexvalid(dims,d0,d1) && indexvalid(dims,d0+1,d1) && indexvalid(dims,d0,d1+1)) {
               // interior: 3-potentials
               addfac3(gm, fid3, dims, d0,d1, d0+1,d1, d0,d1+1);
            } else if (indexvalid(dims,d0,d1) && indexvalid(dims,d0+1,d1)) {
               // boundary: 2-potentials
               addfac2(gm, fid2, dims, d0,d1, d0+1,d1);
            } else if (indexvalid(dims,d0,d1) && indexvalid(dims,d0,d1+1)) {
               // boundary: 2-potentials
               addfac2(gm, fid2, dims, d0,d1, d0,d1+1);
            }
         }
      }
   } else if(strcmp(modelType,"FD-ENV")==0) {
      printf("Regularizer: Forward differences envelope norm; 3-potentials\n");

      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input 3 must be scalar.");
      mxCheck(mxGetDimensions(prhs[3])[0] == 1, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == 1, "Input array 3 has wrong size.");
      double regweight = mxGetScalar(prhs[3]);
      printf("Regularizer weight: %.5f\n", regweight);
	  	  
      LabelType nos2[] = {numberOfLabels, numberOfLabels};
      LabelType nos3[] = {numberOfLabels, numberOfLabels, numberOfLabels};

      // layout:  a  b
      //          c
      ExplicitFunctionType f3(nos3, nos3+3, 0);
      for(LabelType a=0; a<nos3[0]; ++a) {
         for(LabelType b=0; b<nos3[1]; ++b) {
            for(LabelType c=0; c<nos3[2]; ++c) {
               double t;
               if ((a == b) && (b == c))
                  t = 0.0;
               else if (((a == b) && (a != c)) || ((a == c) && (a != b)))
                  t = 1.0;
               else if ((b == c) && (a != b))
                  t = sqrt(2.0);
               else if ((a != b) && (a != c) && (b != c))
                  t = sqrt(2.0+sqrt(3.0));
               else assert(false); // Logic error
               f3(a,b,c) = t * regweight;
            }
         }
      }
      FunctionIdentifier fid3 = gm.addFunction(f3);

      ExplicitFunctionType f2(nos2, nos2+2, 0);
      for(LabelType a=0; a<nos2[0]; ++a) {
         for(LabelType b=0; b<nos2[1]; ++b) {
            f2(a,b) = ((a != b) ? 1.0 : 0.0) * regweight;
         }
      }
      FunctionIdentifier fid2 = gm.addFunction(f2);
	 
      for(size_t d0=0; d0<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1<(size_t)(dims[1]); ++d1) {
            if (indexvalid(dims,d0,d1) && indexvalid(dims,d0+1,d1) && indexvalid(dims,d0,d1+1)) {
               // interior: 3-potentials
               addfac3(gm, fid3, dims, d0,d1, d0+1,d1, d0,d1+1);
            } else if (indexvalid(dims,d0,d1) && indexvalid(dims,d0+1,d1)) {
               // boundary: 2-potentials
               addfac2(gm, fid2, dims, d0,d1, d0+1,d1);
            } else if (indexvalid(dims,d0,d1) && indexvalid(dims,d0,d1+1)) {
               // boundary: 2-potentials
               addfac2(gm, fid2, dims, d0,d1, d0,d1+1);
            }
         }
      }
   } else if(strcmp(modelType,"CD-L2")==0) {
      printf("Regularizer: Centered differences L2 norm; 4-potentials\n");

      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input 3 must be scalar.");
      mxCheck(mxGetDimensions(prhs[3])[0] == 1, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == 1, "Input array 3 has wrong size.");
      double regweight = mxGetScalar(prhs[3]);
      printf("Regularizer weight: %.5f\n", regweight);
       
      LabelType nos4[] = {numberOfLabels, numberOfLabels, numberOfLabels, numberOfLabels};

      //TODO add boundary handling to get more exact discretization at boundaries
      
      // layout:  a  b
      //          c  d
      ExplicitFunctionType f4(nos4, nos4+4, 0);
      for(LabelType a=0; a<nos4[0]; ++a) {
         for(LabelType b=0; b<nos4[1]; ++b) {
            for(LabelType c=0; c<nos4[2]; ++c) {
               for(LabelType d=0; d<nos4[3]; ++d) {
                  double t;
                  if ((a == b) && (b == c) && (c == d))
                     // 4 equal
                     t = 0;
                  else if (((a == b) && (b == c) && (b != d)) || 
                           ((b == c) && (c == d) && (c != a)) || 
                           ((c == d) && (d == a) && (d != b)) || 
                           ((d == a) && (a == b) && (a != c)))
                     // 3 equal
                     t = sqrt(2.0) / 2.0;
                  else if (((a == b) && (c == d) && (a != c)) || 
                           ((a == c) && (b == d) && (a != b)))
                     // 2+2 equal, horizontal or vertical
                     t = 1.0;
                  else if ((a == d) && (b == c) && (a != b))
                     // 2+2 equal, diagonal
                     t = 0; //Jan: This is correct: The gradient is computed on the cell center.
                            //     If a == d =: v1 and b == c =: v2 then the gradient is averaged as
                            //     ((b - a) + (d - c))/2 == ((v2-v1)+(v1-c2))/2 == 0.
                  else if (((a == b) && (a != c) && (a != d) && (c != d)) || 
                           ((a == c) && (a != b) && (a != d) && (b != d)) || 
                           ((b == d) && (b != a) && (b != c) && (a != c)) || 
                           ((c == d) && (c != a) && (c != b) && (a != b)))
                     // 2 equal, 2 different, horizontal or vertical
                     t =  1.0;
                  else if (((a == d) && (a != b) && (a != c) && (b != c)) || 
                           ((b == c) && (b != a) && (b != d) && (a != d)))
                     // 2 equal, 2 different, diagonal
                     t = sqrt(2)/2.0;
                  else if ((a != b) && (a != c) && (a != d) && (b != c) && (b != d) && (c != d))
                     t = 1.0;
                  else assert(false); // Logic error
                  f4(a,b,c,d) = regweight * t;
               }
            }
         }
      }
      FunctionIdentifier fid4 = gm.addFunction(f4);

      for(size_t d0=0; d0+1<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1+1<(size_t)(dims[1]); ++d1) {
            addfac4(gm, fid4, dims, d0,d1, d0+1,d1, d0,d1+1, d0+1,d1+1);
         }
      }
	  
	  // horizontal/vertical boundary, not at corner; corners evaluate to zero
      ExplicitFunctionType f2(nos4, nos4+2, 0);
      for(LabelType a=0; a<nos4[0]; ++a) {
        for(LabelType b=0; b<nos4[0]; ++b) {
		    f2(a,b) = 0.5 * f4(a,a,b,b); // boundary cells are rectangular, i.e., only half as wide
	    }
      }
      FunctionIdentifier fid2 = gm.addFunction(f2);
      for(size_t d0=0; d0+1<(size_t)(dims[0]); ++d0) {
	     addfac2(gm, fid2, dims, d0, 0, d0+1, 0);
		 addfac2(gm, fid2, dims, d0, dims[1]-1, d0, dims[1]-1);
      }
      for(size_t d1=0; d1+1<(size_t)(dims[1]); ++d1) {
	     addfac2(gm, fid2, dims, 0, d1, 0, d1+1);
		 addfac2(gm, fid2, dims, dims[0]-1, d1, dims[0]-1, d1+1);
      }
   } else if(strcmp(modelType,"CD-ENV")==0) {
      printf("Regularizer: Centered differences envelope norm; 4-potentials\n");

      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input 3 must be scalar.");
      mxCheck(mxGetDimensions(prhs[3])[0] == 1, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == 1, "Input array 3 has wrong size.");
      double regweight = mxGetScalar(prhs[3]);
      printf("Regularizer weight: %.5f\n", regweight);
	  
      LabelType nos4[] = {numberOfLabels, numberOfLabels, numberOfLabels, numberOfLabels};

      // layout:  a  b
      //          c  d
      ExplicitFunctionType f4(nos4, nos4+4, 0);
      for(LabelType a=0; a<nos4[0]; ++a) {
         for(LabelType b=0; b<nos4[1]; ++b) {
            for(LabelType c=0; c<nos4[2]; ++c) {
               for(LabelType d=0; d<nos4[3]; ++d) {
                  double t;
                  if ((a == b) && (b == c) && (c == d))
                     // 4 equal
                     t = 0.0;
                  else if (((a == b) && (b == c) && (b != d)) ||
                           ((b == c) && (c == d) && (c != a)) ||
                           ((c == d) && (d == a) && (d != b)) ||
                           ((d == a) && (a == b) && (a != c)))
                     // 3 equal
                     t = sqrt(2.0) / 2.0;
                  else if (((a == b) && (c == d) && (a != c)) ||
                           ((a == c) && (b == d) && (a != b)))
                     // 2+2 equal, horizontal or vertical
                     t = 1.0;
                  else if ((a == d) && (b == c) && (a != b))
                     // 2+2 equal, diagonal
                     t = 0; //Jan: This is correct, see above.
                  else if (((a == b) && (a != c) && (a != d) && (c != d)) ||
                           ((a == c) && (a != b) && (a != d) && (b != d)) ||
                           ((b == d) && (b != a) && (b != c) && (a != c)) ||
                           ((c == d) && (c != a) && (c != b) && (a != b)))
                     // 2 equal, 2 different, horizontal or vertical
                     t =  (sqrt(3.0)+1.0)/2.0;
                  else if (((a == d) && (a != b) && (a != c) && (b != c)) ||
                           ((b == c) && (b != a) && (b != d) && (a != d)))
                     // 2 equal, 2 different, diagonal
                     t = sqrt(2.0) / 2.0;
                  else if ((a != b) && (a != c) && (a != d) && (b != c) && (b != d) && (c != d))
                     t = sqrt(2.0);				  
                  else assert(false); // Logic error
                  f4(a,b,c,d) = t * regweight;
               }
            }
         }
      }
      FunctionIdentifier fid4 = gm.addFunction(f4);

      for(size_t d0=0; d0+1<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1+1<(size_t)(dims[1]); ++d1) {
            addfac4(gm, fid4, dims, d0,d1, d0+1,d1, d0,d1+1, d0+1,d1+1);
         }
      }
	  
	  // horizontal/vertical boundary, not at corner; corners evaluate to zero
      ExplicitFunctionType f2(nos4, nos4+2, 0);
      for(LabelType a=0; a<nos4[0]; ++a) {
        for(LabelType b=0; b<nos4[0]; ++b) {
		    f2(a,b) = 0.5 * f4(a,a,b,b); // boundary cells are rectangular, i.e., only half as wide
	    }
      }
      FunctionIdentifier fid2 = gm.addFunction(f2);
      for(size_t d0=0; d0+1<(size_t)(dims[0]); ++d0) {
	     addfac2(gm, fid2, dims, d0, 0, d0+1, 0);
		 addfac2(gm, fid2, dims, d0, dims[1]-1, d0, dims[1]-1);
      }
      for(size_t d1=0; d1+1<(size_t)(dims[1]); ++d1) {
	     addfac2(gm, fid2, dims, 0, d1, 0, d1+1);
		 addfac2(gm, fid2, dims, dims[0]-1, d1, dims[0]-1, d1+1);
      }
   } else if(strcmp(modelType,"DTV-2") == 0) {
      printf("Regularizer: Discrete TV Approximation; 2-potentials\n");
  
      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input 3 must be scalar.");
      mxCheck(mxGetDimensions(prhs[3])[0] == 1, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == 1, "Input array 3 has wrong size.");
      double regweight = mxGetScalar(prhs[3]);
      printf("Regularizer weight: %.5f\n", regweight);
     
      LabelType nos[]= {numberOfLabels, numberOfLabels};
      ExplicitFunctionType f2(nos, nos+2, 0);

      for(LabelType l0=0; l0<nos[0]; ++l0) {
         for(LabelType l1=0; l1<nos[1]; ++l1) {
            f2(l0,l1) = ((l0 != l1) ? 1.0 : 0.0) * regweight;
         }
      }
      FunctionIdentifier fid2 = gm.addFunction(f2);
      for(size_t d0=0; d0<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1<(size_t)(dims[1]); ++d1) {
            addfac2ifvalid(gm, fid2, dims, d0,d1, d0+1,d1);
            addfac2ifvalid(gm, fid2, dims, d0,d1, d0,d1+1);
         }
      }
   } else if(strcmp(modelType,"DTV-3")==0) {
      printf("Regularizer: Discrete TV Approximation; 3-potentials\n");

      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input 3 must be scalar.");
      mxCheck(mxGetDimensions(prhs[3])[0] == 1, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == 1, "Input array 3 has wrong size.");
      double regweight = mxGetScalar(prhs[3]);
      printf("Regularizer weight: %.5f\n", regweight);
	  	  
      LabelType nos2[] = {numberOfLabels, numberOfLabels};
      LabelType nos3[] = {numberOfLabels, numberOfLabels, numberOfLabels};

      // layout:  a  b
      //          c
      ExplicitFunctionType f3(nos3, nos3+3, 0);
      for(LabelType a=0; a<nos3[0]; ++a) {
         for(LabelType b=0; b<nos3[1]; ++b) {
            for(LabelType c=0; c<nos3[2]; ++c) {
               double t;
               if ((a == b) && (b == c))
                  t = 0.0;
               else if( (a == c) && ( b != a ) ) {
                  t = sqrt(0.5);
               }else if (((b == a) && (a != c)) || ((b == c) && (c != b)))
                  t = 0.5;
               else if ((a != b) && (a != c) && (b != c))
                  t = 1;
               else assert(false); // Logic error
               f3(a,b,c) = t * regweight;
            }
         }
      }
      FunctionIdentifier fid3 = gm.addFunction(f3);

      ExplicitFunctionType f2(nos2, nos2+2, 0);
      for(LabelType a=0; a<nos2[0]; ++a) {
         for(LabelType b=0; b<nos2[1]; ++b) {
            f2(a,b) = ((a != b) ? 0.5 : 0.0) * regweight;
         }
      }
      FunctionIdentifier fid2 = gm.addFunction(f2);
	 
      for(size_t d0=0; d0<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1<(size_t)(dims[1]); ++d1) {
            if (indexvalid(dims,d0-1,d1) && indexvalid(dims,d0,d1) && indexvalid(dims,d0,d1+1)) {
               // interior: 3-potentials
               addfac3(gm, fid3, dims, d0-1,d1, d0,d1, d0,d1+1);
            }
            if (indexvalid(dims,d0,d1-1) && indexvalid(dims,d0,d1) && indexvalid(dims,d0+1,d1)) {
               // interior: 3-potentials
               addfac3(gm, fid3, dims, d0,d1-1, d0,d1, d0+1,d1);
            } 
            if ((!indexvalid(dims,d0,d1-1) || !indexvalid(dims,d0,d1+1)) && indexvalid(dims,d0+1,d1)) {
               // boundary: 2-potentials
               addfac2(gm, fid2, dims, d0,d1, d0+1,d1);
            } 
            if ((!indexvalid(dims,d0-1,d1) || !indexvalid(dims,d0+1,d1)) && indexvalid(dims,d0,d1+1)) {
               // boundary: 2-potentials
               addfac2(gm, fid2, dims, d0,d1, d0,d1+1);
            }
         }
      }
   } else if(strcmp(modelType,"DTV4")==0) {
      printf("Regularizer: Discrete Total Variation; 4-potentials\n");

      mxCheck(mxGetNumberOfDimensions(prhs[3]) == 2, "Input 3 must be scalar.");
      mxCheck(mxGetDimensions(prhs[3])[0] == 1, "Input array 3 has wrong size.");
      mxCheck(mxGetDimensions(prhs[3])[1] == 1, "Input array 3 has wrong size.");
      double regweight = mxGetScalar(prhs[3]);
      printf("Regularizer weight: %.5f\n", regweight);
	  
      LabelType nos4[] = {numberOfLabels, numberOfLabels, numberOfLabels, numberOfLabels};

      // layout:  a  b
      //          c  d
      ExplicitFunctionType f2(nos4, nos4+2, 0); 
      ExplicitFunctionType f4(nos4, nos4+4, 0);
      for(LabelType a=0; a<nos4[0]; ++a) {
         for(LabelType b=0; b<nos4[1]; ++b) {
            if(a!=b) f2(a,b) = 0.5* regweight;
            for(LabelType c=0; c<nos4[2]; ++c) {
               for(LabelType d=0; d<nos4[3]; ++d) {
                  double t;
                  if ((a == b) && (b == c) && (c == d))
                     // 4 equal
                     t = 0.0;
                  else if (((a == b) && (b == c) && (b != d)) ||
                           ((b == c) && (c == d) && (c != a)) ||
                           ((c == d) && (d == a) && (d != b)) ||
                           ((d == a) && (a == b) && (a != c)))
                     // 3 equal
                     t = sqrt(0.5);
                  else if (((a == b) && (c == d) && (a != c)) ||
                           ((a == c) && (b == d) && (a != b)))
                     // 2+2 equal, horizontal or vertical
                     t = 1.0;
                  else if ((a == d) && (b == c) && (a != b))
                     // 2+2 equal, diagonal
                     t = 2.0; 
                  else if (((a == b) && (a != c) && (a != d) && (c != d)) ||
                           ((a == c) && (a != b) && (a != d) && (b != d)) ||
                           ((b == d) && (b != a) && (b != c) && (a != c)) ||
                           ((c == d) && (c != a) && (c != b) && (a != b)))
                     // 2 equal, 2 different, horizontal or vertical
                     t =  1.5;
                  else if (((a == d) && (a != b) && (a != c) && (b != c)) ||
                           ((b == c) && (b != a) && (b != d) && (a != d)))
                     // 2 equal, 2 different, diagonal
                     t = 2*sqrt(0.5);
                  else if ((a != b) && (a != c) && (a != d) && (b != c) && (b != d) && (c != d))
                     t = 2.0;				  
                  else assert(false); // Logic error
                  f4(a,b,c,d) = t * regweight;
               }
            }
         }
      }
      FunctionIdentifier fid2 = gm.addFunction(f2);
      FunctionIdentifier fid4 = gm.addFunction(f4);

      // Inner Factors
      for(size_t d0=0; d0+1<(size_t)(dims[0]); ++d0) {
         for(size_t d1=0; d1+1<(size_t)(dims[1]); ++d1) {
            addfac4(gm, fid4, dims, d0,d1, d0+1,d1, d0,d1+1, d0+1,d1+1);
         }
      }
      // Border Factors
      for(size_t d0=0; d0+1<(size_t)(dims[0]); ++d0) {
         addfac2(gm, fid2, dims, d0,0, d0+1,0);
         addfac2(gm, fid2, dims, d0,dims[1]-1, d0+1,dims[1]-1);
      }  
      for(size_t d1=0; d1+1<(size_t)(dims[1]); ++d1) {
         addfac2(gm, fid2, dims, 0,d1, 0,d1+1);
         addfac2(gm, fid2, dims, dims[0]-1,d1, dims[0]-1,d1+1);
      }
   } else {
     mexErrMsgTxt("Invalid regularizer.");
	 reg_valid = false;
   }
   if (reg_valid) {
     opengm::hdf5::save(gm, filename, "gm");
   }

   mxFree(modelType);
   mxFree(filename);
}
