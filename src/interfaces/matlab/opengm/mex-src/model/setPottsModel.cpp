//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "matlabModelType.hxx"

/**
 * @brief This file implements an interface to load an opengm Potts model in MatLab.
 *
 * This routine accepts a a double, a unary-MAtrix and a couppling Matrix. 
 * The model is loaded and a handle to the model will be
 * passed back to MatLab for further usage.
 *
 * @param[in] nlhs number of output arguments expected from MatLab
 * (needs to be 1).
 * @param[out] plhs pointer to the mxArrays containing the results. If the model
 * can be loaded, plhs[0] contains the handle to the model.
 * @param[in] nrhs number of input arguments provided from MatLab.
 * (needs to be 3)
 * @param[in] prhs pointer to the mxArrays containing the input data provided by
 * matlab. 
 * prhs[0] is the constant energy term. 
 * prhs[1] is the unray term numLabels times numVariables.
 * prhs[2] includes a (upper triangular) matrix with the couppling strength.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //check if data is in correct format
  if(nrhs != 3) {
     mexErrMsgTxt("Wrong number of input variables specified (three needed)\n");
  }
  if(nlhs != 1) {
     mexErrMsgTxt("Wrong number of output variables specified (one needed)\n");
  }

  // get data
  double* constTerm    = reinterpret_cast<double*>(mxGetData(prhs[0]));
  double* unaryTerm    = reinterpret_cast<double*>(mxGetData(prhs[1]));
  double* pairwiseTerm = reinterpret_cast<double*>(mxGetData(prhs[2]));
 
  mwSize nVar = mxGetN(prhs[1]); 
  mwSize nLab = mxGetM(prhs[1]); 

  // load model
  typedef opengm::interface::MatlabModelType::GmType GmType;
  GmType* gm = new GmType();
  std::vector<size_t> nos(nVar,nLab);                     
  GmType::SpaceType space(nos.begin(),nos.end());
  gm->assign(space);
  std::vector<size_t> shape(1,nLab);
  for(size_t var=0; var<nVar; ++var){
     opengm::interface::MatlabModelType::ExplicitFunction function(shape.begin(), shape.end());
     bool isZero = true;
     if(var==0)
        for(size_t l=0; l<nLab; ++l){
           function(l) = unaryTerm[l+var*nLab]+constTerm[0]; 
           if(function(l)!=0)
              isZero = false;
        }
     else
        for(size_t l=0; l<nLab; ++l){
           function(l) = unaryTerm[l+var*nLab];
           if(function(l)!=0)
              isZero = false;
        }
     if(!isZero){
        GmType::FunctionIdentifier functionID = gm->addFunction(function);
        gm->addFactor(functionID,&var,&var+1);
     }
  }
  for(size_t var0=0; var0<nVar; ++var0){
     for(size_t var1=var0+1; var1<nVar; ++var1){
        if(pairwiseTerm[var0+nVar*var1] !=0 ){
           opengm::interface::MatlabModelType::PottsFunction function(nLab, nLab, 0, pairwiseTerm[var0+nVar*var1]); 
           GmType::FunctionIdentifier functionID = gm->addFunction(function);
           size_t vars[2];
           vars[0] = var0;
           vars[1] = var1;
           gm->addFactor(functionID,vars,vars+2);
        }
     }
  }

  // create handle to model
  plhs[0] = opengm::interface::handle<GmType>::createHandle(gm);
}
