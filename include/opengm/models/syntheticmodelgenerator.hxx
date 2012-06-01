#pragma once
#ifndef OPENGM_SYNTHETIC_MODEL_GENERATOR_HXX
#define OPENGM_SYNTHETIC_MODEL_GENERATOR_HXX

/// \cond HIDDEN_SYMBOLS

#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <set>
#include <functional>

#include "opengm/graphicalmodel/graphicalmodel.hxx"

namespace opengm {
 
  template<class GM> class SyntheticModelGenerator
  {
    typedef GM                                GraphicalModelType;
    typedef typename GM::ValueType           ValueType;
    typedef typename GM::ExplicitFunctionType ExplicitFunctionType;
    //typedef typename GM::SparseFunctionType   SparseFunctionType;
    //typedef typename GM::ImplicitFunctionType ImplicitFunctionType;
    typedef typename GM::FunctionIdentifier   FunctionIdentifier;
    typedef typename GM::OperatorType	      OperatorType;
    //   typedef typename opengm::GraphicalModel<ValueType,Operatort,opengm::DefaultImplicitFunction<ValueType>,GraphicalModelType::isMutable > RebindGmType;
  public:
    enum FunktionTypes {RANDOM, Potts, GPotts};
    bool randomNumberOfStates_;
    SyntheticModelGenerator();
    SyntheticModelGenerator(bool randomNumberOfStates);
    GM buildGrid2(size_t height, size_t width, size_t numStates, unsigned int id, FunktionTypes ft2, ValueType l1, ValueType l2);
    GM buildGrid02(size_t height, size_t width, size_t numStates, unsigned int id, FunktionTypes ft2, ValueType l2);
    GM buildFull2(size_t var, size_t numStates, unsigned int id, FunktionTypes ft2, ValueType l1, ValueType l2);
    GM buildFull02(size_t var, size_t numStates, unsigned int id, FunktionTypes ft2, ValueType l2);
    GM buildStar2(size_t var, size_t numStates, unsigned int id, FunktionTypes ft2, ValueType l1, ValueType l2);
    //GM buildTree2(size_t var, size_t numStates, unsigned int id, FunktionTypes ft2, ValueType l1, ValueType l2);
    //GM buildFull3(size_t var, size_t numStates, unsigned int id,  FunktionTypes ft3, ValueType l1, ValueType l3);
    GM buildPottsGrid2(size_t height, size_t width, size_t numStates, unsigned int id, ValueType l1=3, ValueType l2=1);
    GM buildPottsFull2(size_t numVar, size_t numStates, unsigned int id, ValueType l1=3, ValueType l2=1);
    GM buildGPottsFull02(size_t numVar, size_t numStates, unsigned int id, ValueType l2=1);
    GM buildGPottsGrid02(size_t height, size_t width, size_t numStates, unsigned int id, ValueType l2=1);
    GM buildPottsFull02(size_t numVar, size_t numStates, unsigned int id, ValueType l2=1);
    GM buildPottsGrid02(size_t height, size_t width, size_t numStates, unsigned int id, ValueType l2=1);
    GM buildRandomFull2(size_t numVar, size_t numStates, unsigned int id, ValueType l1=1, ValueType l2=1);
    GM buildRandomGrid2(size_t height, size_t width, size_t numStates, unsigned int id, ValueType l1=3, ValueType l2=1);
    GM buildRandomStar2(size_t numVar, size_t numStates, unsigned int id, ValueType l1=3, ValueType l2=1);
    //GM buildRandomFull23(size_t numVar, size_t numStates, unsigned int id);
    //GM buildRandomFull3(size_t numVar, size_t numStates, unsigned int id);
  private:
    void addUnaries(GM& gm, ValueType lambda);
    FunctionIdentifier addFunktion(GM& gm, ValueType lambda, FunktionTypes ft, size_t* beginShape, size_t* endShape);
  };
  template<class GM>
  SyntheticModelGenerator<GM>::SyntheticModelGenerator()
  {
    randomNumberOfStates_ = false;
  }
  template<class GM>
  SyntheticModelGenerator<GM>::SyntheticModelGenerator(bool randomNumberOfStates)
  {
    randomNumberOfStates_ = randomNumberOfStates;
  }
  template<class GM>
  void SyntheticModelGenerator<GM>::addUnaries(GM& gm, ValueType lambda1)
  {
    size_t shape[1];
    size_t var[]={0};
    for(size_t i=0;i<gm.numberOfVariables();++i) {
      shape[0] = gm.numberOfLabels(i);
      var[0]   = i;
      ExplicitFunctionType function(shape,shape+1);
      for(size_t ni=0; ni<shape[0]; ++ni) {
	function(ni)= lambda1 * (rand() % 1000000)*0.000001 + 1;
      }
      FunctionIdentifier funcId=gm.addFunction(function);
      gm.addFactor(funcId,var,var+1);
    }
  }
  template<class GM>
  typename GM::FunctionIdentifier SyntheticModelGenerator<GM>::addFunktion
  (
   GM& gm,
   ValueType lambda,
   FunktionTypes ft,
   size_t* beginShape,
   size_t* endShape
   )
  {
    if(ft==RANDOM) {
      ExplicitFunctionType function(beginShape,endShape);
      for(size_t ni=0; ni<beginShape[0]; ++ni) {
	for(size_t nj=0; nj<beginShape[1]; ++nj) {
	  function(ni,nj) = lambda * (rand() % 1000000)*0.000001 + 1;
	}
      }
      FunctionIdentifier funcId=gm.addFunction(function);
      return funcId;
    }
    else if(ft==Potts) {
      ExplicitFunctionType function(beginShape,endShape);
      for(size_t ni=0; ni<beginShape[0]; ++ni) {
	for(size_t nj=0; nj<beginShape[1]; ++nj) {
	  if(ni==nj) function(ni,nj) = 0;//OperatorType::neutral();
	  else       function(ni,nj) = lambda;
	}
      }
      FunctionIdentifier funcId=gm.addFunction(function);
      return funcId;
    }
    else if(ft==GPotts) {
      double v = ((rand()%10000)-5000)/5000.0;
      ExplicitFunctionType function(beginShape,endShape);
      for(size_t ni=0; ni<beginShape[0]; ++ni) {
	for(size_t nj=0; nj<beginShape[1]; ++nj) {
	  if(ni==nj) function(ni,nj) = 0;//OperatorType::neutral();
	  else       function(ni,nj) = lambda * v;
	}
      }
      FunctionIdentifier funcId=gm.addFunction(function);
      return funcId;
    }
    else{
      //throw exception
      FunctionIdentifier funcId;
      return funcId;
    }
  }
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildGrid2
  (
   size_t height,
   size_t width,
   size_t numStates,
   unsigned int id,
   FunktionTypes ft2,
   ValueType lambda1,
   ValueType lambda2
   )
  {
    srand(id);
    size_t N=height*width;
    std::vector<size_t>  variableStates(N,numStates);
    if(randomNumberOfStates_) {
      for(size_t i=0; i<N;++i) {
	variableStates[i] = (rand() % (numStates-1))+1;
      }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //UNARY
    addUnaries(gm, lambda1);
    //PAIRWISE
    size_t shape[2];
    size_t var[2];
    if(randomNumberOfStates_==false ) {
      shape[0] = shape[1] =gm.numberOfLabels(0);
      FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
      for(size_t i=0;i<height;++i) {
	for(size_t j=0;j<width;++j) {
	  size_t v = i+height*j;
	  if(i+1<height) {
	    var[0] = v;
	    var[1] = i+1+height*j;
	    gm.addFactor(funcId,var,var+2);
	  }
	  if(j+1<width) {
	    var[0] = v;
	    var[1] = i+height*(j+1);
	    gm.addFactor(funcId,var,var+2);
	  }
	}
      }
    }
    else{
      for(size_t i=0;i<height;++i) {
	for(size_t j=0;j<width;++j) {
	  size_t v = i+height*j;
	  if(i+1<height) {
	    var[0] = v;
	    var[1] = i+1+height*j;
	    shape[0] = gm.numberOfLabels(var[0]);
	    shape[1] = gm.numberOfLabels(var[1]);
	    FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	    gm.addFactor(funcId,var,var+2);
	  }
	  if(j+1<width) {
	    var[0] = v;
	    var[1] = i+height*(j+1);
	    shape[0] = gm.numberOfLabels(var[0]);
	    shape[1] = gm.numberOfLabels(var[1]);
	    FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	    gm.addFactor(funcId,var,var+2);
	  }
	}
      }
    }
    return gm;
  }
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildGrid02
  (
   size_t height,
   size_t width,
   size_t numStates,
   unsigned int id,
   FunktionTypes ft2,
   ValueType lambda2
   )
  {
    srand(id);
    size_t N=height*width;
    std::vector<size_t>  variableStates(N,numStates);
    if(randomNumberOfStates_) {
      for(size_t i=0; i<N;++i) {
	variableStates[i] = (rand() % (numStates-1))+1;
      }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //PAIRWISE
    size_t shape[2];
    size_t var[2];
    if(randomNumberOfStates_==false ) {
      shape[0] = shape[1] =gm.numberOfLabels(0);
      for(size_t i=0;i<height;++i) {
	for(size_t j=0;j<width;++j) {
	  size_t v = i+height*j;
	  if(i+1<height) {
	    var[0] = v;
	    var[1] = i+1+height*j;
	    FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	    gm.addFactor(funcId,var,var+2);
	  }
	  if(j+1<width) {
	    var[0] = v;
	    var[1] = i+height*(j+1);
	    FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	    gm.addFactor(funcId,var,var+2);
	  }
	}
      }
    }
    else{
      for(size_t i=0;i<height;++i) {
	for(size_t j=0;j<width;++j) {
	  size_t v = i+height*j;
	  if(i+1<height) {
	    var[0] = v;
	    var[1] = i+1+height*j;
	    shape[0] = gm.numberOfLabels(var[0]);
	    shape[1] = gm.numberOfLabels(var[1]);
	    FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	    gm.addFactor(funcId,var,var+2);
	  }
	  if(j+1<width) {
	    var[0] = v;
	    var[1] = i+height*(j+1);
	    shape[0] = gm.numberOfLabels(var[0]);
	    shape[1] = gm.numberOfLabels(var[1]);
	    FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	    gm.addFactor(funcId,var,var+2);
	  }
	}
      }
    }
    return gm;
  }
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildFull2
  (
   size_t numberOfVariables,
   size_t numberOfLabels,
   unsigned int id,
   FunktionTypes ft2,
   ValueType lambda1,
   ValueType lambda2
   )
  {
    srand(id);
    std::vector<size_t>  variableStates(numberOfVariables,numberOfLabels);
    if(randomNumberOfStates_) {
      for(size_t i=0; i<numberOfVariables;++i) {
	variableStates[i] =  (rand() % (numberOfLabels-1))+1;
      }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //UNARY
    addUnaries(gm, lambda1);
    //PAIRWISE
    size_t shape[2];
    size_t var[2];
    if(randomNumberOfStates_==false )
      {
        shape[0]=gm.numberOfLabels(0);
        shape[1]=gm.numberOfLabels(0);
	FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape, shape+2);
        for(size_t i=0;i<numberOfVariables;++i)
	  {
            for(size_t j=i+1;j<numberOfVariables;++j)
	      {
                var[0] = i;
                var[1] = j;
                gm.addFactor(funcId,var,var+2);
	      }
	  }
      }
    else{
      for(size_t i=0;i<numberOfVariables;++i) {
	for(size_t j=i+1;j<numberOfVariables;++j) {
	  var[0] = i;
	  var[1] = j;
	  shape[0] = gm.numberOfLabels(var[0]);
	  shape[1] = gm.numberOfLabels(var[1]);
	  FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	  gm.addFactor(funcId,var,var+2);
	}
      }
    }
    return gm;
  }
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildFull02
  (
   size_t numberOfVariables,
   size_t numberOfLabels,
   unsigned int id,
   FunktionTypes ft2,
   ValueType lambda2
   )
  {
    srand(id);
    std::vector<size_t>  variableStates(numberOfVariables,numberOfLabels);
    if(randomNumberOfStates_) {
      for(size_t i=0; i<numberOfVariables;++i) {
	variableStates[i] = (rand() % (numberOfLabels-1))+1;
      }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //PAIRWISE
    size_t shape[2];
    size_t var[2];
    if(randomNumberOfStates_==false )
      {
        shape[0]=gm.numberOfLabels(0);
        shape[1]=gm.numberOfLabels(0);
        for(size_t i=0;i<numberOfVariables;++i)
	  {
            for(size_t j=i+1;j<numberOfVariables;++j)
	      {
                var[0] = i;
                var[1] = j;
		FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape, shape+2);
                gm.addFactor(funcId,var,var+2);
	      }
	  }
      }
    else{
      for(size_t i=0;i<numberOfVariables;++i) {
	for(size_t j=i+1;j<numberOfVariables;++j) {
	  var[0] = i;
	  var[1] = j;
	  shape[0] = gm.numberOfLabels(var[0]);
	  shape[1] = gm.numberOfLabels(var[1]);
	  FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	  gm.addFactor(funcId,var,var+2);
	}
      }
    }
    return gm;
  }
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildStar2
  (
   size_t numberOfVariables,
   size_t numberOfLabels,
   unsigned int id,
   FunktionTypes ft2,
   ValueType lambda1,
   ValueType lambda2
   )
  {
    srand(id);
    std::vector<size_t>  variableStates(numberOfVariables,numberOfLabels);
    if(randomNumberOfStates_) {
      for(size_t i=0; i<numberOfVariables;++i) {
	variableStates[i] = (rand() % (numberOfLabels-1))+1;
      }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    size_t root = (rand() % numberOfVariables);
    //UNARY
    addUnaries(gm, lambda1);
    //PAIRWISE
    size_t shape[2];
    size_t var[2];
    if(randomNumberOfStates_==false ) {
      shape[0] = shape[1] =gm.numberOfLabels(0);
      FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
      for(size_t i=0;i<root;++i) {
	var[0] = i;
	var[1] = root;
	gm.addFactor(funcId,var,var+2);
      }
      for(size_t i=root+1;i<numberOfVariables;++i) {
	var[0] = root;
	var[1] = i;
	gm.addFactor(funcId,var,var+2);
      }
    }
    else{
      for(size_t i=0;i<root;++i) {
	var[0] = i;
	var[1] = root;
	shape[0] = gm.numberOfLabels(var[0]);
	shape[1] = gm.numberOfLabels(var[1]);
	FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	gm.addFactor(funcId,var,var+2);
      }
      for(size_t i=root+1;i<numberOfVariables;++i) {
	var[0] = root;
	var[1] = i;
	shape[0] = gm.numberOfLabels(var[0]);
	shape[1] = gm.numberOfLabels(var[1]);
	FunctionIdentifier funcId = addFunktion(gm,lambda2,ft2, shape,shape+2);
	gm.addFactor(funcId,var,var+2);
      }
    }
    return gm;
  }
  /////////////////////////
  /////////////////////////
  /////////////////////////
  /////////////////////////
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildPottsGrid2
  (size_t height, size_t width, size_t numStates, unsigned int id, ValueType l1, ValueType l2)
  {return buildGrid2(height, width, numStates, id, Potts, l1, l2);}
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildPottsGrid02
  (size_t height, size_t width, size_t numStates, unsigned int id, ValueType l2)
  {return buildGrid02(height, width, numStates, id, Potts, l2);}
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildGPottsGrid02
  (size_t height, size_t width, size_t numStates, unsigned int id, ValueType l2)
  {return buildGrid02(height, width, numStates, id, GPotts, l2);}
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildRandomGrid2
  (size_t height, size_t width, size_t numStates, unsigned int id, ValueType l1, ValueType l2)
  {return buildGrid2(height, width, numStates, id, RANDOM, l1, l2);}
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildPottsFull2
  (size_t numVar, size_t numStates, unsigned int id, ValueType l1, ValueType l2)
  {return buildFull2(numVar, numStates, id, Potts, l1, l2);}
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildPottsFull02
  (size_t numVar, size_t numStates, unsigned int id, ValueType l2)
  {return buildFull02(numVar, numStates, id, Potts, l2);}
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildGPottsFull02
  (size_t numVar, size_t numStates, unsigned int id, ValueType l2)
  {return buildFull02(numVar, numStates, id, GPotts, l2);}
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildRandomFull2
  (size_t numVar, size_t numStates, unsigned int id, ValueType l1, ValueType l2)
  {return buildFull2(numVar, numStates, id, RANDOM, l1, l2);}
  template<class GM>
  GM SyntheticModelGenerator<GM>::buildRandomStar2
  (size_t numVar, size_t numStates, unsigned int id, ValueType l1, ValueType l2)
  {return buildStar2(numVar, numStates, id, RANDOM, l1, l2);}
  /*
    template<class GM>
    GM ModelGenerator<GM>::buildPottsGrid(size_t height, size_t width, size_t numStates, unsigned int id, bool variableNumStates)
    {
    srand(id);
    size_t N=height*width;
    std::vector<size_t>  variableStates(N,numStates);
    if(variableNumStates) {
    for(size_t i=0; i<N;++i) {
    variableStates[i] = (rand() % (numStates-1))+1;
    }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //UNARY
    {
    size_t shape[1];
    size_t var[]={0};
    for(size_t i=0;i<N;++i) {
    shape[0] = variableStates[i];
    var[0]   = i;
    ExplicitFunctionType function(shape,shape+1);
    for(size_t ni=0; ni<variableStates[i]; ++ni) {
    function(ni)= (rand() % 1000)*0.001;
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+1);
    }
    }
    //PAIRWISE
    if(variableNumStates==false) {
    size_t shape[2];
    shape[0] = shape[1] = numStates;
    size_t var[]={0,0};
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<numStates; ++ni) {
    for(size_t nj=0; nj<numStates; ++nj) {
    if(ni==nj) function(ni,nj) = 0;
    else       function(ni,nj) = 1;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    for(size_t i=0;i<height;++i) {
    for(size_t j=0;j<width;++j) {
    size_t v = i+height*j;
    if(i+1<height) {
    var[0] = v;
    var[1] = i+1+height*j;
    gm.addFactor(funcId,var,var+2);
    }
    if(j+1<width) {
    var[0] = v;
    var[1] = i+height*(j+1);
    gm.addFactor(funcId,var,var+2);
    }
    }
    }
    }
    else{
    size_t shape[2];
    size_t var[]={0,0};
    for(size_t i=0;i<height;++i) {
    for(size_t j=0;j<width;++j) {
    size_t v = i+height*j;
    if(i+1<height) {
    var[0] = v;
    var[1] = i+1+height*j;
    shape[0] = variableStates[var[0]];
    shape[1] = variableStates[var[1]];
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<shape[0]; ++ni) {
    for(size_t nj=0; nj<shape[1]; ++nj) {
    if(ni==nj) function(ni,nj) = 0;
    else       function(ni,nj) = 1;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+2);
    }
    if(j+1<width) {
    var[0] = v;
    var[1] = i+height*(j+1);
    shape[0] = variableStates[var[0]];
    shape[1] = variableStates[var[1]];
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<shape[0]; ++ni) {
    for(size_t nj=0; nj<shape[1]; ++nj) {
    if(ni==nj) function(ni,nj) = 0;
    else       function(ni,nj) = 1;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+2);
    }
    }
    }
    }
    return gm;
    }
    template<class GM>
    GM ModelGenerator<GM>::buildPottsFull(size_t numVar, size_t numStates, unsigned int id, bool variableNumStates) {
    srand(id);
    size_t N=numVar;
    std::vector<size_t>  variableStates(N,numStates);
    if(variableNumStates) {
    for(size_t i=0; i<N;++i) {
    variableStates[i] = (rand() % (numStates-1))+1;
    }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //UNARY
    {
    size_t shape[1];
    size_t var[]={0};
    for(size_t i=0;i<N;++i) {
    var[0]   = i;
    shape[0] = variableStates[i];
    ExplicitFunctionType function(shape,shape+1);
    for(size_t ni=0; ni<numStates; ++ni) {
    function(ni)= numVar * (rand() % 1000)*0.001;
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+1);
    }
    }
    //PAIRWISE
    if(variableNumStates==false) {
    size_t shape[2];
    shape[0] = shape[1] = numStates;
    size_t var[]={0,0};
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<numStates; ++ni) {
    for(size_t nj=0; nj<numStates; ++nj) {
    if(ni==nj) function(ni,nj) = 0;
    else       function(ni,nj) = 1;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    for(size_t i=0;i<N;++i) {
    for(size_t j=i+1;j<N;++j) {
    var[0] = i;
    var[1] = j;
    gm.addFactor(funcId,var,var+2);
    }
    }
    }
    else{
    size_t shape[2];
    size_t var[]={0,0};
    for(size_t i=0;i<N;++i) {
    for(size_t j=i+1;j<N;++j) {
    var[0] = i;
    var[1] = j;
    shape[0] = variableStates[var[0]];
    shape[1] = variableStates[var[1]];
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<shape[0]; ++ni) {
    for(size_t nj=0; nj<shape[1]; ++nj) {
    if(ni==nj) function(ni,nj) = 0;
    else       function(ni,nj) = 1;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+2);
    }
    }
    }
    return gm;
    }
    template<class GM>
    GM ModelGenerator<GM>::buildRandomGrid(size_t height, size_t width, size_t numStates, unsigned int id, bool variableNumStates) {
    srand(id);
    size_t N=height*width;
    std::vector<size_t>  variableStates(N,numStates);
    if(variableNumStates) {
    for(size_t i=0; i<N;++i) {
    variableStates[i] = (rand() % (numStates-1))+1;
    }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //UNARY
    {
    size_t shape[1];
    size_t var[]={0};
    for(size_t i=0;i<N;++i) {
    var[0]   = i;
    shape[0] = variableStates[i];
    ExplicitFunctionType function(shape,shape+1);
    for(size_t ni=0; ni<shape[0]; ++ni) {
    function(ni)= (rand() % 1000)*0.001;
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+1);
    }
    }
    //PAIRWISE
    {
    size_t shape[2];
    size_t var[]={0,0};
    for(size_t i=0;i<height;++i) {
    for(size_t j=0;j<width;++j) {
    if(i+1<height) {
    var[0] = i+height*j;
    var[1] = i+1+height*j;
    shape[0] = variableStates[var[0]];
    shape[1] = variableStates[var[1]];
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<shape[0]; ++ni) {
    for(size_t nj=0; nj<shape[1]; ++nj) {
    function(ni,nj)= (rand() % 1000)*0.001;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+2);
    }
    if(j+1<width) {
    var[0] = i+height*j;
    var[1] = i+height*(j+1);
    shape[0] = variableStates[var[0]];
    shape[1] = variableStates[var[1]];
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<numStates; ++ni) {
    for(size_t nj=0; nj<numStates; ++nj) {
    function(ni,nj)= (rand() % 1000)*0.001;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+2);
    }
    }
    }
    }
    return gm;
    }
    template<class GM>
    GM ModelGenerator<GM>::buildRandomFull(size_t numVar, size_t numStates, unsigned int id, bool variableNumStates) {
    srand(id);
    size_t N=numVar;
    std::vector<size_t>  variableStates(N,numStates);
    if(variableNumStates) {
    for(size_t i=0; i<N;++i) {
    variableStates[i] = (rand() % (numStates-1))+1;
    }
    }
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //UNARY
    {
    size_t shape[1];
    size_t var[]={0};
    for(size_t i=0;i<N;++i) {
    shape[0] = variableStates[i];
    var[0]   = i;
    ExplicitFunctionType function(shape,shape+1);
    for(size_t ni=0; ni<shape[0]; ++ni) {
    function(ni)= (rand() % 1000)*0.001;
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+1);
    }
    }
    //PAIRWISE
    {
    size_t shape[2];
    size_t var[]={0,0};
    for(size_t i=0;i<N;++i) {
    for(size_t j=i+1;j<N;++j) {
    var[0] = i;
    var[1] = j;
    shape[0] = variableStates[i];
    shape[1] = variableStates[j];
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<numStates; ++ni) {
    for(size_t nj=0; nj<numStates; ++nj) {
    function(ni,nj)= (rand() % 1000)*0.001;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+2);
    }
    }
    }
    return gm;
    }
    template<class GM>
    GM ModelGenerator<GM>::buildRandomStar(size_t numVar, size_t numStates, unsigned int id, bool variableNumStates) {
    srand(id);
    size_t N=numVar;
    std::vector<size_t>  variableStates(N,numStates);
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    if(variableNumStates) {
    for(size_t i=0; i<N;++i) {
    variableStates[i] = (rand() % (numStates-1))+1;
    }
    }
    size_t root = (rand() % numVar);
    //UNARY
    {
    size_t shape[1];
    size_t var[]={0};
    for(size_t i=0;i<N;++i) {
    var[0]   = i;
    shape[0] = variableStates[i];
    ExplicitFunctionType function(shape,shape+1);
    for(size_t ni=0; ni<shape[0]; ++ni) {
    function(ni)= (rand() % 1000)*0.001;
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+1);
    }
    }
    //PAIRWISE
    {
    size_t shape[2];
    size_t var[]={0,0};
    for(size_t i=0;i<N;++i) {
    if(i==root)
    continue;
    else{
    var[0] = i;
    var[1] = root;
    shape[0] = variableStates[var[0]];
    shape[1] = variableStates[var[1]];
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni< shape[0]; ++ni) {
    for(size_t nj=0; nj< shape[1]; ++nj) {
    function(ni,nj)= (rand() % 1000)*0.001;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    gm.addFactor(funcId,var,var+2);
    }
    }
    }
    return gm;
    }
    template<class GM>
    GM ModelGenerator<GM>::buildRandomFull23(size_t numVar, size_t numStates, unsigned int id) {
    srand(id);
    size_t N=numVar;
    std::vector<size_t>  variableStates(N,numStates);
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //UNARY
    {
    size_t shape[1];
    shape[0] = numStates;
    size_t var[]={0};
    for(size_t v=0;v<N;++v) {
    ExplicitFunctionType function(shape,shape+1);
    for(size_t ni=0; ni<numStates; ++ni) {
    function(ni)= (rand() % 1000)*0.001;
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    var[0] = v;
    gm.addFactor(funcId,var,var+1);
    }
    }
    //PAIRWISE
    {
    size_t shape[2];
    shape[0] = shape[1] = numStates;
    size_t var[]={0,0};
    for(size_t i=0;i<N;++i) {
    for(size_t j=i+1;j<N;++j) {
    ExplicitFunctionType function(shape,shape+2);
    for(size_t ni=0; ni<numStates; ++ni) {
    for(size_t nj=0; nj<numStates; ++nj) {
    function(ni,nj)= (rand() % 1000)*0.001;
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    var[0] = i;
    var[1] = j;
    gm.addFactor(funcId,var,var+2);
    }
    }
    }
    //Tripple
    {
    size_t shape[3];
    shape[0] = shape[1] = shape[2] = numStates;
    size_t var[]={0,0,0};
    for(size_t i=0;i<N;++i) {
    for(size_t j=i+1;j<N;++j) {
    for(size_t k=j+1;k<N;++k) {
    ExplicitFunctionType function(shape,shape+3);
    for(size_t ni=0; ni<numStates; ++ni) {
    for(size_t nj=0; nj<numStates; ++nj) {
    for(size_t nk=0; nk<numStates; ++nk) {
    function(ni,nj,nk)= (rand() % 1000)*0.001;
    }
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    var[0] = i;
    var[1] = j;
    var[2] = k;
    gm.addFactor(funcId,var,var+3);
    }
    }
    }
    }
    return gm;
    }
    template<class GM>
    GM ModelGenerator<GM>::buildRandomFull3(size_t numVar, size_t numStates, unsigned int id) {
    srand(id);
    size_t N=numVar;
    std::vector<size_t>  variableStates(N,numStates);
    GraphicalModelType gm(variableStates.begin(),variableStates.end());
    //UNARY
    {
    size_t shape[1];
    shape[0] = numStates;
    size_t var[]={0};
    for(size_t v=0;v<N;++v) {
    ExplicitFunctionType function(shape,shape+1);
    for(size_t ni=0; ni<numStates; ++ni) {
    function(ni)= (rand() % 1000)*0.001;
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    var[0] = v;
    gm.addFactor(funcId,var,var+1);
    }
    }
    //Tripple
    {
    size_t shape[3];
    shape[0] = shape[1] = shape[2] = numStates;
    size_t var[]={0,0,0};
    for(size_t i=0;i<N;++i) {
    for(size_t j=i+1;j<N;++j) {
    for(size_t k=j+1;k<N;++k) {
    ExplicitFunctionType function(shape,shape+3);
    for(size_t ni=0; ni<numStates; ++ni) {
    for(size_t nj=0; nj<numStates; ++nj) {
    for(size_t nk=0; nk<numStates; ++nk) {
    function(ni,nj,nk)= (rand() % 1000)*0.001;
    }
    }
    }
    FunctionIdentifier funcId=gm.addFunction(function);
    var[0] = i;
    var[1] = j;
    var[2] = k;
    gm.addFactor(funcId,var,var+3);
    }
    }
    }
    }
    return gm;
    }
  */
}

/// \endcond

#endif

/*
  template<class GM>
  typename ModelGenerator<GM>::RebindGmType
  ModelGenerator<GM>::buildPottsGridMixedFunctions(size_t height, size_t width, size_t numStates, unsigned int id)
  {
  srand(id);
  size_t N=height*width;
  std::vector<size_t>  variableStates(N,numStates);
  GraphicalModelType gm(variableStates.begin(),variableStates.end());
  //UNARY
  {
  size_t shape[1];
  shape[0] = numStates;
  size_t var[]={0};
  for(size_t v=0;v<N;++v) {
  if(v%2=0) {
  SparseFunctionType function(shape,shape+1,(rand() % 1000)*0.001);
  for(size_t ni=1; ni<numStates; ++ni) {
  function(ni)= 3.0*(rand() % 1000)*0.001;
  }
  FunctionIdentifier funcId=gm.addFunction(function);
  var[0] = v;
  gm.addFactor(funcId,var,var+1);
  }
  if(v%2=1) {
  ExplicitFunctionType function(shape,shape+1);
  for(size_t ni=0; ni<numStates; ++ni) {
  function(ni)= 3.0*(rand() % 1000)*0.001;
  }
  FunctionIdentifier funcId=gm.addFunction(function);
  var[0] = v;
  gm.addFactor(funcId,var,var+1);
  }
  }
  }
  //PAIRWISE
  {
  size_t shape[2];
  shape[0] = shape[1] = numStates;
  size_t var[]={0,0};
  ExplicitFunctionType functionE(shape,shape+2);
  SparseFunctionType functionS(shape,shape+2,1);
  ImplicitFunctionType functionI(ImplicitFunctionType::Potts);
  functionI.parameter(0)=0;
  functionI.parameter(1)=1;
  for(size_t ni=0; ni<numStates; ++ni) {
  for(size_t nj=0; nj<numStates; ++nj) {
  if(ni==nj) {
  functionE(ni,nj) = 0;
  functionS(ni,nj) = 0;
  }
  else{
  functionE(ni,nj) = 1;
  }
  }
  }
  FunctionIdentifier funcIdE=gm.addFunction(functionE);
  FunctionIdentifier funcIdS=gm.addFunction(functionS);
  FunctionIdentifier funcIdI=gm.addFunction(functionI);
  for(size_t i=0;i<height;++i) {
  for(size_t j=0;j<width;++j) {
  size_t v = i+height*j;
  if(i+1<height) {
  var[0] = v;
  var[1] = i+1+height*j;
  if(j%3==0) {
  gm.addFactor(funcIdE,var,var+2);
  }
  else if(j%3==1) {
  gm.addFactor(funcIdS,var,var+2);
  }
  else if(j%3==2) {
  gm.addFactor(funcIdI,var,var+2);
  }
  }
  if(j+1<width) {
  var[0] = v;
  var[1] = i+height*(j+1);
  if(j%3==0) {
  gm.addFactor(funcIdE,var,var+2);
  }
  else if(j%3==1) {
  gm.addFactor(funcIdS,var,var+2);
  }
  else if(j%3==2) {
  gm.addFactor(funcIdI,var,var+2);
  }
  }
  }
  }
  }
  return gm;
  }
*/
