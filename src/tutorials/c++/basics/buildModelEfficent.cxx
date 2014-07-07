/***********************************************************************
 * Tutorial:     Build Grid Model
 * Author:       Joerg Hendrik Kappes
 * Date:         04.07.2014
 * Dependencies: None
 *
 * Description:
 * ------------
 * This tutorial show some tricks how to speed up the build of a model 
 * and which 	pitfall can appear and make building a model slow. 
 * 
 * The example consider a 500 x 500 grid with Potts regularization
 *
 ************************************************************************/

#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/utilities/timer.hxx>

//*******************
//** Typedefs
//*******************
typedef double                                                                 ValueType;          // type used for values
typedef size_t                                                                 IndexType;          // type used for indexing nodes and factors (default : size_t)
typedef size_t                                                                 LabelType;          // type used for labels (default : size_t)
typedef opengm::Adder                                                          OpType;             // operation used to combine terms
typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                ExplicitFunction;   // shortcut for explicit function 
typedef opengm::PottsFunction<ValueType,IndexType,LabelType>                   PottsFunction;      // shortcut for Potts function
typedef opengm::meta::TypeListGenerator<ExplicitFunction,PottsFunction>::type  FunctionTypeList;   // list of all function the model can use (this trick avoids virtual methods) - here only one
typedef opengm::DiscreteSpace<IndexType, LabelType>                            SpaceType;          // type used to define the feasible state-space
typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>    Model;              // type of the model
typedef Model::FunctionIdentifier                                              FunctionIdentifier; // type of the function identifier



//  This function add unaries in the simplest way to the model
//-------------------------------------------------------------
void addUnaries(Model& gm, std::vector<std::vector<ValueType> >& data, bool dirty = false){
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      for(IndexType l=0; l<gm.numberOfLabels(variable); ++l)
         f(l) = data[variable][l];
      // add function
      FunctionIdentifier id = gm.addFunction(f);
      // add factor
      IndexType variableIndex[] = {variable};
      if(dirty)
         gm.addFactorNonFinalized(id, variableIndex, variableIndex + 1);
      else
         gm.addFactor(id, variableIndex, variableIndex + 1);
   }
}

//  This function add unaries to the model and avoid a local copy
//----------------------------------------------------------------
void addUnariesEfficent(Model& gm, std::vector<std::vector<ValueType> >& data, bool dirty = false){
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f_temp; 
      FunctionIdentifier id = gm.addFunction(f_temp);
      ExplicitFunction& f = gm.getFunction<ExplicitFunction>(id);  // get function allocated in the model
      f.resize(shape, shape+1);                                    // reshape the function in the model
      for(IndexType l=0; l<gm.numberOfLabels(variable); ++l)       // fill function
         f(l) = data[variable][l];
      // add function
      // add factor
      IndexType variableIndex[] = {variable};
      if(dirty)
         gm.addFactorNonFinalized(id, variableIndex, variableIndex + 1);
      else
         gm.addFactor(id, variableIndex, variableIndex + 1);
   }
}


//  This function add unaries to the model and enforce the model to check if this function already exists.
//  If it exists it is not allocated a second time and the first one is used. 
//  One can further speed up this if one do the comparison outside, because function comparison can the time-consuming.
//--------------------------------------------------------------------------------------------------------
void addUnariesShared(Model& gm, std::vector<std::vector<ValueType> >& data, bool dirty = false){
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      for(IndexType l=0; l<gm.numberOfLabels(variable); ++l)
         f(l) = data[variable][l];
      // add function
      FunctionIdentifier id = gm.addSharedFunction(f);                  // Add function only if it does not exist
      // add factor
      IndexType variableIndex[] = {variable};
      if(dirty)
         gm.addFactorNonFinalized(id, variableIndex, variableIndex + 1);
      else
         gm.addFactor(id, variableIndex, variableIndex + 1);
   }
}

//  This function add pairwise functions as Potts functions to the model
//  Use for each factor its own Potts function
//-----------------------------------------------------------------------
void addPairwise(Model& gm, size_t N, size_t M,size_t numLabel, bool dirty = false){
   IndexType vars[]  = {0,1}; 
   for(IndexType n=0; n<N;++n){
      for(IndexType m=0; m<M;++m){
         vars[0] = n + m*N;
         if(n+1<N){ //check for right neighbor
            vars[1] =  (n+1) + (m  )*N; 
            PottsFunction potts(numLabel,numLabel, 0.0, 2.0);
            FunctionIdentifier pottsid = gm.addFunction(potts);
            OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
            if(dirty)
               gm.addFactorNonFinalized(pottsid, vars, vars + 2);
            else
               gm.addFactor(pottsid, vars, vars + 2);
         } 
         if(m+1<M){ //check for lower neighbor
            vars[1] =  (n  ) + (m+1)*N; 
            PottsFunction potts(numLabel, numLabel, 0.0, 2.0);
            FunctionIdentifier pottsid = gm.addFunction(potts);
            OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
            if(dirty)
                gm.addFactorNonFinalized(pottsid, vars, vars + 2);
            else
               gm.addFactor(pottsid, vars, vars + 2);
         }
      }
   }
}

//  This function add pairwise functions as explicit functions to the model
//  Use for each factor its own explicit function
//---------------------------------------------------------------------------
void addPairwiseExplicit(Model& gm, size_t N, size_t M, LabelType numLabel, bool dirty = false){
   const LabelType shape[] = {numLabel,numLabel};
   ExplicitFunction f(shape, shape + 2);
   for(size_t i=0; i<numLabel; ++i)
      for(size_t j=0; j<numLabel; ++j)
         if (i==j) f(i,j) = 0;
         else      f(i,j) = 2;

   IndexType vars[]  = {0,1}; 
   for(IndexType n=0; n<N;++n){
      for(IndexType m=0; m<M;++m){
         vars[0] = n + m*N;
         if(n+1<N){ //check for right neighbor
            vars[1] =  (n+1) + (m  )*N; 
            FunctionIdentifier pottsid = gm.addFunction(f);
            OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
             if(dirty)
               gm.addFactorNonFinalized(pottsid, vars, vars + 2);
             else
                gm.addFactor(pottsid, vars, vars + 2);
         } 
         if(m+1<M){ //check for lower neighbor
            vars[1] =  (n  ) + (m+1)*N; 
            FunctionIdentifier pottsid = gm.addFunction(f);
            OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered! 
            if(dirty)
               gm.addFactorNonFinalized(pottsid, vars, vars + 2);
            else
               gm.addFactor(pottsid, vars, vars + 2);
         }
      }
   }
}

//  This function add pairwise functions as Potts functions to the model
//  Use for each factor the same Potts function (saves memory)
//-----------------------------------------------------------------------
void addPairwiseShared(Model& gm, size_t N, size_t M, LabelType numLabel, bool dirty = false){
   PottsFunction potts(numLabel, numLabel, 0.0, 2.0);
   FunctionIdentifier pottsid = gm.addFunction(potts);            // add function only once
   
   IndexType vars[]  = {0,1}; 
   for(IndexType n=0; n<N;++n){
      for(IndexType m=0; m<M;++m){
         vars[0] = n + m*N;
         if(n+1<N){ //check for right neighbor
            vars[1] =  (n+1) + (m  )*N;
            OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered! 
            if(dirty)
               gm.addFactorNonFinalized(pottsid, vars, vars + 2);
            else
               gm.addFactor(pottsid, vars, vars + 2);
         } 
         if(m+1<M){ //check for lower neighbor
            vars[1] =  (n  ) + (m+1)*N; 
            OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
            if(dirty)
               gm.addFactorNonFinalized(pottsid, vars, vars + 2);
            else
               gm.addFactor(pottsid, vars, vars + 2);
         }
      }
   }
}


int main(int argc, char** argv) {
   LabelType numLabel = 20;
   IndexType N         = 1000; 
   IndexType M         = 1000;
   std::cout << std::endl;
   std::cout << std::endl;
   std::cout << "This Tutorial shows some tricks how to speed up the model building in C++." <<std::endl;
   std::cout << "In C++ this is not as critical as in Matlab or Python, but speedups of 10 are possible." <<std::endl;
   std::cout << "Depending on the model-size and function type, different tricks can help more or less." <<std::endl;
   std::cout << "Respectively, the speedups for our model might not be representative for other models !!!" <<std::endl;
   std::cout << std::endl;
   std::cout << std::endl;


   std::cout << "Create data ... "<<std::flush;
   std::vector<std::vector<ValueType> > data(N*M,std::vector<ValueType>(numLabel,0));
   for (size_t i=0; i<N*M; ++i){
      size_t d = size_t(rand() / (RAND_MAX/10));
      for (size_t l=0; l<numLabel; ++l)
         data[i][l] = l;
   }
   std::cout <<"done"<<std::endl;
   std::vector<LabelType> l(N*M,0);
   opengm::Timer timer; 
   std::cout <<std::endl;
   std::cout <<" Trick 1 : Reduce the number of functions in the model" << std::endl;
   std::cout <<"------------------------------------------------------" << std::endl;
  {
      std::cout << "Most naive way ... " <<std::flush;
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnaries(gm,data);
      addPairwiseExplicit(gm,N,M,numLabel);
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   } 
   {
      std::cout << "Model using special Potts function ..." <<std::flush;
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnaries(gm,data);
      addPairwise(gm,N,M,numLabel); 
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   }
   {
      std::cout << "Models with shared special Potts function ..." <<std::flush; 
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnaries(gm,data);
      addPairwiseShared(gm,N,M,numLabel);
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   } 
   {
      std::cout << "Models with shared special Potts function and shared unaries ..." <<std::flush; 
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnariesShared(gm,data);
      addPairwiseShared(gm,N,M,numLabel);
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   } 
 

   std::cout <<std::endl;
   std::cout <<std::endl;
   std::cout <<" Trick 2 : Build a dirty model and finalize in the end (does not help when degree is small)" << std::endl;
   std::cout <<"--------------------------------------------------------------------------------------------" << std::endl;
   {
      std::cout << "Most naive way ... " <<std::flush;
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnaries(gm,data,true);
      addPairwiseExplicit(gm,N,M,numLabel,true);
      gm.finalize();
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   } 
   {
      std::cout << "Model using special Potts function ..." <<std::flush;
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnaries(gm,data,true);
      addPairwise(gm,N,M,numLabel,true);
      gm.finalize(); 
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   }
   {
      std::cout << "Models with shared special Potts function ..." <<std::flush; 
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnaries(gm,data,true);
      addPairwiseShared(gm,N,M,numLabel,true);
      gm.finalize(); 
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   }
   {
      std::cout << "Models with shared special Potts function and shared unaries ..." <<std::flush; 
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnariesShared(gm,data,true);
      addPairwiseShared(gm,N,M,numLabel,true);
      gm.finalize(); 
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   }  

   std::cout <<std::endl;
   std::cout <<std::endl;
   std::cout <<" Trick 3 : Prereservation" << std::endl;
   std::cout <<"--------------------------------------------------------------------------------------------" << std::endl;
   {
      std::cout << "Models with shared special Potts function and shared unaries ..." <<std::flush; 
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      size_t maxVariableDegree = 5;
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()),maxVariableDegree );

      gm.reserveFunctions<ExplicitFunction>(numLabel);
      gm.reserveFunctions<PottsFunction>(1);
      gm.reserveFactors(N*M*4);
      gm.reserveFactorsVarialbeIndices(N*M+2*N*M*2);
  
      addUnariesShared(gm,data);
      addPairwiseShared(gm,N,M,numLabel);
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   } 

   std::cout <<std::endl;
   std::cout <<std::endl;
   std::cout <<" Trick 4 : Avoid temporal instances and copies" << std::endl;
   std::cout <<"--------------------------------------------------------------------------------------------" << std::endl;
   {
      std::cout << "Models with shared special Potts function and uncopied unaries ..." <<std::flush; 
      timer.tic();
      std::vector<LabelType> numbersOfLabels(N*M,numLabel);
      Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
      addUnariesEfficent(gm,data);
      addPairwiseShared(gm,N,M,numLabel);
      timer.toc();
      std::cout << timer.elapsedTime()<< " sec. ... done!" << std::endl;
      timer.reset();
      gm.evaluate(l); 
   }


}
