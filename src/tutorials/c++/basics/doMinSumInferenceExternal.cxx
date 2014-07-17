/***********************************************************************
 * Tutorial:     Inference on the Min-Sum-Semiring (minimizing a energy function)
 * Author:       Joerg Hendrik Kappes
 * Date:         17.07.2014
 * Dependencies: TRWS-lib
 *
 * Description:
 * ------------
 * This Example construct a model and find the labeling with the lowest energy
 * by using external solvers which are wrapped by OpenGM 
 *
 ************************************************************************/

#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/potts.hxx>

#include <opengm/inference/external/trws.hxx>

int main(int argc, char** argv) {
   std::cout <<"Min-Sum-Semiring"<<std::endl;
   //*******************
   //** Typedefs
   //*******************
   typedef double                                                                 ValueType;          // type used for values
   typedef size_t                                                                 IndexType;          // type used for indexing nodes and factors (default : size_t)
   typedef size_t                                                                 LabelType;          // type used for labels (default : size_t)
   typedef opengm::Adder                                                          OpType;             // operation used to combine terms
   typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                ExplicitFunction;   // shortcut for explicite function 
   typedef opengm::PottsFunction<ValueType,IndexType,LabelType>                   PottsFunction;      // shortcut for Potts function
   typedef opengm::meta::TypeListGenerator<ExplicitFunction,PottsFunction>::type  FunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
   typedef opengm::DiscreteSpace<IndexType, LabelType>                            SpaceType;          // type used to define the feasible statespace
   typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>    Model;              // type of the model
   typedef Model::FunctionIdentifier                                              FunctionIdentifier; // type of the function identifier

   //******************
   //** DATA
   //******************
   IndexType N = 6;
   IndexType M = 4;  
   int data[] = { 4, 4, 4, 4, 6, 0,
                  0, 7, 2, 4, 4, 0,
                  9, 9, 4, 4, 9, 9,
                  2, 2, 9, 9, 9, 9 };

   std::cout << "Start building the model ... "<<std::endl;
   // Build empty Model
   LabelType numLabel = 2;
   std::vector<LabelType> numbersOfLabels(N*M,numLabel);
   Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));

   // Add 1st order functions and factors to the model
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      f(0) = std::fabs(data[variable] - 2.0);
      f(1) = std::fabs(data[variable] - 8.0);
      // add function
      FunctionIdentifier id = gm.addFunction(f);
      // add factor
      IndexType variableIndex[] = {variable};
      gm.addFactor(id, variableIndex, variableIndex + 1);
   }
   // add 2nd order functions for all variables neighbored on the grid
   {
      // add a potts function to the model
      PottsFunction potts(numLabel, numLabel, 0.0, 2.0);
      FunctionIdentifier pottsid = gm.addFunction(potts);

      IndexType vars[]  = {0,1}; 
      for(IndexType n=0; n<N;++n){
         for(IndexType m=0; m<M;++m){
            vars[0] = n + m*N;
            if(n+1<N){ //check for right neighbor
               vars[1] =  (n+1) + (m  )*N;
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               gm.addFactor(pottsid, vars, vars + 2);
            } 
            if(m+1<M){ //check for lower neighbor
               vars[1] =  (n  ) + (m+1)*N; 
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               gm.addFactor(pottsid, vars, vars + 2);
            }
         }
      }
   }

   // Infer with TRWS
   std::cout << "Start TRWS inference ... " <<std::endl; 
   typedef opengm::external::TRWS<Model> TRWS; 
   TRWS::Parameter parameter; 
   {
      parameter.energyType_= TRWS::Parameter::TABLES; // Use tables -> make a copy of all functions
      TRWS trws(gm, parameter); 
  
      trws.infer();
      std::vector<LabelType> l;
      trws.arg(l);
      std::cout << "Energy :  "<<trws.value() << std::endl; 
      std::cout << "State :  ";
      for (size_t i=0; i<l.size(); ++i) std::cout <<l[i]<<" ";
      std::cout  << std::endl; 
   }
   {
     
      parameter.energyType_= TRWS::Parameter::VIEW; // Use OpenGM-datastructure -> slower but save memory
      TRWS trws(gm, parameter); 
  
      trws.infer();
      std::vector<LabelType> l;
      trws.arg(l);
      std::cout << "Energy :  "<<trws.value() << std::endl; 
      std::cout << "State :  ";
      for (size_t i=0; i<l.size(); ++i) std::cout <<l[i]<<" ";
      std::cout  << std::endl; 
   }

 
   return 0;
};
