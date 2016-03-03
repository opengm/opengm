/***********************************************************************
 * Tutorial:     Inference on the Sum-Prod-Semiring (marginalization)
 * Author:       Joerg Hendrik Kappes
 * Date:         09.07.2014
 * Dependencies: None
 *
 * Description:
 * ------------
 * This Example construct a model and find the labeling with the lowest energy 
 *
 ************************************************************************/

#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/potts.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/trws/smooth_nesterov.hxx>

//*******************
//** Typedefs
//*******************
typedef double                                                                 ValueType;          // type used for values
typedef size_t                                                                 IndexType;          // type used for indexing nodes and factors (default : size_t)
typedef size_t                                                                 LabelType;          // type used for labels (default : size_t)
typedef opengm::Multiplier                                                     OpType;             // operation used to combine terms
typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                ExplicitFunction;   // shortcut for explicite function 
typedef opengm::PottsFunction<ValueType,IndexType,LabelType>                   PottsFunction;      // shortcut for Potts function
typedef opengm::meta::TypeListGenerator<ExplicitFunction,PottsFunction>::type  FunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
typedef opengm::DiscreteSpace<IndexType, LabelType>                            SpaceType;          // type used to define the feasible statespace
typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>    Model;              // type of the model
typedef Model::FunctionIdentifier                                              FunctionIdentifier; // type of the function identifier

Model buildGrid(){
 
   IndexType N = 6;
   IndexType M = 4;  
   int data[] = { 4, 4, 4, 4, 6, 0,
                  0, 7, 2, 4, 4, 0,
                  9, 9, 4, 4, 9, 9,
                  2, 2, 9, 9, 9, 9 };

   std::cout << "Start building the grid model ... "<<std::endl;
   // Build empty Model
   LabelType numLabel = 2;
   std::vector<LabelType> numbersOfLabels(N*M,numLabel);
   Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));

   // Add 1st order functions and factors to the model
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      f(0) = std::exp(-std::fabs(data[variable] - 2.0));
      f(1) = std::exp(-std::fabs(data[variable] - 8.0));
      // add function
      FunctionIdentifier id = gm.addFunction(f);
      // add factor
      IndexType variableIndex[] = {variable};
      gm.addFactor(id, variableIndex, variableIndex + 1);
   }
   // add 2nd order functions for all variables neighbored on the grid
   {
      // add a potts function to the model
      PottsFunction potts(numLabel, numLabel, 0.1, 0.4);
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
   return gm;
}

Model buildChain(size_t N){
   std::cout << "Start building the chain model ... "<<std::endl;
   LabelType numLabel = 10;
   std::vector<LabelType> numbersOfLabels(N,numLabel);
   Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));

   // Add 1st order functions and factors to the model
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      for (size_t i=0; i<numLabel; ++i)
         f(i) = static_cast<ValueType>(rand()) / (RAND_MAX) * 1000.0;
      FunctionIdentifier id = gm.addFunction(f);
      IndexType variableIndex[] = {variable};
      gm.addFactor(id, variableIndex, variableIndex + 1);
   }
   // add 2nd order functions for all variables neighbored on the chain
   for(IndexType variable = 0; variable < gm.numberOfVariables()-1; ++variable) {
      // construct 1st order function
      const IndexType vars[]  = {variable,variable+1}; 
      const LabelType shape[] = {gm.numberOfLabels(variable),gm.numberOfLabels(variable+1)};
      ExplicitFunction f(shape, shape + 2);
      for (size_t i=0; i<numLabel; ++i) 
         for (size_t j=0; j<numLabel; ++j)
            f(i,j) = static_cast<ValueType>(rand()) / (RAND_MAX) * 1000.0;
      FunctionIdentifier id = gm.addFunction(f);
      gm.addFactor(id, vars, vars + 2);
   }
   return gm;
}

void inferBP(const Model& gm, bool normalization = true){
   typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Integrator> UpdateRules;
   typedef opengm::MessagePassing<Model, opengm::Integrator, UpdateRules, opengm::MaxDistance>  LBP; 
  
   LBP::Parameter parameter(size_t(100)); //maximal number of iterations=0
   parameter.useNormalization_ = normalization;
   LBP lbp(gm, parameter); 
  
   lbp.infer();


   Model::IndependentFactorType marg;
   for(IndexType var=0; var<gm.numberOfVariables(); ++var)
   {
      std::cout<< "Variable 0 has the following marginal distribution P(x_"<<var<<") :";
      lbp.marginal(var,marg);
      for(LabelType i=0; i<gm.numberOfLabels(var); ++i)
         std::cout <<marg(&i) << " ";
      std::cout<<std::endl;
   }   
}


//void inferNesterov(const Model& gm){
//   //This is a dummy - Bogdan will finalize it ...
//
//   typedef opengm::NesterovAcceleratedGradient<Model,opengm::Integrator> INF;
//   INF::Parameter parameter(100); //maximal number of iterations
//   parameter.verbose_=true;
//
//   INF inf(gm, parameter);
//
//   inf.infer();
//
//   Model::IndependentFactorType marg;
//   for(IndexType var=0; var<gm.numberOfVariables(); ++var)
//   {
//      std::cout<< "Variable 0 has the following marginal distribution P(x_"<<var<<") :";
//      inf.marginal(var,marg);
//      for(LabelType i=0; i<gm.numberOfLabels(var); ++i)
//         std::cout <<marg(&i) << " ";
//      std::cout<<std::endl;
//   }
//
//}

int main(int argc, char** argv) {
   std::cout <<"Sum-Prod-Semiring"<<std::endl;
 
   // Infer with LBP

   std::cout << "Start LBP inference ... " <<std::endl;
   Model gmGrid  = buildGrid();
   inferBP(gmGrid);
   Model gmChain = buildChain(10);
   std::cout<< "Inference on small chain with no normalization" <<std::endl;
   inferBP(gmChain,false); 
   std::cout<< "Inference on small chain with normalization" <<std::endl;
   inferBP(gmChain,true);
   Model gmChain2 = buildChain(4);
   std::cout<< "Inference on large chain with no normalization" <<std::endl;
   inferBP(gmChain2,false);

   std::cout<< "Inference on large chain with normalization" <<std::endl;
   inferBP(gmChain2,true); 

   //std::cout<< "Inference on large chain with Nesterov" <<std::endl;
   //inferNesterov(gmChain2);

   return 0;
};
