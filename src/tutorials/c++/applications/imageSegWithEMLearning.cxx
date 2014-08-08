/***********************************************************************
 * Tutorial:     Image Segmentation by learning dataterms by EM
 * Author:       Joerg Hendrik Kappes
 * Date:         11.07.2014
 * Dependencies: None
 *
 * Description:
 * ------------
 * Simple example for learning the local dataterms by EM.
 * As input an pgm-image, the number of labels and smoothing strength are required.
 * The local data terms are modeled by normal distributions with fixed variance (for simplicity).
 * In each step marginals are calculated and model parameters (means are set optimal for given expectation)
 * After 10 rounds the model is solved with the final parameters.
 *
 * this example manipulates the model in-place and make use of warm starting LBP.
 *
 * TODO: Estimate variance, too.
 * 
 ************************************************************************/

#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/integrator.hxx>
#include <opengm/functions/potts.hxx>
#include "utilities/pgmimage.hxx"

#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/icm.hxx>


template <class V>
void learnWithEM(std::vector<V>& means,
                 std::vector<V>& variances,
                 V lambda,
                 const opengm::PGMImage<unsigned char>& image,
                 const size_t rounds)
{
   typedef V                                                                                ValueType;
   typedef size_t                                                                           IndexType;          // type used for indexing nodes and factors (default : size_t)
   typedef size_t                                                                           LabelType;          // type used for labels (default : size_t)
   typedef opengm::Multiplier                                                               OpType;             // operation used to combine terms
   typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                          ExplicitFunction;   // shortcut for explicit function 
   typedef opengm::PottsFunction<ValueType,IndexType,LabelType>                             PottsFunction;      // shortcut for Potts function
   typedef typename opengm::meta::TypeListGenerator<ExplicitFunction,PottsFunction>::type   FunctionTypeList;   // list of all function the model can use (this trick avoids virtual methods) - here only one
   typedef opengm::DiscreteSpace<IndexType, LabelType>                                      SpaceType;          // type used to define the feasible state-space
   typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>              Model;              // type of the model
   typedef typename Model::FunctionIdentifier                                               FunctionIdentifier; // type of the function identifier

   // Build empty Model
   LabelType numLabel = means.size();
   int height         = image.height();
   int width          = image.width();

   std::vector<LabelType> numbersOfLabels(height*width,numLabel);
   Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
   

   // Add functions 
   std::vector<FunctionIdentifier> unaryids(256);
   std::vector<ValueType> scale(numLabel);
   for(size_t j=0; j<numLabel; ++j) {
      scale[j] = 1/(variances[j]*2.5066);
   }
   ExplicitFunction f(&numLabel, &numLabel + 1);
   for(size_t i=0; i<256; ++i) {
      for(size_t j=0; j<numLabel; ++j) {
         const ValueType temp = (i-means[j])/variances[j];
         const ValueType value = scale[j]*std::exp(-temp*temp/2.0);
         f(j) = value;
      }
      unaryids[i] = gm.addFunction(f); 
   }
   PottsFunction pottsfunction(numLabel, numLabel, std::exp(-0.0), std::exp(-lambda));
   FunctionIdentifier pottsid = gm.addFunction(pottsfunction); 

   // Add factor 
   for(IndexType n=0; n<height;++n){
      for(IndexType m=0; m<width;++m){ 
         IndexType var = n + m*height;
         gm.addFactor(unaryids[image(n,m)], &var, &var + 1);
      }
   }
   {
      IndexType vars[]  = {0,1}; 
      for(IndexType n=0; n<height;++n){
         for(IndexType m=0; m<width;++m){
            vars[0] = n + m*height;
            if(n+1<height){ //check for right neighbor
               vars[1] =  (n+1) + (m  )*height;
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               gm.addFactor(pottsid, vars, vars + 2);
            } 
            if(m+1<width){ //check for lower neighbor
               vars[1] =  (n  ) + (m+1)*height; 
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               gm.addFactor(pottsid, vars, vars + 2);
            }
         }
      }
   } 
   // Inference
   typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Integrator> UpdateRules;
   typedef opengm::MessagePassing<Model, opengm::Integrator, UpdateRules, opengm::MaxDistance>  LBP; 
   
   typename LBP::Parameter parameter(70, 0.01, 0.8); 
   parameter.useNormalization_ = true;
   LBP lbp(gm, parameter);

   for(size_t i=0; i<rounds; ++i){
      if(i==1){
         std::cout << "Reduce iterations of LBP because warmstart is used!"<<std::endl;
         lbp.setMaxSteps(10);
      }
      std::cout <<"Start infernce ... "<<std::flush; 
      lbp.infer();
      std::cout <<"done"<<std::endl;

      // Calculate new means 
      typename Model::IndependentFactorType marg;
      std::vector<ValueType> newMeans(numLabel,0);
      std::vector<ValueType> sumMargs(numLabel,0);
      
      for(IndexType n=0; n<height;++n){
         for(IndexType m=0; m<width;++m){
            IndexType var = n + m*height;
            lbp.marginal(var,marg);
            //std::cout <<var<<" : ";
            for(LabelType i=0; i<numLabel; ++i){
               newMeans[i] += marg(&i) * image(n,m);
               sumMargs[i] += marg(&i);
               //std::cout <<marg(&i)<<" ";
            }
            //std::cout << std::endl;
         }
      } 
      for(LabelType l=0; l<numLabel; ++l){
         std::cout << "Label "<<l<<" changed mean " <<means[l]<<" -> ";
         means[l] = newMeans[l] / sumMargs[l];
         std::cout << means[l] <<" ( "<< newMeans[l]<<" / "<<sumMargs[l]<<" )"<<std::endl;
      }
      if(i+1<rounds){//Update model
         std::cout << "Change model inplace ... "<<std::flush;
         for(size_t l=0; l<numLabel; ++l) {
            scale[l] = 1/(variances[l]*2.5066);
         } 
         for(size_t k=0; k<256; ++k) {
            ExplicitFunction& f = gm.template getFunction<ExplicitFunction>(unaryids[k]);  
            for(size_t l=0; l<numLabel; ++l) {
               const ValueType temp = (k-means[l])/variances[l];
               const ValueType value = scale[l]*std::exp(-temp*temp/2.0);
               f(l) = value;
            }
         }
         std::cout <<"done!"<<std::endl;
      }
   }
} 



int main(int argc, char** argv) {
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

   //*****************
   //** HELP
   //*****************
    if (argc != 5){
       std::cerr << "Usage: "<<argv[0]<<" infile outfile numClasses smoothing" << std::endl;
       exit(-1);
    }

   //******************
   //** DATA
   //****************** 
   char *infilename   = argv[1];
   char *outfilename  = argv[2];
   LabelType numLabel = atoi(argv[3]);
   double lambda      = atof(argv[4]);

   opengm::PGMImage<unsigned char> image; 
   image.readPGM(infilename);

   int height = image.height();
   int width = image.width();
  
   std::vector<ValueType> means(numLabel,0);
   std::vector<ValueType> variances(numLabel,0);
 
   //*******************
   //** Code
   //*******************

  
   for (size_t i=0; i<numLabel; ++i){
      means[i]     = 255.0 *(i+1.0)/(numLabel+1.0);
      variances[i] = 42;
   }

   std::cout << "Model have "<<numLabel<<" labels with mean values :"<<std::endl;
   for (size_t i=0; i<means.size(); ++i){
      std::cout << "Label "<<i<<" has mean "<<means[i]<<std::endl;
   }

   //learn means
   learnWithEM(means,variances,lambda,image,10);

   // Build empty Model
   std::vector<LabelType> numbersOfLabels(height*width,numLabel);
   Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
   
   // Add functions 
   std::vector<FunctionIdentifier> unaryids(256);
   ExplicitFunction f(&numLabel, &numLabel + 1);
   for(size_t i=0; i<256; ++i) {
      for(size_t j=0; j<numLabel; ++j) {
         const ValueType temp = (i-means[j])/variances[j];
         const ValueType value = temp*temp/2.0;
         const ValueType truncation = 100.0;
         f(j) = value;
      }
      unaryids[i] = gm.addFunction(f); 
   }
   PottsFunction pottsfunction(numLabel, numLabel, 0.0, lambda);
   FunctionIdentifier pottsid = gm.addFunction(pottsfunction); 

   // Add factor 
   for(IndexType n=0; n<height;++n){
      for(IndexType m=0; m<width;++m){ 
         IndexType var = n + m*height;
         gm.addFactor(unaryids[image(n,m)], &var, &var + 1);
      }
   }
   {
      IndexType vars[]  = {0,1}; 
      for(IndexType n=0; n<height;++n){
         for(IndexType m=0; m<width;++m){
            vars[0] = n + m*height;
            if(n+1<height){ //check for right neighbor
               vars[1] =  (n+1) + (m  )*height;
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               gm.addFactor(pottsid, vars, vars + 2);
            } 
            if(m+1<width){ //check for lower neighbor
               vars[1] =  (n  ) + (m+1)*height; 
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               gm.addFactor(pottsid, vars, vars + 2);
            }
         }
      }
   }

   // Inference
   typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Minimizer> UpdateRules;
   typedef opengm::MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance>  LBP; 
   
   LBP::Parameter parameter(30, 0.01, 0.8);
   LBP lbp(gm, parameter); 

   LBP::VerboseVisitorType visitor;
   lbp.infer(visitor); 

   std::vector<LabelType> labeling(gm.numberOfVariables());
   lbp.arg(labeling);


   opengm::PGMImage<unsigned char> out(height, width);
   for(IndexType n=0; n<height;++n){
      for(IndexType m=0; m<width;++m){ 
         IndexType var = n + m*height;
         out(n,m) = labeling[var]*(255/numLabel);   
      }
   }
   out.writePGM(outfilename);
}
