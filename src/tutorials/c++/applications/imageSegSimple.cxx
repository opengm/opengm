/***********************************************************************
 * Tutorial:     Simple image Segmentation
 * Author:       Joerg Hendrik Kappes
 * Date:         07.07.2014
 * Dependencies: None
 *
 * Description:
 * ------------
 * This is a simple example for image segmentation.
 * The data-term, regularizer and optimizer are very simple.
 * To avoid dependencies we only support 8bit-gray images in PGM-format. 
 * -> do not expect wonders
 *
 * Data-Term:    Truncated L1-distance of pixel-color to exemplary color
 * Regularizer:  Potts (penalize boundary length)
 * Optimizer:    Loopy Belief Propagation (followed by ICM)
 * 
 * Usage:        imageSegSimple opengm/src/tutorials/data/coins.pgm out.pgm 20.0 30 90 
 *
 ************************************************************************/

#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/potts.hxx>
#include "utilities/pgmimage.hxx"

#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/icm.hxx>

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
    if (argc <= 5){
       std::cerr << "Usage: "<<argv[0]<<" infile outfile regularization color0 color1 [color2 ... colorN]" << std::endl;
       exit(-1);
    }

   //******************
   //** DATA
   //****************** 
   char *infilename = argv[1];
   char *outfilename = argv[2];
   double lambda = atof(argv[3]);
   std::vector<int> colors(argc-4,0);
   for(size_t i=0; i<argc-4; ++i){
      colors[i] = atoi(argv[i+4]);
   }

   opengm::PGMImage<unsigned char> image;
   LabelType numLabel=colors.size(); 
   image.readPGM(infilename);

   int height = image.height();
   int width = image.width();
 
   //*******************
   //** Code
   //*******************

   // Build empty Model
   std::vector<LabelType> numbersOfLabels(height*width,numLabel);
   Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));
   
   // Add functions 
   std::vector<FunctionIdentifier> unaryids(256);
   ExplicitFunction f(&numLabel, &numLabel + 1);
   for(size_t i=0; i<256; ++i) {
      for(size_t j=0; j<numLabel; ++j) {
         const ValueType value = i-colors[j];
         const ValueType truncation = 100.0;
         f(j) = std::min(truncation,std::fabs(value));
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
   typedef opengm::ICM<Model,opengm::Minimizer> ICM;
   
   LBP::Parameter parameter(100, 0.01, 0.8);
   LBP lbp(gm, parameter); 

   LBP::VerboseVisitorType visitor;
   lbp.infer(visitor); 

   std::vector<LabelType> labeling(gm.numberOfVariables());
   lbp.arg(labeling);
/*
   ICM icm(gm);
   icm.setStartingPoint(labeling.begin());
   icm.infer();
   icm.arg(labeling);
*/

   opengm::PGMImage<unsigned char> out(height, width);
   for(IndexType n=0; n<height;++n){
      for(IndexType m=0; m<width;++m){ 
         IndexType var = n + m*height;
         out(n,m) = labeling[var]*(255/numLabel);   
      }
   }
   out.writePGM(outfilename);
}
