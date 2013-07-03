#include <string>
#include <vector>
#include <iostream>
#include <fstream>


#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"


int
main(int argc, const char* argv[] ) {

   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;
   typedef opengm::Adder OperatorType;
   typedef opengm::Minimizer AccumulatorType;
   typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

   // Set functions for graphical model
   typedef opengm::meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
      opengm::PottsFunction<ValueType, IndexType, LabelType>,
      opengm::PottsNFunction<ValueType, IndexType, LabelType>,
      opengm::PottsGFunction<ValueType, IndexType, LabelType>,
      opengm::TruncatedSquaredDifferenceFunction<ValueType, IndexType, LabelType>,
      opengm::TruncatedAbsoluteDifferenceFunction<ValueType, IndexType, LabelType>
   >::type FunctionTypeList;


   typedef opengm::GraphicalModel<
      ValueType,
      OperatorType,
      FunctionTypeList,
      SpaceType
   > GmType;
   

   GmType gm; 
   std::string opengmfile = argv[1]; 
   std::string uaifile    = argv[2];
 
   opengm::hdf5::load(gm, opengmfile,"gm");
   std::ofstream myuaifile;
   myuaifile.open(uaifile.c_str());
   myuaifile << "MARKOV" << std::endl;
   myuaifile << gm.numberOfVariables() << std::endl;
   for(IndexType var=0; var<gm.numberOfVariables(); ++var){
      myuaifile << gm.numberOfLabels(var) << " ";
   }
   myuaifile << std::endl;
   myuaifile << gm.numberOfFactors() << std::endl;
   for(IndexType f=0; f<gm.numberOfFactors(); ++f){
      myuaifile << gm[f].numberOfVariables() << " " ;
      for(IndexType i=0; i<gm[f].numberOfVariables(); ++i){
         myuaifile << gm[f].variableIndex(i) << " "; 
      } 
      myuaifile << std::endl;
   }
   LabelType l[3] = {0,0,0};
   for(IndexType f=0; f<gm.numberOfFactors(); ++f){
      myuaifile << std::endl;
      myuaifile << gm[f].size() << std::endl;
      if(gm[f].numberOfVariables()==0){
         l[0]=0;
         myuaifile << std::exp(-gm[f](l)) << std::endl;
      }
      else if(gm[f].numberOfVariables()==1){
         for(l[0]=0; l[0]<gm[f].numberOfLabels(0); ++l[0]){
            myuaifile << std::exp(-gm[f](l)) << " "; 
         } 
         myuaifile << std::endl;

      } 
      else if(gm[f].numberOfVariables()==2){
         for(l[0]=0; l[0]<gm[f].numberOfLabels(0); ++l[0]){
            for(l[1]=0; l[1]<gm[f].numberOfLabels(1); ++l[1]){
               myuaifile << std::exp(-gm[f](l)) << " "; 
            }
            myuaifile << std::endl;
         }
      }
      else{
         std::cout << "Factors of order higher than 2 are so far not supported !" <<std::endl;
         myuaifile.close();
         return 1;
      }

   }
   myuaifile.close();
   return 0;
}
