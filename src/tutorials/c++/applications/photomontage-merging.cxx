#include <string>
#include <sstream>

#include "./utilities/parser/cmd_parser.hxx"

#include <vigra/basicimage.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/impex.hxx>

#include <opengm/opengm.hxx>

#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/adder.hxx>


#include "photomontage-merging.hxx"


int main(int argc, char **argv) {
   typedef unsigned long long StateType;
   typedef float ModelValueType;
   typedef float ParameterValueType;
   
   // which model
   std::string model; 
   
   //action
   std::string action;
   std::string allowedAction[] ={"build-gm","state-to-image"};
   
   // input images
   std::vector<std::string> inputImagePath;
   // prefix and output
   std::string dataCostPath,savePathPrefix,outputGmPath,outputGmName,stateFile,stateName,outputImagePath;
   bool verbose;
   size_t numberOfImages;
   
   
   parser::CmdParser parser(argc, argv,"MRF Photomontage","MRF-Benchmark Photomontage Merging","1.0","Thorsten Beier");
   
   // input images
   parser::Arg argModel=parser.addArg(parser::ArgName("-input-images","-ii","input images (must be at least 2 images, using png's is save )"),
                                      parser::ArgValue< std::vector<std::string>  >(inputImagePath));
   // which action , build or state to image
   parser::Arg argAction=parser.addArg(parser::ArgName("-action","-a","build model or convert state to image"),
                                       parser::ArgValue< std::string >(action,allowedAction,allowedAction+2));
   // if action==build-gm 
   {
      parser.addArg( parser::ArgName("-constraining-image","-ci","image with data constraints (using png's is save )"),
                     parser::ArgValue<std::string>(dataCostPath));
      // output graphical model filepath
      parser.addArg( parser::ArgName("-output-gm","-ogm","hdf5 output graphical model file-path"),
                     parser::ArgValue<std::string>(outputGmPath),parser::IfParentArg<std::string>(argAction,"build-gm"));
      parser.addArg(parser::ArgName("-output-gm-dataset","-ogmd","output graphical model hdf5-dataset name"),
                    parser::ArgValue<std::string>(outputGmName,"gm"),parser::IfParentArg<std::string>(argAction,"build-gm"));
   }
   // if action==state-to-image
   {
      // state file
      parser.addArg( parser::ArgName("-states","-s","hdf5 file which contains the states"),
         parser::ArgValue<std::string>(stateFile),parser::IfParentArg<std::string>(argAction,"state-to-image")
      );
      // state file dataset name
      parser.addArg( parser::ArgName("-states-dataset","-sd","hdf5 file dataset name which contains the states"),
         parser::ArgValue<std::string>(stateName,"state"),parser::IfParentArg<std::string>(argAction,"state-to-image")
      );
      // output image
      parser.addArg( parser::ArgName("-output-image","-oi","output image filepath (using png's is save ) "),
         parser::ArgValue<std::string>(outputImagePath),parser::IfParentArg<std::string>(argAction,"state-to-image")
      );
      
   }
   parser.addArg( parser::ArgName("-verbose","-v","verbose information"),
                  parser::ArgValue<bool>(verbose,false));
   
   // parse all the arguments
   parser.parse();
   numberOfImages=inputImagePath.size();

   if(verbose){
      std::cout<<"number of input images: "<<inputImagePath.size()<<"\n";
      for(size_t i=0;i<inputImagePath.size();++i){
         std::cout<<"Image["<<i<<"] FilePath: "<<inputImagePath[i]<<"\n";
      }
      if(action==std::string("build-gm")){
         std::cout<<"Data Constraining Image Path: "<<dataCostPath<<"\n";
      }
   }
   // Load all the images
   vigra::BRGBImage inputDataCostImage;
   std::vector<vigra::BRGBImage > inputImages(numberOfImages);
   //load images
   try {
      for(size_t i =0; i < numberOfImages; ++i){
         vigra::ImageImportInfo infoImage(inputImagePath[i].c_str());
         if (!infoImage.isGrayscale() ){
            inputImages[i].resize(infoImage.width(), infoImage.height());
            vigra::importImage(infoImage, vigra::destImage(inputImages[i]));
            if (  i!=0 &&  inputImages[i].width()!=inputImages[i-1].width() &&  inputImages[i].height()!=inputImages[i-1].height())
                throw opengm::RuntimeError("Images must all have the same shape");
         } 
         else 
            throw opengm::RuntimeError("Images must be color-images ");
      }
      if(action==std::string("build-gm")){
         vigra::ImageImportInfo infoImage(dataCostPath.c_str());
         if (!infoImage.isGrayscale() ){
            inputDataCostImage.resize(infoImage.width(), infoImage.height());
            vigra::importImage(infoImage, vigra::destImage(inputDataCostImage));
         } 
         else 
            throw opengm::RuntimeError("Images must be color-images ");
      }  
   } 
   catch (vigra::StdException & e) {
      std::cout << e.what() << std::endl;
      return 1;
   }
   
   
   if(action==std::string("build-gm")){
      typedef PhotomontageMerging<ModelValueType > ModelGeneratorType;
      typedef ModelGeneratorType::ResultImageType ResultImageType;
      typedef ModelGeneratorType::GraphicalModelType GraphicalModelType;
      ModelGeneratorType modelGenerator(inputImages, inputDataCostImage,verbose);
      GraphicalModelType gm;
      modelGenerator.buildModel(gm);
      opengm::hdf5::save(gm,outputGmPath, outputGmName);
   }
   else{
      hid_t statefile = marray::hdf5::openFile(stateFile,marray::hdf5::READ_ONLY,marray::hdf5::DEFAULT_HDF5_VERSION);
      marray::Vector<StateType> stateVector;
      marray::hdf5::load(statefile,stateName,stateVector);
      vigra::BRGBImage  outputImage(inputImages.front().width(),inputImages.front().height());
      // fill output image
      for(size_t y=0,vi=0;y<inputImages.front().height();++y){
         for(size_t x=0;x<inputImages.front().width();++x,++vi){
            outputImage(x,y)=inputImages[stateVector(vi)](x,y);
         }
      }
      // export image
      vigra::exportImage(vigra::srcImageRange(outputImage), vigra::ImageExportInfo(outputImagePath.c_str()));
   }
   
   return 0;
   
}
