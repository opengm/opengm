#include <string>
#include <sstream>

#include "./utilities/parser/cmd_parser.hxx"
#include <vigra/basicimage.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/impex.hxx>

#include <opengm/opengm.hxx>

#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>

#include "denoise.hxx"

int main(int argc, char **argv) {
   typedef unsigned long long StateType;
   typedef double ModelValueType;
   typedef double ParameterValueType;
   // which model
   std::string model; 
   //action
   std::string action;
   std::string allowedAction[] ={"build-gm","state-to-image"};
   // filepat
   std::string imagePath, maskPath, savePathPrefix, outputGmPath, outputGmName, resultImageName,stateFile,stateName,outputImagePath;
   bool useTruncation, verbose;
   ParameterValueType lambda, truncateAt;
   
   parser::CmdParser parser(argc, argv,"MRF Photomontage","MRF-Benchmark Denoise ( \"house\",\"penguin\" )","1.0","Thorsten Beier");
   
   parser.addArg( parser::ArgName("-verbose","-v","verbose output"),
      parser::ArgValue<bool>(verbose,false)
   );
   
   // input images
   parser.addArg(parser::ArgName("-input-image","-ii","greyscale input image (using png's is save )"),
      parser::ArgValue< std::string >(imagePath));
   
   
   parser::Arg argAction=parser.addArg(parser::ArgName("-action","-a","build the mode or generate output image from state vector"),
      parser::ArgValue<std::string>(action,allowedAction,allowedAction+2));
   // if action == build-gm
   {
      // mask image
      parser.addArg(parser::ArgName("-input-mask","-im","greyscale input mask image (using png's is save )"),
      parser::ArgValue< std::string >(maskPath),parser::IfParentArg<std::string>(argAction,"build-gm"));
      // lambda 
      parser.addArg( parser::ArgName("-lambda","-l","second order Energy weight"),
         parser::ArgValue<double>(lambda,5),parser::IfParentArg<std::string>(argAction,"build-gm"));
      // use truncation ?
      parser::Arg argUseTrucation = parser.addArg(parser::ArgName("-truncate","-t","use truncation"),
         parser::ArgValue<bool>(useTruncation,false),parser::IfParentArg<std::string>(argAction,"build-gm"));
      // truncate at ?
      parser.addArg( parser::ArgName("-truncate-at","-ta","truncate all values higher than this value"),
         parser::ArgValue<double>(truncateAt,200),
         parser::IfParentArg<bool>(argUseTrucation,true));
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
         parser::ArgValue<std::string>(stateFile),parser::IfParentArg<std::string>(argAction,"state-to-image"));
      // state file dataset name
      parser.addArg( parser::ArgName("-states-dataset","-sd","hdf5 file dataset name which contains the states"),
         parser::ArgValue<std::string>(stateName,"state"),parser::IfParentArg<std::string>(argAction,"state-to-image"));
      // output image
      parser.addArg( parser::ArgName("-output-image","-oi","output image filepath (using png's is save ) "),
         parser::ArgValue<std::string>(outputImagePath),parser::IfParentArg<std::string>(argAction,"state-to-image"));
   }
   parser.parse();

   if (verbose)std::cout << "Image Path: " << imagePath << "\n";
   if (verbose)std::cout << "Mask Path: " << maskPath << "\n";
   if (verbose)std::cout << "lambda: " << lambda << "\n";
   if (verbose && useTruncation)std::cout << "truncate at: " << truncateAt << "\n";

   vigra::BImage image, mask;
   //load images
   try {
      vigra::ImageImportInfo infoImage(imagePath.c_str());
      vigra::ImageImportInfo infoMask(maskPath.c_str());
      if (infoImage.isGrayscale() && infoMask.isGrayscale()) {
         if (infoImage.width() != infoMask.width() || infoImage.height() != infoMask.height()) {
            throw opengm::RuntimeError("image and mask  must have the same sizes");
         }
         image.resize(infoImage.width(), infoImage.height());
         mask.resize(infoMask.width(), infoMask.height());
         vigra::importImage(infoImage, vigra::destImage(image));
         vigra::importImage(infoMask, vigra::destImage(mask));
      }
      else {
         throw opengm::RuntimeError("Images must be grayscale");
      }
   } catch (vigra::StdException & e) {
      std::cout << e.what() << std::endl;
      return 1;
   }

   
   if(action==std::string("build-gm")){
      typedef Denoise<ModelValueType, ParameterValueType > ModelGeneratorType;
      typedef ModelGeneratorType::ResultImageType ResultImageType;
      typedef ModelGeneratorType::GraphicalModelType GraphicalModelType;
      ModelGeneratorType modelGenerator(image, mask, lambda, useTruncation ? truncateAt:double(10000000),verbose);
      GraphicalModelType gm;
      modelGenerator.buildModel(gm);
      opengm::hdf5::save(gm, savePathPrefix + outputGmPath, outputGmName);
   }
   else{
      hid_t statefile = marray::hdf5::openFile(stateFile,marray::hdf5::READ_ONLY,marray::hdf5::DEFAULT_HDF5_VERSION);
      marray::Vector<StateType> stateVector;
      marray::hdf5::load(statefile,stateName,stateVector);
      vigra::BImage  outputImage(image.width(),image.height());
      // fill output image
      for(size_t y=0,vi=0;y<image.height();++y){
         for(size_t x=0;x<image.width();++x,++vi){
            outputImage(x,y)=stateVector[vi];
         }
      }
      // export image
      vigra::exportImage(vigra::srcImageRange(outputImage), vigra::ImageExportInfo(outputImagePath.c_str()));
   }
  }


