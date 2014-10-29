
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "./utilities/pgmimage.hxx"

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/fieldofexperts.hxx>
#include <opengm/operations/adder.hxx>



int main(int argc, char **argv) {
    // Parse command arguments
    if (argc != 3){
       std::cerr << "Usage: "<<argv[0]<<" infile outfile" << std::endl;
        exit(-1);
    }

    char *infilename = argv[1];
    char *outfilename = argv[2];

    opengm::PGMImage<unsigned char> im;
    im.readPGM(infilename);

    int height = im.height();
    int width = im.width();

    std::cout << "Process "<<height<<" x"<<width<<" image... "<<std::endl; 

    size_t numberOfLabels=256; 

    typedef double ValueType;
    typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
    Space space(size_t(height * width), numberOfLabels);
    typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, opengm::FoEFunction<double> ) , Space> Model;
    Model gm(space);
    std::cout <<gm.numberOfVariables()<<std::endl;

    std::vector<Model::FunctionIdentifier> fids(256);
    for(size_t i=0; i<256; ++i){
       const size_t shape[] = {numberOfLabels};
       opengm::ExplicitFunction<double> f(shape, shape + 1);
       for(size_t s = 0; s < numberOfLabels; ++s) {
          ValueType dist = ValueType(s) - ValueType(i);
          f(s) = dist*dist/800.0;
       }
       fids[i] = gm.addFunction(f);
    }

    //unary
    for (size_t j = 0; j < width; ++j) {
       for (size_t i = 0; i < height; ++i) {
          size_t variableIndices[] = { i + j*height};
          gm.addFactor(fids[im(i,j)], variableIndices, variableIndices + 1);
       }
    }
   
    // For each 2x2 patch, add in a Field of Experts clique 
    double alpha[3] = {0.586612685392731, 1.157638405566669, 0.846059486257292};
    double experts[12] = {
       -0.0582774013402734, 0.0339010363051084, -0.0501593018104054, 0.0745568557931712,
       0.0492112815304123, -0.0307820846538285, -0.123247230948424, 0.104812330861557,
       0.0562633568728865, 0.0152832583489560, -0.0576215592718086, -0.0139673758425540
    };
    opengm::FoEFunction<double> f(experts,alpha,256,4,3);
    Model::FunctionIdentifier fid = gm.addFunction(f);

    for (size_t j = 0; j < width - 1; ++j) {
       for (size_t i = 0; i < height - 1; ++i) {
          int vars[4];
          vars[0] = i     + j    *height;
          vars[1] = (i+1) + j    *height;
          vars[2] = i     + (j+1)*height;
          vars[3] = (i+1) + (j+1)*height;
          gm.addFactor(fid, vars, vars+4);
       }
    } 

    std::cout <<"save model ..."<<std::flush;
    opengm::hdf5::save(gm, outfilename, "gm");  
    std::cout <<" done!"<<std::endl;
    return 0;
}
