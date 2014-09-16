
#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>

#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/auxiliary/fusion_move/permutable_label_fusion_mover.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>





template<class LVEC>
void generateRandState(LVEC & lvec ){

    opengm::RandomUniform<typename LVEC::value_type> randGen(0, 2);

    for(size_t i=0; i<lvec.size(); ++i){
        lvec[i] = randGen();
    }
}



int main() {

    using namespace opengm;


    typedef SimpleDiscreteSpace<size_t, size_t> Space;
    
    typedef GraphicalModel<double, Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> , PottsFunction<double> ) , Space> Model;

    const size_t nx = 100; // width of the grid
    const size_t ny = 100; // height of the grid
    const size_t numberOfLabels = nx*ny;

    Space space(nx * ny, numberOfLabels);
    Model gm(space);

    RandomUniform<float> randGen(-1.0, 1.0);

    for(size_t y = 0; y < ny; ++y) 
    for(size_t x = 0; x < nx; ++x) {
        if(x + 1 < nx) { // (x, y) -- (x + 1, y)
            PottsFunction<double> f(numberOfLabels, numberOfLabels, 0.0, randGen());
            Model::FunctionIdentifier fid = gm.addFunction(f);
            size_t variableIndices[] = {x + nx * y, (x+1) + nx * y};
            std::sort(variableIndices, variableIndices + 2);
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }
        if(y + 1 < ny) { // (x, y) -- (x, y + 1)
            PottsFunction<double> f(numberOfLabels, numberOfLabels, 0.0, randGen());
            Model::FunctionIdentifier fid = gm.addFunction(f);
            size_t variableIndices[] = {x + nx * y, x + nx * (y+1)};
            std::sort(variableIndices, variableIndices + 2);
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }
    }



    // label invariant fusion moves
    typedef PermutableLabelFusionMove<Model, Minimizer> FusionMover;
    FusionMover fusionMover(gm);

    std::vector<size_t> labelsA(gm.numberOfVariables());
    std::vector<size_t> labelsB(gm.numberOfVariables());
    std::vector<size_t> labelsR(gm.numberOfVariables());



    for(size_t i=0; i<100; ++i){
        generateRandState(labelsA);
        generateRandState(labelsB);

        fusionMover.fuse(labelsA, labelsB, labelsR);
        
    }
}
