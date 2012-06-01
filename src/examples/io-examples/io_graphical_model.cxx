#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/multiplier.hxx>

int main() {   
   // build a graphical model (other examples have more details)
   size_t numbersOfLabels[] = {3, 3, 3, 3};
   typedef opengm::GraphicalModel<float, opengm::Multiplier> GraphicalModel;
   GraphicalModel gmA(opengm::DiscreteSpace<>(numbersOfLabels, numbersOfLabels + 4)); 
   size_t shape[] = {3};
   opengm::ExplicitFunction<float> f(shape, shape + 1); 
   for(size_t i = 0; i < gmA.numberOfVariables(); ++i) {
      size_t vi[] = {i};
      f(0) = float(i);
      f(1) = float(i + 1);
      f(2) = float(i - 2);
      GraphicalModel::FunctionIdentifier idExplicit = gmA.addFunction(f);
      gmA.addFactor(idExplicit, vi, vi + 1);
   }

   // save graphical model into an hdf5 dataset named "toy-gm"
   opengm::hdf5::save(gmA, "gm.h5", "toy-gm");
   GraphicalModel gmB;

   // load the graphical model from the hdf5 dataset
   opengm::hdf5::load(gmB, "gm.h5","toy-gm");
}
