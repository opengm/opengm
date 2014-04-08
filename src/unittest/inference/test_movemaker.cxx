#include <vector>

#include <opengm/unittests/test.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/inference/movemaker.hxx>

inline size_t variableIndex(const size_t x, const size_t y, const size_t nx) {
   return x + y*nx; 
}

struct MovemakerTest {
   void run() {
      {
         typedef opengm::GraphicalModel<float, opengm::Adder> GraphicalModelType;
         typedef opengm::ExplicitFunction<GraphicalModelType::ValueType> ExplicitFunctionType;
         typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;
         typedef float ValueType;
         typedef ValueType Energy;

         size_t numbersOfStates[] = {3, 3, 3, 3, 3};
         //size_t someVar[] = {0, 2, 4};
         GraphicalModelType gm(opengm::DiscreteSpace<size_t, size_t > (numbersOfStates, numbersOfStates + 5));
         // single site factors
         for (size_t j = 0; j < gm.numberOfVariables(); ++j) {
            ExplicitFunctionType f1(numbersOfStates, numbersOfStates + 1);
            size_t variableIndices[] = {j};
            f1(0) = 0.0f;
            f1(1) = 0.2f;
            f1(2) = 0.2f;
            FunctionIdentifier i1 = gm.addFunction(f1);
            gm.addFactor(i1, variableIndices, variableIndices + 1);
         }

         // 2nd order factors
         for (size_t j = 0; j < gm.numberOfVariables() - 1; ++j) {
            size_t variableIndices[] = {j, j + 1};
            ExplicitFunctionType f1(numbersOfStates, numbersOfStates + 2);
            for (size_t j = 0; j < 9; ++j) {
               f1(j) = static_cast<float> (j) / 10;
            }
            FunctionIdentifier i1 = gm.addFunction(f1);
            gm.addFactor(i1, variableIndices, variableIndices + 2);
         }

         // 3rd order factor
         {
            size_t variableIndices[] = {0, 2, 4};
            ExplicitFunctionType f1(numbersOfStates, numbersOfStates + 3, 1.0);
            for (size_t j = 0; j < 27; ++j) {
               f1(j) = static_cast<float> (j) / 20;
            }
            FunctionIdentifier i1 = gm.addFunction(f1);
            gm.addFactor(i1, variableIndices, variableIndices + 3);
         }

         // test by exhaustive comparison
         typedef opengm::Movemaker<GraphicalModelType> Movemaker;
         Movemaker movemaker(gm);
         {
            Movemaker movemakerA = movemaker;
            Movemaker movemakerB = movemaker;
         }
         std::vector<size_t> state(gm.numberOfVariables());
         std::vector<size_t> vi(gm.numberOfVariables());
         for (size_t j = 0; j < gm.numberOfVariables(); ++j) {
            vi[j] = j;
         }
         bool overflow = false;
         while (!overflow) {
            OPENGM_TEST(
               movemaker.valueAfterMove(vi.begin(), vi.end(), state.begin())
               == gm.evaluate(state.begin())
               );
            for (size_t j = 0; j < gm.numberOfVariables(); ++j) {
               if (state[j] + 1 < gm.numberOfLabels(j)) {
                  ++state[j];
                  break;
               } else {
                  state[j] = 0;
                  if (j == gm.numberOfVariables() - 1) {
                     overflow = true;
                  }
               }
            }
         }
      }

      // moveOptimally test
      {
         const size_t nx = 8; // width of the grid
         const size_t ny = 8; // height of the grid
         double lambda = 0.1; // coupling strength of the Potts model
         // construct a label space with
         // - nx * ny variables 
         // - each having numberOfLabels many labels
         typedef opengm::DiscreteSpace<size_t, size_t> Space;
         const size_t numberOfLabels2[]=
         {
            2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 2, 3
         };
         Space space2(numberOfLabels2, numberOfLabels2+8*8);

         typedef opengm::GraphicalModel<double, opengm::Adder,
            opengm::meta::TypeListGenerator<
               opengm::PottsFunction<double>,
               opengm::ExplicitFunction<double>
            >::type, Space> Model;
         Model gm2(space2);

         // for each node (x, y) in the grid, i.e. for each variable
         // variableIndex(x, y) of the model, add one 1st order functions
         // and one 1st order factor
         for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
               // function
               const size_t shape[] = {gm2.numberOfLabels(variableIndex(x, y, nx))};
               opengm::ExplicitFunction<double>  f(shape, shape + 1);
               for (size_t s = 0; s < shape[0]; ++s) {
                  f(s) = (1.0 - lambda) * rand() / RAND_MAX;
               }
               Model::FunctionIdentifier fid = gm2.addFunction(f);
               size_t variableIndices[] = {variableIndex(x, y, nx)};
               gm2.addFactor(fid, variableIndices, variableIndices + 1);
         }

         // for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
         // add one factor that connects the corresponding variable indices and 
         // refers to the Potts function
         for (size_t y = 0; y < ny; ++y)
         for (size_t x = 0; x < nx; ++x) {
            if (x + 1 < nx) { // (x, y) -- (x + 1, y)
               size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x + 1, y, nx)};
               
               const size_t shape[] = {gm2.numberOfLabels(variableIndex(x, y, nx)), gm2.numberOfLabels(variableIndex(x+1, y, nx))};
               OPENGM_ASSERT(variableIndex(x, y, nx)<variableIndex(x+1, y, nx));
               opengm::ExplicitFunction<double>  f(shape, shape + 2);
               for(size_t fs=0;fs<f.size();++fs) {
                  f(fs)= rand()/RAND_MAX;
               }
               Model::FunctionIdentifier fid = gm2.addFunction(f);
               gm2.addFactor(fid, variableIndices, variableIndices + 2);
            }
            if (y + 1 < ny) { // (x, y) -- (x, y + 1)
               size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x, y + 1, nx)};
               
               const size_t shape[] = {gm2.numberOfLabels(variableIndex(x, y, nx)), gm2.numberOfLabels(variableIndex(x, y+1, nx))};
               OPENGM_ASSERT(variableIndex(x, y, nx)<variableIndex(x, y+1, nx));
               opengm::ExplicitFunction<double>  f(shape, shape + 2);
               for(size_t fs=0;fs<f.size();++fs) {
                  f(fs)= rand()/RAND_MAX;
               }
               Model::FunctionIdentifier fid = gm2.addFunction(f);
               gm2.addFactor(fid, variableIndices, variableIndices + 2);
            }
         }

         typedef opengm::Movemaker<Model> Movemaker2;
         Movemaker2 movemaker2(gm2);
         {
            //size_t varToMoveOpt[]={1, 2, 3, 31, 32, 33, 61, 62, 63};
            /*
              size_t varToMoveOpt[]={
               5, 6, 9, 10,
               5+8, 6+8, 9+8, 10+8,
               5+16, 6+16, 9+16, 10+16,
            };
            */
            //const size_t varToFlip=12;
            Movemaker2 movemaker2A = movemaker2;
            Movemaker2 movemaker2B = movemaker2;
            std::vector<size_t> init(gm2.numberOfVariables(), 1);
            movemaker2A.initialize(init.begin());
            movemaker2B.initialize(init.begin());
            //movemaker2A.moveAstarOptimally<opengm::Minimizer > (varToMoveOpt, varToMoveOpt + varToFlip);
            //movemaker2B.moveOptimally<opengm::Minimizer > (varToMoveOpt, varToMoveOpt + varToFlip);
            //OPENGM_TEST_EQUAL_TOLERANCE(gm2.evaluate(movemaker2A.stateBegin()), gm2.evaluate(movemaker2B.stateBegin()), 0.0001);
            //for(size_t i=0;i<gm2.numberOfVariables();++i) {
               //OPENGM_TEST_EQUAL(movemaker2A.state(i), movemaker2B.state(i));
            //}
         }
      }

      // moveOptimallyWithAllLabelsChanging test
      {
         typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
         typedef float Value;
         typedef opengm::ExplicitFunction<Value> ExplicitFunction;
         typedef opengm::GraphicalModel<Value, opengm::Adder, opengm::meta::TypeList<ExplicitFunction, opengm::meta::ListEnd>, Space> GraphicalModel;
         typedef GraphicalModel::FunctionIdentifier FunctionIdentifier;
         typedef opengm::Movemaker<GraphicalModel> Movemaker;

         Space space(3, 3);
         GraphicalModel model(space);
         const size_t variableIndices[] = {0, 1, 2};

         // add function f(x0, x1, x2) =
         //    0.0f if (x0 = 0 or x1 = 2 or x2 = 1) and not (x0 = 0 and x1 = 2 and x2 = 1)
         //    5.0f otherwise
         {
            const size_t shape[] = {3, 3, 3};
            ExplicitFunction f(shape, shape + 3, 5.0f);
            {
               marray::View<float> view = f.boundView(0, 0); // x0 = 0
               view = 0.0f;
            }
            {
               marray::View<float> view = f.boundView(1, 2); // x1 = 2
               view = 0.0f;
            }
            {
               marray::View<float> view = f.boundView(2, 1); // x2 = 1
               view = 0.0f;
            }
            f(0, 2, 1) = 5.0f;
            f(2, 1, 2) = 3.0f;
            FunctionIdentifier fid = model.addFunction(f);
            model.addFactor(fid, variableIndices, variableIndices + 3);
         }

         Movemaker movemaker(model);
         const size_t labeling[] = {0, 2, 1};
         Value value = movemaker.move(variableIndices, variableIndices + 3, labeling);
         OPENGM_TEST(movemaker.state(0) == 0);
         OPENGM_TEST(movemaker.state(1) == 2);
         OPENGM_TEST(movemaker.state(2) == 1);
         OPENGM_TEST(value == 5.0f);

         value = movemaker.moveOptimallyWithAllLabelsChanging<opengm::Minimizer>(variableIndices, variableIndices + 3);
         OPENGM_TEST(movemaker.state(0) == 2);
         OPENGM_TEST(movemaker.state(1) == 1);
         OPENGM_TEST(movemaker.state(2) == 2);
         OPENGM_TEST(value == 3.0f);
      }

      // moveOptimallyWithAllLabelsChanging additional test
      {
         typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
         typedef float Value;
         typedef opengm::ExplicitFunction<Value> ExplicitFunction;
         typedef opengm::GraphicalModel<Value, opengm::Adder, opengm::meta::TypeList<ExplicitFunction, opengm::meta::ListEnd>, Space> GraphicalModel;
         typedef GraphicalModel::FunctionIdentifier FunctionIdentifier;
         typedef opengm::Movemaker<GraphicalModel> Movemaker;

         const size_t numberOfVariables = 50;
         Space space(numberOfVariables, 3);
         GraphicalModel model(space);

         srand(0);
         for(size_t j=0; j<numberOfVariables-3; ++j) {
            const size_t shape[] = {3, 3, 3};
            ExplicitFunction f(shape, shape + 3);
            for(size_t k=0; k<27; ++k) {
               f(k) = rand() % 20;
            }
            FunctionIdentifier fid = model.addFunction(f);

            const size_t variableIndices[] = {j, j+1, j+2};
            model.addFactor(fid, variableIndices, variableIndices + 3);
         }

         std::vector<size_t> labels(numberOfVariables);
         Movemaker movemaker(model);
         for(size_t j=0; j<100; ++j) {
            // set random state
            for(size_t m=0; m<numberOfVariables; ++m) {
               labels[m] = rand() % 3;
            }
            movemaker.initialize(labels.begin());
            OPENGM_TEST_EQUAL(model.evaluate(movemaker.stateBegin()), movemaker.value());

            const size_t k = rand() % (numberOfVariables - 3);
            const size_t variableIndices[] = {k, k+1, k+2, k+3};
            const Value v = movemaker.moveOptimallyWithAllLabelsChanging<opengm::Minimizer>(variableIndices, variableIndices + 4);

            OPENGM_TEST_EQUAL(v, movemaker.value());
            OPENGM_TEST_EQUAL(v, model.evaluate(movemaker.stateBegin()));
         }
      }
   }
};

int main() {
   std::cout << "Movemaker Tests ..." << std::endl;
   {
      MovemakerTest t;
      t.run();
   }
   return 0;
}
