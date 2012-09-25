#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include <opengm/opengm.hxx>
#include <opengm/datastructures/marray/marray.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/lazyflipper.hxx>

using namespace std; // 'using' is used only in example code

// this class is used to map a node (x, y) in the topological
// grid to a unique variable index
class TopologicalCoordinateToIndex {
public:
   TopologicalCoordinateToIndex(
      const size_t geometricGridSizeX,
      const size_t geometricGridSizeY
   ) 
   :  gridSizeX_(geometricGridSizeX), 
      gridSizeY_(geometricGridSizeY) 
   {}

   const size_t operator()(
      const size_t tx,
      const size_t ty
   ) const {
      return tx / 2 + (ty / 2)*(gridSizeX_) + ((ty + ty % 2) / 2)*(gridSizeX_ - 1);
   }

   size_t gridSizeX_;
   size_t gridSizeY_;
};

template<class T>
void randomData(
   const size_t gridSizeX,
   const size_t gridSizeY, 
   marray::Marray<T>& data
) {
   srand(gridSizeX * gridSizeY);
   const size_t shape[] = {gridSizeX, gridSizeY};
   data.assign();
   data.resize(shape, shape + 2);
   for (size_t y = 0; y < gridSizeY; ++y) {
      for (size_t x = 0; x < gridSizeX; ++x) {
         data(x, y) = static_cast<float> (rand() % 10) *0.1;
      }
   }
}

template<class T>
void printData(
   const marray::Marray<T> & data
) {
   cout << "energy for boundary to be active:" << endl;
   for (size_t y = 0; y < data.shape(1)*2 - 1; ++y) {
      for (size_t x = 0; x < data.shape(0)*2 - 1; ++x) {
         if (x % 2 == 0 && y % 2 == 0) {
            cout << left << setw(3) << setprecision(1) << data(x / 2, y / 2);
         } else if (x % 2 == 0 && y % 2 == 1) {
            cout << left << setw(3) << setprecision(1) << "___";
         } else if (x % 2 == 1 && y % 2 == 0) {
            cout << left << setw(3) << setprecision(1) << " | ";
         } else if (x % 2 == 1 && y % 2 == 1) {
            cout << left << setw(3) << setprecision(1) << " * ";
         }
      }
      cout << "\n";
   }
}

// output the (approximate) argmin
template<class T>
void printSolution(
   const marray::Marray<T>& data, 
   const vector<size_t>& solution
) 
{
   TopologicalCoordinateToIndex cTHelper(data.shape(0), data.shape(1));
   cout << endl << "solution states:" << endl;
   cout << "solution:" << endl;
   for (size_t x = 0; x < data.shape(0)*2 - 1; ++x) {
      cout << left << setw(3) << setprecision(1) << "___";
   }
   cout << endl;
   for (size_t y = 0; y < data.shape(1)*2 - 1; ++y) {
      cout << "|";
      for (size_t x = 0; x < data.shape(0)*2 - 1; ++x) {
         if (x % 2 == 0 && y % 2 == 0) {
            data(x / 2, y / 2) = static_cast<float> (rand() % 10) *0.1;
            cout << left << setw(3) << setprecision(1) << " ";
         } else if (x % 2 == 0 && y % 2 == 1) {
            if (solution[cTHelper(x, y)]) {
               cout << left << setw(3) << setprecision(1) << "___";
            }
            else {
               cout << left << setw(3) << setprecision(1) << "   ";
            }
         } else if (x % 2 == 1 && y % 2 == 0) {
            if (solution[cTHelper(x, y)])
               cout << left << setw(3) << setprecision(1) << " | ";
            else
               cout << left << setw(3) << setprecision(1) << "   ";
         } else if (x % 2 == 1 && y % 2 == 1) {
            cout << left << setw(3) << setprecision(1) << " * ";
         }
      }
      cout << "|" << endl;
   }
   for (size_t x = 0; x < data.shape(0)*2 - 1; ++x) {
      cout << left << setw(3) << setprecision(1) << "___";
   }
   cout << endl;
}

// user defined Function Type
template<class T>
struct ClosednessFunctor {
public:
   typedef T value_type;

   template<class Iterator>
   inline const T operator()(Iterator begin)const {
      size_t sum = begin[0];
      sum += begin[1];
      sum += begin[2];
      sum += begin[3];
      if (sum != 2 && sum != 0) {
         return high;
      }
      return 0;
   }

   size_t dimension()const {
      return 4;
   }

   size_t shape(const size_t i)const {
      return 2;
   }

   size_t size()const {
      return 16;
   }
   T high;
};

int main(int argc, char** argv) {
   // model parameters
   const size_t gridSizeX = 5, gridSizeY = 5; //size of grid
   const float beta = 0.9; // bias to choose between under- and over-segmentation
   const float high = 10; // closedness-enforcing soft-constraint

   // size of the topological grid
   const size_t tGridSizeX = 2 * gridSizeX - 1, tGridSizeY = 2 * gridSizeY - 1;
   const size_t nrOfVariables = gridSizeY * (gridSizeX - 1) + gridSizeX * (gridSizeY - 1);
   const size_t dimT[] = {tGridSizeX, tGridSizeY};
   TopologicalCoordinateToIndex cTHelper(gridSizeX, gridSizeY);
   marray::Marray<float> data;
   randomData(gridSizeX, gridSizeY, data);

   cout << "interpixel boundary segmentation with closedness:" << endl;
   printData(data);

   // construct a graphical model with 
   // - addition as the operation (template parameter Adder)
   // - the user defined function type ClosednessFunctor<float>
   // - gridSizeY * (gridSizeX - 1) + gridSizeX * (gridSizeY - 1) variables, 
   //   each having 2 many labels.
   typedef opengm::meta::TypeListGenerator<
      opengm::ExplicitFunction<float>,
      ClosednessFunctor<float>
   >::type FunctionTypeList;
   typedef opengm::GraphicalModel<float, opengm::Adder, FunctionTypeList,
      opengm::SimpleDiscreteSpace<> > Model;
   typedef Model::FunctionIdentifier FunctionIdentifier;
   Model gm(opengm::SimpleDiscreteSpace<>(nrOfVariables, 2));

   // for each boundary in the grid, i.e. for each variable 
   // of the model, add one 1st order functions 
   // and one 1st order factor
   {
      const size_t shape[] = {2};
      opengm::ExplicitFunction<float> f(shape, shape + 1);
      for (size_t yT = 0; yT < dimT[1]; ++yT) {
         for (size_t xT = 0; xT < dimT[0]; ++xT) {
            if ((xT % 2 + yT % 2) == 1) {
               float gradient = fabs(data(xT / 2, yT / 2) - data(xT / 2 + xT % 2, yT / 2 + yT % 2));              
               f(0) = beta * gradient; // value for inactive boundary               
               f(1) = (1.0 - beta) * (1.0 - gradient); // value for active boundary;
               FunctionIdentifier id = gm.addFunction(f);
               size_t vi[] = {cTHelper(xT, yT)};
               gm.addFactor(id, vi, vi + 1);
            }
         }
      }
   }

   // for each junction of four inter-pixel edges on the grid, 
   // one factor is added that connects the corresponding variable 
   // indices and refers to the ClosednessFunctor function
   {
      // add one (!) 4th order ClosednessFunctor function
      ClosednessFunctor<float> f;
      f.high = high;
      FunctionIdentifier id = gm.addFunction(f);
      // add factors
      for (size_t y = 0; y < dimT[1]; ++y) {
         for (size_t x = 0; x < dimT[0]; ++x) {
            if (x % 2 + y % 2 == 2) {
               size_t vi[] = {
                  cTHelper(x + 1, y),
                  cTHelper(x - 1, y),
                  cTHelper(x, y + 1),
                  cTHelper(x, y - 1)
               };
               sort(vi, vi + 4);
               gm.addFactor(id, vi, vi + 4);
            }
         }
      }
   }

   // set up the optimizer (lazy flipper)
   typedef opengm::LazyFlipper<Model, opengm::Minimizer> LazyFlipperType;
   LazyFlipperType::VerboseVisitorType verboseVisitor;
   size_t maxSubgraphSize = 5;
   LazyFlipperType lazyflipper(gm, maxSubgraphSize);
   cout << "start inference:" << endl;

   // obtain the (approximate) argmin
   lazyflipper.infer(verboseVisitor);

   // output the (approximate) argmin
   vector<size_t> solution;
   lazyflipper.arg(solution);
   printSolution(data, solution);
}
