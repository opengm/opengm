#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <opengm/opengm.hxx>
#include <opengm/datastructures/marray/marray.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/astar.hxx>

using namespace std; // 'using' is used only in example code

template<class T>
void createAndPrintData(size_t nrOfVariables, marray::Marray<T>& data) {
	size_t shape[]={nrOfVariables, nrOfVariables};
	data.resize(shape, shape+2);
	cout << "pariwise costs:" << endl;
	srand(0);
	for(size_t v=0; v<data.shape(0); ++v) {
		for(size_t s=0; s<data.shape(0); ++s) {
			data(v, s) = static_cast<float>(rand() % 100) * 0.01;
			cout << left << setw(6) << setprecision(2) << data(v, s);
		}
		cout << endl;
	}
}

void printSolution(const vector<size_t>& solution) {
	set<size_t> unique;
	cout << endl << "Solution Labels :" << endl;
	for(size_t v=0;v<solution.size();++v) {
		cout << left << setw(2) << v << "  ->   " << solution[v] << endl;
	}
}

int main() {
   // model parameters
	const size_t nrOfVariables = 5;
	const size_t nrOfLabels = nrOfVariables;
   float high = 20;	
   cout << endl << "Matching with one to one correspondences:" << endl
	   << nrOfVariables << " variables with " 
      << nrOfLabels <<" labels" << endl << endl;
	
   // pairwise costs
	marray::Marray<float> data;
	createAndPrintData(nrOfVariables, data);
	
   // build the model with
   // - addition as the operation (template parameter Adder)
   // - support for Potts functions (template parameter PottsFunction<double>))
   // - nrOfVariables variables, each having nrOfLabels labels
	typedef opengm::ExplicitFunction<float> ExplicitFunction;
	typedef opengm::PottsFunction<float> PottsFunction;
	typedef opengm::GraphicalModel<float, opengm::Adder, 
		OPENGM_TYPELIST_2(PottsFunction, ExplicitFunction),
      opengm::SimpleDiscreteSpace<>
	> Model;
	typedef Model::FunctionIdentifier FunctionIdentifier;	
   Model gm(opengm::SimpleDiscreteSpace<>(nrOfVariables, nrOfLabels));

   // add 1st order functions and factors
	{
		const size_t shape[] = {nrOfLabels};
		ExplicitFunction f(shape, shape+1);
		for(size_t v=0; v<nrOfVariables; ++v) {
			for(size_t s=0; s<nrOfLabels; ++s) {
				f(s) = 1.0f-data(v, s);
			}
			FunctionIdentifier id = gm.addFunction(f);
			size_t vi[] = {v};
			gm.addFactor(id, vi, vi+1);
		}
	}

   // add 2nd order functions and factors
	{
		// add one (!) 2nd order Potts function
		PottsFunction f(nrOfLabels, nrOfLabels, high, 0);
		FunctionIdentifier id = gm.addFunction(f);
		// add pair potentials for all variables
		for(size_t v1=0;v1<nrOfVariables;++v1)
		for(size_t v2=v1+1;v2<nrOfVariables;++v2) {
			size_t vi[] = {v1, v2};
			gm.addFactor(id, vi, vi+2);
		}
	}

   // set up the optimizer (A-star search)
	typedef opengm::AStar<Model, opengm::Minimizer> AstarType;
	AstarType astar(gm);

	// obtain and print the argmin
	AstarType::VerboseVisitorType verboseVisitor;
	cout << "\nA-star search:\n";
	astar.infer(verboseVisitor);
	vector<size_t> argmin;
	astar.arg(argmin);
	printSolution(argmin);
}
