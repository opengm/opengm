#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <set>

#include "mymath.hxx"

#include <opengm/opengm.hxx>
#include <opengm/datastructures/marray/marray.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/datastructures/sparsemarray/sparsemarray.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/astar.hxx>

using namespace std; // 'using' is used only in example code

template<class T>
struct PositionAndEvidence {
	T x;
	T y;
	T e;
};

enum Face{
   LeftEye = 0, 
   RightEye = 1, 
   TipOfTheNose = 2, 
   LeftCornerOfMouth = 3, 
   RightCornerOfMouth = 4
};

template<class T>
void createData
(
   const size_t nrOfVariables, 
   const size_t minCandidates, 
   const size_t maxCandidates, 
   const size_t dimX, 
   const size_t dimY, 
   std::vector< std::vector<PositionAndEvidence<T> > > & candidates
) {
	srand(minCandidates + maxCandidates + dimX + dimY);
	candidates.resize(nrOfVariables);
	for(size_t v=0; v < nrOfVariables; ++v) {
      T px, py;
      switch (v) {
         case 0: 
            px = 2;
            py = 5;
            break;
         case 1: 
            px = 4;
            py = 5;
            break;
         case 2: 
            px = 3;
            py = 3;
            break;
         case 3: 
            px = 2;
            py = 2;
            break;
         case 4: 
            px = 4;
            py = 2;
            break;
      }
		size_t nrOfCandidates = (rand() % (maxCandidates + 1 - minCandidates)) + minCandidates;
		candidates[v].resize(nrOfCandidates);
      T x, y;
		for(size_t c=0; c < nrOfCandidates; ++c)
		{
         limitedRandGauss2d(T(0), T(dimX), T(0), T(dimY), px, py, T(1), T(1), x, y);
			candidates[v][c].x = x;
			candidates[v][c].y = y;
			candidates[v][c].e = T(rand()) / RAND_MAX;
		}
	}
}

template<class T>
void printData(
   const std::vector< std::vector<PositionAndEvidence<T> > >& candidates
)
{
	std::cout << endl << "Position and evidence of the candidates: \n(x / y / evidence)" << endl;
	for(size_t v=0;v<candidates.size();++v) {
		std::string candidateName;
		switch(v) {
         case 0 : candidateName="left eye"; break;
         case 1 : candidateName="right eye"; break;
         case 2 : candidateName="tip of the nose"; break;
         case 3 : candidateName="left corner of the mouth"; break;
         case 4 : candidateName="light corner of the mouth"; break;
		}
		std::cout << endl << "Positions of the " << candidateName << "-candidates" << endl;
		size_t nrOfCandidates = candidates[v].size();
		for(size_t c=0; c < nrOfCandidates; ++c) {
			std::cout << "(";
			std::cout << std::left << std::setw(3) << std::setprecision(1) << std::fixed << candidates[v][c].x;
			std::cout << " / ";
			std::cout << std::left << std::setw(3) << std::setprecision(1) << std::fixed << candidates[v][c].y;
			std::cout << " / ";
			std::cout << std::left << std::setw(3) << std::setprecision(1) << std::fixed << candidates[v][c].e;
			std::cout << ") ";
			if((c+1) % 3==0 && c != candidates.size() - 1) {
            std::cout << endl;
         }
		}
		std::cout << endl;
	}
}

template<class T>
void printSolution(
   const std::vector< std::vector<PositionAndEvidence<T> > >& candidates, 
   const std::vector<size_t>& solution
)
{
	std::cout << endl << "Solutions:" << endl;
	for(size_t v=0; v < solution.size(); ++v) {
		std::string candidateName;
		switch(v) {
		   case 0: candidateName="left eye"; break;
		   case 1: candidateName="right eye"; break;
		   case 2: candidateName="tip of the nose"; break;
		   case 3: candidateName="left corner of the mouth"; break;
		   case 4: candidateName="right corner of the mouth"; break;
		}
		const size_t s = solution[v];
		const T x = (candidates[v][s].x );
		const T y = (candidates[v][s].y );
		const T e = (candidates[v][s].e );
		std::cout << std::setw(30) << candidateName << "-> candidate " << s
		 << " ( " << std::setw(3) << std::setprecision(1) << std::fixed << x
		 << " / " << std::setw(3) << std::setprecision(1) << std::fixed << y
		 << " / " << std::setw(3) << std::setprecision(1) << std::fixed << e << " )" << endl;
	}
}

int main() {
	typedef float ValueType;
	const size_t nrOfVariables = 5;
	const size_t minCandidates = 8;
	const size_t maxCandidates = 8;
	const size_t dimX = 10;
	const size_t dimY = 10;
	// weight between first and second order factor
	ValueType lambda = 0.1;
	ValueType high = 100;
	size_t desiredDistancesShape[] = {nrOfVariables, nrOfVariables};
	marray::Marray<ValueType> desiredDistances(desiredDistancesShape, desiredDistancesShape+2, 0);
	desiredDistances(LeftEye, RightEye) = 3.0;
	desiredDistances(LeftEye, TipOfTheNose) = sqrt(5.0);
	desiredDistances(LeftEye, LeftCornerOfMouth) = 3.0;
	desiredDistances(LeftEye, RightCornerOfMouth) = sqrt(13.0f);
	desiredDistances(RightEye, TipOfTheNose) = sqrt(5.0);
	desiredDistances(RightEye, LeftCornerOfMouth) = sqrt(13.0f);
	desiredDistances(RightEye, RightCornerOfMouth) = 3.0;
	desiredDistances(TipOfTheNose, LeftCornerOfMouth) = sqrt(2.0);
	desiredDistances(TipOfTheNose, RightCornerOfMouth) = sqrt(2.0);
	desiredDistances(LeftCornerOfMouth, RightCornerOfMouth) = 2.0;

	std::vector< std::vector<PositionAndEvidence<ValueType> > > candidates;
	createData(nrOfVariables, minCandidates, maxCandidates, dimX, dimY, candidates);
	printData(candidates);

	// construct a graphical model with 
   // - addition as the operation (template parameter Adder)
   // - support for explicit and sparse functions
   // - nrOfVariables variables (with different numbers of labels)
   typedef opengm::ExplicitFunction<ValueType> ExplicitFunction;
   typedef opengm::SparseMarray<std::map<size_t, ValueType> > SparseFunction;
   typedef opengm::meta::TypeListGenerator<ExplicitFunction, SparseFunction>::type FunctionTypeList;
	typedef opengm::GraphicalModel<ValueType, opengm::Adder, FunctionTypeList>	GraphicalModel;
	typedef GraphicalModel::FunctionIdentifier FunctionIdentifier;
	std::vector<size_t> variableStates(nrOfVariables);
	for(size_t v=0;v<nrOfVariables;++v) {
		variableStates[v]=candidates[v].size();
	}
	GraphicalModel gm(opengm::DiscreteSpace<>(variableStates.begin(), variableStates.end()));

   // add 1st order functions and factors
	{
		for(size_t v=0; v<nrOfVariables; ++v) {
			const size_t shape[] = {candidates[v].size()};
			ExplicitFunction f(shape, shape+1);
			for(size_t c = 0; c < candidates[v].size(); ++c) {
				f(c) = lambda * (1.0 - candidates[v][c].e);
			}
			FunctionIdentifier id = gm.addFunction(f);
			size_t vi[] = {v};
			gm.addFactor(id, vi, vi+1);
		}
	}

   // add 2nd order functions and factors
	{
		for(size_t v1 = 0; v1 < nrOfVariables; ++v1)
		for(size_t v2 = v1 + 1; v2 < nrOfVariables; ++v2) {
			size_t shape[] = {candidates[v1].size(), candidates[v2].size()};
			SparseFunction f(shape, shape+2, high);
			for(size_t c2=0; c2<shape[1]; ++c2)
			for(size_t c1=0; c1<shape[0]; ++c1) {
				const ValueType c1x = candidates[v1][c1].x;
				const ValueType c1y = candidates[v1][c1].y;
				const ValueType c2x = candidates[v2][c2].x;
				const ValueType c2y = candidates[v2][c2].y;
				// distance between c1 and c2
				const ValueType d12 = sqrt((c1x-c2x)*(c1x-c2x) + (c1y-c2y)*(c1x-c2y));
				// desired distance
				const ValueType dd = desiredDistances(v1, v2);
				const ValueType d = fabs(dd - d12);
				// add value only if d is not too large
				if(dd - dd / 4 <= d && d <= dd + dd / 4) {
					f(c1, c2) = (1.0-lambda) * 
                  (d / sqrt(static_cast<float>(dimX * dimX + dimY * dimY)));
				}
			}
			FunctionIdentifier id = gm.addFunction(f);	
         // sequences of variable indices need to be (and in this case are) sorted
			size_t vi[]={ v1, v2 }; // variables indices 
			gm.addFactor(id, vi, vi+2);
		}
	}

	// set up the optimizer (A*Star)
	typedef opengm::AStar<GraphicalModel, opengm::Minimizer> AstarType;
	AstarType::VerboseVisitorType verboseVisitor;
	AstarType astar(gm);

	// obtain the (approximate) argmin
	std::cout<<"Start Inference:\n";
	astar.infer(verboseVisitor);

	// output the (approximate) argmin
	std::vector<size_t> solution;
	astar.arg(solution);
	printSolution(candidates, solution);
}
