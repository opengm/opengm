//
// File: canonical_view.hxx
//
// This file is part of OpenGM.
//
// Copyright (C) 2015 Stefan Haller
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//

#include <cstdlib>
#include <ctime>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/utilities/canonical_view.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#define ARRAY_LEN(x) (sizeof((x)) / sizeof((x)[0]))

template<class T>
void checkEqualityAndThrow(T a, T b)
{
	if (std::abs(a - b) > 0.00001)
		throw std::runtime_error("Test failed!");
}

int main()
{
	typedef double ValueType;
	typedef size_t IndexType;
	typedef size_t LabelType;
	typedef opengm::Adder OperatorType;
	typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

	typedef opengm::meta::TypeListGenerator<
		opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
		opengm::PottsFunction<ValueType, IndexType, LabelType>
	>::type FunctionTypes;

	typedef opengm::GraphicalModel<ValueType, OperatorType, FunctionTypes, SpaceType> GraphicalModelType;

	std::srand(time(0));

	//
	// Create GraphicalModel
	//

	LabelType spaceData[] = { 4, 4, 4, 4, 4, 4 };
	SpaceType space(spaceData, spaceData + ARRAY_LEN(spaceData));
	GraphicalModelType gm(space);

	//
	// Insert random unary factors multiple times.
	//

	for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
		IndexType varData[] = { i };
		LabelType shapeData[] = { gm.numberOfLabels(i) };
		opengm::ExplicitFunction<ValueType, IndexType, LabelType> f(shapeData, shapeData + ARRAY_LEN(shapeData));

		for (LabelType j = 0; j < gm.numberOfLabels(i); ++j)
			f(j) = static_cast<ValueType>(std::rand()) / RAND_MAX;

		GraphicalModelType::FunctionIdentifier fid = gm.addFunction(f);

		for (size_t i = 0; i < 4; ++i)
			gm.addFactor(fid, varData, varData + ARRAY_LEN(varData));
	}

	//
	// Insert Potts pairwise factors multiple times
	//

	opengm::PottsFunction<ValueType, IndexType, LabelType> potts(spaceData[0], spaceData[0], 0, 1);
	GraphicalModelType::FunctionIdentifier pottsFid = gm.addFunction(potts);

	for (IndexType i = 1; i < gm.numberOfVariables(); ++i) {
		IndexType varData[] = { i-1, i };

		for (size_t i = 0; i < 4; ++i)
			gm.addFactor(pottsFid, varData, varData + ARRAY_LEN(varData));
	}

	//
	// Generate all different kinds of CanonicalViews
	//

	opengm::CanonicalView<GraphicalModelType, opengm::canonical_view::CloneNever> view1(gm);
	opengm::CanonicalView<GraphicalModelType, opengm::canonical_view::CloneViews> view2(gm);
	opengm::CanonicalView<GraphicalModelType, opengm::canonical_view::CloneDeep> view3(gm);

	//
	// Test all possible combinations of labels
	//

	IndexType size = 1;
	for (size_t i = 0; i < ARRAY_LEN(spaceData); ++i)
		size *= spaceData[i];

	opengm::ShapeWalker<LabelType*> walker(spaceData, ARRAY_LEN(spaceData));

	for (IndexType i = 0; i < size; ++i, ++walker) {
		ValueType valRef = gm.evaluate(walker.coordinateTuple().begin());
		ValueType valView1 = view1.evaluate(walker.coordinateTuple().begin());
		ValueType valView2 = view2.evaluate(walker.coordinateTuple().begin());
		ValueType valView3 = view3.evaluate(walker.coordinateTuple().begin());

		checkEqualityAndThrow(valRef, valView1);
		checkEqualityAndThrow(valRef, valView2);
		checkEqualityAndThrow(valRef, valView3);
	}

	return 0;
}
