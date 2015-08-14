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

#pragma once
#ifndef OPENGM_UTILITIES_CANONICAL_FACTORS_HXX
#define OPENGM_UTILITIES_CANONICAL_FACTORS_HXX

#include <map>
#include <vector>

#include <opengm/functions/accumulated_view.hxx>
#include <opengm/functions/constant.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/utilities/metaprogramming.hxx>


namespace opengm {

namespace canonical_view_internal {
	// The type of the GraphicalModel is complicated, but can’t be typedef’d
	// easilty. That’s why we use this handy type generator.
	template<class GM>
	struct Generator {
		typedef GraphicalModel<
			typename GM::ValueType,
			typename GM::OperatorType,
			typename meta::TypeListGenerator<
				ConstantFunction<
					typename GM::ValueType,
					typename GM::IndexType,
					typename GM::LabelType>,
				AccumulatedViewFunction<GM> >::type,
			typename GM::SpaceType
		> Type;
	};
}


/// \brief Canonical view of an arbitrary GraphicalModel
///
/// This class wraps an arbitrary GraphicalModel and acts as a view on the
/// model. The original model is changed with respect to the following aspects:
///
///   - all variables are associated with exactly one unary factor (multiple
///     unary factors get squashed into one, a non-existent unary factor is
///     mapped to a zero-constant factor)
///
///   - there is at most one factor for a given set of variables (multiple
///     factors attached to the clique are squashed into one factor)
template<class GM>
class CanonicalView : public canonical_view_internal::Generator<GM>::Type {
public:
	typedef typename canonical_view_internal::Generator<GM>::Type Parent;

	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;

	typedef ConstantFunction<ValueType, IndexType, LabelType> ConstFuncType;
	typedef AccumulatedViewFunction<GM> ViewFuncType;

	CanonicalView(const GM &gm)
	: Parent(gm.space())
	{
		// FIXME: Use opengm::FastSequence, but operator< is missing. :-(
		typedef std::vector<const typename GM::FactorType*> Factors;
		typedef std::vector<Factors> UnaryFactors;
		typedef std::map<std::vector<IndexType>, Factors> FactorMap;

		UnaryFactors unaryFactors(gm.numberOfVariables());
		FactorMap otherFactors;

		// Append all unary factors to the corresponding unary factor vector.
		// All other factors are inserted into the factor map. The keys are the
		// variable indices of the factor, so we group factors of the same
		// variables together.
		for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
			const typename GM::FactorType &f = gm[i];
			if (f.numberOfVariables() == 1) {
				unaryFactors[f.variableIndex(0)].push_back(&f);
			} else {
				std::vector<IndexType> vars(f.variableIndicesBegin(), f.variableIndicesEnd());
				otherFactors[vars].push_back(&f);
			}
		}

		// Associate each variable with *exactly* one unary factor. (Create
		// an empty factor if missing.)
		for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
			typename Parent::FunctionIdentifier fid;
			IndexType vars[1] = { i };

			if (unaryFactors[i].empty()) {
				LabelType shape[1] = { gm.numberOfLabels(i) };
				ConstFuncType func(shape, shape+1, 0);
				fid = this->addFunction(func);
			} else {
				ViewFuncType func(unaryFactors[i].begin(), unaryFactors[i].end());
				fid = this->addFunction(func);
			}

			this->addFactor(fid, vars, vars+1);
		}

		// Accumulate all other factors (with order != 1).
		for (typename FactorMap::const_iterator it = otherFactors.begin(); it != otherFactors.end(); ++it) {
			ViewFuncType func(it->second.begin(), it->second.end());
			this->addFactor(this->addFunction(func), it->first.begin(), it->first.end());
		}
	}
};

} // namespace opengm

#endif
