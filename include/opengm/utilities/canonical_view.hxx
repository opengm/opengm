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
#include <opengm/functions/explicit_function.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/utilities/metaprogramming.hxx>


namespace opengm {

namespace canonical_view {

	/// \brief Controls cloning behavior of a CanonicalView.
	///
	/// To reduce the overhead of view functions, CanonicalView is extended to
	/// allow cloning the calculated functions into ExplicitFunctions.
	///
	/// A template argument handles the different cases:
	///
	/// (1) CloneNever: Original behavior, no functions will be cloned. Original
	///     functions get injected directly whenever possible.
	///
	/// (2) CloneViews: Only the view functions will be cloned using
	///     ExplicitFunctions. Original functions get injected whenever
	///     possible.
	///
	/// (3) CloneDeep: All functions get cloned.
	enum CloneOption { CloneNever, CloneViews, CloneDeep };

} // namespace canonical_view


/// \cond HIDDEN_SYMBOLS
namespace canonical_view_internal {

	// The type of the GraphicalModel is complicated, but can’t be typedef’d
	// easilty. That’s why we use this handy type generator.
	template<class GM>
	struct Generator {
	private:
		typedef typename GM::ValueType ValType;
		typedef typename GM::IndexType IndType;
		typedef typename GM::LabelType LabType;
		typedef typename GM::FunctionTypeList OldTypeList;

		// We extend the type list of the old model and add some more
		// functions.
		typedef ConstantFunction<ValType, IndType, LabType> ConstFunType;
		typedef AccumulatedViewFunction<GM> AccViewType;
		typedef typename meta::TypeListGenerator<ConstFunType, AccViewType>::type NewTypeList;

	public:
		typedef GraphicalModel<
			typename GM::ValueType,
			typename GM::OperatorType,
			typename meta::MergeTypeListsWithoutDups<OldTypeList, NewTypeList>::type,
			typename GM::SpaceType
		> type;
	};

	// Helper for perform the actual cloning (if requested by template
	// parameter). The first GM parameter is expected be the CanonicalView.
	template<class GM, canonical_view::CloneOption CLONE>
	struct CloneHelper;

	template<class GM>
	struct CloneHelper<GM, canonical_view::CloneNever> {
		template<class FUNC>
		static const FUNC& handleInjected(const FUNC &func) { return func; }

		template<class FUNC>
		static const FUNC& handleView(const FUNC &func) { return func; }
	};

	template<class GM>
	struct CloneHelper<GM, canonical_view::CloneViews> {
		template<class FUNC>
		static const FUNC& handleInjected(const FUNC &func) { return func; }

		template<class FUNC>
		static typename GM::ExplFuncType handleView(const FUNC &func)
		{
			typename GM::ExplFuncType result;
			cloneAsExplicitFunction(func, result);
			return result;
		}
	};

	template<class GM>
	struct CloneHelper<GM, canonical_view::CloneDeep> {
		template<class FUNC>
		static typename GM::ExplFuncType handleInjected(const FUNC &func)
		{
			typename GM::ExplFuncType result;
			cloneAsExplicitFunction(func, result);
			return result;
		 }

		template<class FUNC>
		static typename GM::ExplFuncType handleView(const FUNC &func)
		{
			typename GM::ExplFuncType result;
			cloneAsExplicitFunction(func, result);
			return result;
		}
	};


	// This functor is run on the factor of the wrapped GraphicalModel. It
	// will reinject the original function into the wrapper for the
	// GraphicalModel.
	template<class GM, canonical_view::CloneOption CLONE>
	class FunctionInjectionFunctor {
	public:
		FunctionInjectionFunctor(GM &gm)
		: gm_(&gm)
		{
		}

		template<class FUNCTION>
		void operator()(FUNCTION &func)
		{
			result_ = gm_->addFunction(CloneHelper<GM, CLONE>::handleInjected(func));
		}

		typename GM::FunctionIdentifier result() const
		{
			return result_;
		}

	private:
		GM *gm_;
		typename GM::FunctionIdentifier result_;
	};

	// This class injects a function of the wrapped GraphicalModel. Additional
	// it uses a map to remember already injected functions. One original
	// function is thus only inserted into the wrapper once.
	template<class WRAPPER, class WRAPPED, canonical_view::CloneOption CLONE>
	class FunctionInjector {
	public:
		FunctionInjector
		(
			WRAPPER &gm
		)
		: gm_(&gm)
		{
		}

		typename WRAPPER::FunctionIdentifier
		inject
		(
			const typename WRAPPED::FactorType &factor
		)
		{
			typename WRAPPED::FunctionIdentifier id(factor.functionIndex(), factor.functionType());

			typename MapType::const_iterator it = map_.find(id);
			if (it != map_.end())
				return it->second;

			FunctionInjectionFunctor<WRAPPER, CLONE> injector(*gm_);
			factor.callFunctor(injector);
			typename WRAPPER::FunctionIdentifier result = injector.result();
			map_[id] = result;
			return result;
		}

	private:
		typedef std::map<
			typename WRAPPED::FunctionIdentifier,
			typename WRAPPER::FunctionIdentifier
		> MapType;

		WRAPPER *gm_;
		MapType map_;
	};

} // namespace canonical_view_internal
/// \endcond HIDDEN_SYMBOLS


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
template<class GM, canonical_view::CloneOption CLONE = canonical_view::CloneNever>
class CanonicalView : public canonical_view_internal::Generator<GM>::type {
public:
	typedef typename canonical_view_internal::Generator<GM>::type Parent;
	typedef CanonicalView<GM, CLONE> MyType;

	typedef typename Parent::IndexType IndexType;
	typedef typename Parent::LabelType LabelType;
	typedef typename Parent::ValueType ValueType;
	typedef typename Parent::FunctionIdentifier FunctionIdentifier;

	typedef ExplicitFunction<ValueType, IndexType, LabelType> ExplFuncType;
	typedef ConstantFunction<ValueType, IndexType, LabelType> ConstFuncType;
	typedef AccumulatedViewFunction<GM> ViewFuncType;

	typedef canonical_view_internal::FunctionInjector<MyType, GM, CLONE> FunctionInjectorType;
	typedef canonical_view_internal::CloneHelper<MyType, CLONE> CloneHelperType;

	CanonicalView(const GM &gm)
	: Parent(gm.space())
	{
		// FIXME: Use opengm::FastSequence, but operator< is missing. :-(
		typedef std::vector<IndexType> Variables;
		typedef std::vector<const typename GM::FactorType*> Factors;
		typedef std::vector<Factors> UnaryFactors;
		typedef std::map<Variables, Factors> FactorMap;
		typedef std::map<FunctionIdentifier, typename GM::FunctionIdentifier> FunctionMap;

		FunctionInjectorType injector(*this);
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
			FunctionIdentifier fid;
			IndexType vars[1] = { i };

			// Note: Use of curly braces to create new scope.
			switch (unaryFactors[i].size()) {
			case 0: { // Create new “empty” factor.
				LabelType shape[1] = { gm.numberOfLabels(i) };
				ConstFuncType func(shape, shape+1, 0);
				fid = this->addFunction(func);
				break;
			}
			case 1: { // Reuse old function.
				fid = injector.inject(*unaryFactors[i][0]);
				break;
			}
			default: { // Create a new view function.
				ViewFuncType func(unaryFactors[i].begin(), unaryFactors[i].end());
				fid = this->addFunction(CloneHelperType::handleView(func));
				break;
			}
			}

			this->addFactor(fid, vars, vars+1);
		}

		// Accumulate all other factors (with order != 1).
		for (typename FactorMap::const_iterator it = otherFactors.begin(); it != otherFactors.end(); ++it) {
			const Variables &vars= it ->first;
			const Factors &factors = it->second;
			FunctionIdentifier fid;

			OPENGM_ASSERT_OP(factors.size(), >, 0);
			if (factors.size() == 1) {
				// Reuse old function.
				fid = injector.inject(*factors[0]);
			} else {
				// Create new view function.
				ViewFuncType func(it->second.begin(), it->second.end());
				fid = this->addFunction(CloneHelperType::handleView(func));
			}

			this->addFactor(fid, vars.begin(), vars.end());
		}
	}
};

} // namespace opengm

#endif
