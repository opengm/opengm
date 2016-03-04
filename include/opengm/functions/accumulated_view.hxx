//
// File: accumulated_view.hxx
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
#ifndef OPENGM_FUNCTIONS_ACCUMULATED_VIEW_HXX
#define OPENGM_FUNCTIONS_ACCUMULATED_VIEW_HXX

#include <opengm/datastructures/fast_sequence.hxx>
#include <opengm/functions/function_properties_base.hxx>

namespace opengm {

template<class GM>
class AccumulatedViewFunction : public FunctionBase< AccumulatedViewFunction<GM>,
                                                     typename GM::ValueType,
                                                     typename GM::IndexType,
                                                     typename GM::LabelType > {
public:
	typedef typename GM::IndexType IndexType;
	typedef typename GM::IndexType LabelType;
	typedef typename GM::ValueType ValueType;
	typedef typename GM::FactorType FactorType;
	typedef typename GM::OperatorType OperatorType;

	AccumulatedViewFunction();
	AccumulatedViewFunction(const FactorType &);
	template<class ITERATOR> AccumulatedViewFunction(ITERATOR begin, ITERATOR end);

	template<class ITERATOR> ValueType operator()(ITERATOR begin) const;
	LabelType shape(const IndexType) const;
	IndexType dimension() const;
	IndexType size() const;

private:
	void check() const;

	opengm::FastSequence<const FactorType*> factors_;
};

template<class GM>
AccumulatedViewFunction<GM>::AccumulatedViewFunction()
{
}

template<class GM>
AccumulatedViewFunction<GM>::AccumulatedViewFunction
(
	const FactorType &factor
)
{
	factors_.push_back(&factor);
}

template<class GM>
template<class ITERATOR>
AccumulatedViewFunction<GM>::AccumulatedViewFunction
(
	ITERATOR begin,
	ITERATOR end
)
{
	factors_.resize(end - begin);
	std::copy(begin, end, factors_.begin());

	check();
}

template<class GM>
template<class Iterator>
typename AccumulatedViewFunction<GM>::ValueType
AccumulatedViewFunction<GM>::operator()
(
	Iterator begin
) const
{
	check();

	ValueType result = GM::OperatorType::template neutral<ValueType>();
	for (size_t i = 0; i < factors_.size(); ++i)
		GM::OperatorType::op(factors_[i]->operator()(begin), result);

	return result;
}

template<class GM>
typename AccumulatedViewFunction<GM>::LabelType
AccumulatedViewFunction<GM>::shape
(
	const IndexType index
) const
{
	check();
	return factors_[0]->numberOfLabels(index);
}

template<class GM>
typename AccumulatedViewFunction<GM>::IndexType
AccumulatedViewFunction<GM>::dimension() const
{
	check();
	return factors_[0]->numberOfVariables();
}

template<class GM>
typename AccumulatedViewFunction<GM>::IndexType
AccumulatedViewFunction<GM>::size() const
{
	check();
	return factors_[0]->size();
}

template<class GM>
void
AccumulatedViewFunction<GM>::check() const
{
#ifndef NDEBUG
	OPENGM_ASSERT_OP(factors_.size(), >, 0);

	for (size_t i = 0; i < factors_.size(); ++i)
		OPENGM_ASSERT(factors_[i] != NULL);

	for (size_t i = 1; i < factors_.size(); ++i) {
		OPENGM_ASSERT_OP(factors_[0]->size(), ==, factors_[i]->size());
		OPENGM_ASSERT_OP(factors_[0]->numberOfVariables(), ==, factors_[i]->numberOfVariables());

		for (IndexType j = 0; j < factors_[0]->numberOfVariables(); ++j) {
			OPENGM_ASSERT_OP(factors_[0]->numberOfLabels(j), ==, factors_[i]->numberOfLabels(j));
			OPENGM_ASSERT_OP(factors_[0]->variableIndex(j), ==, factors_[i]->variableIndex(j));
		}
	}
#endif
}

} // namespace opengm

#endif
