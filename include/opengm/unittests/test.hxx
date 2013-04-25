#pragma once
#ifndef OPENGM_TEST_HXX
#define OPENGM_TEST_HXX

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "opengm/opengm.hxx"
#include "opengm/utilities/indexing.hxx"

/// \cond HIDDEN_SYMBOLS
#define OPENGM_TEST(x) \
    if(!(x)) { \
    std::stringstream s; \
    s << #x << " does not hold [line " << __LINE__ << "]"; \
    throw std::logic_error(s.str().c_str()); \
    exit(1); \
}

#define OPENGM_TEST_EQUAL(x, y) \
    if(!(x == y)) { \
    std::stringstream s; \
    s << x << " != " << y << " [line " << __LINE__ << ": " << #x << " == " << #y << " ]"; \
    throw std::logic_error(s.str().c_str()); \
    exit(1); \
}

#define OPENGM_TEST_EQUAL_TOLERANCE(x, y, epsilon) \
    if( (x<y && y-x > epsilon) || (x>y && x-y > epsilon) ) { \
    std::stringstream s; \
    s << x << " != " << y << " [line " << __LINE__ << ": " << #x << " == " << #y << " ]"; \
    throw std::logic_error(s.str().c_str()); \
    exit(1); \
}
#define OPENGM_UNUSED(x) (void)x

#define OPENGM_TEST_EQUAL_SEQUENCE(b1,e1,b2) testEqualSequence(b1,e1,b2)

template<class F1,class F2>
void testEqualFactor(const F1 & f1,const F2 & f2)
{
	OPENGM_TEST_EQUAL(f1.numberOfVariables(),f2.numberOfVariables());
	std::vector<size_t> factorShape(f1.numberOfVariables());
	for(size_t i=0;i<f1.numberOfVariables();++i) {
		OPENGM_TEST_EQUAL(f1.variableIndex(i),f2.variableIndex(i));
		OPENGM_TEST_EQUAL(f1.numberOfLabels(i),f2.numberOfLabels(i));
		factorShape[i]=f1.numberOfLabels(i);
	}
	opengm::ShapeWalker< std::vector<size_t>::const_iterator > walker(factorShape.begin(), factorShape.size() );
	if(f1.numberOfVariables()!=0) {
		OPENGM_TEST_EQUAL(f1.size(),f2.size());
		for(size_t i=0;i<f1.size();++i,++walker) {
			OPENGM_TEST_EQUAL_TOLERANCE(f1(walker.coordinateTuple().begin()),f2(walker.coordinateTuple().begin()),0.00001);
		}
	}
	else{
		size_t coordinate[]={0};
		OPENGM_TEST_EQUAL_TOLERANCE(f1(coordinate),f2(coordinate),0.00001);
	}
}

template<class Gm1, class Gm2>
struct GraphicalModelEqualityTest
{
	void operator()(const Gm1 & gm1,const Gm2 & gm2)const
	{
		OPENGM_TEST(gm1.numberOfVariables()==gm2.numberOfVariables());
		OPENGM_TEST(gm1.numberOfFactors()==gm2.numberOfFactors());
		for(size_t i=0;i<gm1.numberOfVariables();++i)
		{
			OPENGM_TEST(gm1.numberOfLabels(i)==gm2.numberOfLabels(i));
		}

		for(size_t i=0;i<gm1.numberOfFactors();++i) {
         testEqualFactor(gm1[i],gm2[i]);
		}
	}
};



template<class It1, class It2>
void testEqualSequence(It1 begin1, It1 end1, It2 begin2) {
    while(begin1 != end1) {
        OPENGM_TEST(*begin1 == *begin2);
        ++begin1;
        ++begin2;
    }
}

template<class Gm1, class Gm2>
inline void testEqualGm(const Gm1 & gm1,const Gm2 & gm2)
{
	GraphicalModelEqualityTest<Gm1,Gm2> testEqualGmFunctor;
	testEqualGmFunctor(gm1,gm2);
}

/// \endcond

#endif // #ifndef OPENGM_TEST_HXX
