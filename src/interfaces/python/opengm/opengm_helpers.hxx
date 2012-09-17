/* 
 * File:   opengm_helpers.hxx
 * Author: tbeier
 *
 * Created on August 19, 2012, 6:43 PM
 */

#ifndef OPENGM_HELPERS_HXX
#define	OPENGM_HELPERS_HXX

#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/icm.hxx>

template<typename T> const T copyObject(const T& v) { return v; }

template<class T,class V,class C>
struct std_item
{
    static V& get(const T & x, int i)
    {
        if( i<0 ) i+=x.size();
        if( i>=0 && i<x.size() ) return x( static_cast<size_t>(i));
        //IndexError();
    }
    static void set( T & x, int i, V const v)
    {
        if( i<0 ) i+=x.size();
        if( i>=0 && i<x.size() )  x( static_cast<size_t>(i))=v;
        //else IndexError();
    }
    static V& getC(const T & x, const  C & i)
    {
        return x( i.begin());
        //IndexError();
    }
    static void setC(T & x, const  C & i, V const v)
    {
        x(i.begin())=v;
        //else IndexError();
    }
};










#endif	/* OPENGM_HELPERS_HXX */

