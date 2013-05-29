#ifndef ARGUMENT_HXX_
#define ARGUMENT_HXX_
/*
#include <string>
#include <vector>
#include <cstdlib>
#include <ostream>
#include <sstream>

#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/opengm.hxx>
*/

#include "int_argument.hxx"
#include "size_t_argument.hxx"
#include "float_argument.hxx"
#include "double_argument.hxx"
#include "string_argument.hxx"
#include "vector_argument.hxx"
#include "marray_argument.hxx"
#include "bool_argument.hxx"

#ifdef WITH_MATLAB
   #include "mxArray_argument.hxx"
#endif

#endif /* ARGUMENT_HXX_ */
