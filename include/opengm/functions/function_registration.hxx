#pragma once
#ifndef OPENGM_FUNCTION_REGISTRATION_HXX
#define OPENGM_FUNCTION_REGISTRATION_HXX

namespace opengm {
   
/// \var FUNCTION_TYPE_ID_OFFSET
///
/// User-defined function have ids smaller than FUNCTION_TYPE_ID_OFFSET
///
const size_t FUNCTION_TYPE_ID_OFFSET = 16000;

/// FunctionRegistration
///
/// assigns a unique index to each I/O-comatible function
///
template<class FUNCTION_TYPE> struct FunctionRegistration;

/// FunctionSerialization
///
/// used to serialize a function into two vectors,
/// a vector of values and a vector of indices
///
template<class T> class FunctionSerialization;

} // namespace opengm

#endif // OPENGM_FUNCTION_REGISTRATION_HXX
