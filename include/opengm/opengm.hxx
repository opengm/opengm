#pragma once
#ifndef OPENGM_HXX
#define OPENGM_HXX

#include <stdexcept>
#include <sstream>

#include "opengm/config.hxx"
#include "opengm/utilities/metaprogramming.hxx"




/// runtime assertion
#ifdef NDEBUG
#  define OPENGM_ASSERT(expression) {if(true || ( expression )){}}
#else
#  define OPENGM_ASSERT(expression) if(!(expression)) { \
   std::stringstream s; \
   s << "OpenGM assertion " << #expression \
   << " failed in file " << __FILE__ \
   << ", line " << __LINE__ << std::endl; \
   throw std::runtime_error(s.str()); \
}
#endif

/// opengm compile time assertion
#define OPENGM_META_ASSERT(assertion, msg) { \
   meta::Assert<   meta::Compare< meta::Bool<(assertion)> , meta::Bool<true> >::value    >  \
   OPENGM_COMPILE_TIME_ASSERTION_FAILED_____REASON_____##msg; \
   (void) OPENGM_COMPILE_TIME_ASSERTION_FAILED_____REASON_____##msg; \
}

/// The OpenGM namespace
namespace opengm {

typedef double DefaultTimingType;
typedef opengm::UIntType SerializationIndexType;

/// OpenGM runtime error
struct RuntimeError
: public std::runtime_error
{
   typedef std::runtime_error base;

   RuntimeError(const std::string& message)
   :  base(std::string("OpenGM error: ") + message) {}
};

// abs function
template<class T>
inline T abs(const T& x) { 
   return x > 0 ? x : -x; 
}

template<class T>
inline T opengmMax(const T& x, const T& y) {
   return x >= y ? x : y;
}

template<class T>
inline T opengmMin(const T& x, const T& y) {
   return x <= y ? x : y;
}

} // namespace opengm

#endif // #ifndef OPENGM_HXX
