#pragma once
#ifndef OPENGM_HXX
#define OPENGM_HXX

#include <stdexcept>
#include <sstream>

#include "opengm/config.hxx"
#include "opengm/utilities/metaprogramming.hxx"




// as runtime assertion but cefined even if NDEBUG

#define OPENGM_CHECK_OP(a,op,b,message) \
    if(!  static_cast<bool>( a op b )   ) { \
       std::stringstream s; \
       s << "OpenGM Error: "<< message <<"\n";\
       s << "OpenGM check :  " << #a <<#op <<#b<< "  failed:\n"; \
       s << #a " = "<<a<<"\n"; \
       s << #b " = "<<b<<"\n"; \
       s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
       throw std::runtime_error(s.str()); \
    }

#define OPENGM_CHECK(expression,message) if(!(expression)) { \
   std::stringstream s; \
   s << message <<"\n";\
   s << "OpenGM assertion " << #expression \
   << " failed in file " << __FILE__ \
   << ", line " << __LINE__ << std::endl; \
   throw std::runtime_error(s.str()); \
 }


/// runtime assertion
#ifdef NDEBUG
   #ifndef OPENGM_DEBUG 
      #define OPENGM_ASSERT_OP(a,op,b) { }
   #else
      #define OPENGM_ASSERT_OP(a,op,b) \
      if(!  static_cast<bool>( a op b )   ) { \
         std::stringstream s; \
         s << "OpenGM assertion :  " << #a <<#op <<#b<< "  failed:\n"; \
         s << #a " = "<<a<<"\n"; \
         s << #b " = "<<b<<"\n"; \
         s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
         throw std::runtime_error(s.str()); \
      }
   #endif
#else
   #define OPENGM_ASSERT_OP(a,op,b) \
   if(!  static_cast<bool>( a op b )   ) { \
      std::stringstream s; \
      s << "OpenGM assertion :  " << #a <<#op <<#b<< "  failed:\n"; \
      s << #a " = "<<a<<"\n"; \
      s << #b " = "<<b<<"\n"; \
      s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
      throw std::runtime_error(s.str()); \
   }
#endif

#ifdef NDEBUG
   #ifndef OPENGM_DEBUG
      #define OPENGM_ASSERT(expression) {}
   #else
      #define OPENGM_ASSERT(expression) if(!(expression)) { \
         std::stringstream s; \
         s << "OpenGM assertion " << #expression \
         << " failed in file " << __FILE__ \
         << ", line " << __LINE__ << std::endl; \
         throw std::runtime_error(s.str()); \
      }
   #endif
#else
      #define OPENGM_ASSERT(expression) if(!(expression)) { \
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
