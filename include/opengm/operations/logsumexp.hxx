#pragma once
#ifndef OPENGM_OPERATIONS_LOGSUMEXP_HXX
#define OPENGM_OPERATIONS_LOGSUMEXP_HXX

#include "adder.hxx"
#include <cmath>

namespace opengm {

/// Logsumexp (addition in log space) as a unary accumulation
///
/// \ingroup operators
struct Logsumexp
{
   /// neutral element (with return)
   template<class T>
      static T neutral()
         { return -std::numeric_limits<T>::infinity(); }

   /// neutral element (call by reference)
   template<class T>
      static void neutral(T& out)
         { out = -std::numeric_limits<T>::infinity(); }

   /// inverse neutral element (with return)
   template<class T>
      static T ineutral()
         { return std::numeric_limits<T>::infinity(); }

   /// inverse neutral element (call by reference)
   template<class T>
      static void ineutral(T& out)
         { out = std::numeric_limits<T>::infinity(); }

   /// operation (in-place)
   template<class T1, class T2>
      static void op(const T1& in1, T2& out)
//         { out = log(exp(out) + exp(in1)); }
   {
      T2 theMax = std::max(in1, out);
      T2 theMin = std::min(in1, out);
      out = theMax + log(1 + exp(theMin - theMax));  // numerically better
   }

   /// operation (not in-place)
   template<class T1,class T2,class T3>
      static void op(const T1 in1, const T2 in2, T3& out)
//         { out = log(exp(in1) + exp(in2)); }
   {
      T2 theMax = std::max(in1, out);
      T2 theMin = std::min(in1, out);
      out = theMax + log(1 + exp(theMin - theMax));  // numerically better
   }

   /// inverse operation (in-place)
   template<class T1, class T2>
      static void iop(const T1& in1,  T2& out)
//         { out - in1; }  // reference implementation has no effect?
   { throw "not implemented"; }

   /// inverse operation (call by reference)
   template<class T1,class T2,class T3>
      static void iop(const T1 in1, const T2 in2, T3& out)
         { out = log(exp(in1) - exp(in2)); }

   /// bool operation flag
   static bool hasbop()
      { return false; }
   /// \obsolete inverse boolean operation 
   /// boolean operation (obsolete)
   template<class T>
      static bool bop(const T& in1, const T& in2)
         { return false; }
   /// \obsolete inverse boolean operation 
   /// inverse boolean operation (obsolete)
   template<class T>
      static bool ibop(const T& in1, const T& in2)
         { return false; }
};

} // namespace opengm

#endif // #ifndef OPENGM_OPERATIONS_INTEGRATOR_HXX
