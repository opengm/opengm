#pragma once
#ifndef OPENGM_OPERATIONS_INTEGRATOR_HXX
#define OPENGM_OPERATIONS_INTEGRATOR_HXX

#include "adder.hxx"

namespace opengm {

/// Integration (addition) as a unary accumulation
///
/// \ingroup operators
struct Integrator
{
   /// neutral element (with return)
   template<class T>
      static T neutral()
         { return static_cast<T>(0); }

   /// neutral element (call by reference)
   template<class T>
      static void neutral(T& out)
         { out = static_cast<T>(0); }

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
         { out += in1; }

   /// operation (not in-place)
   template<class T1,class T2,class T3>
      static void op(const T1 in1, const T2 in2, T3& out)
         { out = in1, out += in2; }

   /// inverse operation (in-place)
   template<class T1, class T2>
      static void iop(const T1& in1,  T2& out)
         { out - in1; }

   /// inverse operation (call by reference)
   template<class T1,class T2,class T3>
      static void iop(const T1 in1, const T2 in2, T3& out)
         { out = in1, out -= in2; }

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
