#pragma once
#ifndef OPENGM_OPERATION_OR_HXX
#define OPENGM_OPERATION_OR_HXX

namespace opengm {

/// Disjunction as a binary operation
///
/// \ingroup operators
struct Or
{
   /// neutral element (with return)
   template<class T>
      static T neutral()
         { return static_cast<T>(false); }

   /// neutral element (call by reference)
   template<class T>
      static void neutral(T& out)
         { out = static_cast<T>(false); }

   /// inverse neutral element (with return)
   template<class T>
      static T ineutral()
         { return static_cast<T>(true); }

   /// inverse neutral element (call by reference)
   template<class T>
      static void ineutral(T& out)
         { out = static_cast<T>(true); }

   /// operation (in-place)
   template<class T1, class T2>
   static void op(const T1& in, T2& out)
      { out |= in; } 
      
   /// operation (not in-place)
   template<class T1,class T2,class T3>
   static void op(const T1& in1, const T2& in2, T3& out)
      { out = in1 | in2; }
   static void op(const bool& in1, const bool& in2, bool& out)
      { out = in1 || in2; }

   /// bool operation flag
   static bool hasbop()
      { return true; } 

   /// boolean operation
   template<class T>
   static bool bop(const T& in1, const T& in2)
      { return (in1 | in2); } 
   static bool bop(const bool& in1, const bool& in2)
      { return (in1 || in2); }

   /// inverse boolean operation
   template<class T>
   static bool ibop(const T& in1, const T& in2)
      { return !(in1 | in2); }
   static bool ibop(const bool& in1, const bool& in2)
      { return !(in1 || in2); }
};

} // namespace opengm

#endif // #ifndef OPENGM_OPERATION_OR_HXX
