#pragma once
#ifndef OPENGM_OPERATIONS_MINIMIZER_HXX
#define OPENGM_OPERATIONS_MINIMIZER_HXX

namespace opengm {

/// Minimization as a unary accumulation
///
/// \ingroup operators
struct Minimizer
{
   /// neutral element (with return)
   template<class T>
   static T neutral()
      { return std::numeric_limits<T>::infinity(); }
   /// neutral element (call by reference)
   template<class T>
   static void neutral(T& out)
      { out = std::numeric_limits<T>::infinity(); }

   /// inverse neutral element (with return)
   template<class T>
   static T ineutral()
      { return -std::numeric_limits<T>::infinity(); }
   /// inverse neutral element (call by reference)
   template<class T>
   static void ineutral(T& out)
      { out = -std::numeric_limits<T>::infinity(); }
      
   /// operation (in-place)
   template<class T1, class T2>
   static void op(const T1& in1, T2& out)
      { out = out < in1 ? out : in1; }
   /// operation (not in-place)
   template<class T1,class T2,class T3>
   static void op(const T1& in1, const T2& in2, T3& out)
      { out = in1 < in2 ? in1 : in2; }

   /// inverse operation (in-place)
   template<class T1, class T2>
   static void iop(const T1& in1, T2& out)
      { out -= out > in1 ? out : in1; }
   /// inverse operation (not in-place)
   template<class T1,class T2,class T3>
   static void iop(const T1& in1, const T2& in2, T3& out)
      { out = in1 > in2 ? in1:in2; }

   /// bool operation flag
   static bool hasbop()
      {return true;}

   /// boolean operation
   template<class T>
   static bool bop(const T& in1, const T& in2)
      { return (in1 < in2); }

   /// inverse boolean operation
   template<class T>
   static bool ibop(const T& in1, const T& in2)
      { return (in1 > in2); }
};

} // namespace opengm

#endif // #ifndef OPENGM_OPERATIONS_MINIMIZER_HXX
