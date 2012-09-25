#pragma once
#ifndef OPENGM_OPERATION_ADDER_HXX
#define OPENGM_OPERATION_ADDER_HXX

namespace opengm {

/// Addition as a binary operation
///
/// \ingroup operators
struct Adder
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
         { return static_cast<T>(0); }
   
   /// inverse neutral element (call by reference)
   template<class T>
      static void ineutral(T& out)
         { out = static_cast<T>(0); }

   /// operation (in-place)
   template<class T1, class T2>
      static void op(const T1& in, T2& out)
         { out += in; }
   
   /// operation (not in-place)
   template<class T1,class T2,class T3>
      static void op(const T1& in1, const T2& in2, T3& out)
         { out = in1 + in2; }

   /// inverse operation (in-place)
   template<class T1, class T2>
      static void iop(const T1& in, T2& out)
         { out -= in; }
   
   /// inverse operation (not in-place)
   template<class T1,class T2,class T3>
      static void iop(const T1& in1, const T2& in2, T3& out)
         { out = in1 - in2; }

   /// bool operation flag
   static bool hasbop()
      { return false; }

   /// hyper-operation (in-place)
   template<class T1, class T2>
      static void hop(const T1& in, T2& out)
         { out *= in; }

   /// hyper-operation (not in-place)
   template<class T1,class T2, class T3>
      static void hop(const T1& in1, const T2& in2, T3& out)
         { out = in1 * in2; }

   /// inverse hyper-operation (in-place)
   template<class T1,class T2>
      static void ihop(const T1& in, T2& out)
         { out /= in; }

   /// inverse hyper-operation (same type, not in-place)
   template<class T1, class T2, class T3>
      static void ihop(const T1& in1, const T2& in2, T3& out)
         { out = in1 / in2; }
};

} // namespace opengm

#endif // #ifndef OPENGM_OPERATION_ADDER_HXX
