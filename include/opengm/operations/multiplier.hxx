#pragma once
#ifndef OPENGM_OPERATION_MULTIPLIER_HXX
#define OPENGM_OPERATION_MULTIPLIER_HXX

#include "opengm/graphicalmodel/graphicalmodel_factor_operator.hxx"

namespace opengm {

/// Multiplication as a binary operation
///
/// \ingroup operators
struct Multiplier
{
   /// neutral element (with return)
   template<class T>
   static T neutral()
      { return static_cast<T>(1); }

   /// neutral element (call by reference)
   template<class T>
   static void neutral(T& out)
      { out = static_cast<T>(1); }

   /// inverse neutral element (with return)
   template<class T>
   static T ineutral()
      { return static_cast<T>(1); }

   /// inverse neutral element (call by reference)
   template<class T>
   static void ineutral(T& out)
      { out = static_cast<T>(1); }

   /// operation (in-place)
   template<class T1, class T2>
   static void op(const T1& in, T2& out)
      { out *= in; }

   /// operation (not in-place)
   template<class T1,class T2,class T3>
   static void op(const T1& in1, const T2& in2, T3& out)
      { out = in1 * in2; }

   /// inverse operation (in-place)
   template<class T1, class T2>
   static void iop(const T1& in, T2& out)
      { out /= in; }

   /// inverse operation (not in-place)
   template<class T1, class T2, class T3>
   static void iop(const T1& in1, const T2& in2, T3& out)
      { out = in1 / in2; }

   /// bool operation flag
   static bool hasbop()
      {return false;}

   /// hyper-operation (not in-place)
   template<class T1, class T2>
   static void hop(const T1& in1, T2& out)
      { 
         opengm::operateUnary(out,opengm::PowFunctor<T1>(in1) );
         //T2 temp = out;
         //opengm::operateUnary(temp,out,opengm::PowFunctor<T1>(in1));
         //out.operateUnary(out,opengm::PowFunctor<T2>(in1)) ;
         //out= pow(out, in1);
      }

   /// hyper-operation (not in-place)
   template<class T1, class T2, class T3>
   static void hop(const T1& in1, const T2& in2, T3& out)
      {
         opengm::operateUnary(in1,out,opengm::PowFunctor<T2>(in2) );
         //out.operateUnary(in1,opengm::PowFunctor<T2>(in2)) ;
         //out= pow(in1, in2);
      }

   /// inverse hyper-operation (in-place)
   template<class T1, class T2>
   static void ihop(const T1& in1, T2& out)
      {
         opengm::operateUnary(out,opengm::PowFunctor<T1>(1.0/in1) );
         //out= pow(out, in1);
      }

   /// inverse hyper-operation (not in-place)
   template<class T1, class T2, class T3>
   static void ihop(const T1& in1, const T2& in2, T3& out)
      {
         //out = pow(in1, 1/in2);
         opengm::operateUnary(in1,out,opengm::PowFunctor<T2>(1.0/in2) );
      }
};

} // namespace opengm

#endif // #ifndef OPENGM_OPERATION_MULTIPLIER_HXX
