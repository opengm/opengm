#pragma once
#ifndef OPENGM_OPERATION_WEIGTED_OPERATIONS_HXX
#define OPENGM_OPERATION_WEIGTED_OPERATIONS_HXX

/// \cond HIDDEN_SYMBOLS

namespace opengm {

struct WeightedOperations{
   ///weighted mean (in-place)
   template<class OP, class T1, class T2>
   static inline void weightedMean(const T2& in1, const T2& in2, const T1& w, T2& out)
   {
      OPENGM_ASSERT(&out != &in2);
      out = in1;
      OP::iop(in2,out);
      OP::hop(w,out);
      OP::op(in2,out);
      //             equivalent to
      //             out = in1*w + in2*(1-w) = (in1-in2)*w+in2
      //             out = in1^w * in2^(1-w) = (in1/in2)^w*in2
   }

   /// weighted operation (not in-place)
   template<class OP, class T1, class T2>
   static inline void wop(const T2& in, const T1& w, T2& out)
   {
      T2 t = in;
      OP::hop(w,t);
      OP::op(t,out);
      //             equivalent to
      //             out = out + in*w
      //             out = out * in^w
   }

   /// inverse weighted operation (not in-place)
   template<class OP, class T1, class T2>
   static inline void iwop(const T2& in, const T1& w, T2& out)
   {
      T2 t = in;
      T1 v = 1/w;
      OP::hop(v,t);
      OP::op(t,out);
      //             equivalent to
      //             out = out + in/w
      //             out = out * in^(1/w)
   }

   /// weighted inverse operation (not in-place)
   template<class OP, class T1, class T2>
   static inline void wiop(const T2& in, const T1& w, T2& out)
   {
      T2 t = in;
      OP::hop(w,t);
      OP::iop(t,out);
      //             equivalent to
      //             out = out - in*w
      //             out = out / in^(w)
   }

   /// inverse weighted inverse operation (not in-place)
   template<class OP, class T1, class T2>
   static inline void iwiop(const T2& in, const T1& w, T2& out)
   {
      T2 t = in;
      T1 v = 1/w;
      OP::hop(v,t);
      OP::iop(t,out);
      //             equivalent to
      //             out = out - in/w
      //             out = out / in^(1/w)
   }
};

} // namespace opengm

/// \endcond

#endif // #ifndef OPENGM_OPERATION_WEIGTED_OPERATIONS_HXX
