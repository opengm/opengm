#pragma once
#ifndef OPENGM_OPERATION_NORMALIZE_HXX
#define OPENGM_OPERATION_NORMALIZE_HXX

#include <typeinfo>

#include "opengm/opengm.hxx"
#include "opengm/operations/multiplier.hxx"

namespace opengm
{

/// Normalization w.r.t. a binary operation (e.g. Multiplier) and a unary accumulation (e.g. Integrator)
struct Normalization {
   template<class ACC, class OP, class T>
   static void normalize(T& out) {
      typename T::ValueType v;
      out.template accumulate<ACC>(v);
      if(typeid(OP) == typeid(opengm::Multiplier) && v <= 0.00001) {
         return;
      }
      if(typeid(OP) == typeid(opengm::Multiplier)) {
         OPENGM_ASSERT(v > 0.00001); 
      }
      OP::iop(v,out);
   }
};

} // namespace opengm

#endif // #ifndef OPENGM_OPERATION_NORMAIZE_HXX
