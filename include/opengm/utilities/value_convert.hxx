#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/multiplier.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/inference/inference.hxx"

namespace opengm{
   template<class OP,class ACC>
   struct SemiRing;
   typedef SemiRing<Adder,Minimizer> MinSum;
   typedef SemiRing<Adder,Maximizer> MaxSum;
   typedef SemiRing<Multiplier,Minimizer> MinProd;
   typedef SemiRing<Multiplier,Maximizer> MaxProd;

   template<class IN,class OUT,class VOUT>
   struct ValueConverter;
   
   template <class INOUT,class VOUT>
   struct ValueConverter<INOUT,INOUT,VOUT> {
      template< class T>
      static VOUT convert(const T in){
         return in;
      }
   }
   template <class VOUT>
   struct ValueConverter<MaxProd,MaxSum,VOUT> {
      template< class T>
      static VOUT convert(const T in){
         return std::log(static_cast<VOUT>(in));
      }
   }
}
