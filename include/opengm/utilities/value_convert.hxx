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
   struct ValueConverter{
	  template< class T>
      static VOUT convert(const T in){
		 throw RuntimeError("The requested semi-ring conversion not supported");
         return 0;
      }
   };
   // no conversion
   template <class INOUT,class VOUT>
   struct ValueConverter<INOUT,INOUT,VOUT> {
      template< class T>
      static VOUT convert(const T in){
         return in;
      }
   };
   // maxprod -> minsum
   template <class VOUT>
   struct ValueConverter<MaxProd,MinSum,VOUT> {
      template< class T>
      static VOUT convert(const T in){
		 if(in <= 0.0 ) {
			std::cout<<"Error,value is "<<in<<"\n";
			throw RuntimeError("ere");
		 }
         return static_cast<VOUT>(-1.0)*std::log(static_cast<VOUT>(in));
      }
   };
   // maxprod -> maxsum
   template <class VOUT>
   struct ValueConverter<MaxProd,MaxSum,VOUT> {
      template< class T>
      static VOUT convert(const T in){
         return std::log(static_cast<VOUT>(in));
      }
   };
}
