#pragma once
#ifndef OPENGM_FUNCTION_PROPERTIES_BASE_HXX
#define OPENGM_FUNCTION_PROPERTIES_BASE_HXX

#include <cmath>

#include "opengm/opengm.hxx"
#include "opengm/utilities/shape_accessor.hxx"
#include "opengm/utilities/accessor_iterator.hxx"
#include "opengm/utilities/accumulation.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/functors.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/and.hxx"
#include "opengm/operations/or.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/integrator.hxx"
#include "opengm/operations/multiplier.hxx"

#define OPENGM_FLOAT_TOL 0.000001

namespace opengm {
    
template<class T>
inline bool isNumericEqual(const T a, const T b) {
   if(meta::IsFloatingPoint<T>::value) {
      if(a < b) {
         return b-a<OPENGM_FLOAT_TOL ? true : false;
      }
      else {
         return a - b < OPENGM_FLOAT_TOL ? true : false;
      }
   }
   else {
      return a == b;
   }
}   

/// Fallback implementation of member functions of OpenGM functions
template<class FUNCTION, class VALUE, class INDEX = size_t, class LABEL = size_t>
class FunctionBase {
private:
   typedef VALUE ReturnType;
   typedef const VALUE& ReturnReferenceType;

public:   
   bool isPotts() const;
   bool isGeneralizedPotts() const;
   bool isSubmodular() const;
   bool isSquaredDifference() const;
   bool isTruncatedSquaredDifference() const;
   bool isAbsoluteDifference() const;
   bool isTruncatedAbsoluteDifference() const;
   
   /// find minimum and maximum of the function in a single sweep
   /// \return class holding the minimum and the maximum
   MinMaxFunctor<VALUE> minMax() const;

   ReturnType min() const;
   ReturnType max() const;
   ReturnType sum() const;
   ReturnType product() const;

   /// accumulate all values of the function
   /// \tparam ACC Accumulator (e.g. Minimizer, Maximizer, ...)
   /// \return accumulated value
   template<class ACC>
      ReturnType accumulate() const;

   /// call a functor for each value of the function (in lexicographical order of the variable indices)
   ///
   /// Example:
   /// \code
   /// template<class T>
   /// struct MaxFunctor {
   ///    void operator()(const T v) {
   ///       if(v > max_)
   ///          max_ = v;
   ///    }
   ///    T max_;
   /// };
   /// MaxFunctor<float> maxFunctor;
   /// maxFunctor.max_ = 0.0;
   /// function.forAllValuesInOrder(maxFunctor);
   /// \endcode  
   ///
   /// \sa forAllValuesInAnyOrder, forAtLeastAllUniqueValues
   template<class FUNCTOR>
      void forAllValuesInOrder(FUNCTOR& functor) const;

   template<class FUNCTOR>
      void forAllValuesInSwitchedOrder(FUNCTOR& functor) const;

   /// call a functor for each value of the function (in un-specified order)
   ///
   /// \sa forAllValuesInOrder, forAtLeastAllUniqueValues
   template<class FUNCTOR>
      void forAllValuesInAnyOrder(FUNCTOR& functor) const;

   /// call a functor for at least all unique values of the function
   ///
   /// \sa forAllValuesInOrder, forAllValuesInAnyOrder
   template<class FUNCTOR>
      void forAtLeastAllUniqueValues(FUNCTOR& functor) const ; 
   
   
   template<class COORDINATE_FUNCTOR>
      void forAllValuesInOrderWithCoordinate(COORDINATE_FUNCTOR& functor) const;
   template<class COORDINATE_FUNCTOR>
      void forAllValuesInAnyOrderWithCoordinate(COORDINATE_FUNCTOR& functor) const;
   template<class COORDINATE_FUNCTOR>
      void forAtLeastAllUniqueValuesWithCoordinate(COORDINATE_FUNCTOR& functor) const ; 
   
   bool operator==(const FUNCTION&) const;

private:
   typedef FUNCTION FunctionType;
   typedef FunctionShapeAccessor<FunctionType> FunctionShapeAccessorType;

public:
   typedef AccessorIterator<FunctionShapeAccessorType, true> FunctionShapeIteratorType;

   FunctionShapeIteratorType functionShapeBegin() const;
   FunctionShapeIteratorType functionShapeEnd() const;
};




template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline bool  
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::operator==
(
   const FUNCTION& fb
) const{
   const FunctionType& fa=*static_cast<FunctionType const *>(this);
   const size_t dimA=fa.dimension();
   // compare dimension
   if(dimA==fb.dimension()) {
      // compare shape
      for(size_t i=0;i<dimA;++i) {
         if(fa.shape(i)!=fb.shape(i)) {
            return false;
         }
      }
      // compare all values
      ShapeWalker<FunctionShapeIteratorType> shapeWalker(fa.functionShapeBegin(), dimA);
      for(INDEX i=0;i<fa.size();++i, ++shapeWalker) {
        if(isNumericEqual(fa(shapeWalker.coordinateTuple().begin()), fb(shapeWalker.coordinateTuple().begin()))==false) {
           return false;
        }
      }
   }
   else{
      return false;
   }
   return true;
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
template<class COORDINATE_FUNCTOR>
inline void 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::forAllValuesInOrderWithCoordinate
(
   COORDINATE_FUNCTOR& functor
) const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   ShapeWalker<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(), f.dimension());
   for(INDEX i=0;i<f.size();++i, ++shapeWalker) {
      functor(f(shapeWalker.coordinateTuple().begin()),shapeWalker.coordinateTuple().begin());
   }
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
template<class COORDINATE_FUNCTOR>
inline void 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::forAllValuesInAnyOrderWithCoordinate
(
   COORDINATE_FUNCTOR& functor
) const{
   this->forAllValuesInOrderWithCoordinate(functor);
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
template<class COORDINATE_FUNCTOR>
inline void 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::forAtLeastAllUniqueValuesWithCoordinate
(
   COORDINATE_FUNCTOR& functor
) const {
   this->forAllValuesInAnyOrderWithCoordinate(functor);
}


template<class FUNCTION, class VALUE, class INDEX, class LABEL>
template<class FUNCTOR>
inline void
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::forAllValuesInOrder
( 
   FUNCTOR&  functor
) const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   ShapeWalker<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(), f.dimension());
   for(INDEX i=0;i<f.size();++i, ++shapeWalker) {
      functor(f(shapeWalker.coordinateTuple().begin()));
   }
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
template<class FUNCTOR>
inline void
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::forAllValuesInSwitchedOrder
( 
   FUNCTOR&  functor
) const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   ShapeWalkerSwitchedOrder<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(), f.dimension());
   for(INDEX i=0;i<f.size();++i, ++shapeWalker) {
      functor(f(shapeWalker.coordinateTuple().begin()));
   }
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
template<class FUNCTOR>
inline void
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::forAllValuesInAnyOrder
( 
   FUNCTOR&  functor
) const {
   static_cast<FunctionType const *>(this)->forAllValuesInOrder(functor);
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
template<class FUNCTOR>
inline void
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::forAtLeastAllUniqueValues
( 
   FUNCTOR&  functor
) const {
   static_cast<FunctionType const *>(this)->forAllValuesInAnyOrder(functor);
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline typename FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::FunctionShapeIteratorType
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::functionShapeBegin() const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   return FunctionShapeIteratorType(FunctionShapeAccessorType(f), 0);
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline typename FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::FunctionShapeIteratorType
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::functionShapeEnd() const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   return FunctionShapeIteratorType(FunctionShapeAccessorType(f), f.dimension());
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline bool 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::isSquaredDifference() const{
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   if(f.dimension()==2) {
      OPENGM_ASSERT(f.shape(0)>static_cast<LABEL>(1));
      LABEL c[2]={1, 0};
      //get possible weight
      VALUE weight=f(c);
      for( c[1]=0;c[1]<f.shape(1);++c[1]) {
         for( c[0]=0;c[0]<f.shape(0);++c[0]) {
            VALUE d= static_cast<VALUE> (c[0]<c[1] ? c[1]-c[0]:c[0]-c[1]);
            d*=d;
            if( isNumericEqual(f(c), d*weight  )==false)
               return false;
         } 
      }
      return true;
   }
   return false;
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline bool 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::isTruncatedSquaredDifference() const{
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   if(f.dimension()==2) {
      OPENGM_ASSERT(f.shape(0)>static_cast<LABEL>(1));
      LABEL c[2]={1, 0};
      //get possible weight
      VALUE weight=f(c);
      //get possible truncation (compute the biggest possible distance => value is the truncation value)
      c[0]=f.shape(0)-static_cast<LABEL>(1);
      VALUE truncated=f(c);
      for( c[1]=0;c[1]<f.shape(1);++c[1]) {
         for( c[0]=0;c[0]<f.shape(0);++c[0]) {
            VALUE d= static_cast<VALUE> (c[0]<c[1] ? c[1]-c[0]:c[0]-c[1]);
            d*=d;
            const VALUE fval=f(c);
            const VALUE compare=d*weight;
            if( isNumericEqual(fval, compare  )==false && ((isNumericEqual(fval, truncated) && truncated<compare)==false)) {
               return false;
            }
         } 
      }
      return true;
   }
   return false;
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline bool 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::isAbsoluteDifference() const{
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   if(f.dimension()==2) {
      OPENGM_ASSERT(f.shape(0)>static_cast<LABEL>(1));
      LABEL c[2]={1, 0};
      //get possible weight
      VALUE weight=f(c);
      for( c[1]=0;c[1]<f.shape(1);++c[1]) {
         for( c[0]=0;c[0]<f.shape(0);++c[0]) {
            VALUE d= static_cast<VALUE> (c[0]<c[1] ? c[1]-c[0]:c[0]-c[1]);
            if( isNumericEqual(f(c), d*weight  )==false)
               return false;
         } 
      }
      return true;
   }
   return false;
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline bool 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::isTruncatedAbsoluteDifference() const{
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   if(f.dimension()==2) {
      OPENGM_ASSERT(f.shape(0)>static_cast<LABEL>(1));
      LABEL c[2]={1, 0};
      //get possible weight
      VALUE weight=f(c);
      //get possible truncation (compute the biggest possible distance => value is the truncation value)
      c[0]=f.shape(0)-static_cast<LABEL>(1);
      VALUE truncated=f(c);
      for( c[1]=0;c[1]<f.shape(1);++c[1]) {
         for( c[0]=0;c[0]<f.shape(0);++c[0]) {
            VALUE d= static_cast<VALUE> (c[0]<c[1] ? c[1]-c[0]:c[0]-c[1]);
            const VALUE fval=f(c);
            const VALUE compare=d*weight;
            if( isNumericEqual(fval, compare  )==false && ((isNumericEqual(fval, truncated) && truncated<compare)==false)) {
               return false;
            }
         } 
      }
      return true;
   }
   return false;
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline bool 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::isPotts() const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   if (f.size()<=2) return true;//BSD: Bug fixed?
   ShapeWalker<FunctionShapeIteratorType> shapeWalker(f.functionShapeBegin(), f.dimension());
   VALUE vEqual=f(shapeWalker.coordinateTuple().begin());
   ++shapeWalker;
   VALUE vNotEqual=f(shapeWalker.coordinateTuple().begin());
   ++shapeWalker;
   for(INDEX i=2;i<f.size();++i, ++shapeWalker) {
      // all labels are equal
      if(isEqualValueVector(shapeWalker.coordinateTuple()) ) {
         if(vEqual!=f(shapeWalker.coordinateTuple().begin()))
            return false;
      }
      // all labels are not equal
      else{
         if(vNotEqual!=f(shapeWalker.coordinateTuple().begin()))
            return false;
      }
   }
   return true;
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline bool 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::isGeneralizedPotts() const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   if(f.dimension()==2) {
      LABEL l[] = {0, 1};
      VALUE v0 = f(l);
      l[1]=0;
      VALUE v1 = f(l);
      for(l[0]=0;l[0]<f.shape(0);++l[0]) {
         for(l[1]=0;l[1]<f.shape(1);++l[1]) {
            if((l[0]==l[1] && f(l)!=v1) || ((l[0]!=l[1] && f(l)!=v0)) ) return false;
         }
      }
      return true;
   }
   else  if(f.dimension()==3) {
      LABEL l[] = {0, 1, 2};
      VALUE v000 = f(l);
      l[2]=0; l[1]=1; l[0]=1;
      VALUE v001 = f(l);
      l[2]=1; l[1]=0; l[0]=1;
      VALUE v010 = f(l);
      l[2]=1; l[1]=1; l[0]=0;
      VALUE v100 = f(l); 
      l[2]=0; l[1]=0; l[0]=0;
      VALUE v111 = f(l);
      for(l[0]=0;l[0]<f.shape(0);++l[0]) {
         for(l[1]=0;l[1]<f.shape(1);++l[1]) { 
            for(l[2]=0;l[2]<f.shape(2);++l[2]) {
               if((l[1]!=l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v000) ) return false;
               if((l[1]!=l[2] && l[0]!=l[2]  && l[0]==l[1] && f(l)!=v001) ) return false;
               if((l[1]!=l[2] && l[0]==l[2]  && l[0]!=l[1] && f(l)!=v010) ) return false;
               if((l[1]==l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v100) ) return false;
               if((l[1]==l[2] && l[0]==l[2]  && l[0]==l[1] && f(l)!=v111) ) return false;
            }
         }
      }
      return true;
   } 
   else  if(f.dimension()==4) {
      LABEL l[] = {0, 1, 2, 3};
      VALUE v000000 = f(l);
      l[3]=2; l[2]=1; l[1]=0;l[0]=0;
      VALUE v000001 = f(l);
      l[3]=2; l[2]=0; l[1]=1;l[0]=0;
      VALUE v000010 = f(l);
      l[3]=2; l[2]=0; l[1]=0;l[0]=1;
      VALUE v000100 = f(l); 
      l[3]=1; l[2]=0; l[1]=0;l[0]=0; //3-1
      VALUE v000111 = f(l);
      l[3]=0; l[2]=1; l[1]=2; l[0]=0;
      VALUE v001000 = f(l);
      l[3]=0; l[2]=1; l[1]=1; l[0]=0;
      VALUE v001100 = f(l);
      l[3]=0; l[2]=1; l[1]=0; l[0]=0; //3-1
      VALUE v011001 = f(l);
      l[3]=0; l[2]=0; l[1]=0; l[0]=1; //3-1
      VALUE v110100 = f(l); 
      l[3]=0; l[2]=0; l[1]=0; l[0]=0;
      VALUE v111111 = f(l);
      l[3]=1; l[2]=1; l[1]=0; l[0]=0;
      VALUE v100001 = f(l); 
      l[3]=1; l[2]=0; l[1]=1; l[0]=0;
      VALUE v010010 = f(l); 
      l[3]=0; l[2]=0; l[1]=1; l[0]=2;
      VALUE v100000 = f(l); 
      l[3]=0; l[2]=1; l[1]=0; l[0]=2;
      VALUE v010000 = f(l);
      l[3]=0; l[2]=0; l[1]=1; l[0]=0; //3-1
      VALUE v101010 = f(l); 


      for(l[0]=0;l[0]<f.shape(0);++l[0]) {
         for(l[1]=0;l[1]<f.shape(1);++l[1]) { 
            for(l[2]=0;l[2]<f.shape(2);++l[2]) { 
               for(l[3]=0;l[3]<f.shape(3);++l[3]) {
                  if((l[2]!=l[3] && l[1]!=l[3] && l[0]!=l[3] && l[1]!=l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v000000) ) {std::cout<<"1"; return false;}
                  if((l[2]!=l[3] && l[1]!=l[3] && l[0]!=l[3] && l[1]!=l[2] && l[0]!=l[2]  && l[0]==l[1] && f(l)!=v000001) ) {std::cout<<"1"; return false;}
                  if((l[2]!=l[3] && l[1]!=l[3] && l[0]!=l[3] && l[1]!=l[2] && l[0]==l[2]  && l[0]!=l[1] && f(l)!=v000010) ) {std::cout<<"1"; return false;}
                  if((l[2]!=l[3] && l[1]!=l[3] && l[0]!=l[3] && l[1]==l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v000100) ) {std::cout<<"1"; return false;}
                  if((l[2]!=l[3] && l[1]!=l[3] && l[0]!=l[3] && l[1]==l[2] && l[0]==l[2]  && l[0]==l[1] && f(l)!=v000111) ) {std::cout<<"1"; return false;}
                  
                  if((l[2]!=l[3] && l[1]!=l[3] && l[0]==l[3] && l[1]!=l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v001000) ) {std::cout<<"1"; return false;}
                  if((l[2]!=l[3] && l[1]!=l[3] && l[0]==l[3] && l[1]==l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v001100) ) {std::cout<<"1"; return false;}
                  
                  if((l[2]!=l[3] && l[1]==l[3] && l[0]!=l[3] && l[1]!=l[2] && l[0]==l[2]  && l[0]!=l[1] && f(l)!=v010010) ) {std::cout<<"1"; return false;}
                  if((l[2]!=l[3] && l[1]==l[3] && l[0]!=l[3] && l[1]!=l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v010000) ) {std::cout<<"1"; return false;}
                  if((l[2]!=l[3] && l[1]==l[3] && l[0]==l[3] && l[1]!=l[2] && l[0]!=l[2]  && l[0]==l[1] && f(l)!=v011001) ) {std::cout<<"1"; return false;}
                  
                  if((l[2]==l[3] && l[1]==l[3] && l[0]!=l[3] && l[1]==l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v110100) ) {std::cout<<"1"; return false;}
                  if((l[2]==l[3] && l[1]==l[3] && l[0]==l[3] && l[1]==l[2] && l[0]==l[2]  && l[0]==l[1] && f(l)!=v111111) ) {std::cout<<"1"; return false;}
                  
                  if((l[2]==l[3] && l[1]!=l[3] && l[0]!=l[3] && l[1]!=l[2] && l[0]!=l[2]  && l[0]==l[1] && f(l)!=v100001) ) {std::cout<<"1"; return false;}
                  if((l[2]==l[3] && l[1]!=l[3] && l[0]!=l[3] && l[1]!=l[2] && l[0]!=l[2]  && l[0]!=l[1] && f(l)!=v100000) ) {std::cout<<"1"; return false;}
                  if((l[2]==l[3] && l[1]!=l[3] && l[0]==l[3] && l[1]!=l[2] && l[0]==l[2]  && l[0]!=l[1] && f(l)!=v101010) ) {std::cout<<"1"; return false;}
               }
            }
         }
      }
      return true;
   }
   else{
      return false;
   }
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline bool 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::isSubmodular() const {
   const FunctionType& f = *static_cast<FunctionType const *>(this);
   if(f.dimension()==1){
      return true;
   }
   if(f.dimension()!=2 ||f.shape(0)!=2 || f.shape(1)!=2) {
      throw RuntimeError("Fallback FunctionBase::isSubmodular only defined for binary functions with order less than 3");
   }
   LABEL l00[] = {0, 0};
   LABEL l01[] = {0, 1};
   LABEL l10[] = {1, 0};
   LABEL l11[] = {1, 1};

   return f(l00)+f(l11)<= f(l10)+f(l01);
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline MinMaxFunctor<VALUE> 
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::minMax() const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   opengm::FastSequence<INDEX>  c(f.dimension(), 0);
   const VALUE tmp=f(c.begin());
   MinMaxFunctor<VALUE> minMax(tmp, tmp);
   static_cast<FunctionType const *>(this)->forAtLeastAllUniqueValues(minMax);
   return minMax;
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline typename FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::ReturnType
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::min() const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   opengm::FastSequence<INDEX>  c(f.dimension(), 0);
   AccumulationFunctor<Minimizer, VALUE> accumulator(f(c.begin()));
   static_cast<FunctionType const *>(this)->forAtLeastAllUniqueValues(accumulator);
   return accumulator.value();
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline typename FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::ReturnType
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::max() const {
   const FunctionType& f=*static_cast<FunctionType const *>(this);
   opengm::FastSequence<INDEX>  c(f.dimension(), 0);
   AccumulationFunctor<Maximizer, VALUE> accumulator(f(c.begin()));
   static_cast<FunctionType const *>(this)->forAtLeastAllUniqueValues(accumulator);
   return accumulator.value();
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline typename FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::ReturnType
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::sum() const {
   AccumulationFunctor<Integrator, VALUE> accumulator(static_cast<VALUE>(0));
   static_cast<FunctionType const *>(this)->forAllValuesInAnyOrder(accumulator);
   return accumulator.value();
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
inline typename FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::ReturnType
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::product() const {
   AccumulationFunctor<Multiplier, VALUE> accumulator(static_cast<VALUE>(1));;
   static_cast<FunctionType const *>(this)->forAllValuesInAnyOrder(accumulator);
   return accumulator.value();
}

template<class FUNCTION, class VALUE, class INDEX, class LABEL>
template<class ACC>
inline typename FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::ReturnType
FunctionBase<FUNCTION, VALUE, INDEX, LABEL>::accumulate() const {
   if(meta::Compare<ACC, Minimizer>::value  ) {
      return static_cast<FunctionType const *>(this)->min();
   }
   else if( meta::Compare<ACC, Maximizer>::value ) {
      return static_cast<FunctionType const *>(this)->max();
   }
   else if( meta::Compare<ACC, Adder>::value ) {
      return static_cast<FunctionType const *>(this)->sum();
   }
   else if( meta::Compare<ACC, Integrator>::value ) {
      return static_cast<FunctionType const *>(this)->sum();
   }
   else if( meta::Compare<ACC, Multiplier>::value ) {
      return static_cast<FunctionType const *>(this)->product();
   }
   else{
      AccumulationFunctor<ACC, VALUE> accumulator;
      static_cast<FunctionType const *>(this)->forAllValuesInOrder(accumulator);
      return accumulator.value();
   }
}

} // namespace opengm

#endif // OPENGM_FUNCTION_PROPERTIES_BASE_HXX
