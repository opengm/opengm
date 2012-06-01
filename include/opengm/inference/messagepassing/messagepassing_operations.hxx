#pragma once
#ifndef OPENGM_MESSAGEPASSING_OPERATIONS_HXX
#define OPENGM_MESSAGEPASSING_OPERATIONS_HXX

#include <opengm/opengm.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

/// \cond HIDDEN_SYMBOLS

namespace opengm {
namespace messagepassingOperations {

// out = M(M.shape, OP:neutral)
/// \todo unroll loop
template<class OP, class M>
inline void clean(M& out) {
   for(size_t n=0; n<out.size(); ++n ) {
      OP::neutral(out(n));
   }
/*
   const size_t loopSize= out.size()/2;
   for(size_t n=0; n<loopSize;n+=2 ) {
      OP::neutral(out(n));
      OP::neutral(out(n+1));
   }
   const size_t loopSize2= loopSize*2;
   if(loopSize2!=out.size()) {
      OP::neutral(out(loopSize2));
   }
*/
}
      
template<class OP, class ACC, class M>
inline void normalize
(
   M& out
) {
   typename M::ValueType v;
   ACC::neutral(v);
   for(size_t n=0; n<out.size(); ++n)
      ACC::op(out(n),v);
         
   if( opengm::meta::Compare<OP,opengm::Multiplier>::value && v <= 0.00001)
      return;
   if(opengm::meta::Compare<OP,opengm::Multiplier>::value)
      OPENGM_ASSERT(v > 0.00001); // ??? this should be checked in released code   
   for(size_t n=0; n<out.size();++n ) {
      OP::iop(v,out(n));      
   }
/*
   const size_t loopSize= out.size()/2;
   for(size_t n=0; n<loopSize;n+=2 ) {
      OP::iop(v,out(n));
      OP::iop(v,out(n+1));
   }
   const size_t loopSize2= loopSize*2;
   if(loopSize2!=out.size()) {
      OP::iop(v,out(loopSize2));
   }
*/
}

/// out = op( hop(in1,alpha), hop(in2,1-alpha) )
template<class OP, class M, class T>
inline void weightedMean
(
   const M& in1,
   const M& in2, 
   const T alpha, 
   M& out
) { 
   /// TODO
   /// Speedup
   T v1,v2;
   const T oneMinusAlpha=static_cast<T>(1)-alpha;
        
   for(size_t n=0; n<out.size();++n ) {
      OP::hop(in1(n),alpha,  v1);
      OP::hop(in2(n),oneMinusAlpha,v2);
      OP::op(v1,v2,out(n));
   }
   /*
   const size_t loopSize= out.size()/2;
   for(size_t n=0; n<loopSize;n+=2 ) {
      OP::hop(in1(n),alpha,  v1);
      OP::hop(in2(n),oneMinusAlpha,v2);
      OP::op(v1,v2,out(n));
            
      OP::hop(in1(n+1),alpha,  v1);
      OP::hop(in2(n+1),oneMinusAlpha,v2);
      OP::op(v1,v2,out(n+1));
   }
   const size_t loopSize2= loopSize*2;
   if(loopSize2!=out.size()) {
      OP::hop(in1(loopSize2),alpha,  v1);
      OP::hop(in2(loopSize2),oneMinusAlpha,v2);
      OP::op(v1,v2,out(loopSize2));
   }
   */
}
      
/// out = op(vec[0].current, ..., vec[n].current ) 
template<class OP, class BUFFER, class M>
inline void operate
(
   const std::vector<BUFFER>& vec,
   M& out
) {
   /// ???
   /// switch order of loops ?
   clean<OP>(out);
   for(size_t j = 0; j < vec.size(); ++j) {
      const typename BUFFER::ArrayType& b = vec[j].current();
      OPENGM_ASSERT(b.size()==out.size());
      for(size_t n=0; n<out.size(); ++n) 
         OP::op(b(n), out(n));
   }
}  
      
/// out = op( out , op_i( hop(vec[i],rho[i]) ) )
template<class GM, class BUFFER, class M>
inline void operateW
(
   const std::vector<BUFFER>& vec,
   const std::vector<typename GM::ValueType>& rho,
   M& out
) {
   typedef typename GM::OperatorType OP;
   clean<OP>(out);
   /// ???
   /// switch order of loops ?
   /// => loop over out?
   for(size_t j = 0; j < vec.size(); ++j) {
      const typename BUFFER::ArrayType& b = vec[j].current();
      typename GM::ValueType e = rho[j];
      typename GM::ValueType v;
      for(size_t n=0; n<out.size(); ++n) {
         OP::hop(b(n),e,v);
         OP::op(v,out(n));
      }
   }
}
   
/// out = op( vec[0].current, ..., vec[i-1].current,vec[i+1].current, ... , vec[n].current ) 
template<class OP, class BUFVEC, class M, class INDEX>
inline void operate
(
   const BUFVEC& vec,
   const INDEX i, 
   M& out
) {
   clean<OP>(out); 
   /// TODO
   /// switch order of loops ? (loop over out?)
   /// => clean could be inside the loop over the result
   for(size_t j = 0; j < i; ++j) {
      const M& f = vec[j].current();
      for(size_t n=0; n<out.size(); ++n) 
         OP::op(f(n), out(n));
   }
   for(size_t j = i+1; j < vec.size(); ++j) {
      const M& f = vec[j].current();
      for(size_t n=0; n<out.size(); ++n) 
         OP::op(f(n), out(n));        
   }
}
 
/// out = op( hop(vec[i],rho[i]-1), op_j/i( hop(vec[j],rho[j]) ) )
template<class GM, class BUFVEC, class M, class INDEX>
inline void operateW
(
   const BUFVEC& vec, 
   const INDEX i, 
   const std::vector<typename GM::ValueType>& rho,
   M& out
) {  
   typedef typename GM::OperatorType OP;
   OPENGM_ASSERT(vec[i].current().size()==out.size());
   typename GM::ValueType v;
   const typename GM::ValueType e = rho[i]-1;
   const M& b = vec[i].current();
   for(size_t n=0; n<out.size(); ++n) {
      //OP::hop(b(n),e,v);
      //OP::op(v,out(n));
      OP::hop(b(n),e,out(n));
   }
         
   for(size_t j = 0; j < i; ++j) {
      const M& b = vec[j].current();
      const typename GM::ValueType e = rho[j];
      OPENGM_ASSERT(b.size()==out.size());
      for(size_t n=0; n<out.size(); ++n) {
         OP::hop(b(n),e,v);
         OP::op(v,out(n));
      }
   } 
   for(size_t j = i+1; j < vec.size(); ++j) {
      const M& b = vec[j].current();
      const typename GM::ValueType e = rho[j];
      OPENGM_ASSERT(b.size()==out.size());
      for(size_t n=0; n<out.size(); ++n) {
         OP::hop(b(n),e,v);
         OP::op(v,out(n));
      }
   }       
}
      
/// out = acc( op(f, vec[0].current, ..., vec[n].current ), -i) 
template<class GM, class ACC, class BUFVEC, class ARRAY, class INDEX>
inline void operateF
(
   const typename GM::FactorType& f, 
   const BUFVEC& vec, 
   const INDEX i, 
   ARRAY& out
) {  //TODO: Speedup, Inplace
   typedef typename GM::OperatorType OP;
   if(f.numberOfVariables()==2) {
      size_t count[2];
      typename GM::ValueType v;
      for(size_t n=0; n<out.size(); ++n)
         ACC::neutral(out(n));
      for(count[0]=0;count[0]<f.numberOfLabels(0);++count[0]) { 
         for(count[1]=0;count[1]<f.numberOfLabels(1);++count[1]) {
            v = f(count);
            if(i==0)
               OP::op(vec[1].current()(count[1]), v);
            else
               OP::op(vec[0].current()(count[0]), v);
            /*
            for(size_t j = 0; j < i; ++j)
               OP::op(vec[j].current()(count[j]), v);
            for(size_t j = i+1; j < vec.size(); ++j)
               OP::op(vec[j].current()(count[j]), v);
            */ 
            ACC::op(v,out(count[i]));
         }
      }
   }
   else{
      // accumulation over all variables except x
      typedef typename GM::IndexType IndexType;
      typedef typename GM::LabelType LabelType;
      // neutral initialization of output
      for(size_t n=0; n<f.numberOfLabels(i); ++n)
         ACC::neutral(out(n));
      // factor shape iterator
      typedef typename GM::FactorType::ShapeIteratorType FactorShapeIteratorType;
      opengm::ShapeWalker<FactorShapeIteratorType> shapeWalker(f.shapeBegin(),f.numberOfVariables());
      for(IndexType scalarIndex=0;scalarIndex<f.size();++scalarIndex,++shapeWalker) {
         // loop over the variables
         // initialize output value with value of the factor at this coordinate
         // operate j=[0,..i-1]
         typename GM::ValueType value=f(shapeWalker.coordinateTuple().begin());
         for(IndexType j=0;j<static_cast<typename GM::IndexType>(i);++j) {
            const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
            OP::op(vec[j].current()(label),value);
         }
         // operate j=[i+1,..,vec.size()]
         for(IndexType j=i+1;j< vec.size();++j) {
            const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
            OP::op(vec[j].current()(label),value);
         }
         // accumulate 
         ACC::op(value,out(shapeWalker.coordinateTuple()[i]));
      }
      //typename GM::IndependentFactorType temp = f; 
      //std::vector<size_t> accVar;
      //accVar.reserve(vec.size());
      //
      //for(size_t j = 0; j < i; ++j) {
      //   size_t var = f.variableIndex(j);
      //   typename GM::IndependentFactorType dummy(&var, &var+1,vec[j].current().shapeBegin(),vec[j].current().shapeEnd());
      //   for(size_t n=0; n<dummy.size();++n)
      //      dummy(n) = vec[j].current()(n);
      //   OP::op(dummy, temp);
      //   accVar.push_back(f.variableIndex(j));
      //}
      // 
      //for(size_t j = i+1; j < vec.size(); ++j) {
      //   size_t var = f.variableIndex(j);
      //   typename GM::IndependentFactorType dummy(&var, &var+1,vec[j].current().shapeBegin(),vec[j].current().shapeEnd());
      //   for(size_t n=0; n<dummy.size();++n)
      //      dummy(n) = vec[j].current()(n);
      //   OP::op(dummy, temp);
      //   accVar.push_back(f.variableIndex(j));
      //}
      //temp.template accumulate<ACC> (accVar.begin(), accVar.end());
      //for(size_t n=0; n<temp.size(); ++n) {
      //   out(n) = temp(n);
      //}
   }  
}

/// out = acc_-i( op( ihop(f,rho), op_j/i( vec[j] ) ) )
template<class GM, class ACC, class BUFVEC, class M, class INDEX>
inline void operateWF
(
   const typename GM::FactorType& f, 
   const typename GM::ValueType rho, 
   const BUFVEC& vec, 
   const INDEX i, 
   M& out
) {//TODO: Speedup, Inplace   
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;
   typedef typename GM::OperatorType OP;
   // neutral initialization of output
   for(size_t n=0; n<f.numberOfLabels(i); ++n)
      ACC::neutral(out(n));
   // factor shape iterator
   typedef typename GM::FactorType::ShapeIteratorType FactorShapeIteratorType;
   opengm::ShapeWalker<FactorShapeIteratorType> shapeWalker(f.shapeBegin(),f.numberOfVariables());
   for(IndexType scalarIndex=0;scalarIndex<f.size();++scalarIndex,++shapeWalker) {
      // loop over the variables
      // initialize output value with value of the factor at this coordinate
      // operate j=[0,..i-1]
      typename GM::ValueType value;
      OP::ihop(f(shapeWalker.coordinateTuple().begin()),rho,value);
      for(IndexType j=0;j<static_cast<typename GM::IndexType>(i);++j) {
         const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
         OP::op(vec[j].current()(label),value);
      }
      // operate j=[i+1,..,vec.size()]
      for(IndexType j=i+1;j< vec.size();++j) {
         const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
         OP::op(vec[j].current()(label),value);
      }
      // accumulate 
      ACC::op(value,out(shapeWalker.coordinateTuple()[i]));
   }

   //typedef typename GM::OperatorType OP;
   //typename GM::IndependentFactorType temp = f; 
   //OP::ihop(f, rho, temp);
   //std::vector<size_t> accVar;
   //accVar.reserve(vec.size());      
   //for(size_t j = 0; j < i; ++j) {
   //   size_t var = f.variableIndex(j);
   //   typename GM::IndependentFactorType dummy(&var, &var+1,vec[j].current().shapeBegin(),vec[j].current().shapeEnd());
   //   for(size_t n=0; n<dummy.size();++n)
   //      dummy(n) = vec[j].current()(n);
   //   OP::op(dummy, temp);
   //   accVar.push_back(f.variableIndex(j));
   //} 
   //for(size_t j = i+1; j < vec.size(); ++j) {
   //   size_t var = f.variableIndex(j);
   //   typename GM::IndependentFactorType dummy(&var, &var+1,vec[j].current().shapeBegin(),vec[j].current().shapeEnd());
   //   for(size_t n=0; n<dummy.size();++n)
   //      dummy(n) = vec[j].current()(n);
   //   OP::op(dummy, temp);
   //   accVar.push_back(f.variableIndex(j));
   //}
   //temp.template accumulate<ACC> (accVar.begin(), accVar.end());
   //for(size_t n=0; n<temp.size(); ++n) {
   //   out(n) = temp(n);
   //}  
}

/// out = op(f, vec[0].current, ..., vec[n].current ) 
template<class GM, class BUFVEC>
inline void operateF
(
   const typename GM::FactorType& f, 
   const BUFVEC& vec, 
   typename GM::IndependentFactorType& out
)
{
   OPENGM_ASSERT(out.numberOfVariables()!=0);
   //TODO: Speedup
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;
   typedef typename GM::OperatorType OP;
   // factor shape iterator
   typedef typename GM::FactorType::ShapeIteratorType FactorShapeIteratorType;
   opengm::ShapeWalker<FactorShapeIteratorType> shapeWalker(f.shapeBegin(),f.numberOfVariables());
   for(IndexType scalarIndex=0;scalarIndex<f.size();++scalarIndex,++shapeWalker) {
      // loop over the variables
      typename GM::ValueType value=f(shapeWalker.coordinateTuple().begin());
      for(IndexType j=0;j<static_cast<typename GM::IndexType>(vec.size());++j) {
         const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
         OP::op(vec[j].current()(label),value);
      }
      out(scalarIndex)=value;
   }
   //typedef typename GM::OperatorType OP;
   //out = f; 
   // accumulation over all variables except x
   //for(size_t j = 0; j < vec.size(); ++j) {
   //   size_t var = f.variableIndex(j);
   //   typename GM::IndependentFactorType dummy(&var, &var+1,vec[j].current().shapeBegin(),vec[j].current().shapeEnd());
   //   for(size_t n=0; n<dummy.size();++n)
   //      dummy(n) = vec[j].current()(n); 
   //   //OPENGM_ASSERT(f.variableIndex(j)==vec[j].current().variableIndex(0));   
   //   OP::op(dummy, out);
   //}
}


/// out = op( ihop(f,rho), op_j(vec[j]) )
template<class GM, class BUFVEC>
inline void operateWF
(
   const typename GM::FactorType& f, 
   const typename GM::ValueType rho, 
   const BUFVEC& vec, 
   typename GM::IndependentFactorType& out
) {//TODO: Speedup
   typedef typename GM::OperatorType OP;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;
   typedef typename GM::FactorType::ShapeIteratorType FactorShapeIteratorType;
   opengm::ShapeWalker<FactorShapeIteratorType> shapeWalker(f.shapeBegin(),f.numberOfVariables());
   for(IndexType scalarIndex=0;scalarIndex<f.size();++scalarIndex,++shapeWalker) {
      // loop over the variables
      typename GM::ValueType value;
      OP::ihop(f(shapeWalker.coordinateTuple().begin()),rho,value);
      for(IndexType j=0;j<static_cast<typename GM::IndexType>(vec.size());++j) {
         const LabelType label=static_cast<LabelType>(shapeWalker.coordinateTuple()[j]);
         OP::op(vec[j].current()(label),value);
      }
      out(scalarIndex)=value;
   }
           
   //OP::ihop(f, rho, out);
   //for(size_t j = 0; j <  vec.size(); ++j) {
   //   size_t var = f.variableIndex(j);
   //   typename GM::IndependentFactorType dummy(&var, &var+1,vec[j].current().shapeBegin(),vec[j].current().shapeEnd());
   //   for(size_t n=0; n<dummy.size();++n)
   //      dummy(n) = vec[j].current()(n); 
   //   //OPENGM_ASSERT(f.variableIndex(j)==vec[j].current().variableIndex(0));   
   //   OP::op(dummy, out);
   //} 
}
/* 

template<class GM, class BUFVEC> 
void operateFi(
   const typename GM::FactorType& myFactor, 
   const BUFVEC& vec, 
   typename GM::IndependentFactorType& b
   )
{//SPEED ME UP
   typedef typename GM::OperatorType OP;
   b=myFactor;
   for(size_t j=0; j<vec.size();++j) {
      size_t var = myFactor.variableIndex(j);
      typename GM::IndependentFactorType dummy(&var, &var+1,vec[j]->current().shapeBegin(),vec[j]->current().shapeEnd());
      for(size_t n=0; n<dummy.size();++n)
         dummy(n) = vec[j]->current()(n); 
      OP::iop(dummy,b);
   }
}

template<class GM, class BUFVEC> 
void operateFiW(
   const typename GM::FactorType& myFactor, 
   const BUFVEC& vec, 
   const typename GM::ValueType rho, 
   typename GM::IndependentFactorType& b
   )
{//SPEED ME UP
   typedef typename GM::OperatorType OP;
   b=myFactor;
   for(size_t j=0; j<vec.size();++j) {
      size_t var = myFactor.variableIndex(j);
      typename GM::IndependentFactorType dummy(&var, &var+1,vec[j]->current().shapeBegin(),vec[j]->current().shapeEnd());
      for(size_t n=0; n<dummy.size();++n)
         OP::hop(vec[j]->current()(n),rho,dummy(n)); 
      OP::iop(dummy,b);
   }
}

template<class A, class B, class T, class OP, class ACC>
T boundOperation(const A& a, const B& b)
{ 
   T v;
         
   if(typeid(ACC)==typeid(opengm::Adder) && typeid(OP)==typeid(opengm::Multiplier)) { 
      T t;
      OP::hop(a(0),b(0),v);
      for(size_t n=1; n<a.size(); ++n) {
         OP::hop(a(n),b(n),t);
         OP::op(t,v);
      }
   }
   else if(typeid(ACC)==typeid(opengm::Minimizer) || typeid(ACC)==typeid(opengm::Maximizer)) {
      v = b(0);
      for(size_t n=1; n<a.size(); ++n) {
         ACC::bop( a(n),v );
      }
   }
   else{
      ACC::neutral(v);
   }
   return v;
}  
*/
    
} // namespace messagepassingOperations
} // namespace opengm

/// \endcond

#endif
