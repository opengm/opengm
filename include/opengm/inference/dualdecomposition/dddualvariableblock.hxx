#pragma once
#ifndef OPENGM_DD_DUALVARIABLEBLOCK_HXX
#define OPENGM_DD_DUALVARIABLEBLOCK_HXX

#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/graphicalmodel/decomposition/graphicalmodeldecomposition.hxx"

namespace opengm {
/// \cond HIDDEN_SYMBOLS

   template<class DUALVAR = marray::Marray<double> >
   class DDDualVariableBlock{
   public:
      typedef DUALVAR  DualVariableType;
      typedef typename DUALVAR::ValueType ValueType; 
      typedef typename GraphicalModelDecomposition::SubFactor                    SubFactorType;
      typedef typename GraphicalModelDecomposition::SubFactorListType            SubFactorListType; 

      // Methods
      DDDualVariableBlock(){};
      template<class ITERATOR> DDDualVariableBlock(const SubFactorListType& subFactorList, ITERATOR shapeBegin, ITERATOR shapeEnd);
      std::vector<DUALVAR*> getPointers();
      void test() const;

      // Members
      std::vector<DualVariableType> duals_;
      const SubFactorListType* subFactorList_;
   };


   template<class DUALVAR = marray::Marray<double> >
   class DDDualVariableBlock2{
   public:
      typedef DUALVAR  DualVariableType;
      typedef typename DUALVAR::ValueType ValueType; 
      typedef typename GraphicalModelDecomposition::SubFactor                    SubFactorType;
      typedef typename GraphicalModelDecomposition::SubFactorListType            SubFactorListType; 

      // Methods
      DDDualVariableBlock2(){};
      template<class ITERATOR> DDDualVariableBlock2(const SubFactorListType& subFactorList, ITERATOR shapeBegin, ITERATOR shapeEnd);
      std::vector<DUALVAR*> getPointers(); 
      void test() const;

      // Members
      std::vector<DualVariableType> duals_;
      std::vector<DualVariableType> duals2_;
      const SubFactorListType* subFactorList_;
   }; 


   /////////////////////////////////
   ////////////////////////////////

   template<class DUALVAR>
   template<class ITERATOR> 
   DDDualVariableBlock<DUALVAR>::DDDualVariableBlock
   (
      const typename GraphicalModelDecomposition::SubFactorListType& subFactorList, 
      ITERATOR shapeBegin,
      ITERATOR shapeEnd
      )
   {
      const size_t numDuals = subFactorList.size(); 
      duals_.resize(numDuals, DUALVAR(shapeBegin,shapeEnd,0));
      subFactorList_ = &subFactorList; 
   }
   template<>
   template<class ITERATOR> 
   DDDualVariableBlock<marray::View<double,false> >::DDDualVariableBlock
   (
      const SubFactorListType& subFactorList, 
      ITERATOR shapeBegin,
      ITERATOR shapeEnd
      )
   {
      const size_t numDuals = subFactorList.size();
      double tmp;
      duals_.resize(numDuals, marray::View<double,false>(shapeBegin,shapeEnd,&tmp));
      subFactorList_ = &subFactorList; 
   }  

   template<>
   template<class ITERATOR> 
   DDDualVariableBlock<marray::View<float,false> >::DDDualVariableBlock
   (
      const SubFactorListType& subFactorList, 
      ITERATOR shapeBegin,
      ITERATOR shapeEnd
      )
   {
      const size_t numDuals = subFactorList.size();
      double tmp;
      duals_.resize(numDuals, marray::View<float,false>(shapeBegin,shapeEnd,&tmp));
      subFactorList_ = &subFactorList; 
   } 
   
   template<class DUALVAR>
   std::vector<DUALVAR*>  DDDualVariableBlock<DUALVAR>::getPointers()
   {
      std::vector<DualVariableType*> ret(duals_.size());
      for(size_t i=0; i<duals_.size(); ++i) ret[i] = &(duals_[i]);
      return ret;
   }

   template<class DUALVAR>
   void DDDualVariableBlock<DUALVAR>::test() const
   {
      marray::Marray<double> temp(duals_[0].shapeBegin(), duals_[0].shapeEnd() ,0);
      for(size_t i=0; i<duals_.size(); ++i) {
         temp += duals_[i];
      }
      //std::cout<<" temp size "<<temp.size()<<"\n";
      for(size_t j=0; j<temp.size(); ++j) {
         if(  (temp(j)<0.001 && temp(j)>-0.001)==false ){
            std::cout<<"temp("<<j<<") = "<<temp(j)<<"\n";
         }
         //OPENGM_ASSERT(temp(i)<0.00001 && temp(i)>-0.00001);
      }
   }

   ////////////////////////////////////////

   template<class DUALVAR>
   template<class ITERATOR> 
   DDDualVariableBlock2<DUALVAR>::DDDualVariableBlock2
   (
      const typename GraphicalModelDecomposition::SubFactorListType& subFactorList, 
      ITERATOR shapeBegin,
      ITERATOR shapeEnd
      )
   {
      const size_t numDuals = subFactorList.size(); 
      duals_.resize(numDuals, DUALVAR(shapeBegin,shapeEnd,0));
      duals2_.resize(numDuals, DUALVAR(shapeBegin,shapeEnd,0));
      subFactorList_ = &subFactorList; 
   }
   template<>
   template<class ITERATOR> 
   DDDualVariableBlock2<marray::View<double,false> >::DDDualVariableBlock2
   (
      const SubFactorListType& subFactorList, 
      ITERATOR shapeBegin,
      ITERATOR shapeEnd
      )
   {
      const size_t numDuals = subFactorList.size();
      double tmp;
      duals_.resize(numDuals, marray::View<double,false>(shapeBegin,shapeEnd,&tmp));
      duals2_.resize(numDuals, marray::View<double,false>(shapeBegin,shapeEnd,&tmp));
      subFactorList_ = &subFactorList; 
   }  
   template<>
   template<class ITERATOR> 
   DDDualVariableBlock2<marray::View<float,false> >::DDDualVariableBlock2
   (
      const SubFactorListType& subFactorList, 
      ITERATOR shapeBegin,
      ITERATOR shapeEnd
      )
   {
      const size_t numDuals = subFactorList.size();
      float tmp;
      duals_.resize(numDuals, marray::View<float,false>(shapeBegin,shapeEnd,&tmp));
      duals2_.resize(numDuals, marray::View<float,false>(shapeBegin,shapeEnd,&tmp));
      subFactorList_ = &subFactorList; 
   } 
   
   template<class DUALVAR>
   std::vector<DUALVAR*>  DDDualVariableBlock2<DUALVAR>::getPointers()
   {
      std::vector<DualVariableType*> ret(duals_.size());
      for(size_t i=0; i<duals_.size(); ++i) ret[i] = &(duals_[i]);
      return ret;
   }

   template<class DUALVAR>
   void  DDDualVariableBlock2<DUALVAR>::test() const 
   {
      marray::Marray<double> temp(duals_[0].shapeBegin(),duals_[0].shapeEnd(),0);
      for(size_t i=0; i<duals_.size(); ++i) temp += duals_[i];
      for(size_t i=0; i<temp.size(); ++i) OPENGM_ASSERT(temp(i)<0.00001 && temp(i)>-0.00001); 
   }
  
/// \endcond HIDDEN_SYMBOLS
} // namespace opengm

#endif
