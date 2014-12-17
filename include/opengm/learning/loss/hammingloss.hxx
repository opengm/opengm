#pragma once
#ifndef OPENGM_HAMMING_LOSS_HXX
#define OPENGM_HAMMING_LOSS_HXX

#include "opengm/functions/explicit_function.hxx"
namespace opengm {
   namespace learning {
      class HammingLoss{
      public:
          class Parameter{
          };

      public:
         HammingLoss(const Parameter& param = Parameter()) : param_(param){}

         template<class IT1, class IT2>
         double loss(IT1 labelBegin, IT1 labelEnd, IT2 GTBegin,IT2 GTEnd) const;
  
         template<class GM, class IT>
         void addLoss(GM& gm, IT GTBegin) const;
      private:
         Parameter param_;
      };

      template<class IT1, class IT2>
      double HammingLoss::loss(IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
      {
         double loss = 0.0;
         for(; labelBegin!= labelEnd; ++labelBegin, ++GTBegin){
            if(*labelBegin != *GTBegin){
               loss += 1.0;
            }
         }
         return loss;
      }

      template<class GM, class IT>
      void HammingLoss::addLoss(GM& gm, IT gt) const
      {

         for(typename GM::IndexType i=0; i<gm.numberOfVariables(); ++i){
            typename GM::LabelType numL = gm.numberOfLabels(i);
            opengm::ExplicitFunction<typename GM::ValueType,typename GM::IndexType, typename GM::LabelType> f(&numL, &numL+1,-1);
            f(*gt) = 0;
            ++gt;
            gm.addFactor(gm.addFunction(f), &i, &(i)+1);
         }
      }

   }  
} // namespace opengm

#endif 
