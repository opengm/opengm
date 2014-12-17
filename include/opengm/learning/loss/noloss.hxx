#pragma once
#ifndef OPENGM_NO_LOSS_HXX
#define OPENGM_NO_LOSS_HXX

#include "opengm/functions/explicit_function.hxx"
namespace opengm {
   namespace learning {
      class NoLoss{
      public:
          class Parameter{
          };

      public:
         NoLoss(const Parameter& param = Parameter()) : param_(param){}

         template<class IT1, class IT2>
         double loss(IT1 labelBegin, IT1 labelEnd, IT2 GTBegin,IT2 GTEnd) const;
  
         template<class GM, class IT>
         void addLoss(GM& gm, IT GTBegin) const;
      private:
         Parameter param_;
      };

      template<class IT1, class IT2>
      double NoLoss::loss(IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
      {
         double loss = 0.0;
         return loss;
      }

      template<class GM, class IT>
      void NoLoss::addLoss(GM& gm, IT gt) const
      {
      }
   }  
} // namespace opengm

#endif 
