#pragma once
#ifndef OPENGM_HAMMING_LOSS_HXX
#define OPENGM_HAMMING_LOSS_HXX

#include "opengm/functions/explicit_function.hxx"
#include "opengm/graphicalmodel/graphicalmodel_hdf5.hxx"

namespace opengm {
   namespace learning {
      class HammingLoss{
      public:
          class Parameter{
            public:
            bool operator==(const HammingLoss & other) const{
                return true;
            }
            bool operator<(const HammingLoss & other) const{
                return false;
            }
            bool operator>(const HammingLoss & other) const{
                return false;
            }
            /**
             * serializes the parameter object to the given hdf5 group handle;
             * the group must contain a dataset "lossType" containing the
             * loss type as a string
             **/
            void save(hid_t& groupHandle) const;
            inline void load(const hid_t& ) {}
            static std::size_t getLossId() { return lossId_; }
          private:
            static const std::size_t lossId_ = 16000;
          };

      public:
         HammingLoss(const Parameter& param = Parameter()) : param_(param){}

         template<class GM, class IT1, class IT2>
         double loss(const GM & gm, IT1 labelBegin, IT1 labelEnd, IT2 GTBegin,IT2 GTEnd) const;
  
         template<class GM, class IT>
         void addLoss(GM& gm, IT GTBegin) const;
      private:
         Parameter param_;
      };

      inline void HammingLoss::Parameter::save(hid_t &groupHandle) const {
          std::vector<std::size_t> name;
          name.push_back(this->getLossId());
          marray::hdf5::save(groupHandle,"lossId",name);
      }

      template<class GM, class IT1, class IT2>
      double HammingLoss::loss(const GM & gm, IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
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
