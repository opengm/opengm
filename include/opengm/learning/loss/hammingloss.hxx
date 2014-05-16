#include "opengm/functions/explicit_function.hxx"


class HammingLoss{
public:

   template<class IT1, class IT2>
   double loss(IT1 labelBegin, IT1 labelEnd, IT2 GTBegin,IT2 GTEnd) const;
  
   template<class GM, class IT>
   void addLoss(GM& gm, IT GTBegin) const;
private:
};

template<class IT1, class IT2>
double HammingLoss::loss(IT1 labelBegin, const IT1 labelEnd, IT2 GTBegin, const IT2 GTEnd) const
{
   double loss = 0;
   for(; labelBegin!= labelEnd; ++labelBegin, ++GTBegin){
      if(*labelBegin != *GTBegin){
         loss += 1.0;
      }
   }
}

template<class GM, class IT>
void HammingLoss::addLoss(GM& gm, IT GTBegin) const
{

   for(typename GM::IndexType i=0; i<gm.numberOfVariables(); ++i){
      typename GM::LabelType numL = gm.numberOFLabels(i);
      //todo add functions and factors
   }
}
  
