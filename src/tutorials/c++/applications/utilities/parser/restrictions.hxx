#ifndef RESTRICTIONS_HXX
#define	RESTRICTIONS_HXX

#include <string>

namespace parser{
  
   
   template<class T>
   struct NoRestriction{
      bool operator()(const T & value)const{
         return true;
      }
      std::string restrictionDescription()const{
         return "-";
      }
      std::string restrictionDescription2()const{
         return "-";
      }
      bool canBeRestricted()const{return false;}
   };
   
   template<class T>
   struct RestrictedToAllowedValues{
      RestrictedToAllowedValues(const std::vector<T> & allowedValues = std::vector<T>()):allowedValues_(allowedValues){}
      bool operator()(const T & value)const{
         if(allowedValues_.size()!=0){
            for(size_t i=0;i<allowedValues_.size();++i){
               if(allowedValues_[i]==value){
                  return true;
               }
            }
            return false;
         }
         else{
            return true;
         }
      }
      
      std::string restrictionDescription()const{
         if(allowedValues_.size()!=0){
         std::string desc="allowed values =";
         for(size_t i=0;i<allowedValues_.size();++i){
            std::stringstream ss;
            std::string tmp;
            Stringify<T>::toString(allowedValues_[i],tmp);
            ss<<tmp;
            desc.append( (i==0?std::string(" "):std::string(" | "))+ qoute(colorString(ss.str(),Purple) ));
         }
         return desc;
         }
         else{
            return "-";
         }
      }
      std::string restrictionDescription2()const{
         if(allowedValues_.size()!=0){
         std::string desc="";
         for(size_t i=0;i<allowedValues_.size();++i){
            std::stringstream ss;
            std::string tmp;
             Stringify<T>::toString(allowedValues_[i],tmp);
            ss<<tmp;
            desc.append( (i==0?std::string(" "):std::string(" | "))+qoute(colorString(ss.str(),Purple)));
         }
         return desc;
         }
         else{
            return "-";
         }
      }
      bool canBeRestricted()const{return allowedValues_.size()!=0 ? true :false ;}
      
      bool checkBound(const T & val){
         if(lowerBoundCheck_){
            if(lowerEqual_)
               return lowerBound_>= val;
            else
               return lowerBound_> val;
         }
         if(upperBoundCheck_){
            if(upperEqual_)
               return upperBound_<= val;
            else
               return upperBound_< val;
         }
      }
      std::vector<T> allowedValues_;
      T lowerBound_, upperBound_;
      bool lowerBoundCheck_, upperBoundCheck_,lowerEqual_,upperEqual_;
   };
}
#endif	/* RESTRICTIONS_HXX */

