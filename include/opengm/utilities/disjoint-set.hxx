#pragma once
#ifndef OPENGM_DISJOINT_SET_HXX
#define OPENGM_DISJOINT_SET_HXX

#include <map>
#include <vector>
#include <string>
#include <iostream>

namespace opengm{
  
  template<class T= size_t>
  class disjoint_set{
    
  public:
    
    typedef struct{
      T rank;
      T p;
      T size;
    }elem;
    
    disjoint_set(T);
    
    T find(T);
    void join(T,T);
    T size(T) ; 
    T numberOfSets() const;
    void representativeLabeling(std::map<T, T>&) ;
    
  private:
    
    elem *elements_;
    T numberOfElements_;
    T numberOfSets_;
    
  };
  // end Class
  
  template<class T>
  T disjoint_set<T>::size(T i) {
    i = find(i);
    return elements_[i].size;
  }
  
  template<class T>
  disjoint_set<T>::disjoint_set(T numberOfElements){
    
    elements_ = new elem[numberOfElements];
    numberOfElements_ = numberOfElements;
    numberOfSets_ = numberOfElements;
    for(T i=0;i < numberOfElements;++i){
      elements_[i].rank = 0;
      elements_[i].size=1;
      elements_[i].p = i;
    }
  }
  
  template<class T>
  T disjoint_set<T>::find(T x){
    T y = x;
    while(y != elements_[y].p){
      y=elements_[y].p;
    }
  elements_[x].p = y;
  return y;
}

template<class T>
void disjoint_set<T>::join(T x,T y){
  
  x = find(x);
  y = find(y);
  
  if(x!=y){
    
    if(elements_[x].rank > elements_[y].rank){
      elements_[y].p = x;
      elements_[x].size += elements_[y].size;
    } 
    else {
      elements_[x].p = y;
      elements_[y].size += elements_[x].size;
      if(elements_[x].rank == elements_[y].rank){
        elements_[y].rank++;
      }
    }
    numberOfSets_--;
    
  }
  
}

  template<class T>
  T disjoint_set<T>::numberOfSets() const{
    return numberOfSets_;
  }
  
  template<class T>
  void disjoint_set<T>::representativeLabeling(std::map<T,T>& repL) {
    
    repL.clear();
    T n=0;
    for(T i=0;i<numberOfElements_;++i){
      T x = find(i);
      if(i==x){
        repL[i]=n;
        n++;
      }
    }
  }
  
}


#endif