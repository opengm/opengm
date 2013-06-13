#pragma once
#ifndef OPENGM_MODELTREES_HXX
#define OPENGM_MODELTRESS_HXX

#include <string>
#include <iostream>
#include <vector>
#include <set>
#include <map>

#include "opengm/functions/scaled_view.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
//#include "opengm/graphicalmodel/graphicalmodel_hdf5.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/utilities/random.hxx"
#include "opengm/datastructures/partition.hxx"

#include <boost/lexical_cast.hpp>


namespace opengm{

  template<class GM>
  class modelTrees{
    
  public:
    
    typedef GM GmType;
    typedef typename GM::ValueType ValueType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::LabelType LabelType;
    typedef typename GM::OperatorType OperatorType;
    
    modelTrees(const GmType&);
    IndexType numberOfTrees() const;
    IndexType treeOfVariable(IndexType i); //Root index if Variable is in a Tree, numberOfVariables if not
    IndexType parentOfVariable(IndexType i); //Parent index if Variable is in a Tree, numberOfVariables if not
    IndexType treeOfRoot(IndexType i);
    void roots(std::vector<IndexType>&);
    void nodes(std::vector<std::vector<IndexType> >&);
    
  private:
    
    const GmType& gm_;
    // Partition Trees_;
    std::map<IndexType, IndexType> representives_;
    std::vector<IndexType> parents_;
    std::vector<bool> b_roots_;
    IndexType numberOfRoots_;
    
  };
  // end class

  
  // Constructor
  template<class GM>
  modelTrees<GM>::modelTrees(const GmType& gm) 
  : 
  gm_(gm)
  {
    std::vector<std::set<IndexType> > neighbors;
    gm_.variableAdjacencyList(neighbors);

    std::vector<IndexType> degree(gm_.numberOfVariables());
    std::vector<IndexType> leafs;
    b_roots_.resize(gm_.numberOfVariables());
    parents_.resize(gm_.numberOfVariables());
    typename std::set<typename GM::IndexType>::iterator it;
    typename std::set<typename GM::IndexType>::iterator fi;
    
    for(IndexType i=0;i<degree.size();++i){
      degree[i]=neighbors[i].size();
      parents_[i]=gm_.numberOfVariables();
      if(degree[i]==1){
        leafs.push_back(i);
      }
    }
    while(!leafs.empty()){
      IndexType l=leafs.back();
      leafs.pop_back();
      if(degree[l]>0){
        it=neighbors[l].begin();
        b_roots_[*it]=1;
        b_roots_[l]=0;
        parents_[l]=*it;
        parents_[*it]=*it;
        degree[*it]=degree[*it]-1;
        fi=neighbors[*it].find(l);
        neighbors[*it].erase(fi);
        if(degree[*it]==1){
          leafs.push_back(*it);
        }
      }
    }
    
    numberOfRoots_=0;    
    for(IndexType i=0;i<gm_.numberOfVariables();++i){
      if(b_roots_[i]==1){
        representives_[i]=numberOfRoots_;
        numberOfRoots_++;
      }
    }
    
  }
  
  template<class GM>
  inline
  typename GM::IndexType modelTrees<GM>::numberOfTrees() const{
    return numberOfRoots_;
  }
  
  template<class GM>
  inline
  typename GM::IndexType modelTrees<GM>::treeOfVariable(IndexType i){
    if(parents_[i]==gm_.numberOfVariables()){
      return gm_.numberOfVariables();
    }
    else{
      IndexType r=i;
      while(parents_[r]!=r){
        r=parents_[r];        
      }
      return r;
    }
  }

  template<class GM>
  inline
  void modelTrees<GM>::roots(std::vector<IndexType>& roots){
    roots.resize(numberOfRoots_);
    IndexType j=0;
    for(IndexType i=0;i<gm_.numberOfVariables();++i){
      if(b_roots_[i]==1){
        roots[j]=i;
        j++;
      }
    }
    
  }

  template<class GM>
  inline
  typename GM::IndexType modelTrees<GM>::parentOfVariable(IndexType i){
    if(parents_[i]==gm_.numberOfVariables()){
      return gm_.numberOfVariables();
    }
    else{
      return parents_[i];
    }
  }

  template<class GM>
  inline
  void modelTrees<GM>::nodes(std::vector<std::vector<IndexType> >& nodes){
    
    nodes.resize(numberOfRoots_);
    for(IndexType i=0;i<gm_.numberOfVariables();++i){
      if(parents_[i]!=gm_.numberOfVariables()){
        IndexType treeID = representives_[treeOfVariable(i)];
        nodes[treeID].push_back(i);
      }
    }
  }
  template<class GM>
  inline
  typename GM::IndexType modelTrees<GM>::treeOfRoot(IndexType i){
    if(parents_[i]==gm_.numberOfVariables()){
      return gm_.numberOfVariables();
    }
    else{
      return representives_[treeOfVariable(i)];
    }
  }
  
}

#endif
