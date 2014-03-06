#pragma once
#ifndef OPENGM_PARTITIONMOVE_HXX
#define OPENGM_PARTITIONMOVE_HXX

#include <algorithm>
#include <vector>
#include <queue>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <limits> 
#ifdef WITH_BOOST
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>		
#else
#include <ext/hash_map> 
#include <ext/hash_set>
#endif

#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

/// \brief Partition Move\n\n
/// Currently Partition Move only implements the Kernighan-Lin-Algorithm
///
/// - Cite: B.W. Kernighan and S. Lin, "An efficent heuristic procedure for partition graphs", 1970
/// - Maximum factor order : second order Potts functions only!
/// - Maximum number of labels : same as the number of variables !
/// - Restrictions : see above
/// - Convergent :   Converge to some local fix point
///
/// \ingroup inference 
template<class GM, class ACC>
class PartitionMove : public Inference<GM, ACC>
{
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef size_t LPIndexType;
   typedef visitors::VerboseVisitor<PartitionMove<GM, ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<PartitionMove<GM, ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<PartitionMove<GM, ACC> >  TimingVisitorType;
#ifdef WITH_BOOST 
   typedef boost::unordered_map<IndexType, LPIndexType> EdgeMapType;
   typedef boost::unordered_set<IndexType>             VariableSetType; 
#else
   typedef __gnu_cxx::hash_map<IndexType, LPIndexType> EdgeMapType;
   typedef __gnu_cxx::hash_set<IndexType>              VariableSetType; 
#endif
 

   struct Parameter{
     Parameter ( ) {};
   };

   ~PartitionMove();
   PartitionMove(const GraphicalModelType&, Parameter para=Parameter());
   virtual std::string name() const {return "PartitionMove";}
   const GraphicalModelType& graphicalModel() const {return gm_;}
   virtual InferenceTermination infer();
   template<class VisitorType> InferenceTermination infer(VisitorType&);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;

private:
   enum ProblemType {INVALID, MC, MWC};

   const GraphicalModelType& gm_; 
   ProblemType problemType_;
   Parameter parameter_;
  
   LabelType   numberOfTerminals_;
   LPIndexType numberOfInternalEdges_;
 
 
   /// For each variable it contains a map indexed by neighbord nodes giving the index to the LP-variable
   /// e.g. neighbours[a][b] = i means a has the neighbour b and the edge has the index i in the linear objective
   std::vector<EdgeMapType >                       neighbours_; 
   std::vector<double>                             edgeWeight_;
   double                                          constant_;
   std::vector<LabelType>                          states_;

   template<class VisitorType> InferenceTermination inferKL(VisitorType&);
   double solveBinaryKL(VariableSetType&, VariableSetType&);
 
};
 

template<class GM, class ACC>
PartitionMove<GM, ACC>::~PartitionMove() {
   ;
}

template<class GM, class ACC>
PartitionMove<GM, ACC>::PartitionMove
(
   const GraphicalModelType& gm,
   Parameter para
   ) : gm_(gm), parameter_(para)
{
   if(typeid(ACC) != typeid(opengm::Minimizer) || typeid(OperatorType) != typeid(opengm::Adder)) {
      throw RuntimeError("This implementation does only supports Min-Plus-Semiring.");
   } 


   //Set Problem Type
   problemType_           = MC;
   numberOfInternalEdges_ = 0;
   numberOfTerminals_     = gm_.numberOfLabels(0); 
   for(size_t i=0; i<gm_.numberOfVariables(); ++i){
      if(gm_.numberOfLabels(i)<gm_.numberOfVariables()) {
         problemType_ = MWC;
         numberOfTerminals_ = std::max(numberOfTerminals_ ,gm_.numberOfLabels(i));
      }
   }
   for(size_t f=0; f<gm_.numberOfFactors();++f) {
      if(gm_[f].numberOfVariables()==0) {
         continue;
      }
      else if(gm_[f].numberOfVariables()==1) {
         problemType_ = MWC;
      }
      else if(gm_[f].numberOfVariables()==2) {
         ++numberOfInternalEdges_;
         if(!gm_[f].isPotts()) {
            problemType_ = INVALID;
            break;
         }
      }
      else{ 
         problemType_ = INVALID;
         break;
      }
   } 
  
   if(problemType_ == INVALID)
      throw RuntimeError("Invalid Model for Multicut-Solver! Solver requires a potts model!");
   if(problemType_ == MWC)
      throw RuntimeError("Invalid Model for Multicut-Solver! Solver currently do not support first order terms!");


   //Calculate Neighbourhood
   neighbours_.resize(gm_.numberOfVariables());
   edgeWeight_.resize(numberOfInternalEdges_,0);
   constant_=0;
   LPIndexType numberOfInternalEdges=0;
   // Add edges that have to be included
   for(size_t f=0; f<gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables()==0) {
         const LabelType l=0;
         constant_+=gm_[f](&l); 
      }
      else if(gm_[f].numberOfVariables()==2) {
         LabelType cc0[] = {0,0};
         LabelType cc1[] = {0,1};
         edgeWeight_[numberOfInternalEdges] += gm_[f](cc1) - gm_[f](cc0); 
         constant_ += gm_[f](cc0);
         IndexType u = gm_[f].variableIndex(0);
         IndexType v = gm_[f].variableIndex(1);
         neighbours_[u][v] = numberOfInternalEdges;
         neighbours_[v][u] = numberOfInternalEdges;
         ++numberOfInternalEdges;
      }    
      else{
         throw RuntimeError("Only supports second order Potts functions!");
      }
   }
   OPENGM_ASSERT(numberOfInternalEdges==numberOfInternalEdges_);

   states_.resize(gm_.numberOfVariables(),0);
   size_t init = 2;  

   if(init==1){
      for(size_t i=0; i<states_.size();++i){
         states_[i]=rand()%10;
      }
   }

   if(init==2){
      LabelType p=0;
      std::vector<bool> assigned(states_.size(),false);
      for(IndexType node=0; node<states_.size(); ++node) {
         if(assigned[node])
            continue;
         else{
            std::list<IndexType> nodeList;
            states_[node]  = p;
            assigned[node] = true;
            nodeList.push_back(node);
            while(!nodeList.empty()) {
               size_t n=nodeList.front(); nodeList.pop_front();
               for(typename EdgeMapType::const_iterator it=neighbours_[n].begin() ; it != neighbours_[n].end(); ++it) {
                  const IndexType node2 = (*it).first; 
                  if(!assigned[node2] && edgeWeight_[(*it).second]>0) {
                     states_[node2] = p;
                     assigned[node2] = true;
                     nodeList.push_back(node2);
                  }
               }
            }
            ++p;
         }
      }
   }

   if(init==3){
      for(size_t i=0; i<states_.size();++i){
         states_[i]=i;
      }
   }
   
 
}


template <class GM, class ACC>
InferenceTermination
PartitionMove<GM,ACC>::infer()
{
   EmptyVisitorType visitor;
   return infer(visitor);
}


template <class GM, class ACC>
template<class VisitorType>
InferenceTermination
PartitionMove<GM,ACC>::infer(VisitorType& visitor)
{ 
   visitor.begin(*this);
   inferKL(visitor);
   visitor.end(*this);
   return NORMAL;
}

template <class GM, class ACC>
template<class VisitorType>
InferenceTermination
PartitionMove<GM,ACC>::inferKL(VisitorType& visitor)
{
   // Current Partition-Sets
   std::vector<VariableSetType> partitionSets;

   // Set-Up Partition-Sets from current/initial partitioning
   LabelType numberOfPartitions =0;
   for(size_t i=0; i<states_.size(); ++i)
      if(states_[i]+1>numberOfPartitions) numberOfPartitions=states_[i]+1;
   partitionSets.resize(numberOfPartitions);
   for(IndexType i=0; i<states_.size(); ++i){
      partitionSets[states_[i]].insert(i);
   }

   bool change = true;
   while(change){
      // std::cout << numberOfPartitions << " conncted subsets."<<std::endl;
      change = false;
      std::vector<size_t> pruneSets;
      // Check all pairs of partitions
      for(size_t part0=0; part0<numberOfPartitions; ++part0){
         //std::cout <<"*"<<std::flush;
         // Find neighbord sets
         std::set<size_t> neighbordSets;
         for(typename VariableSetType::const_iterator it=partitionSets[part0].begin(); it!=partitionSets[part0].end(); ++it){
            const IndexType node = (*it);
            for(typename EdgeMapType::const_iterator nit=neighbours_[node].begin() ; nit != neighbours_[node].end(); ++nit) {
                 const IndexType node2 = (*nit).first;
                 if(states_[node2]>part0){
                    neighbordSets.insert(states_[node2]);
                 }
            }
         } 
         for(std::set<size_t>::const_iterator it=neighbordSets.begin(); it!=neighbordSets.end();++it){
            size_t part1 = *it;
            //for(size_t part1=part0+1; part1<numberOfPartitions; ++part1){
            if(partitionSets[part0].size()==0 || partitionSets[part1].size()==0)
               continue;
            double improvement = solveBinaryKL(partitionSets[part0],partitionSets[part1]);
            //std::cout <<part0<<" vs "<<part1<<" : " <<improvement<<std::endl;
            OPENGM_ASSERT(improvement<1e-8);
            if(-1e-8>improvement){
               change = true; // Partition has been improved  
            }
         }
      } 
      // Check for each Partition ...
      for(size_t part0=0; part0<numberOfPartitions; ++part0){
         // ... if it is empty and can be pruned
         if(partitionSets[part0].size()==0){
            //std::cout <<"Remove "<<part0<<std::endl;
            pruneSets.push_back(part0);
         }
         // ... or if it can be splited into two sets
         else if(partitionSets[part0].size()>1){
            // std::cout <<part0<<" vs "<<"NULL"<<std::endl;
          
            VariableSetType emptySet(partitionSets[part0].size());
            double improvement = solveBinaryKL(partitionSets[part0], emptySet);
            if(emptySet.size()>0){
               OPENGM_ASSERT(improvement<0);
               partitionSets.push_back(emptySet);
               change = true;
            }
         }
      }
      // Remove sets marked as to prune
      //std::cout << "Remove " <<pruneSets.size() << " subsets."<<std::endl;
      for(size_t i=0; i<pruneSets.size(); ++i){
         size_t part = pruneSets[pruneSets.size()-1-i];
         partitionSets.erase( partitionSets.begin()+part);
      }
      // Update Labeling
      numberOfPartitions = partitionSets.size();
      for(size_t part=0; part<numberOfPartitions; ++part){
         for(typename VariableSetType::const_iterator it=partitionSets[part].begin(); it!=partitionSets[part].end(); ++it){
            states_[*it] = part;
         }
      }
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
         change = false;
      }
   }
   return NORMAL;
}

template <class GM, class ACC>
double PartitionMove<GM,ACC>::solveBinaryKL
(
   VariableSetType& set0, 
   VariableSetType& set1
)
{
   double improvement = 0.0;
   //std::cout << "Set0: "<< set0.size() <<" Set1: "<< set1.size() << std::endl; 

   for(size_t outerIt=0; outerIt<100;++outerIt){ 
      // Compute D[n] = E_n - I_n
      std::vector<double> D(gm_.numberOfVariables(),0);
      for(typename VariableSetType::const_iterator it=set0.begin(); it!=set0.end(); ++it){ 
         double E_a = 0.0;
         double I_a = 0.0;
         const IndexType node = *it;
         for (typename EdgeMapType::const_iterator eit=neighbours_[node].begin(); eit!=neighbours_[node].end(); ++eit){
            const IndexType node2 = (*eit).first;
            const double weight = edgeWeight_[(*eit).second];

            if (set0.find(node2) != set0.end()) {
                I_a += weight;
            } 
            else if(set1.find(node2) != set1.end()){
               E_a += weight;
            }
         }
         D[node] = -(E_a - I_a);
      }
      for(typename VariableSetType::const_iterator it=set1.begin(); it!=set1.end(); ++it){ 
         double E_a = 0.0;
         double I_a = 0.0;
         const IndexType node = *it;
         for(typename EdgeMapType::const_iterator eit=neighbours_[node].begin(); eit!=neighbours_[node].end(); ++eit){
            const IndexType node2 = (*eit).first;
            const double weight = edgeWeight_[(*eit).second];
            
            if (set1.find(node2) != set1.end()) {
                I_a += weight;
            } 
            else if(set0.find(node2) != set0.end()){
               E_a += weight;
            }
         }
         D[node] = -(E_a - I_a);
      }

      double d=0;
      for(size_t i=0; i<D.size(); ++i){
         if(D[i]<d)
            d=D[i];
      }
    

      // Search a gready move sequence
      std::vector<bool>      isMovedNode(gm_.numberOfVariables(),false);
      std::vector<IndexType> nodeSequence;
      std::vector<double>    improveSequence;
      std::vector<double>    improveSumSequence(1,0.0);
      size_t                 bestMove=0;
       
      // Build sequence of greedy best moves
      for(size_t innerIt=0; innerIt<1000; ++innerIt){
         double    improve = std::numeric_limits<double>::infinity();
         IndexType node;
         bool      moved = false;
         // Search over moves from set0
         for(typename VariableSetType::const_iterator it=set0.begin(); it!=set0.end(); ++it){
            if(isMovedNode[*it]){
               continue;
            }
            else{
               if(D[*it]<improve){
                  improve = D[*it];
                  node    = *it;
                  moved   = true;
               }
            }  
         }
         // Search over moves from set1
         for(typename VariableSetType::const_iterator it=set1.begin(); it!=set1.end(); ++it){
            if(isMovedNode[*it]){
               continue;
            }
            else{
               if(D[*it]<improve){
                  improve = D[*it];
                  node    = *it;
                  moved   = true;
               }
            }  
         }

         // No more moves?
         if(moved == false){
            break;
         }
         
         // Move node and recalculate D
         //std::cout << " " <<improveSumSequence.back()+improve;
         isMovedNode[node]=true;
         nodeSequence.push_back(node);
         improveSumSequence.push_back(improveSumSequence.back()+improve);
         improveSequence.push_back(improve);
         if (improveSumSequence[bestMove]>improveSumSequence.back()) {
            bestMove = improveSumSequence.size()-1;
         }
 
         VariableSetType& mySet = set0.find(node) != set0.end() ? set0 : set1;
         for(typename EdgeMapType::const_iterator eit=neighbours_[node].begin(); eit!=neighbours_[node].end(); ++eit){
            IndexType node2  = (*eit).first;
            double    weight = edgeWeight_[(*eit).second]; 
            if(mySet.find(node2) != mySet.end()){
               D[node2] -= 2.0 * weight;
            }
            else{
               D[node2] += 2.0 * weight;
            }

         }   
      }
        
       // Perform Move
      if(improveSumSequence[bestMove]>-1e-10)
         break;
      else{ 
         improvement += improveSumSequence[bestMove];
         for (size_t i = 0; i < bestMove; ++i) {
            int node = nodeSequence[i];
            if (set0.find(node) != set0.end()) {
               set0.erase(node);
               set1.insert(node);
            }
            else{
               set1.erase(node);
               set0.insert(node);
            }
         }
      }
      // Search for the next move if this move has give improvement
   }
   return improvement;
}

template <class GM, class ACC>
InferenceTermination
PartitionMove<GM,ACC>::arg
(
   std::vector<typename PartitionMove<GM,ACC>::LabelType>& x,
   const size_t N
   ) const
{
   if(N!=1) {
      return UNKNOWN;
   }
   else{
      x.resize(gm_.numberOfVariables());
      for(size_t i=0; i<gm_.numberOfVariables(); ++i)
         x[i] = states_[i];
      return NORMAL;
   }
}

} // end namespace opengm

#endif
