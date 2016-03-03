#pragma once
#ifndef OPENGM_PARTITIONS_HXX
#define OPENGM_PARTITIONS_HXX

#include <vector>
#include <algorithm>  

namespace opengm {

/// Enumaration of partitions of a set of N nodes
   template<class I=size_t, class L=size_t>
   class Partitions {
   public:
      typedef I EdgeLabelType;
      typedef L NodeLabelType;

      void   resize(size_t N)             { buildPartitions(N);}
      size_t BellNumber(size_t N)         { return Bell[N]; } 
      EdgeLabelType getPartition(size_t n){ return partitions[n];}

      template<class T>
      void  getPartition(const size_t n, std::vector<T>& l){
         const EdgeLabelType el = getPartition(n);
         const size_t N = l.size();
         size_t base=1;
         l[0] = 0;
         for(size_t v1=1; v1<N; ++v1){
            l[v1]=v1;
            for(size_t v2=0; v2<v1; ++v2){
               if( (el & base) == base){
                  l[v1] = l[v2];
               }
               base *= 2;
            }
         }  
      }
     
      size_t number2Index(const EdgeLabelType el){
         typename std::vector<EdgeLabelType>::iterator it;
         it = find(partitions.begin(), partitions.end(), el);

         if(it == partitions.end() )
            return -1;
         else
            //return (size_t)(it-partitions.begin())/sizeof(EdgeLabelType);
            return std::distance(partitions.begin(),it);
      }

      size_t label2Index(const std::vector<NodeLabelType>& l){ 
         buildPartitions(l.size());
         EdgeLabelType el = label2Number(l);
         return number2Index(el);
      }
 
      template<class IT>
      size_t label2Index(const IT begin, const size_t order){ 
         buildPartitions(order);
         EdgeLabelType el = label2Number(begin,order); 
         return number2Index(el);
      }
  
      EdgeLabelType label2Number(const std::vector<NodeLabelType>& l){
         EdgeLabelType indicator = 0;
         EdgeLabelType base      = 1;
         
         for(size_t v1=1; v1<l.size(); ++v1){
            for(size_t v2=0; v2<v1; ++v2){
               indicator += base * (l[v1] == l[v2]);
               base*=2;
            }
         }
         return indicator;
      }
      
      template<class IT>
      EdgeLabelType label2Number(IT begin, size_t order){
         EdgeLabelType indicator = 0;
         EdgeLabelType base      = 1;
         
         for(size_t v1=1; v1<order; ++v1){
            for(size_t v2=0; v2<v1; ++v2){
               indicator += base * (*(begin+v1) == *(begin+v2) );
               base*=2;
            }
         }
         return indicator;
      }

   private:  
      static const size_t Bell[16];// = {1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597, 27644437, 190899322, 1382958545};
      static std::vector<EdgeLabelType> partitions;

      bool increment( std::vector<NodeLabelType>& l){
         size_t N=l.size();
         size_t p=0;
         std::vector<NodeLabelType> maxV(N+1,0);
         for(size_t i=N; i>0 ; --i)
            maxV[i-1] = std::max(maxV[i],l[i-1]);
         
         //find position to increment
         for(p=0; p<=N; ++p){
            if(p==N) return false;
            if(l[p]<N-1-p && ( (p==N-1) || (l[p]<=maxV[p+1] )))
               break;
         } 
         //std::cout<<"X " <<p<<std::endl;
         ++l[p];
         maxV[p] = std::max(maxV[p+1], l[p]);
         //set succesors
         for(size_t q=p; q>0;--q){
            l[q-1]=0;
            maxV[q-1] =  maxV[p];
         } 
         return true;
      }

 
      void buildPartitions(size_t N){
         if(partitions.size() >= Bell[N])
            return;  
         if(N*(N-1)/2>sizeof(I)*8){
            throw std::runtime_error("Error: EdgeIndexType is to small!");
         }

         partitions.clear();
         partitions.reserve(Bell[N]);

         std::vector<NodeLabelType> l(N,0);
         partitions.push_back(label2Number(l));
         while( increment(l) ){
            partitions.push_back( label2Number(l) );
         } 
         //std::cout << "B("<<N<<") = "<<partitions.size()<<" == "<<Bell[N]<<std::endl;
         //assert(partitions.size() == Bell[N]);
         std::sort(partitions.begin(), partitions.end());
         return;
      }     
   };

   template<class I, class L>
   const size_t Partitions<I, L>::Bell[16] = {1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597, 27644437, 190899322, 1382958545};

   template<class I, class L>
   std::vector<I> Partitions<I, L>::partitions;

}
#endif
