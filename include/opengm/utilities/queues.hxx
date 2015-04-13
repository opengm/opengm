
#ifndef OPENGM_PRIORITY_QUEUE_HXX
#define OPENGM_PRIORITY_QUEUE_HXX

#include <queue>
#include <vector>

namespace opengm{

template<class T>
class FiFoQueue {
private:
   size_t begin_;
   size_t end_;
   size_t maxSize_; 
   std::vector<T> data_;
public:
   typedef T ValueType;
   FiFoQueue(const size_t maxSize) : begin_(0), end_(0), maxSize_(maxSize)
   {
      data_.resize(maxSize);
   }  

   T front(){
      return data_[begin_];
   }

   void pop(){
      ++begin_;
   }

   void push(T t){ 
      data_[end_]=t;
      ++end_;
      //if(end_==maxSize_){
      //   end_=0;
      //}
      // assert(end_ != begin_); // larger tan maxSize
   }

   bool empty(){
      return begin_==end_;
   }

   size_t size(){
      if(begin_<=end_)
         return end_-begin_;
      else
         return end_+maxSize_-1-begin_;
   }

   void clear(){
      begin_ = 0;
      end_   = 0;
   }

};



/** \brief Heap-based changable priority queue with a maximum number of elemements.

    This pq allows to change the priorities of elements in the queue
*/
template<class T,class COMPARE = std::less<T> >
class ChangeablePriorityQueue {


public:

    typedef T priority_type;
    typedef int ValueType;
    typedef ValueType value_type;
    typedef ValueType const_reference;



    /// Create an empty ChangeablePriorityQueue which can contain atmost maxSize elements
    ChangeablePriorityQueue(const size_t maxSize)  
    : maxSize_(maxSize),
      currentSize_(0),
      heap_(maxSize_+1),
      indices_(maxSize_+1),
      priorities_(maxSize_+1)
    {
        for(int i = 0; i <= maxSize_; i++)
            indices_[i] = -1;
    }

    /// set all priorities to the given value
    void setPriorities(T newPriority){
      for(typename std::vector<T>::iterator it = priorities_.begin(); it !=priorities_.end(); ++it)
         *it = newPriority;
      return;
    }
   
    /// reset heap - priorities are not changed
    void reset(){
       for(size_t i=0; i<=currentSize_; ++i)
          indices_[heap_[i]] = -1;
       currentSize_ = 0;
    }
 
    /// check if the PQ is empty
    bool empty() const {
        return currentSize_ == 0;
    }
 
    /// check if i is an index on the PQ
    bool contains(const int i) const{
        return indices_[i] != -1;
    }
 
    /// return the number of elements in the PQ
    int size()const{
        return currentSize_;
    }
 

    /** \brief Insert a index with a given priority.

        If the queue contains i before this 
        call the priority of the given index will
        be changed
    */
    void push(const value_type i, const priority_type p) {
        if(!contains(i)){
            currentSize_++;
            indices_[i] = currentSize_;
            heap_[currentSize_] = i;
            priorities_[i] = p;
            bubbleUp(currentSize_);
        }
        else{
            changePriority(i,p);
        }
    }
 
    /** \brief get index with top priority
    */
    const_reference top() const {
        return heap_[1];
    }
 
    /**\brief get top priority
    */
    priority_type topPriority() const {
        return priorities_[heap_[1]];
    }
 
    /** \brief Remove the current top element.
    */
    void pop() {
        const int min = heap_[1];
        swapItems(1, currentSize_--);
        bubbleDown(1);
        indices_[min] = -1;
        heap_[currentSize_+1] = -1;
    }
 
    /// returns the value associated with index i
    priority_type priority(const value_type i) const{
        return priorities_[i];
    }
  
    /// deleqte the priority associated with index i
    void deleteItem(const value_type i)   {
        int ind = indices_[i];
        swapItems(ind, currentSize_--);
        bubbleUp(ind);
        bubbleDown(ind);
        indices_[i] = -1;
    }
    /** \brief change priority of a given index.
        The index must be in the queue!
        Call push to auto insert / change .
    */
    void changePriority(const value_type i,const priority_type p)  {
        if(_gt(p,priorities_[i])){
            priorities_[i] = p;
            bubbleDown(indices_[i]);
        }
        else if(_lt(p,priorities_[i])) {
            priorities_[i] = p;
            bubbleUp(indices_[i]);
        }
    }
private:
    

    void swapItems(const int i,const  int j) {
        std::swap(heap_[i],heap_[j]);
        indices_[heap_[i]] = i; 
        indices_[heap_[j]] = j;
    }
 
    void bubbleUp(int k)    {
        while(k > 1 && _gt( priorities_[heap_[k/2]],priorities_[heap_[k]]))   {
            swapItems(k, k/2);
            k = k/2;
        }
    }
 
    void bubbleDown(int k)  {
        int j;
        while(2*k <= currentSize_) {
            j = 2*k;
            if(j < currentSize_ && _gt(priorities_[heap_[j]] , priorities_[heap_[j+1]]) )
                j++;
            if( _leqt(priorities_[heap_[k]] , priorities_[heap_[j]]))
                break;
            swapItems(k, j);
            k = j;
        }
    }


    bool _lt(const T & a,const T & b)const{
        return comp_(a,b);
    }
    bool _leqt(const T & a,const T & b)const{
        return !comp_(b,a);
    }
    bool _eq(const T & a,const T & b)const{
        return !comp_(a,b) && !comp_(b,a);
    }
    bool _gt(const T & a,const T & b)const{
        return !_eq(a,b) && !comp_(a,b);
    }
    bool _geqt(const T & a,const T & b)const{
        return !comp_(a,b);
    }
 

    size_t maxSize_;
    size_t currentSize_;
    std::vector<int> heap_;
    std::vector<int> indices_;
    std::vector<T>   priorities_;
    COMPARE          comp_;

};

}

#endif
