#pragma once
#ifndef OPENGM_BUFFER_RANDOM_ACCESS_SET_HXX
#define OPENGM_BUFFER_RANDOM_ACCESS_SET_HXX

#include <vector>
#include <algorithm>
#include <utility>

#include <opengm/datastructures/buffer_vector.hxx>

namespace opengm {
   template<class T>
   class BufferRandomAccessSet{
   public:
      void insert(const T & );
   private:
     BufferVector<T> vector_;
  }; 
} // namespace opengm

#endif // #ifndef OPENGM_BUFFER_RANDOM_ACCESS_SET_HXX