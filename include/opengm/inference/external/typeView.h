#ifndef TYPEVIEW_H_
#define TYPEVIEW_H_

#include <string.h>
#include <assert.h>

#include <limits>

template <class T> class MRFEnergy;

template <class GM>
class TypeView
{
private:
   struct Vector; // node parameters and messages
   struct Edge; // stores edge information and either forward or backward message

public:

   // types declarations
   typedef int Label;
   typedef double REAL;
   struct GlobalSize; // global information about number of labels
   struct LocalSize; // local information about number of labels (stored at each node)
   struct NodeData; // argument to MRFEnergy::AddNode()
   struct EdgeData; // argument to MRFEnergy::AddEdge()


   struct GlobalSize
   {

   };

   struct LocalSize // number of labels is stored at MRFEnergy::m_Kglobal
   {
      LocalSize(int K);

   private:
   friend struct Vector;
   friend struct Edge;
      int      m_K; // number of labels
   };

   struct NodeData
   {
      NodeData(const GM& gm, const std::vector<typename GM::IndexType>& factorIDs);

   private:
      friend struct Vector;
      friend struct Edge;
      const GM& gm_;
      const std::vector<typename GM::IndexType> factorIDs_;
   };

   struct EdgeData
   {

      EdgeData(const GM& gm, const typename GM::IndexType factorID);

   private:
      friend struct Vector;
      friend struct Edge;
      const GM& gm_;
      const typename GM::IndexType factorID_;
   };

   //////////////////////////////////////////////////////////////////////////////////
   ////////////////////////// Visible only to MRFEnergy /////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////

private:
   friend class MRFEnergy<TypeView<GM> >;

   struct Vector
   {
      static int GetSizeInBytes(GlobalSize Kglobal, LocalSize K); // returns -1 if invalid K's
      void Initialize(GlobalSize Kglobal, LocalSize K, NodeData data);  // called once when user adds a node
      void Add(GlobalSize Kglobal, LocalSize K, NodeData data); // called once when user calls MRFEnergy::AddNodeData()

      void SetZero(GlobalSize Kglobal, LocalSize K);                            // set this[k] = 0
      void Copy(GlobalSize Kglobal, LocalSize K, Vector* V);                    // set this[k] = V[k]
      void Add(GlobalSize Kglobal, LocalSize K, Vector* V);                     // set this[k] = this[k] + V[k]
      REAL GetValue(GlobalSize Kglobal, LocalSize K, Label k);                  // return this[k]
      REAL ComputeMin(GlobalSize Kglobal, LocalSize K, Label& kMin);            // return min_k { this[k] }, set kMin
      REAL ComputeAndSubtractMin(GlobalSize Kglobal, LocalSize K);              // same as previous, but additionally set this[k] -= vMin (and kMin is not returned)

      static int GetArraySize(GlobalSize Kglobal, LocalSize K);
      REAL GetArrayValue(GlobalSize Kglobal, LocalSize K, int k); // note: k is an integer in [0..GetArraySize()-1].
                                                                  // For Potts functions GetArrayValue() and GetValue() are the same,
                                                                  // but they are different for, say, 2-dimensional labels.
      void SetArrayValue(GlobalSize Kglobal, LocalSize K, int k, REAL x);

   private:
   friend struct Edge;
      REAL     m_data[1]; // actual size is MRFEnergy::m_Kglobal
   };

   struct Edge
   {
      static int GetSizeInBytes(GlobalSize Kglobal, LocalSize Ki, LocalSize Kj, EdgeData data); // returns -1 if invalid data
      static int GetBufSizeInBytes(int vectorMaxSizeInBytes); // returns size of buffer need for UpdateMessage()
      void Initialize(GlobalSize Kglobal, LocalSize Ki, LocalSize Kj, EdgeData data, Vector* Di, Vector* Dj); // called once when user adds an edge
      Vector* GetMessagePtr();
      void Swap(GlobalSize Kglobal, LocalSize Ki, LocalSize Kj); // if the client calls this function, then the meaning of 'dir'
                                                                       // in distance transform functions is swapped

      // When UpdateMessage() is called, edge contains message from dest to source.
      // The function must replace it with the message from source to dest.
      // The update rule is given below assuming that source corresponds to tail (i) and dest corresponds
      // to head (j) (which is the case if dir==0).
      //
      // 1. Compute Di[ki] = gamma*source[ki] - message[ki].  (Note: message = message from j to i).
      // 2. Compute distance transform: set
      //       message[kj] = min_{ki} (Di[ki] + V(ki,kj)). (Note: message = message from i to j).
      // 3. Compute vMin = min_{kj} m_message[kj].
      // 4. Set m_message[kj] -= vMin.
      // 5. Return vMin.
      //
      // If dir==1 then source corresponds to j, sink corresponds to i. Then the update rule is
      //
      // 1. Compute Dj[kj] = gamma*source[kj] - message[kj].  (Note: message = message from i to j).
      // 2. Compute distance transform: set
      //       message[ki] = min_{kj} (Dj[kj] + V(ki,kj)). (Note: message = message from j to i).
      // 3. Compute vMin = min_{ki} m_message[ki].
      // 4. Set m_message[ki] -= vMin.
      // 5. Return vMin.
      //
      // If Edge::Swap has been called odd number of times, then the meaning of dir is swapped.
      //
      // Vector 'source' must not be modified. Function may use 'buf' as a temporary storage.
      REAL UpdateMessage(GlobalSize Kglobal, LocalSize Ksource, LocalSize Kdest, Vector* source, REAL gamma, int dir, void* buf);

      // If dir==0, then sets dest[kj] += V(ksource,kj).
      // If dir==1, then sets dest[ki] += V(ki,ksource).
      // If Swap() has been called odd number of times, then the meaning of dir is swapped.
      void AddColumn(GlobalSize Kglobal, LocalSize Ksource, LocalSize Kdest, Label ksource, Vector* dest, int dir);

   protected:
      // edge information
      const GM* gm_;
      typename GM::IndexType factorID_;

      int      m_dir; // 0 if Swap() was called even number of times, 1 otherwise

      // message
      Vector*     m_message;
   };

};




//////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Implementation ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template <class GM>
inline TypeView<GM>::LocalSize::LocalSize(int K)
{
   m_K = K;
}

///////////////////// NodeData and EdgeData ///////////////////////

template <class GM>
inline TypeView<GM>::NodeData::NodeData(const GM& gm, const std::vector<typename GM::IndexType>& factorIDs) : gm_(gm), factorIDs_(factorIDs)
{

}

template <class GM>
inline TypeView<GM>::EdgeData::EdgeData(const GM& gm, const typename GM::IndexType factorID) : gm_(gm), factorID_(factorID)
{

}

///////////////////// Vector ///////////////////////

template <class GM>
inline int TypeView<GM>::Vector::GetSizeInBytes(GlobalSize Kglobal, LocalSize K)
{
   if (K.m_K < 1)
   {
      return -1;
   }
   return K.m_K*sizeof(REAL);
}

template <class GM>
inline void TypeView<GM>::Vector::Initialize(GlobalSize Kglobal, LocalSize K, NodeData data)
{
   for (int k=0; k<K.m_K; k++) {
      m_data[k] = 0.0;
      for(typename std::vector<typename GM::IndexType>::const_iterator iter = data.factorIDs_.begin(); iter != data.factorIDs_.end(); iter++) {
         m_data[k] += data.gm_[*iter](&k);
      }
   }
}

template <class GM>
inline void TypeView<GM>::Vector::Add(GlobalSize Kglobal, LocalSize K, NodeData data)
{
   for(typename std::vector<typename GM::IndexType>::const_iterator iter = data.factorIDs_.begin(); iter != data.factorIDs_.end(); iter++) {
      for (int k=0; k<K.m_K; k++)
      {
         m_data[k] += data.gm_[*iter](&k);
      }
   }
}

template <class GM>
inline void TypeView<GM>::Vector::SetZero(GlobalSize Kglobal, LocalSize K)
{
   memset(m_data, 0, K.m_K*sizeof(REAL));
}

template <class GM>
inline void TypeView<GM>::Vector::Copy(GlobalSize Kglobal, LocalSize K, Vector* V)
{
   memcpy(m_data, V->m_data, K.m_K*sizeof(REAL));
}

template <class GM>
inline void TypeView<GM>::Vector::Add(GlobalSize Kglobal, LocalSize K, Vector* V)
{
   for (int k=0; k<K.m_K; k++)
   {
      m_data[k] += V->m_data[k];
   }
}

template <class GM>
inline typename TypeView<GM>::REAL TypeView<GM>::Vector::GetValue(GlobalSize Kglobal, LocalSize K, Label k)
{
   assert(k>=0 && k<K.m_K);
   return m_data[k];
}

template <class GM>
inline typename TypeView<GM>::REAL TypeView<GM>::Vector::ComputeMin(GlobalSize Kglobal, LocalSize K, Label& kMin)
{
   REAL vMin = m_data[0];
   kMin = 0;
   for (int k=1; k<K.m_K; k++)
   {
      if (vMin > m_data[k])
      {
         vMin = m_data[k];
         kMin = k;
      }
   }

   return vMin;
}

template <class GM>
inline typename TypeView<GM>::REAL TypeView<GM>::Vector::ComputeAndSubtractMin(GlobalSize Kglobal, LocalSize K)
{
   REAL vMin = m_data[0];
   for (int k=1; k<K.m_K; k++)
   {
      if (vMin > m_data[k])
      {
         vMin = m_data[k];
      }
   }
   for (int k=0; k<K.m_K; k++)
   {
      m_data[k] -= vMin;
   }

   return vMin;
}

template <class GM>
inline int TypeView<GM>::Vector::GetArraySize(GlobalSize Kglobal, LocalSize K)
{
   return K.m_K;
}

template <class GM>
inline typename TypeView<GM>::REAL TypeView<GM>::Vector::GetArrayValue(GlobalSize Kglobal, LocalSize K, int k)
{
   assert(k>=0 && k<K.m_K);
   return m_data[k];
}

template <class GM>
inline void TypeView<GM>::Vector::SetArrayValue(GlobalSize Kglobal, LocalSize K, int k, REAL x)
{
   assert(k>=0 && k<K.m_K);
   m_data[k] = x;
}

///////////////////// EdgeDataAndMessage implementation /////////////////////////

template <class GM>
inline int TypeView<GM>::Edge::GetSizeInBytes(GlobalSize Kglobal, LocalSize Ki, LocalSize Kj, EdgeData data)
{
   int messageSizeInBytes = ((Ki.m_K > Kj.m_K) ? Ki.m_K : Kj.m_K)*sizeof(REAL);
   return sizeof(Edge) + messageSizeInBytes;
}

template <class GM>
inline int TypeView<GM>::Edge::GetBufSizeInBytes(int vectorMaxSizeInBytes)
{
   return 0;
}

template <class GM>
inline void TypeView<GM>::Edge::Initialize(GlobalSize Kglobal, LocalSize Ki, LocalSize Kj, EdgeData data, Vector* Di, Vector* Dj)
{
   gm_ = &data.gm_;
   factorID_ = data.factorID_;

   m_dir = 0;
   m_message = (Vector*)((char*)this + sizeof(Edge));
   memset(m_message->m_data, 0, ((Ki.m_K > Kj.m_K) ? Ki.m_K : Kj.m_K)*sizeof(REAL));
}

template <class GM>
inline typename TypeView<GM>::Vector* TypeView<GM>::Edge::GetMessagePtr()
{
   return m_message;
}

template <class GM>
inline void TypeView<GM>::Edge::Swap(GlobalSize Kglobal, LocalSize Ki, LocalSize Kj)
{
   m_dir = 1 - m_dir;
}

template <class GM>
inline typename TypeView<GM>::REAL TypeView<GM>::Edge::UpdateMessage(GlobalSize Kglobal, LocalSize Ksource, LocalSize Kdest, Vector* source, REAL gamma, int dir, void* _buf)
{
   Vector* buf = (Vector*) _buf;
   REAL vMin;

   int ksource, kdest;

   for (ksource=0; ksource<Ksource.m_K; ksource++)
   {
      buf->m_data[ksource] = gamma*source->m_data[ksource] - m_message->m_data[ksource];
   }

   if (dir == m_dir)
   {
      for (kdest=0; kdest<Kdest.m_K; kdest++)
      {
         typename GM::IndexType index[] = {0, static_cast<typename GM::IndexType>(kdest)};
         vMin = buf->m_data[0] + (*gm_)[factorID_](index);
         for (ksource=1; ksource<Ksource.m_K; ksource++)
         {
            index[0] = ksource;

            if (vMin > buf->m_data[ksource] + (*gm_)[factorID_](index))
            {
               vMin = buf->m_data[ksource] + (*gm_)[factorID_](index);
            }
         }
         m_message->m_data[kdest] = vMin;
      }
   }
   else
   {
      for (kdest=0; kdest<Kdest.m_K; kdest++)
      {
         typename GM::IndexType index[] = {static_cast<typename GM::IndexType>(kdest), 0};
         vMin = buf->m_data[0] + (*gm_)[factorID_](index);
         for (ksource=1; ksource<Ksource.m_K; ksource++)
         {
            index[1] = ksource;
            if (vMin > buf->m_data[ksource] + (*gm_)[factorID_](index))
            {
               vMin = buf->m_data[ksource] + (*gm_)[factorID_](index);
            }
         }
         m_message->m_data[kdest] = vMin;
      }
   }

   vMin = m_message->m_data[0];
   for (kdest=1; kdest<Kdest.m_K; kdest++)
   {
      if (vMin > m_message->m_data[kdest])
      {
         vMin = m_message->m_data[kdest];
      }
   }

   for (kdest=0; kdest<Kdest.m_K; kdest++)
   {
      m_message->m_data[kdest] -= vMin;
   }

   return vMin;
}

template <class GM>
inline void TypeView<GM>::Edge::AddColumn(GlobalSize Kglobal, LocalSize Ksource, LocalSize Kdest, Label ksource, Vector* dest, int dir)
{
   assert(ksource>=0 && ksource<Ksource.m_K);

   int k;

   //REAL* data = ((EdgeGeneral*)this)->m_data;

   if (dir == m_dir)
   {
      typename GM::IndexType index[] = {static_cast<typename GM::IndexType>(ksource), 0};
      for (k=0; k<Kdest.m_K; k++)
      {
         index[1] = k;
         dest->m_data[k] += (*gm_)[factorID_](index);
      }
   }
   else
   {
      typename GM::IndexType index[] = {0, static_cast<typename GM::IndexType>(ksource)};
      for (k=0; k<Kdest.m_K; k++)
      {
         index[0] = k;
         dest->m_data[k] += (*gm_)[factorID_](index);
      }
   }
}
#endif /* TYPEVIEW_H_ */
