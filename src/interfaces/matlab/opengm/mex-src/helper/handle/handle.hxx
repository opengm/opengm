#ifndef HANDLE_HXX_
#define HANDLE_HXX_

#include <mex.h>

#include <set>
#include <iostream>

#include "handleCollector.hxx"

namespace opengm {

namespace interface {

// handle class to allow sharing c++ objects between mex function calls.
// Adapted from http://www-personal.acfr.usyd.edu.au/tbailey/software/other.htm
template<class T>
class handle {
public:
   // handle takes ownership of object.
   handle(T* const object);
   ~handle();

   // Convert handle to mxArray to pass back from mex function.
   mxArray* toMxArrayHandle();
   // Convert mxArray to handle.
   static handle<T>* fromMxArrayHandle(const mxArray* const matlabHandle);
   // get corresponding c++ object.
   T& getObject();

   static mxArray* createHandle(T* const object);
   static T& getObject(const mxArray* const matlabHandle);
   static void deleteObject(const mxArray* const matlabHandle);
protected:
   // pointer to c++ Object
   T* object_;
   // Signature to check if handle is still valid.
   // Clear all will cause mex functions to be freed before the corresponding MatLab objects will be destroyed.
   // As the handle collector will delete all remaining handles, if the mex function is freed, the remaining handles might be deleted twice as the destructor in matlab also tries to delete the handle.
   // Thus double checking if the handle is still valid is required.
   handle<T>* myself_;
};

/******************
 * implementation *
 ******************/

template<class T>
inline handle<T>::handle(T* const object) : object_(object), myself_(this) {
   addHandle(this, false);
}

template<class T>
inline handle<T>::~handle() {
   if(myself_ == this) {
      delete object_;
      addHandle(this, true);
      // destroy signature
      myself_ = NULL;
   }
}

template<class T>
inline mxArray* handle<T>::toMxArrayHandle() {
   mxArray* mxArrayHandle = NULL;
   if(sizeof(this) > 8) {
      mexErrMsgTxt("unsupported pointer size");
   } else if(sizeof(this) > 4) {
      mxArrayHandle  = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
   } else {
      mxArrayHandle  = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL);
   }
   *reinterpret_cast<handle<T>**>(mxGetData(mxArrayHandle)) = this;
   return mxArrayHandle;
}

template<class T>
inline handle<T>* handle<T>::fromMxArrayHandle(const mxArray* const matlabHandle) {
   if(sizeof(handle<T>*) > 8)
   {
      mexErrMsgTxt("unsupported pointer size");
   } else if(sizeof(handle<T>*) > 4) {
      if((mxGetClassID(matlabHandle) != mxUINT64_CLASS) || mxIsComplex(matlabHandle) || (mxGetM(matlabHandle) != 1) || (mxGetN(matlabHandle) != 1)) {
         mexErrMsgTxt("Given MatLab handle is not an handle type");
      }
   } else {
      if((mxGetClassID(matlabHandle) != mxUINT32_CLASS) || mxIsComplex(matlabHandle) || (mxGetM(matlabHandle) != 1) || (mxGetN(matlabHandle) != 1)) {
         mexErrMsgTxt("Given MatLab handle is not an handle type");
      }
   }

   return *reinterpret_cast<handle<T>**>(mxGetData(matlabHandle));
}

template<class T>
inline T& handle<T>::getObject() {
   return *object_;
}

template<class T>
inline mxArray* handle<T>::createHandle(T* const object) {
   handle<T>* newHandle = new handle<T>(object);
   return newHandle->toMxArrayHandle();
}

template<class T>
inline T& handle<T>::getObject(const mxArray* const matlabHandle) {
   handle<T>* currentHandle= handle<T>::fromMxArrayHandle(matlabHandle);
   return currentHandle->getObject();
}

template<class T>
inline void handle<T>::deleteObject(const mxArray* const matlabHandle) {
   delete fromMxArrayHandle(matlabHandle);
   // set handle to zero
   if(sizeof(handle<T>*) > 8)
   {
      mexErrMsgTxt("unsupported pointer size");
   } else if(sizeof(handle<T>*) > 4) {
      *reinterpret_cast<uint64_T*>(mxGetData(matlabHandle)) = 0;
   } else {
      *reinterpret_cast<uint32_T*>(mxGetData(matlabHandle)) = 0;
   }
}

} // namespace interface

} // namespace opengm

#endif /* HANDLE_HXX_ */
