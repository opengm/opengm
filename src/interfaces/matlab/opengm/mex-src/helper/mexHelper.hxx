#ifndef MEXHELPER_HXX_
#define MEXHELPER_HXX_

namespace opengm {

namespace interface {

namespace helper {

/********************
 * class definition *
 ********************/

template <class FUNCTOR>
class getDataFromMXArray {
public:
   void operator() (FUNCTOR& functorIn, const mxArray* matlabArray) const;
};

// access functors

// access first element
template <class FUNCTOR>
class forFirstValue {
public:
   forFirstValue(FUNCTOR& functorIn);
   template <class DATATYPE>
   void operator() (DATATYPE* dataIn, const size_t numElements);
protected:
   FUNCTOR& functor_;
};

// access all elements
template <class FUNCTOR>
class forAllValues {
public:
   forAllValues(FUNCTOR& functorIn);
   template <class DATATYPE>
   void operator() (DATATYPE* dataIn, const size_t numElements);
protected:
   FUNCTOR& functor_;
};

// copy value
// no boundary check is done
template <class VALUETYPE, class ITERATOR = VALUETYPE*>
class copyValue {
public:
   copyValue(const ITERATOR& storageBegin);
   template <class DATATYPE>
   void operator() (const DATATYPE& valueIn);
protected:
   ITERATOR storageIterator_;

};

// store value
// no boundary check is done
template <class VALUETYPE, class ITERATOR = VALUETYPE*>
class storeValue {
public:
   storeValue(const ITERATOR& storageBegin);
   template <class DATATYPE>
   void operator() (DATATYPE& valueIn);
protected:
   ITERATOR storageIterator_;

};

/***********************
 * class documentation *
 ***********************/
//TODO add documentation

/******************
 * implementation *
 ******************/

template <class FUNCTOR >
inline void getDataFromMXArray<FUNCTOR>::operator() (FUNCTOR& functorIn, const mxArray* matlabArray) const {
   if(mxIsSingle(matlabArray)) {
      functorIn(static_cast<float*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsDouble(matlabArray)) {
      functorIn(static_cast<double*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsInt8(matlabArray)) {
      functorIn(static_cast<int8_T*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsInt16(matlabArray)) {
      functorIn(static_cast<int16_T*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsInt32(matlabArray)) {
      functorIn(static_cast<int32_T*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsInt64(matlabArray)) {
      functorIn(static_cast<int64_T*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsUint8(matlabArray)) {
      functorIn(static_cast<uint8_T*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsUint16(matlabArray)) {
      functorIn(static_cast<uint16_T*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsUint32(matlabArray)) {
      functorIn(static_cast<uint32_T*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else if(mxIsUint64(matlabArray)) {
      functorIn(static_cast<uint64_T*>(mxGetData(matlabArray)), mxGetNumberOfElements(matlabArray));
   } else {
     mexErrMsgTxt("Unsupported data format!");
   }
}

template <class FUNCTOR>
inline forFirstValue<FUNCTOR>::forFirstValue(FUNCTOR& functorIn) : functor_(functorIn) {

}

template <class FUNCTOR>
template <class DATATYPE>
inline void forFirstValue<FUNCTOR>::operator() (DATATYPE* dataIn, const size_t numElements) {
   OPENGM_ASSERT(dataIn != NULL);
   OPENGM_ASSERT(numElements > 0);
   functor_(*dataIn);
}

template <class FUNCTOR>
inline forAllValues<FUNCTOR>::forAllValues(FUNCTOR& functorIn) : functor_(functorIn) {

}

template <class FUNCTOR>
template <class DATATYPE>
inline void forAllValues<FUNCTOR>::operator() (DATATYPE* dataIn, const size_t numElements) {
   OPENGM_ASSERT(dataIn != NULL);
   OPENGM_ASSERT(numElements > 0);
   for(size_t i = 0; i < numElements; i++) {
      functor_(dataIn[i]);
   }
}

template <class VALUETYPE, class ITERATOR>
inline copyValue<VALUETYPE, ITERATOR>::copyValue(const ITERATOR& storageBegin) : storageIterator_(storageBegin) {

}

template <class VALUETYPE, class ITERATOR>
template <class DATATYPE>
inline void copyValue<VALUETYPE, ITERATOR>::operator() (const DATATYPE& valueIn) {
   *storageIterator_ = static_cast<VALUETYPE>(valueIn);
   storageIterator_++;
}

template <class VALUETYPE, class ITERATOR>
inline storeValue<VALUETYPE, ITERATOR>::storeValue(const ITERATOR& storageBegin) : storageIterator_(storageBegin) {

}

template <class VALUETYPE, class ITERATOR>
template <class DATATYPE>
inline void storeValue<VALUETYPE, ITERATOR>::operator() (DATATYPE& valueIn) {
   valueIn = static_cast<DATATYPE>(*storageIterator_);
   storageIterator_++;
}

} // namespace helper

} // namespace interface

} // namespace opengm

#endif /* MEXHELPER_HXX_ */
