
#ifndef HANDLECOLLECTOR_HXX_
#define HANDLECOLLECTOR_HXX_

#include <set>

namespace opengm {

namespace interface {

template <class T>
class handleCollector;
template <class T>
class handle;

} // namespace interface

} // namespace opengm

#include "handle.hxx"
#include "../../model/matlabModelType.hxx"

namespace opengm {

namespace interface {

// Ensures that all remaining handles will be deleted if "clear all" is called in MatLab.
// Singleton policy, only one garbage collector allowed.
template <class T>
class handleCollector {
public:
   // destructor deletes all remaining handles.
   ~handleCollector();
   // collects handle and adds it to garbage collector.
   // removes handle from garbage collector if remove is set to true
   static void addHandle(handle<T>* const objectHandle, bool remove = false);

protected:
   // list of all remaining handles
   std::set<handle<T>*> handleList_;
   // no construction allowed
   handleCollector();
};

void addHandle(handle<MatlabModelType::GmType>* const objectHandle, bool remove = false);

template<class T>
inline handleCollector<T>::~handleCollector() {
   for(typename std::set<handle<T>*>::iterator iter = handleList_.begin(); iter != handleList_.end(); iter = handleList_.begin()) {
      delete *iter;
   }
}

template<class T>
inline handleCollector<T>::handleCollector() {
}

template<class T>
inline void handleCollector<T>::addHandle(handle<T>* const objectHandle, bool remove) {
   // static objects will be deleted if "clear all" is called in MatLab
   static handleCollector<T> collector;
   if(remove) {
      typename std::set<handle<T>*>::iterator iter = collector.handleList_.find(objectHandle);
      if(iter != collector.handleList_.end()) {
         collector.handleList_.erase(objectHandle);
      }
   } else {
      collector.handleList_.insert(objectHandle);
   }
}

} // namespace interface

} // namespace opengm
#endif /* HANDLECOLLECTOR_HXX_ */
