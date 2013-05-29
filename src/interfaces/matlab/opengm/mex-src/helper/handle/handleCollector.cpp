#include "handleCollector.hxx"

namespace opengm {

namespace interface {

void addHandle(handle<MatlabModelType::GmType>* const objectHandle, bool remove) {
   handleCollector<MatlabModelType::GmType>::addHandle(objectHandle, remove);
}

} // namespace interface

} // namespace opengm
