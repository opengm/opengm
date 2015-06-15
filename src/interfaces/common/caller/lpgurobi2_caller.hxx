#ifndef OPENGM_LPGUROBI2_CALLER_HXX_
#define OPENGM_LPGUROBI2_CALLER_HXX_

#include <opengm/inference/lpgurobi2.hxx>
#include "lp_inference_caller_base.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LPGurobi2Caller : public LPInferenceCallerBase<LPGurobi2<GM, ACC>, IO, GM, ACC> {
public:
   typedef LPInferenceCallerBase<LPGurobi2<GM, ACC>, IO, GM, ACC> BaseClass;
   const static std::string name_;
   LPGurobi2Caller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LPGurobi2Caller<IO, GM, ACC>::LPGurobi2Caller(IO& ioIn)
   : BaseClass(ioIn, name_, "detailed description of LPGurobi2 caller...") {

}

template <class IO, class GM, class ACC>
const std::string LPGurobi2Caller<IO, GM, ACC>::name_ = "LPGUROBI2";

} // namespace interface

} // namespace opengm

#endif /* OPENGM_LPGUROBI2_CALLER_HXX_ */
