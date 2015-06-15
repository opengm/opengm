#ifndef OPENGM_LPCPLEX2_CALLER_HXX_
#define OPENGM_LPCPLEX2_CALLER_HXX_

#include <opengm/inference/lpcplex2.hxx>
#include "lp_inference_caller_base.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LPCplex2Caller : public LPInferenceCallerBase<LPCplex2<GM, ACC>, IO, GM, ACC> {
public:
   typedef LPInferenceCallerBase<LPCplex2<GM, ACC>, IO, GM, ACC> BaseClass;
   const static std::string name_;
   LPCplex2Caller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LPCplex2Caller<IO, GM, ACC>::LPCplex2Caller(IO& ioIn)
   : BaseClass(ioIn, name_, "detailed description of LPCplex2 caller...") {

}

template <class IO, class GM, class ACC>
const std::string LPCplex2Caller<IO, GM, ACC>::name_ = "LPCPLEX2";

} // namespace interface

} // namespace opengm

#endif /* OPENGM_LPCPLEX2_CALLER_HXX_ */
