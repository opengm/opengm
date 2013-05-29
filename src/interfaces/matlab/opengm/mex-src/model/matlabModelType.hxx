#ifndef MATLABDEFAULTMODELTYPE_HXX_
#define MATLABDEFAULTMODELTYPE_HXX_

// opengm stuff
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"

namespace opengm {

namespace interface {

class MatlabModelType {
public:
  typedef double ValueType;
  typedef size_t IndexType;
  typedef size_t LabelType;
  typedef Adder OperatorType;
  typedef DiscreteSpace<IndexType, LabelType> SpaceType;

  typedef opengm::ExplicitFunction<ValueType, IndexType, LabelType> ExplicitFunction;
  typedef opengm::PottsFunction<ValueType, IndexType, LabelType> PottsFunction;
  typedef opengm::PottsNFunction<ValueType, IndexType, LabelType> PottsNFunction;
  typedef opengm::PottsGFunction<ValueType, IndexType, LabelType> PottsGFunction;
  typedef opengm::TruncatedSquaredDifferenceFunction<ValueType, IndexType, LabelType> TruncatedSquaredDifferenceFunction;
  typedef opengm::TruncatedAbsoluteDifferenceFunction<ValueType, IndexType, LabelType> TruncatedAbsoluteDifferenceFunction;

  // Set functions for graphical model
  typedef meta::TypeListGenerator<
        ExplicitFunction,
        PottsFunction,
        PottsNFunction,
        PottsGFunction,
        TruncatedSquaredDifferenceFunction,
        TruncatedAbsoluteDifferenceFunction
     >::type FunctionTypeList;

  typedef opengm::GraphicalModel<
     ValueType,
     OperatorType,
     FunctionTypeList,
     SpaceType
  > GmType;
};

} // namespace interface

} // namespace opengm
#endif /* MATLABDEFAULTMODELTYPE_HXX_ */
