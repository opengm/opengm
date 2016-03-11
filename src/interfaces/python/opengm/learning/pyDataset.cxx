#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <stdexcept>
#include <stddef.h>

#include <opengm/learning/dataset/editabledataset.hxx>
#include <opengm/learning/dataset/dataset_io.hxx>
#include <opengm/learning/loss/hammingloss.hxx>
#include <opengm/learning/loss/generalized-hammingloss.hxx>
#include <opengm/learning/loss/noloss.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>

#define DefaultErrorFn DefaultErrorFn_TrwsExternal_DS
#include "helper.hxx"

using namespace boost::python;

namespace opengm{

template<class GM, class LOSS>
void pySetInstanceWithLossParam(opengm::datasets::EditableDataset<GM, LOSS>& ds,
                   const size_t i,
                   const GM& gm,
                   const opengm::python::NumpyView<typename GM::LabelType,1>  gt,
                   const typename LOSS::Parameter & param) {
    std::vector<typename GM::LabelType> gt_vector(gt.begin(), gt.end());
    ds.setInstance(i, gm, gt_vector, param);
}

template<class GM, class LOSS>
void pySetInstance(opengm::datasets::EditableDataset<GM, LOSS>& ds,
                   const size_t i,
                   const GM& gm,
                   const opengm::python::NumpyView<typename GM::LabelType,1>& gt
                   ) {
    pySetInstanceWithLossParam(ds, i, gm, gt, typename LOSS::Parameter());
}

template<class GM, class LOSS>
void pyPushBackInstanceWithLossParam(opengm::datasets::EditableDataset<GM,LOSS>& ds,
                        const GM& gm,
                        const opengm::python::NumpyView<typename GM::LabelType,1>& gt,
                        const typename LOSS::Parameter & param) {
    std::vector<typename GM::LabelType> gt_vector(gt.begin(), gt.end());
    ds.pushBackInstance(gm, gt_vector, param);
}

template<class GM, class LOSS>
void pyPushBackInstance(opengm::datasets::EditableDataset<GM,LOSS>& ds,
                        const GM& gm,
                        const opengm::python::NumpyView<typename GM::LabelType,1>& gt
                        ) {
    pyPushBackInstanceWithLossParam(ds, gm, gt, typename LOSS::Parameter());
}

template<class GM, class LOSS>
void pySaveDataset(opengm::datasets::EditableDataset<GM,LOSS >& ds,
                   const std::string datasetpath,
                   const std::string prefix) {
    opengm::datasets::DatasetSerialization::save(ds, datasetpath, prefix);
}

template<class GM, class LOSS>
void pyLoadDataset(opengm::datasets::EditableDataset<GM,LOSS >& ds,
                   const std::string datasetpath,
                   const std::string prefix) {
    opengm::datasets::DatasetSerialization::loadAll(datasetpath, prefix, ds);
}

template<class GM, class LOSS>
void export_dataset(const std::string& className){
    typedef opengm::datasets::EditableDataset<GM,LOSS > PyDataset;

   class_<PyDataset > (className.c_str(),init<size_t>())
           .def("lockModel", &PyDataset::lockModel)
           .def("unlockModel", &PyDataset::unlockModel)
           .def("getModel", &PyDataset::getModel, return_internal_reference<>())
           .def("getModelWithLoss", &PyDataset::getModelWithLoss, return_internal_reference<>())
           .def("getGT", &PyDataset::getGT, return_internal_reference<>())
           .def("getWeights", &PyDataset::getWeights, return_internal_reference<>())
           .def("getNumberOfWeights", &PyDataset::getNumberOfWeights)
           .def("getNumberOfModels", &PyDataset::getNumberOfModels)
           .def("setInstance", &pySetInstance<GM,LOSS>)
           .def("setInstanceWithLossParam", &pySetInstanceWithLossParam<GM,LOSS>)
           .def("setInstance", &pySetInstanceWithLossParam<GM,LOSS>)
           .def("pushBackInstance", &pyPushBackInstance<GM,LOSS>)
           .def("pushBackInstanceWithLossParam", &pyPushBackInstanceWithLossParam<GM,LOSS>)
           .def("pushBackInstance", &pyPushBackInstanceWithLossParam<GM,LOSS>)
           .def("setWeights", &PyDataset::setWeights)
           .def("save", &pySaveDataset<GM, LOSS>)
           .def("load", &pyLoadDataset<GM, LOSS>)
           .def(DatasetInferenceSuite<PyDataset>())
   ;

}


//template void export_dataset<opengm::python::GmAdder, opengm::learning::HammingLoss> (const std::string& className);
//template void export_dataset<opengm::python::GmAdder, opengm::learning::NoLoss> (const std::string& className);
template void export_dataset<opengm::python::GmAdder, opengm::learning::FlexibleLoss> (const std::string& className);

}
