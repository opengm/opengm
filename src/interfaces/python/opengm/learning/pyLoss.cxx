#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <stdexcept>
#include <stddef.h>

//#include <opengm/learning/loss/hammingloss.hxx>
//#include <opengm/learning/loss/generalized-hammingloss.hxx>
#include <opengm/learning/loss/flexibleloss.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>

using namespace boost::python;

namespace opengm{
    
void pySetNodeLossMultiplier(opengm::learning::FlexibleLoss::Parameter& p,
                             const opengm::python::NumpyView<double,1>& m)
{
    p.nodeLossMultiplier_ = std::vector<double>(m.begin(), m.end());
}

void pySetLabelLossMultiplier(opengm::learning::FlexibleLoss::Parameter& p,
                             const opengm::python::NumpyView<double,1>& m)
{
    p.labelLossMultiplier_ = std::vector<double>(m.begin(), m.end());
}
void pySetFactorLossMultiplier(opengm::learning::FlexibleLoss::Parameter& p,
                               const opengm::python::NumpyView<double,1>& m)
{
    p.labelLossMultiplier_ = std::vector<double>(m.begin(), m.end());
}


template <class GM>
void export_loss(){
   typedef typename std::vector<typename GM::LabelType>::const_iterator Literator;
   typedef typename std::vector<typename GM::LabelType>::const_iterator Niterator;
   typedef opengm::learning::HammingLoss PyHammingLoss;
   typedef opengm::learning::FlexibleLoss PyFlexibleLoss;
   typedef opengm::learning::GeneralizedHammingLoss PyGeneralizedHammingLoss;
   typedef opengm::learning::NoLoss PyNoLoss;






    class_<PyFlexibleLoss >("FlexibleLoss")
        //.def("loss", &PyHammingLoss::loss<const GM &, Literator,Literator>)
        //.def("addLoss", &PyHammingLoss::addLoss<GM, Literator>)
    ;

    // learner param enum
    enum_<PyFlexibleLoss::Parameter::LossType>("LossType")
      .value("hamming", PyFlexibleLoss::Parameter::Hamming)
      .value("l1",  PyFlexibleLoss::Parameter::L1)
      .value("l2",  PyFlexibleLoss::Parameter::L2)
      .value("partition",  PyFlexibleLoss::Parameter::Partition)
      .value("ConfMat",  PyFlexibleLoss::Parameter::ConfMat)
    ;


    class_<PyFlexibleLoss::Parameter>("FlexibleLossParameter")
        .def_readwrite("lossType", &PyFlexibleLoss::Parameter::lossType_)
        .def("setNodeLossMultiplier", &pySetNodeLossMultiplier)
        .def("setLabelLossMultiplier", &pySetLabelLossMultiplier)
        .def("setFactorLossMultiplier", &pySetFactorLossMultiplier)
    ;


    class_<std::vector< PyFlexibleLoss::Parameter > >("FlexibleLossParameterVector")
        .def(vector_indexing_suite<std::vector< PyFlexibleLoss::Parameter> >())
    ;


}


template void export_loss<opengm::python::GmAdder>();

}
