#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <stdexcept>
#include <stddef.h>

#include <opengm/learning/loss/hammingloss.hxx>
#include <opengm/learning/loss/generalized-hammingloss.hxx>
#include <opengm/learning/loss/noloss.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>

using namespace boost::python;

namespace opengm{
    
void pySetNodeLossMultiplier(opengm::learning::GeneralizedHammingLoss::Parameter& p,
                             const opengm::python::NumpyView<double,1>& m)
{
    p.nodeLossMultiplier_ = std::vector<double>(m.begin(), m.end());
}

void pySetLabelLossMultiplier(opengm::learning::GeneralizedHammingLoss::Parameter& p,
                             const opengm::python::NumpyView<double,1>& m)
{
    p.labelLossMultiplier_ = std::vector<double>(m.begin(), m.end());
}

template <class GM>
void export_loss(){
   typedef typename std::vector<typename GM::LabelType>::const_iterator Literator;
   typedef typename std::vector<typename GM::LabelType>::const_iterator Niterator;
   typedef opengm::learning::HammingLoss PyHammingLoss;
   typedef opengm::learning::GeneralizedHammingLoss PyGeneralizedHammingLoss;
   typedef opengm::learning::NoLoss PyNoLoss;



    typedef opengm::learning::GeneralizedHammingLoss::Parameter PyGeneralizedHammingLossParameter;

    class_<PyHammingLoss >("HammingLoss")
        .def("loss", &PyHammingLoss::loss<Literator,Literator>)
        .def("addLoss", &PyHammingLoss::addLoss<GM, Literator>)
    ;

    class_<PyNoLoss >("NoLoss")
        .def("loss", &PyNoLoss::loss<Literator,Literator>)
        .def("addLoss", &PyNoLoss::addLoss<GM, Literator>)
    ;

    class_<PyGeneralizedHammingLoss >("GeneralizedHammingLoss", init<PyGeneralizedHammingLossParameter>())
        .def("loss", &PyGeneralizedHammingLoss::loss<Literator,Literator>)
        .def("addLoss", &PyGeneralizedHammingLoss::addLoss<GM, Literator>)
    ;


    class_<PyNoLoss::Parameter>("NoLossParameter")
    ;

    class_<PyHammingLoss::Parameter>("HammingLossParameter")
    ;

    class_<PyGeneralizedHammingLossParameter>("GeneralizedHammingLossParameter")
        .def("setNodeLossMultiplier", &pySetNodeLossMultiplier)
        .def("setLabelLossMultiplier", &pySetLabelLossMultiplier)
    ;

    class_<std::vector< PyNoLoss::Parameter > >("NoLossParameterVector")
        .def(vector_indexing_suite<std::vector< PyNoLoss::Parameter> >())
    ;
    class_<std::vector< PyHammingLoss::Parameter > >("HammingLossParameterVector")
        .def(vector_indexing_suite<std::vector< PyHammingLoss::Parameter> >())
    ;
    class_<std::vector< PyGeneralizedHammingLoss::Parameter > >("GeneralizedHammingLossParameterVector")
        .def(vector_indexing_suite<std::vector< PyGeneralizedHammingLoss::Parameter> >())
    ;

}


template void export_loss<opengm::python::GmAdder>();

}
