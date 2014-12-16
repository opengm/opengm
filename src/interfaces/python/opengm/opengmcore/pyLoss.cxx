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

template <class GM>
void export_loss(){
   typedef typename std::vector<typename GM::LabelType>::const_iterator Literator;
   typedef typename std::vector<typename GM::LabelType>::const_iterator Niterator;
   typedef opengm::learning::HammingLoss PyHammingLoss;
   typedef opengm::learning::NoLoss PyNoLoss;
   typedef opengm::learning::GeneralizedHammingLoss PyGeneralizedHammingLoss;

   class_<PyHammingLoss >("HammingLoss")
           .def("loss", &PyHammingLoss::loss<Literator,Literator>)
           .def("addLoss", &PyHammingLoss::addLoss<GM, Literator>)
   ;

   class_<PyNoLoss >("NoLoss")
           .def("loss", &PyNoLoss::loss<Literator,Literator>)
           .def("addLoss", &PyNoLoss::addLoss<GM, Literator>)
   ;

   class_<PyGeneralizedHammingLoss >("GeneralizedHammingLoss", init<Niterator,Niterator,Literator,Literator>())
           .def("loss", &PyGeneralizedHammingLoss::loss<Literator,Literator>)
           .def("addLoss", &PyGeneralizedHammingLoss::addLoss<GM, Literator>)
   ;
}

template void export_loss<opengm::python::GmAdder>();
