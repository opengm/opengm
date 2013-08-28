#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleHdf5


#include <boost/python.hpp>
#include <stddef.h>

#include <opengm/python/opengmpython.hxx>

#ifdef WITH_HDF5
#include "pyHdf5.hxx"
#endif


BOOST_PYTHON_MODULE_INIT(_hdf5) {
    Py_Initialize();
    PyEval_InitThreads();
    #ifdef WITH_HDF5
    export_hdf5<opengm::python::GmAdder>();
    #endif
}
