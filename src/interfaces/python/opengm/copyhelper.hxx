// from http://stackoverflow.com/questions/4696966/copying-a-boost-python-object

#ifndef COPYHELPER_HXX
#define	COPYHELPER_HXX

#include <boost/python.hpp>

#define PYTHON_ERROR(TYPE, REASON) \
{ \
    PyErr_SetString(TYPE, REASON); \
    throw boost::python::error_already_set(); \
}

template<class T>
inline PyObject * managingPyObject(T *p)
{
    return typename boost::python::manage_new_object::apply<T *>::type()(p);
}

template<class Copyable>
boost::python::object
generic__copy__(boost::python::object copyable)
{
    Copyable *newCopyable(new Copyable(boost::python::extract<const Copyable
&>(copyable)));
    boost::python::object
result(boost::python::detail::new_reference(managingPyObject(newCopyable)));

    boost::python::extract<boost::python::dict>(result.attr("__dict__"))().update(
        copyable.attr("__dict__"));

    return result;
}



#endif	/* COPYHELPER_HXX */

