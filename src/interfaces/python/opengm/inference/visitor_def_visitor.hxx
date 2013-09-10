#ifndef VISITOR_DEF_VISITOR
#define VISITOR_DEF_VISITOR


#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>
#include <sstream>
#include <string>
#include "gil.hxx"
#include <opengm/inference/inference.hxx>
#include "pyVisitor.hxx"
///////////////////////////////////
// VERBOSE VISITOR
///////////////////////////////////

template<class INF,bool HAS_VISITOR>
class InfVerboseVisitorSuite;

template<class INF>
class InfVerboseVisitorSuite<INF,true> : public boost::python::def_visitor<InfVerboseVisitorSuite<INF,true> >{
public:
    friend class boost::python::def_visitor_access;
    typedef typename INF::VerboseVisitorType VisitorType;

    InfVerboseVisitorSuite(const std::string & className)
    :className_(className){
    }

    const std::string className_;

    template <class classT>
    void visit(classT& c) const{ 

        boost::python::class_<VisitorType > (className_.c_str() , boost::python::init<size_t,bool>(
                (
                    boost::python::arg("printNth")=1,
                    boost::python::arg("multiline")=true
                )
            )
        )
        ;

        c
            .def("verboseVisitor", &verboseVisitor,boost::python::return_value_policy<boost::python::manage_new_object>(),
                (
                    boost::python::arg("printNth")=1,
                    boost::python::arg("multiline")=true)
                )
            .def("_infer", &infer,
                (
                    boost::python::arg("visitor"),
                    boost::python::arg("releaseGil")=true
                )
            ) 
        ;
    }

    static VisitorType * verboseVisitor(const INF & inf,const size_t printNth,const bool printMultiLine){
        return new VisitorType(printNth,printMultiLine);
    }

    static opengm::InferenceTermination infer(INF & inf,VisitorType & visitor,const bool releaseGil){
        opengm::InferenceTermination result;
        {
            if(releaseGil){
                releaseGIL rgil;
                result= inf.infer(visitor);
            }
            else{
                result= inf.infer(visitor);
            }
        }
        return result;
    }
};
template<class INF>
class InfVerboseVisitorSuite<INF,false> : public boost::python::def_visitor<InfVerboseVisitorSuite<INF,false> >{
public:
    friend class def_visitor_access;

    InfVerboseVisitorSuite(const std::string & className){
        //(void)className;
    }

    template <class classT>
    void visit(classT& c) const{
    } 
};



///////////////////////////////////
// PYTHON VISITOR
///////////////////////////////////

template<class INF,bool HAS_VISITOR>
class InfPythonVisitorSuite;

template<class INF>
class InfPythonVisitorSuite<INF,true> : public boost::python::def_visitor<InfPythonVisitorSuite<INF,true> >{
public:
    friend class boost::python::def_visitor_access;
    typedef PythonVisitor<INF> VisitorType;
    InfPythonVisitorSuite(const std::string & className)
    :className_(className){
    }

    const std::string className_;

    template <class classT>
    void visit(classT& c) const{ 

        boost::python::class_<VisitorType > (className_.c_str() , boost::python::init<boost::python::object,const size_t>(
                (
                    boost::python::arg("callbackObject"),
                    boost::python::arg("multiline")=1,
                    boost::python::arg("ensureGilState")=true
                )
            )
        )
        ;

        c
            .def("pythonVisitor", &pythonVisitor,boost::python::return_value_policy<boost::python::manage_new_object>(),
                (
                    boost::python::arg("callbackObject"),
                    boost::python::arg("visitNth")=1)
                )
            .def("_infer", &infer,
                (
                    boost::python::arg("visitor"),
                    boost::python::arg("releaseGil")=true
                )
            ) 
        ;
    }

    static VisitorType * pythonVisitor(const INF & inf,boost::python::object f,const size_t visitNth){
        return new VisitorType(f,visitNth);
    }

    static opengm::InferenceTermination infer(INF & inf,VisitorType & visitor,const bool releaseGil){
        visitor.setGilEnsure(releaseGil);
        opengm::InferenceTermination result;
        {
            if(releaseGil){
                releaseGIL rgil;
                result= inf.infer(visitor);
            }
            else{
                result= inf.infer(visitor);
            }
        }
        return result;
    }
};
template<class INF>
class InfPythonVisitorSuite<INF,false> : public boost::python::def_visitor<InfPythonVisitorSuite<INF,false> >{
public:
    friend class boost::python::def_visitor_access;

    InfPythonVisitorSuite(const std::string & className){
        //(void)className;
    }

    
    template <class classT>
    void visit(classT& c) const{
    } 
};


#endif // VISITOR_DEF_VISITOR