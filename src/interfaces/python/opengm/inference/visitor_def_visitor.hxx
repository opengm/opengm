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
    friend class def_visitor_access;
    typedef typename INF::VerboseVisitorType VisitorType;

    InfVerboseVisitorSuite(const std::string & className)
    :className_(className){
    }

    const std::string className_;

    template <class classT>
    void visit(classT& c) const{ 

        class_<VisitorType > (className_.c_str() , init<size_t,bool>(
                (
                    arg("printNth")=1,
                    arg("multiline")=true
                )
            )
        )
        ;

        c
            .def("verboseVisitor", &verboseVisitor,return_value_policy<manage_new_object>(),(arg("printNth")=1,arg("multiline")=true))
            .def("_infer", &infer,
                (
                    arg("visitor"),
                    arg("releaseGil")=true
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
    friend class def_visitor_access;
    typedef PythonVisitor<INF> VisitorType;
    InfPythonVisitorSuite(const std::string & className)
    :className_(className){
    }

    const std::string className_;

    template <class classT>
    void visit(classT& c) const{ 

        class_<VisitorType > (className_.c_str() , init<boost::python::object,const size_t>(
                (
                    arg("callbackObject"),
                    arg("multiline")=1,
                    arg("ensureGilState")=true
                )
            )
        )
        ;

        c
            .def("pythonVisitor", &pythonVisitor,return_value_policy<manage_new_object>(),(arg("callbackObject"),arg("visitNth")=1))
            .def("_infer", &infer,
                (
                    arg("visitor"),
                    arg("releaseGil")=true
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
    friend class def_visitor_access;

    InfPythonVisitorSuite(const std::string & className){
        //(void)className;
    }

    
    template <class classT>
    void visit(classT& c) const{
    } 
};


#endif // VISITOR_DEF_VISITOR