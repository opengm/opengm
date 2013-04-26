#ifndef OPENGM_PYTHON_VISITOR
#define OPENGM_PYTHON_VISITOR

#include <boost/python.hpp>
#include <boost/python/wrapper.hpp> 
#include <vector>

#include "nifty_iterator.hxx"
#include "iteratorToTuple.hxx"
#include "export_typedes.hxx"
#include "copyhelper.hxx"

#include "../converter.hxx"




template<class INF>
class PythonVisitor{
    
public:
    typedef INF & PassedInfTye;
    typedef typename INF::GraphicalModelType GraphicalModelType;
    typedef typename INF::ValueType ValueType;
    typedef typename INF::IndexType IndexType;
    typedef typename INF::LabelType LabelType;

    void setGilEnsure(const bool gilEnsure){
        gilEnsure_=gilEnsure;
    }

    PythonVisitor(
        boost::python::object obj,
        const size_t visitNth,
        const bool gilEnsure=true
    )
    : obj_(obj),
    visitNth_(visitNth),
    visitNr_(0),
    gilEnsure_(gilEnsure)
    {

    }


    void begin_impl(PassedInfTye inf){

        if(gilEnsure_){
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure ();
            // CALL BACK TO PYTHON
            obj_.attr("begin")(inf);
            PyGILState_Release (gstate);
        }
        else{
            obj_.attr("begin")(inf);
        }
    }
    void end_impl(PassedInfTye inf){
        if(gilEnsure_){
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure ();
            // CALL BACK TO PYTHON
            obj_.attr("end")(inf);
            PyGILState_Release (gstate);
        }
        else{
            obj_.attr("end")(inf);
        }
    }
    void visit_impl(PassedInfTye inf){
        ++visitNr_;
        if(visitNr_%visitNth_==0){
            if(gilEnsure_){
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure ();

                obj_.attr("visit")(inf);

                PyGILState_Release (gstate);
            }
            else{
                obj_.attr("visit")(inf);
            }
        }
    }


    void begin(PassedInfTye inf){return begin_impl(inf);}
    template<class A>
    void begin(PassedInfTye inf,const A & a){return begin_impl(inf);}
    template<class I,class A,class B>
    void begin(I & inf,const A & a,const B & b){return begin_impl(inf);}
    template<class A,class B,class C>
    void begin(PassedInfTye inf,const A & a,const B & b,const C & c){return begin_impl(inf);}
    template<class A,class B,class C,class D>
    void begin(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d){return begin_impl(inf);}
    template<class A,class B,class C,class D,class E>
    void begin(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e){return begin_impl(inf);}
    template<class A,class B,class C,class D,class E,class F>
    void begin(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f){return begin_impl(inf);}

    void end(PassedInfTye inf){return end_impl(inf);}
    template<class A>
    void end(PassedInfTye inf,const A & a){return end_impl(inf);}
    template<class A,class B>
    void end(PassedInfTye inf,const A & a,const B & b){return end_impl(inf);}
    template<class A,class B,class C>
    void end(PassedInfTye inf,const A & a,const B & b,const C & c){return end_impl(inf);}
    template<class A,class B,class C,class D>
    void end(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d){return end_impl(inf);}
    template<class A,class B,class C,class D,class E>
    void end(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e){return end_impl(inf);}
    template<class A,class B,class C,class D,class E,class F>
    void end(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f){return end_impl(inf);}

    template<class I>
    void operator()(I & inf){return visit_impl(inf);}
    template<class A>
    void operator()(PassedInfTye inf,const A & a){return visit_impl(inf);}
    template<class A,class B>
    void operator()(PassedInfTye inf,const A & a,const B & b){return visit_impl(inf);}
    template<class A,class B,class C>
    void operator()(PassedInfTye inf,const A & a,const B & b,const C & c){return visit_impl(inf);}
    template<class A,class B,class C,class D>
    void operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d){return visit_impl(inf);}
    template<class A,class B,class C,class D,class E>
    void operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e){return visit_impl(inf);}
    template<class A,class B,class C,class D,class E,class F>
    void operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f){return visit_impl(inf);}
    // Interface



private:
    boost::python::object obj_;
    size_t visitNth_;
    size_t visitNr_;
    bool gilEnsure_;
    //std::vector<LabelType> labeling_;

};

#endif