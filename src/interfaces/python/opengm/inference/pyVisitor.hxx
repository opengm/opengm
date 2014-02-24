#ifndef OPENGM_PYTHON_VISITOR
#define OPENGM_PYTHON_VISITOR

#include <boost/python.hpp>
#include <boost/python/wrapper.hpp> 
#include <vector>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>
//#include <opengm/inference/new_visitors/new_visitors.hxx>



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
    size_t visit_impl(PassedInfTye inf){
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
        return 0;//static_cast<size_t>(opengm::visitors::VisitorReturnFlag::continueInf);
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


    template<class A,class B,class C,class D,class E,class F,class G>
    void begin(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g){return begin_impl(inf);}    
    template<class A,class B,class C,class D,class E,class F,class G,class H>
    void begin(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g,const H & h){return begin_impl(inf);}
    template<class A,class B,class C,class D,class E,class F,class G,class H,class I>
    void begin(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g,const H & h,const I & i){return begin_impl(inf);}

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
    template<class A,class B,class C,class D,class E,class F,class G>
    void end(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g){return end_impl(inf);}    
    template<class A,class B,class C,class D,class E,class F,class G,class H>
    void end(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g,const H & h){return end_impl(inf);}
    template<class A,class B,class C,class D,class E,class F,class G,class H,class I>
    void end(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g,const H & h,const I & i){return end_impl(inf);}



    template<class I>
    size_t operator()(I & inf){return visit_impl(inf);}
    template<class A>
    size_t operator()(PassedInfTye inf,const A & a){return visit_impl(inf);}
    template<class A,class B>
    size_t operator()(PassedInfTye inf,const A & a,const B & b){return visit_impl(inf);}
    template<class A,class B,class C>
    size_t operator()(PassedInfTye inf,const A & a,const B & b,const C & c){return visit_impl(inf);}
    template<class A,class B,class C,class D>
    size_t operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d){return visit_impl(inf);}
    template<class A,class B,class C,class D,class E>
    size_t operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e){return visit_impl(inf);}
    template<class A,class B,class C,class D,class E,class F>
    size_t operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f){return visit_impl(inf);}
    template<class A,class B,class C,class D,class E,class F,class G>
    size_t operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g){return visit_impl(inf);}    
    template<class A,class B,class C,class D,class E,class F,class G,class H>
    size_t operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g,const H & h){return visit_impl(inf);}
    template<class A,class B,class C,class D,class E,class F,class G,class H,class I>
    size_t operator()(PassedInfTye inf,const A & a,const B & b,const C & c,const D & d,const E & e,const F & f,const G & g,const H & h,const I & i){return visit_impl(inf);}
    // Interface



private:
    boost::python::object obj_;
    size_t visitNth_;
    size_t visitNr_;
    bool gilEnsure_;
    //std::vector<LabelType> labeling_;

};

#endif