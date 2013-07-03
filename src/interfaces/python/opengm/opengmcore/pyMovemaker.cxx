
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <ostream>
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/movemaker.hxx>

#include "export_typedes.hxx"
#include "../converter.hxx"
#include "../gil.hxx"
#include "numpyview.hxx"


namespace pymovemaker{


    template<class MM>
    inline MM * constructor
    (
        const typename MM::GraphicalModelType & gm,
        NumpyView<typename MM::LabelType,1> labeling
    ){
        return new MM(gm,labeling.begin());
    }

    template<class MM>
    inline void initialize
    (
        MM & movemaker,
        NumpyView<typename MM::LabelType,1> labeling
    ){
        movemaker.initialize(labeling.begin());
    }

    template<class MM>
    inline typename MM::LabelType state
    (
        MM & movemaker,
        const typename MM::IndexType vi
    ){
        return movemaker.state(vi);
    }

    // MULTIVAR
    template<class MM,class ACC>
    inline void moveOptimally(
        MM & movemaker,
        NumpyView<typename MM::IndexType,1> vis
    ){
        {
            releaseGIL rgil;
            movemaker.moveOptimally<ACC>(vis.begin(),vis.end());
        }
    }


    template<class MM>
    inline void move(
        MM & movemaker,
        NumpyView<typename MM::IndexType,1> vis,
        NumpyView<typename MM::LabelType,1> labels
    ){
        {
            releaseGIL rgil;
            movemaker.move(vis.begin(),vis.end(),labels.begin());
        }
    }


    template<class MM>
    inline typename MM::ValueType valueAfterMove(
        MM & movemaker,
        NumpyView<typename MM::IndexType,1> vis,
        NumpyView<typename MM::LabelType,1> labels
    ){
        typename MM::ValueType result=0.0;
        {
            releaseGIL rgil;
            result=movemaker.valueAfterMove(vis.begin(),vis.end(),labels.begin());
        }
        return result;
    }



    // SingleVar
    template<class MM,class ACC>
    inline 
    typename MM::LabelType
    moveOptimallySingleVar(
        MM & movemaker,
        const typename MM::IndexType vi
    ){
        movemaker.moveOptimally<ACC>(&vi,&vi+1);
        return movemaker.state(vi);
    }


    template<class MM>
    inline void moveSingleVar(
        MM & movemaker,
        const typename MM::IndexType vi,
        const typename MM::LabelType label
    ){
        movemaker.move(&vi,&vi+1,&label); 
    }


    template<class MM>
    inline typename MM::ValueType valueAfterMoveSingleVar(
        MM & movemaker,
        const typename MM::IndexType vi,
        const typename MM::LabelType label
    ){
        return movemaker.valueAfterMove(&vi,&vi+1,&label);
    }

}


template<class GM>
void export_movemaker() {
   using namespace boost::python;
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   boost::python::docstring_options docstringOptions(true,true,false);
   
   import_array();
   typedef GM PyGm;
   typedef typename PyGm::SpaceType PySpace;
   typedef typename PyGm::ValueType ValueType;
   typedef typename PyGm::IndexType IndexType;
   typedef typename PyGm::LabelType LabelType;
   typedef opengm::Movemaker<PyGm> PyMovemaker;


    class_<PyMovemaker > ("Movemaker",
    init<const PyGm &>("Construct a movemaker from a graphical model ")[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const PyGM& */>()]
    )
    .def("__init__", make_constructor(&pymovemaker::constructor<PyMovemaker> ,default_call_policies(),(arg("gm"),arg("labels"))),
    "construct a movemaker from  a graphical model and initialize movemaker with given labeling\n\n"
    "Args:\n\n"
    "   gm : the graphicalmodel \n\n"
    "   labels : the intital labeling starting point\n\n"
    "Example: ::\n\n"
    "   >>> # assuming there is graphical model named gm with ``'adder'`` as operator\n\n"
    "   >>> labels=numpy.zeros(gm.numberOfVariables,dtype=opengm.index_type)\n\n"
    "   >>> movemaker.opengm.Movemaker(gm=gm,labels=labels)\n\n"
    )
    .def("initalize",&pymovemaker::initialize<PyMovemaker>,(arg("labeling")),"initialize movemaker with a labeling")
    .def("reset",&PyMovemaker::reset,"reset the movemaker")
    .def("value",&PyMovemaker::value,"get the value (energy/probability) of graphical model for the current labeling")
    .def("label",&pymovemaker::state<PyMovemaker>,(arg("vi")),"get the label for the given varible")
    .def("move",&pymovemaker::move<PyMovemaker>,(arg("vis"),arg("labels")),"doc todo")
    .def("valueAfterMove",&pymovemaker::valueAfterMove<PyMovemaker>,(arg("vis"),arg("labels")),"doc todo")
    .def("moveOptimallyMin",&pymovemaker::moveOptimally<PyMovemaker,opengm::Minimizer>,(arg("vis")),"doc todo")
    .def("moveOptimallyMax",&pymovemaker::moveOptimally<PyMovemaker,opengm::Maximizer>,(arg("vis")),"doc todo")

    .def("move",&pymovemaker::moveSingleVar<PyMovemaker>,(arg("vis"),arg("labels")),"doc todo")
    .def("valueAfterMove",  &pymovemaker::valueAfterMoveSingleVar<PyMovemaker>,(arg("vis"),arg("labels")),"doc todo")
    .def("moveOptimallyMin",&pymovemaker::moveOptimallySingleVar <PyMovemaker,opengm::Minimizer>,(arg("vi")),"doc todo")
    .def("moveOptimallyMax",&pymovemaker::moveOptimallySingleVar<PyMovemaker,opengm::Maximizer>,(arg("vi")),"doc todo")
    ;
}


template void export_movemaker<GmAdder>();
template void export_movemaker<GmMultiplier>();
