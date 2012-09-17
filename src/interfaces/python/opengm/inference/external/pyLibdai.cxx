#ifdef WITH_LIBDAI
#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/external/libdai/bp.hxx>
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include"../export_typedes.hxx"
using namespace boost::python;

template<class GM,class ACC>
void export_libdai_inference(){
	import_array();    
   
	typedef GM PyGm;
	typedef typename PyGm::ValueType ValueType;
	typedef typename PyGm::IndexType IndexType;
	typedef typename PyGm::LabelType LabelType;
	typedef opengm::external::libdai::Bp<PyGm, ACC>  PyLibdaiBp;
	typedef typename PyLibdaiBp::Parameter PyLibdaiBpParameter;
	typedef typename PyLibdaiBp::VerboseVisitorType PyLibdaiBpVerboseVisitor;

	class_<PyLibdaiBpParameter > ("LibDaiBpParameter", init< >() )
	;


	OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLibdaiBpVerboseVisitor,"LibDaiBpVerboseVisitor" );
	OPENGM_PYTHON_INFERENCE_EXPORTER(PyLibdaiBp,"LibDaiBp");

}

template void export_libdai_inference<GmAdder, opengm::Minimizer>();
template void export_libdai_inference<GmAdder, opengm::Maximizer>();
template void export_libdai_inference<GmMultiplier, opengm::Minimizer>();
template void export_libdai_inference<GmMultiplier, opengm::Maximizer>();

#endif
