#ifdef WITH_LIBDAI
#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/external/libdai/bp.hxx>
#include <opengm/inference/external/libdai/fractional_bp.hxx>
#include <opengm/inference/external/libdai/tree_reweighted_bp.hxx>
#include <opengm/inference/external/libdai/double_loop_generalized_bp.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>
#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include"../export_typedes.hxx"
using namespace boost::python;






namespace pylibdaibp{
   template<class PARAM>
   void set
   (
      PARAM  & p,
      const size_t maxIterations,
      const double damping,
      const double tolerance,
      const opengm::external::libdai::BpUpdateRule updateRule,
      const size_t verbose
   ){
      p.maxIterations_=maxIterations;
      p.damping_=damping;
      p.tolerance_=tolerance;
      p.updateRule_=updateRule;
      p.verbose_=verbose;
      p.logDomain_=0;
      p.updateRule_=updateRule;
   }
}

namespace pylibdaitrbp{
   template<class PARAM>
   void set
   (
      PARAM  & p,
      const size_t maxIterations,
      const double damping,
      const double tolerance,
      const size_t ntrees,
      const opengm::external::libdai::BpUpdateRule updateRule,
      const size_t verbose
   ){
      p.maxIterations_=maxIterations;
      p.damping_=damping;
      p.tolerance_=tolerance;
      p.ntrees_=ntrees;
      p.updateRule_=updateRule;
      p.verbose_=verbose;
      p.logDomain_=0;
      p.updateRule_=updateRule;
   }
}

namespace pylibdaidlgbp{
   template<class PARAM>
   void set
   (
      PARAM  & p,
      const bool doubleloop,
      const opengm::external::libdai::Clusters clusters,
      const size_t loopdepth,
      const opengm::external::libdai::Init init,
      const size_t maxiter,
      const double tolerance,
      const size_t verbose
   ){
      p.doubleloop_=doubleloop;
      p.clusters_=clusters;
      p.loopdepth_=loopdepth;
      p.init_=init;
      p.maxiter_=maxiter;
      p.tolerance_=tolerance;
      p.verbose_=verbose;
   }
}


namespace pylibdaijunctiontree{
   template<class PARAM>
   void set
   (
      PARAM  & p,
      opengm::external::libdai::JunctionTreeUpdateRule updateRule,
      opengm::external::libdai::JunctionTreeHeuristic heuristic,
      const size_t verbose
   ){
      p.updateRule_=updateRule;
      p.heuristic_=heuristic;
      p.verbose_=verbose;
   }
}
template<class GM,class ACC>
void export_libdai_inference(){
	import_array();    
   
	typedef GM PyGm;
	typedef typename PyGm::ValueType ValueType;
	typedef typename PyGm::IndexType IndexType;
	typedef typename PyGm::LabelType LabelType;
   // bp
	typedef opengm::external::libdai::Bp<PyGm, ACC>  PyLibdaiBp;
	typedef typename PyLibdaiBp::Parameter PyLibdaiBpParameter;
	typedef typename PyLibdaiBp::VerboseVisitorType PyLibdaiBpVerboseVisitor;
   
	class_<PyLibdaiBpParameter > ("LibDaiBpParameter", init< >() )
   .def ("set", &pylibdaibp::set<PyLibdaiBpParameter>, 
      (
      arg("steps")=100,
      arg("damping")=0.0,
      arg("tolerance")=0.000001,
      arg("updateRule")= opengm::external::libdai::PARALL,
      arg("verbose")= 0
      ) 
   )
   .def_readwrite("steps", &PyLibdaiBpParameter::maxIterations_)
   .def_readwrite("damping", &PyLibdaiBpParameter::damping_)
   .def_readwrite("tolerance", &PyLibdaiBpParameter::tolerance_)
   .def_readwrite("updateRule", &PyLibdaiBpParameter::updateRule_) 
   .def_readwrite("verbose", &PyLibdaiBpParameter::verbose_)  
	;

	OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLibdaiBpVerboseVisitor,"LibDaiVerboseVisitor" );
	OPENGM_PYTHON_INFERENCE_EXPORTER(PyLibdaiBp,"LibDaiBp");
   
   //fractional bp
	typedef opengm::external::libdai::FractionalBp<PyGm, ACC>  PyLibdaiFractionalBp;
	typedef typename PyLibdaiFractionalBp::Parameter PyLibdaiFractionalBpParameter;
	typedef typename PyLibdaiFractionalBp::VerboseVisitorType PyLibdaiFractionalBpVerboseVisitor;
   
	class_<PyLibdaiFractionalBpParameter > ("LibDaiFractionalBpParameter", init< >() )
   .def ("set", &pylibdaibp::set<PyLibdaiFractionalBpParameter>, 
      (
      arg("steps")=100,
      arg("damping")=0.0,
      arg("tolerance")=0.000001,
      arg("updateRule")= opengm::external::libdai::PARALL,
      arg("verbose")= 0
      ) 
   )
   .def_readwrite("steps", &PyLibdaiFractionalBpParameter::maxIterations_)
   .def_readwrite("damping", &PyLibdaiFractionalBpParameter::damping_)
   .def_readwrite("tolerance", &PyLibdaiFractionalBpParameter::tolerance_)
   .def_readwrite("updateRule", &PyLibdaiFractionalBpParameter::updateRule_) 
   .def_readwrite("verbose", &PyLibdaiFractionalBpParameter::verbose_)  
	;

	//OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLibdaiFractionalBpVerboseVisitor,"LibDaiFractionalBpVerboseVisitor" );
	OPENGM_PYTHON_INFERENCE_EXPORTER(PyLibdaiFractionalBp,"LibDaiFractionalBp");
   
   // trbp
   typedef opengm::external::libdai::TreeReweightedBp<PyGm, ACC>  PyLibdaiTrbp;
	typedef typename PyLibdaiTrbp::Parameter PyLibdaiTrbpParameter;
	typedef typename PyLibdaiTrbp::VerboseVisitorType PyLibdaiTrbpVerboseVisitor;
   
   class_<PyLibdaiTrbpParameter > ("LibDaiTrBpParameter", init< >() )
   .def ("set", &pylibdaitrbp::set<PyLibdaiTrbpParameter>, 
      (
      arg("steps")=100,
      arg("damping")=0.0,
      arg("tolerance")=0.000001,
      arg("ntrees")=0,
      arg("updateRule")= opengm::external::libdai::PARALL,
      arg("verbose")= 0
      ) 
   )
   .def_readwrite("steps", &PyLibdaiTrbpParameter::maxIterations_)
   .def_readwrite("damping", &PyLibdaiTrbpParameter::damping_)
   .def_readwrite("tolerance", &PyLibdaiTrbpParameter::tolerance_)
   .def_readwrite("ntrees", &PyLibdaiTrbpParameter::ntrees_)
   .def_readwrite("updateRule", &PyLibdaiTrbpParameter::updateRule_) 
   .def_readwrite("verbose", &PyLibdaiTrbpParameter::verbose_)  
	;

	//OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLibdaiTrbpVerboseVisitor,"LibDaiTrBpVerboseVisitor" );
	OPENGM_PYTHON_INFERENCE_EXPORTER(PyLibdaiTrbp,"LibDaiTrBp");
   
   /*
   //double loop genenralized  bp
   typedef opengm::external::libdai::DoubleLoopGeneralizedBP<PyGm, ACC>  PyLibdaiDoubleLoopGBP;
	typedef typename PyLibdaiDoubleLoopGBP::Parameter PyLibdaiDoubleLoopGBPParameter;
	typedef typename PyLibdaiDoubleLoopGBP::VerboseVisitorType PyLibdaiDoubleLoopGBPVerboseVisitor;

   class_<PyLibdaiDoubleLoopGBPParameter > ("LibDaiDoubleLoopGbpParameter", init< >() )
   .def ("set", &pylibdaidlgbp::set<PyLibdaiDoubleLoopGBPParameter>, 
      (
      arg("doubleloop")=1,
      arg("clusters")=opengm::external::libdai::BETHE,
      arg("loopdepth")=3,
      arg("init")=opengm::external::libdai::UNIFORM,
      arg("steps")= 10000,
      arg("tolerance")= 1e-9,
      arg("verbose")= 0
      ) 
   )
   .def_readwrite("doubleloop", &PyLibdaiDoubleLoopGBPParameter::doubleloop_)
   .def_readwrite("clusters", &PyLibdaiDoubleLoopGBPParameter::clusters_)
   .def_readwrite("loopdepth", &PyLibdaiDoubleLoopGBPParameter::loopdepth_)
   .def_readwrite("init", &PyLibdaiDoubleLoopGBPParameter::init_)
   .def_readwrite("steps", &PyLibdaiDoubleLoopGBPParameter::maxiter_)
   .def_readwrite("tolerance", &PyLibdaiDoubleLoopGBPParameter::tolerance_)
   .def_readwrite("verbose", &PyLibdaiDoubleLoopGBPParameter::verbose_)
	;
   */

	//OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLibdaiDoubleLoopGBPVerboseVisitor,"LibDaiDoubleLoopGbperboseVisitor" );
	//OPENGM_PYTHON_INFERENCE_EXPORTER(PyLibdaiDoubleLoopGBP,"LibDaiDoubleLoopGbp");
   
   
   // junction tree
   
   // opengm::external::libdai::JunctionTreeUpdateRule updateRule= HUGIN,
   // opengm::external::libdai::JunctionTreeHeuristic heuristic=MINWEIGHT,
   typedef opengm::external::libdai::JunctionTree<PyGm, ACC>  PyJunctionTree;
	typedef typename PyJunctionTree::Parameter PyJunctionTreeParameter;
	typedef typename PyJunctionTree::VerboseVisitorType PyJunctionTreeVerboseVisitor;

   class_<PyJunctionTreeParameter > ("LibDaiJunctionTreeParameter", init< >() )
   .def ("set", &pylibdaijunctiontree::set<PyJunctionTreeParameter>, 
      (
      arg("updateRule")=opengm::external::libdai::HUGIN,
      arg("heuristic")=opengm::external::libdai::MINWEIGHT,
       arg("verbose")=0
      ) 
   )
   .def_readwrite("updateRule", &PyJunctionTreeParameter::updateRule_)
   .def_readwrite("heuristic", &PyJunctionTreeParameter::heuristic_)
   .def_readwrite("verbose", &PyJunctionTreeParameter::verbose_)
	;

	//OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyJunctionTreeVerboseVisitor,"LibDaiJunctionTreeVerboseVisitor" );
	OPENGM_PYTHON_INFERENCE_EXPORTER(PyJunctionTree,"LibDaiJunctionTree");
}

template void export_libdai_inference<GmAdder, opengm::Minimizer>();
template void export_libdai_inference<GmAdder, opengm::Maximizer>();
template void export_libdai_inference<GmMultiplier, opengm::Minimizer>();
template void export_libdai_inference<GmMultiplier, opengm::Maximizer>();

#endif
