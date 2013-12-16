#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <opengm/inference/inference.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/auxiliary/fusion_move/fusion_mover.hxx>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


// Fusion Move Solver
#include "opengm/inference/astar.hxx"
#include "opengm/inference/lazyflipper.hxx"
#include "opengm/inference/infandflip.hxx"
#include <opengm/inference/messagepassing/messagepassing.hxx>



#ifdef WITH_AD3
#include "opengm/inference/external/ad3.hxx"
#endif
#ifdef WITH_CPLEX
#include "opengm/inference/lpcplex.hxx"
#endif
#ifdef WITH_QPBO
#include "QPBO.h"
#endif


template<class GM,class ACC>
class PythonFusionMover{
	typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef opengm::FusionMover<GM,ACC> CppFusionMover;




	typedef typename CppFusionMover::SubGmType 				SubGmType;
	// sub-inf-astar
	typedef opengm::AStar<SubGmType,AccumulationType> 			AStarSubInf;
	// sub-inf-lf
	typedef opengm::LazyFlipper<SubGmType,AccumulationType> 	LazyFlipperSubInf;
	// sub-inf-bp
	typedef opengm::BeliefPropagationUpdateRules<SubGmType,AccumulationType> 						UpdateRulesType;
    typedef opengm::MessagePassing<SubGmType,AccumulationType,UpdateRulesType, opengm::MaxDistance> BpSubInf;
    // sub-inf-bp-lf
    typedef opengm::InfAndFlip<SubGmType,AccumulationType,BpSubInf>        BpLfSubInf;


	//#ifdef WITH_AD3
	//typedef opengm::external::AD3Inf<SubGmType,AccumulationType> 	Ad3SubInf;
	//#endif
	#ifdef WITH_QPBO
	typedef kolmogorov::qpbo::QPBO<double> 			  				QpboSubInf;
	#endif
	#ifdef WITH_CPLEX
	typedef opengm::LPCplex<SubGmType,AccumulationType> 			CplexSubInf;
	#endif



public:
	PythonFusionMover(const GM & gm)
	:	gm_(gm),
		fusionMover_(gm),
		argA_(gm.numberOfVariables()),
		argB_(gm.numberOfVariables()),
		argR_(gm.numberOfVariables()),
		factorOrder_(gm.factorOrder())
	{

	}	

	boost::python::tuple fuse(
		opengm::python::NumpyView<LabelType,1> labelsA,
		opengm::python::NumpyView<LabelType,1> labelsB,
		const std::string & fusionSolver
	)
	{
		// copy input
		//#std::cout<<"copy input\n";
		std::copy(labelsA.begin(),labelsA.end(),argA_.begin());
		std::copy(labelsB.begin(),labelsB.end(),argB_.begin());

		// do setup
		//#std::cout<<"setup\n";
		fusionMover_.setup(
			argA_,argB_,argR_,
			gm_.evaluate(argA_.begin()),gm_.evaluate(argB_.begin())
		);
		//std::cout<<"do fusion\n";
		// do the fusion 
		const ValueType resultValue = this->doFusion(fusionSolver);

		//std::cout<<"make result\n";
		// return result
	  	return boost::python::make_tuple(
	  		opengm::python::iteratorToNumpy(argR_.begin(), argR_.size()),
	  		fusionMover_.valueResult(),
	  		fusionMover_.valueA(),
	  		fusionMover_.valueB()
	  	);
	}

private:

	ValueType doFusion(const std::string & fusionSolver){
		if(fusionSolver==std::string("qpbo")){
			#ifdef WITH_QPBO
			if(factorOrder_<=2){
				return fusionMover_. template fuseQpbo<QpboSubInf> ();
			}
			else{
				return fusionMover_. template fuseFixQpbo<QpboSubInf> ();
			}
			#else
				OPENGM_CHECK(false,"qpbo fusion solver need WITH_QPBO");
			#endif 
		}
		else if(fusionSolver==std::string("lf2")){
			return  fusionMover_. template fuse<LazyFlipperSubInf> (typename LazyFlipperSubInf::Parameter(2),true);
		}
		else if(fusionSolver==std::string("lf3")){
			return  fusionMover_. template fuse<LazyFlipperSubInf> (typename LazyFlipperSubInf::Parameter(3),true);
		}

	}




	const GM & gm_;
	CppFusionMover fusionMover_;
	std::vector<LabelType> argA_;
	std::vector<LabelType> argB_;
	std::vector<LabelType> argR_;
	size_t factorOrder_;
};



template<class GM,class ACC>
void export_fusion_moves(){
   using namespace boost::python;
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   boost::python::docstring_options docstringOptions(true,true,false);
   
   import_array();


   typedef PythonFusionMover<GM,ACC> PyFusionMover;



	boost::python::class_<PyFusionMover > ("FusionMover",
	boost::python::init<const GM &>("Construct a FusionMover from a graphical model ")
	[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const PyGM& */>()]
	)
	.def("fuse",&PyFusionMover::fuse)
	;
}


template void export_fusion_moves<opengm::python::GmAdder ,opengm::Minimizer>();

