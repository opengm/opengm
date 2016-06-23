#if defined(WITH_CPLEX) || defined(WITH_GUROBI)

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/learning/maximum_likelihood_learning.hxx>

#define DefaultErrorFn DefaultErrorFn_TrwsExternal_ML
#include "helper.hxx"

namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;

namespace opengm{


    template<class PARAM>
    PARAM * pyMaxLikelihoodParamConstructor(
	size_t maximumNumberOfIterations=100,
	double gradientStepSize=0.1,
	double weightStoppingCriteria=0.00000001,
	double gradientStoppingCriteria=0.00000001,
	bool infoFlag=true,
	bool infoEveryStep=false,
	double weightRegularizer = 1.0,
	size_t beliefPropagationMaximumNumberOfIterations = 20,
	double beliefPropagationConvergenceBound = 0.0001,
	double beliefPropagationDamping = 0.5,
	double beliefPropagationTemperature = 0.3,
	opengm::Tribool beliefPropagationIsAcyclic=opengm::Tribool(opengm::Tribool::Maybe)
    ){
        PARAM * p  = new PARAM();
	p->maximumNumberOfIterations_ = maximumNumberOfIterations;
	p->gradientStepSize_ = gradientStepSize;
	p->weightStoppingCriteria_ = weightStoppingCriteria;
	p->gradientStoppingCriteria_ = gradientStoppingCriteria;
	p->infoFlag_ = infoFlag;
	p->infoEveryStep_ = infoEveryStep;
	p->weightRegularizer_ = weightRegularizer;
	p->beliefPropagationMaximumNumberOfIterations_ = beliefPropagationMaximumNumberOfIterations;
	p->beliefPropagationConvergenceBound_ = beliefPropagationConvergenceBound;
	p->beliefPropagationDamping_ = beliefPropagationDamping;
	p->beliefPropagationTemperature_ = beliefPropagationTemperature;
	p->beliefPropagationIsAcyclic_ = beliefPropagationIsAcyclic;
        return p;
    }

    template<class DATASET>
    void export_max_likelihood_learner(const std::string & clsName){
        typedef learning::MaximumLikelihoodLearner<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;
        typedef typename PyLearner::DatasetType DatasetType;

        const std::string paramClsName = clsName + std::string("Parameter");

        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
	  .def("__init__", make_constructor(&pyMaxLikelihoodParamConstructor<PyLearnerParam> ,boost::python::default_call_policies()))
	  //.def_readwrite("maxIterations", &PyLearnerParam::maximumNumberOfIterations_)
        ;

        boost::python::class_<PyLearner>( clsName.c_str(), boost::python::init<DatasetType &, const PyLearnerParam &>() )
            .def("learn",&PyLearner::learn)
        ;
    }

  //template void
  //export_max_likelihood_learner<op::GmAdderHammingLossDataset> (const std::string& className);

    template void
    export_max_likelihood_learner<op::GmAdderFlexibleLossDataset> (const std::string& className);
}



#endif

