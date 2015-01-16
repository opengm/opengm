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
	size_t maximumNumberOfIterations=1234567,
	double gradientStepSize=0.1234567,
	double weightStoppingCriteria=0.00001234567,
	double gradientStoppingCriteria=0.00000001234567,
	bool infoFlag=true,
	bool infoEveryStep=false,
	size_t beleifPropagationMaximumNumberOfIterations = 30,
	double beleifPropagationConvergenceBound = 0.00011,
	double beleifPropagationDamping = 0.55,
	double beleifPropagationReg = 1.00000001,
	double beleifPropagationTemperature = 0.3000000001
    ){
        PARAM * p  = new PARAM();
	p->maximumNumberOfIterations_ = maximumNumberOfIterations;
	p->gradientStepSize_ = gradientStepSize;
	p->weightStoppingCriteria_ = weightStoppingCriteria;
	p->gradientStoppingCriteria_ = gradientStoppingCriteria;
	p->infoFlag_ = infoFlag;
	p->infoEveryStep_ = infoEveryStep;
	p->beliefPropagationMaximumNumberOfIterations_ = beleifPropagationMaximumNumberOfIterations;
	p->beliefPropagationConvergenceBound_ = beleifPropagationConvergenceBound;
	p->beliefPropagationDamping_ = beleifPropagationDamping;
	p->beliefPropagationReg_ = beleifPropagationReg;
	p->beliefPropagationTemperature_ = beleifPropagationTemperature;
        return p;
    }

    template<class DATASET>
    void export_max_likelihood_learner(const std::string & clsName){
        typedef learning::MaximumLikelihoodLearner<DATASET> PyLearner;
        typedef typename PyLearner::Parameter PyLearnerParam;
        typedef typename PyLearner::DatasetType DatasetType;

        const std::string paramClsName = clsName + std::string("Parameter");

        bp::class_<PyLearnerParam>(paramClsName.c_str(), bp::init<>())
	  //.def("__init__", make_constructor(&pyMaxLikelihoodParamConstructor<PyLearnerParam> ,boost::python::default_call_policies()))
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

