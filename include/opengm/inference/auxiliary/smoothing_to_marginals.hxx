/*
 * smoothingToMarginals.hxx
 *
 *  Created on: Dec 1, 2014
 *      Author: bsavchyn
 */

#ifndef SMOOTHINGTOMARGINALS_HXX_
#define SMOOTHINGTOMARGINALS_HXX_

namespace opengm{

//! [function setSmoothingParametersForMarginals]
/// setSmoothingParametersForMarginals - adjusts parameters of smoothing-based algorithms (NesterovAcceleratedGradient and ADSal)
/// to obtain estimations of sum-prod margonals.
///
/// For mathematical details see the papers:
/// B. Savchynskyy, J. H. Kappes, S. Schmidt, C. Schnörr
/// A Study of Nesterov's Scheme for Lagrangian Decomposition and MAP Labeling, in CVPR 2011
/// and
/// B. Savchynskyy, S. Schmidt, J. H. Kappes, C. Schnörr
/// Efficient MRF Energy Minimization via Adaptive Diminishing Smoothing, In UAI, 2012, pp. 746-755
///
///
/// Usage examples:
///
////* With NesterovAcceleratedGradient: */
///NesterovAcceleratedGradient<GraphicalModelType,Minimizer>::Parameter params;
///setSmoothingParametersForMarginals(params,100,1.0);
///NesterovAcceleratedGradient<GraphicalModelType,Minimizer> solver(gm,params);
///solver.infer();
///
///GraphicalModelType::IndependentFactorType out;
///for (size_t i=0;i<gm.numberOfVariables();++i)
///{
///	solver.marginal(i,out_nest);
///			  .../* do with the 'out' marginals what you want */
///}
///
////* With ADSal: */
///ADSal<GraphicalModelType,Minimizer>::Parameter params;
///setSmoothingParametersForMarginals(params,100,1.0);
///ADSal<GraphicalModelType,Minimizer> solver(gm,params);
///solver.infer();
///GraphicalModelType::IndependentFactorType out;
///for (size_t i=0;i<gm.numberOfVariables();++i)
///{
///	solver.marginal(i,out);
///			  .../* do with the 'out' marginals what you want */
///}
///
///
///
/// Corresponding author: Bogdan Savchynskyy
///
///\ingroup inference

template<class PARAMETERS>
void setSmoothingParametersForMarginals(PARAMETERS& params,
		 	 	 	 	 	 	 	 	size_t numIterations,
		 	 	 	 	 	 	 	 	typename PARAMETERS::ValueType temperature=1.0,
		 	 	 	 	 	 	 	    typename PARAMETERS::Storage::StructureType decompositionType=PARAMETERS::Storage::GENERALSTRUCTURE)
{
	params.maxNumberOfIterations()=1;
	params.numberOfInternalIterations()=numIterations;
	params.setStartSmoothingValue(temperature);
	params.smoothingStrategy()=PARAMETERS::SmoothingParametersType::FIXED;
	params.maxNumberOfPresolveIterations()=0;
	params.setPrecision(0);
	params.lazyLPPrimalBoundComputation()=true;
	params.maxPrimalBoundIterationNumber()=1;
	params.decompositionType()=decompositionType;
}


}


#endif /* SMOOTHINGTOMARGINALS_HXX_ */
