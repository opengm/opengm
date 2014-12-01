/*
 * smoothingToMarginals.hxx
 *
 *  Created on: Dec 1, 2014
 *      Author: bsavchyn
 */

#ifndef SMOOTHINGTOMARGINALS_HXX_
#define SMOOTHINGTOMARGINALS_HXX_

namespace opengm{

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
