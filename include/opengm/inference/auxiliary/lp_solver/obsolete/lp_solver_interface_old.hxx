#pragma once
#ifndef OPENGM_LP_SOLVER_INTERFACE_HXX
#define OPENGM_LP_SOLVER_INTERFACE_HXX

namespace opengm{

	class LpSolverInterface{
	public:
		enum LpVarType{
			Continous = 0,
			Binary    = 1,
			Integer   = 2
		};
	};
}

#endif




