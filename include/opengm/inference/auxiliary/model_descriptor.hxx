



class ModelDescriptor{

	// variable counting
	UInt64Type  nVariables_;
	UInt64Type  nFactor_;
	UInt64Type  nLpVariables_;

	// factor order
	UInt64Type minFactorOrder_;
	UInt64Type maxFactorOder_;
	float avergeHighOrderFactorOrder_;
	float avergeFactorOrder_;

	// factor density (and unary density)
	float averageVariableWithoutUnaries_;
	float averageVariableWithoutInformativeUnaries_;
	float averageHighOrderFactorsPerVariable_;

	// label space
	UInt64Type minLabels_;
	UInt64Type maxLabels_;
	float averageLabels_;

}