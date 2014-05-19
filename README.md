TODO
====

Thorsten:

	Dataset
         - void   getModel(const size_t, GM&) 
         - void   getGT(const size_t, std::vector<LabelType>&)
         - size_t getNumberOfParameters() 

         Maybe GM should be templated?!?

Jan:

	StructMaxMargin
	Optimizer
		BundleMethodOptimizer
		SubgradientOptimizer

Jörg:

	SampleLearning
	LossGenerator
		HammingLossGenerator



Open Points/Questions:

* Rename Patameters<V,I> into ModelParameters;
* Replace V, I in Patameters<V,I> by fix Types double and size_t ?!?
* More efficient Model Views (maybe on function level)
* What interfaces does learners need?
