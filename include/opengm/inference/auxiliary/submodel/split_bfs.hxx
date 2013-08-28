





template<class GM>
void splitBFS(const GM & gm ,std::vector<  typename GM::LabelType >  resultArg ,const typename GM::IndexType approxSize ){

	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;

	std::vector<bool> usedVar(gm.numberOfVariables(),false);
	std::vector<bool> usedFac(gm.numberOfFactors(),false);

	resultArg.resize(gm.numberOfVariables());
	std::queue<IndexType> queue;
	LabelType currentLabel = 0 ;

	// iterate over all variables and try to grow a 
	// connected comp. with size ~= approxSize
	// arround each unuused variable
	for(IndexType vi=0;vi<gm.numberOfVariables();++vi){

		// skip the variable if has already been 
		// added to a connected comp. 
		if(usedVar[fi]==false){

			// grow a  connected comp.  until the size of the 
			// connected comp.  is <= approxSize 
			// (it might get even a bit bigger .. see  (*ref*1*ref*) )
			// or it is not possible to grow the connected comp. any more
			queue.clear();
			queue.push_back(vi);
			while( queue.size()<approxSize  &&  !queue.empty() ){

				// get front and pop front and make used
				const IndexType cVi = queue.front();
				queue.pop_front();
				usedVar[cVi]=true;
				resultArg[cVi]=currentLabel;
				IndexType numFac = gm.numberOfFactors(cVi);

				// iterate over all factor of of the variable cVi
				// - the factor must have order >=2 
				// - the factor must be unused
				for(IndexType f=0;f<numFac;++f){

					// check the factor must have order >=2  and the factor must be unused
					const IndexType fi = gm.factorOfVariable(cVi,f);
					if(gm[fi].numberOfVariables()>1  && usedFac[fi]==false ){

						// make factor used
						usedFac[fi]=true;

						// iterate over all variables of the factor 
						// - variable must be unused
						const IndexType nVar = gm[fi].numberOfVariables();
						for(IndexType v=0;v<nVar;++v){

							// push_back the variable to queue even if the 
							// queue 's size is bigger than approxSize  (*ref*1*ref*)
							const IndexType viOfFac=gm.variableOfFactor(fi,v);
							queue.push_back(viOfFac); 
						}
					}
				}
			}
			// end of grow 
			// - clear queue (might be not nessesary but hey,let's do it anyway)
			// - increment the current label since we grow another connected comp. now
			queue.clear();
			IndexType+=1;
		}
	}
}