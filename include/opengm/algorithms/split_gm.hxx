#include <vector>
#include <queue>

namespace opengm{


	struct VarState{
		enum State{
			StateA=0,
			StateB=1,
			NotAssigned=2,
			InQueue=3,
			NotIncluded=4
		};
	};

	template<class GM>
	void graphBisection(
		const GM & gm,
		const typename GM::IndexType viA,
		const typename GM::IndexType viB,
		const std::vector<typename GM::IndexType> &  visToSplit,
		std::vector<unsigned char> & varState
	){
		typedef typename GM::IndexType IndexType;


		std::vector<RandomAccessSet<IndexType> > visAdj;
		gm.variableAdjacencyList(visAdj);

		
		varState.resize(gm.numberOfVariables());
		std::fill(varState.begin(),varState.end(),VarState::NotIncluded);

		for(IndexType i=0;i<visToSplit.size();++i){
			OPENGM_CHECK_OP(visToSplit[i],<,gm.numberOfVariables(),"");
			varState[visToSplit[i]]=VarState::NotAssigned;
		}

		const IndexType  nVisToSplit=visToSplit.size();

		OPENGM_CHECK_OP(viA,<,gm.numberOfVariables(),"");
		OPENGM_CHECK_OP(viB,<,gm.numberOfVariables(),"");

		OPENGM_CHECK_OP(nVisToSplit,>=,2,"isToSplit must be >= 2 ");
		OPENGM_CHECK(varState[viA]==VarState::NotAssigned,"varA not i  visToSplit");
		OPENGM_CHECK(varState[viB]==VarState::NotAssigned,"varA not i  visToSplit");

		IndexType nStates[2]={1,1};
		IndexType nNotAssigned=nVisToSplit-2;
		varState[viA]=VarState::StateA;
		varState[viB]=VarState::StateB;

		typedef std::queue<IndexType> QueueType;
		std::queue<IndexType>  queueA,queueB;
		std::vector<IndexType> setVis[2];
		queueA.push(viA);
		queueB.push(viB);
		


		while(nNotAssigned!=0 && ( !queueA.empty() || !queueB.empty() ) ){


			std::cout<<"nA "<<nStates[0]<<" nB "<<nStates[1] << " nNotAssigned "<<nNotAssigned<<"\n";
			OPENGM_CHECK_OP(nStates[0]+nStates[1],==,nVisToSplit- nNotAssigned,"internal invariant is violated");

			// iterate over both queue's

			const IndexType sOrder []={queueA.size()<queueB.size() ? 0 : 1,queueA.size()<queueB.size() ?1 : 0} ;


			for(IndexType si=0;si<2;++si){

				const IndexType setNr = sOrder[si];


				QueueType & queue = setNr==0 ? queueA : queueB;

				if(!queue.empty()){
					const IndexType vv=queue.front();
					queue.pop();
					
					if(varState[vv]!=setNr){
						OPENGM_CHECK(varState[vv]!=VarState::NotIncluded,"");
						OPENGM_CHECK(varState[vv]!=VarState::NotAssigned,"")
						OPENGM_CHECK(varState[vv]!= (setNr==0 ? 1 :0 ) ,"");
						varState[vv]=setNr;
						varState.push_back(vv);
						++nStates[setNr];
						--nNotAssigned;
					}


					// extend queue

					const IndexType nVV=visAdj[vv].size();
					for(IndexType nv=0;nv<nVV;++nv){
						const IndexType aVi=visAdj[vv][nv];
						const unsigned char state=varState[aVi];
						if(state==VarState::NotAssigned){
							queue.push(aVi);
							varState[aVi]=VarState::InQueue;
						}
					}
				}
			}
		}
		OPENGM_CHECK_OP(nNotAssigned,==,0,"there are unassigned variables left");
	}




	template<class GM>
	void recuriveGraphBisectionHelper(
		const GM & gm,
		const typename GM::IndexType viA,
		const typename GM::IndexType viB,
		std::vector<typename GM::IndexType> & finalResult,
		const size_t levels,
		const std::vector<typename GM::IndexType> &  visToSplit,
		std::vector<unsigned char> & varState,
		size_t & enumeration ,
		bool first=false
	){
		typedef typename GM::IndexType IndexType;

		if(finalResult.size()<gm.numberOfVariables()){
			finalResult.resize(gm.numberOfVariables());
		}

		if(levels>0){
			if(!first)
				enumeration+=2;
			std::vector<RandomAccessSet<IndexType> > visAdj;
			gm.variableAdjacencyList(visAdj);

			
			varState.resize(gm.numberOfVariables());
			std::fill(varState.begin(),varState.end(),VarState::NotIncluded);

			for(IndexType i=0;i<visToSplit.size();++i){
				OPENGM_CHECK_OP(visToSplit[i],<,gm.numberOfVariables(),"");
				varState[visToSplit[i]]=VarState::NotAssigned;
			}

			const IndexType  nVisToSplit=visToSplit.size();

			OPENGM_CHECK_OP(viA,<,gm.numberOfVariables(),"");
			OPENGM_CHECK_OP(viB,<,gm.numberOfVariables(),"");

			OPENGM_CHECK_OP(nVisToSplit,>=,2,"isToSplit must be >= 2 ");
			OPENGM_CHECK(varState[viA]==VarState::NotAssigned,"varA not i  visToSplit");
			OPENGM_CHECK(varState[viB]==VarState::NotAssigned,"varA not i  visToSplit");

			IndexType nStates[2]={1,1};
			IndexType nNotAssigned=nVisToSplit-2;
			varState[viA]=VarState::StateA;
			varState[viB]=VarState::StateB;

			typedef std::queue<IndexType> QueueType;
			std::queue<IndexType>  queueA,queueB;
			std::vector<IndexType> setVis[2];
			queueA.push(viA);
			queueB.push(viB);
			


			while(nNotAssigned!=0 && ( !queueA.empty() || !queueB.empty() ) ){


				//std::cout<<"nA "<<nStates[0]<<" nB "<<nStates[1] << " nNotAssigned "<<nNotAssigned<<"\n";
				OPENGM_CHECK_OP(nStates[0]+nStates[1],==,nVisToSplit- nNotAssigned,"internal invariant is violated");

				// iterate over both queue's

				const IndexType sOrder []={queueA.size()<queueB.size() ? 0 : 1,queueA.size()<queueB.size() ?1 : 0} ;


				for(IndexType si=0;si<2;++si){

					const IndexType setNr = sOrder[si];


					QueueType & queue = setNr==0 ? queueA : queueB;

					if(!queue.empty()){
						const IndexType vv=queue.front();
						queue.pop();
						
						if(varState[vv]!=setNr){
							OPENGM_CHECK(varState[vv]!=VarState::NotIncluded,"");
							OPENGM_CHECK(varState[vv]!=VarState::NotAssigned,"")
							OPENGM_CHECK(varState[vv]!= (setNr==0 ? 1 :0 ) ,"");
							varState[vv]=setNr;
							varState.push_back(vv);
							++nStates[setNr];
							--nNotAssigned;
						}


						// extend queue

						const IndexType nVV=visAdj[vv].size();
						for(IndexType nv=0;nv<nVV;++nv){
							const IndexType aVi=visAdj[vv][nv];
							const unsigned char state=varState[aVi];
							if(state==VarState::NotAssigned){
								queue.push(aVi);
								varState[aVi]=VarState::InQueue;
							}
						}
					}
				}
			}
			OPENGM_CHECK_OP(nNotAssigned,==,0,"there are unassigned variables left");


			//std::cout<<"round one finished";

			std::vector<IndexType> visToSplitA,visToSplitB;
			visToSplitA.reserve(nStates[0]);
			visToSplitB.reserve(nStates[0]);

			IndexType minA=gm.numberOfVariables();
			IndexType minB=gm.numberOfVariables();
			IndexType maxA=0;
			IndexType maxB=0;



			for(IndexType v=0;v<nVisToSplit;++v){
				const IndexType vi=visToSplit[v];

				if(varState[vi]==VarState::StateA){
					visToSplitA.push_back(vi);
					finalResult[vi]=enumeration;
					minA = std::min(minA,vi);
					maxA = std::max(minA,vi);
				}
				else if(varState[vi]==VarState::StateB){
					visToSplitB.push_back(vi);
					finalResult[vi]=enumeration+1;
					minB = std::min(minB,vi);
					maxB = std::max(maxB,vi);
				}
				else{
					OPENGM_CHECK(false,"");
				}

			}

			//std::cout<<"start recursion  A \n";
			//enumeration+=2;
			recuriveGraphBisectionHelper(
				gm,
				minA,
				maxA,
				finalResult,
				levels-1,
				visToSplitA,
				varState,
				enumeration
			);

			//std::cout<<"start recursion  B \n";
			//enumeration+=2;
			recuriveGraphBisectionHelper(
				gm,
				minB,
				maxB,
				finalResult,
				levels-1,
				visToSplitB,
				varState,
				enumeration
			);
			


		}
	}


	template<class GM>
	size_t recuriveGraphBisection(
		const GM & gm,
		const typename GM::IndexType viA,
		const typename GM::IndexType viB,
		const std::vector<typename GM::IndexType> &  visToSplit,
		const size_t levels,
		std::vector<typename GM::IndexType> & finalResult
	){

		typedef typename GM::IndexType IndexType;
		finalResult.resize(gm.numberOfVariables());
		std::vector<unsigned char>  varState;
		size_t enumeration=0;
		recuriveGraphBisectionHelper(
			gm,viA,viB,finalResult,levels,visToSplit,varState,enumeration,true
		);

		IndexType minLabel=gm.numberOfVariables();

		std::map<IndexType,IndexType>  mapping;
		IndexType di=0;

		for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
			const IndexType nd=finalResult[vi];
			if(mapping.find(nd)==mapping.end()){
				mapping[nd]=di;
				++di;
			}

		}


		for(IndexType vi=0;vi<gm.numberOfVariables();++vi){
			const IndexType nd=finalResult[vi];
			finalResult[vi]=mapping[nd];
		}
		return di;
	}

}
