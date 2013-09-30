namespace opengm{






/*

DETERMINISTIC MOVES :
	- hamming - distance optimality 
	- tree  optimal 
	- non opt-bfs icm optimality  (icm with large sg-size and solver like bp)
	- alpha expansion optimality

Random Moves :
	- LOC

*/


















template<class GM,class ACC>
class GeneralizedMoveMaking{

	public:
		typedef GM GraphicalModelType;
		typedef ACC AccumulationType;
		OPENGM_GM_TYPE_TYPEDEFS;


		struct Parameter{
			Parameter(
				const std::string & workflow = '( AEF,LF2,LOC2  )'
			) 

			std::string workflow_;
		};




		// helpers
		template<class VI_ITER>
		IndexType growBall(const IndexType startVi,const IndexType radius,const IndexType maxSize, VI_ITER visInBall)









		template<class VI_ITER,class ARG_ITER>
		bool moveOptimal(VI_ITER viBegin,VI_ITER viEnd,ARG_ITER argBegin);



		void naiveSubgraphIcm(const MoveInput moveInput &, const size_t maxSubgraphSize=25){
			const IndexType radius=maxSubgraphSize;
			bool anyChanges=true;
			while(anyChanges){
				bool anyChanges=false;
				for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
					if(!localOptimal_[vi]){
						// grow ball around vis
						const IndexType nSubmodelVis = growBall(vi,radius,0,submodelVis_.begin());
						// move variables optimal
						const bool changes = moveOptimal(submodelVis_.begin(), submodelVis_.begin()+nSubmodelVis,submodelArg_);
						// get changes and properate them
						if(changes){
							anyChanges=true;
							// update dirtyness
							// - grow subgraph around all variables which changed
							// - and set them to dirty (all vis in submodelVis_
							//   will be set undirty later)
							for(IndexType localVi=0;localVi<nSubmodelVis;++localVi){
								const IndexType globalVi = submodelVis_[localVi];

								// if arg changed 
								if (labels_[globalVi]!=submodelArg_[localVi]){
									// grow ball around vis
									const IndexType nAffectedVis = growBall(globalVi,radius,0,visBuffer_.begin());
									// set all var in this ball to false (will set all var in the just solve
									// graph to optimal later )
									for(IndexType vb=0;vb<nAffectedVis;++vb){
										const IndexType visInBall = visBuffer_[vb];
										localOptimal_[vi]=false;
									}
								}
							}
							// 
							// write labels into global labels
							// and update dirtyness
							// - make just optimized variables optimal
							for(IndexType localVi=0;localVi<nSubmodelVis;++localVi){
								const IndexType globalVi = submodelVis_[localVi];
								localOptimal_[vi]=true;
							}
						}
					}
				}
			}
		}


	private:
		std::vector<LabelType> labels_;


		std::vector<bool> localOptimal_;
		std::vector<IndexType> submodelVis_;
		std::vector<IndexType> visBuffer_;
		std::vector<IndexType> submodelArg_;
};

}