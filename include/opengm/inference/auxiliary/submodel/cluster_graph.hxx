#include <opengm/opengm.hxx>

namespace opengm{


	template<class GM>
	class ClusterGraph{
	public:

		typedef typename GM::IndexType IndexType;
		typedef typename GM::LabelType LabelType;
		typedef typename GM::ValueType ValueType;


		ClusterGraph(const GM & gm)
		:	gm_(gm),
			numCluster_(0)
			clusterMembership_(gm.numberOfVariables()),
			explicitClustering_(),
			neighbourClusters_()
		{
			gm_.variableAdjacenyList(neighbourVis_);
		}


		template<class CLUSTERING_ITER>
		void setClustering(const IndexType numClusters,CLUSTERING_ITER begin,CLUSTERING_ITER end){

			OPENGM_CHECK_OP(std::distance(begin,end),==,gm_.numberOfVariables());

			numCluster_=numClusters;
			explicitClustering_.clear();
			neighbourClusters_.clear();
			explicitClustering_.resize(numCluster_);
			neighbourClusters_.resize(numCluster_);

			for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
				const IndexType clusterLabel = static_cast<IndexType>(begin[vi]);
				clusterMembership_[vi]=clusterLabel;
				explicitClustering_[clusterLabel].push_back(vi);
			}

			for(IndexType vi=0;vi<gm_.numberOfVariables();++vi){
				const IndexType nNvis=neighbourVis_[vi].size();
				for(IndexType vo=0;vo<nNvis;++vo){
					const IndexType vio=neighbourVis_[vi][vo];

					const IndexType clusterVi  = clusterMembership_[vi];
					const IndexType clusterVio = clusterMembership_[vio];

					if(clusterVi!=clusterVio){
						neighbourClusters_[clusterVi ].insert(clusterVio);
						neighbourClusters_[clusterVio].insert(clusterVi );
					}
				}
			}
		}

		IndexType numberOfClusters()const{
			return numCluster_;
		}

		IndexType numberOfClusterNeigbours(const IndexType ci)const{
			return neighbourClusters_[ci].size();
		}

		IndexType clusterNeigbour(const IndexType ci,const IndexType i)const{
			return neighbourClusters_[ci][i];
		}

		IndexType clusterIndex(const IndexType vi)const{
			return clusterMembership_[vi];
		}

		IndexType clusterSize(const IndexType ci)const{
			return explicitClustering_[ci].size();
		}


	private:

		const GM & gm_;
		std::vector<RandomAccessSet<IndexType>   neighbourVis_;
		const IndexType 						 numCluster_;
		std::vector<LabelType>  				 clusterMembership_;
		std::vector<std::vector<IndexType> > 	 explicitClustering_;
		std::vector< RandomAccessSet<IndexType>  neighbourClusters_;
		
	};	

}