

template<class GM>
class SubmodelStructureGenerator{

	typedef typename  GM::IndexType IndexType;


	// all structues can be combined


	// - block like 
	void addBfsBlock(const IndexType startVi,const IndexType maxRadius,const IndexType maxSize=0,const IndexType maxOrder=0);
	void addDfsBlock(const IndexType startVi,const IndexType maxRadius,const IndexType maxSize=0,const IndexType maxOrder=0);

 	// - tree / line like 
	void addFanTree(const IndexType startVi,const IndexType maxLength=0,const IndexType maxSize=0,const IndexType maxOrder=0);
	void addBfsTree(const IndexType startVi,const IndexType maxRadius=0,const IndexType maxSize=0,const IndexType maxOrder=0);
	void addLine(const IndexType startVi,const IndexType maxSize=0 );
	void addPath(const IndexType startVi,const IndexType endVi=0 );



	

	// setting and free's
	void freeStructure();
	void allowRandomization(bool allow);


}