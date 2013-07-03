#ifndef TRWS_INTERFACE_HXX_
#define TRWS_INTERFACE_HXX_
#include <opengm/inference/inference.hxx>
#include <opengm/inference/trws/trws_base.hxx>

namespace opengm{

template<class GM>
struct TRWSi_Parameter : public trws_base::MaxSumTRWS_Parameters<typename GM::ValueType>
{
	typedef typename GM::ValueType ValueType;
	typedef trws_base::MaxSumTRWS_Parameters<ValueType> parent;
	typedef trws_base::DecompositionStorage<GM> Storage;

	TRWSi_Parameter(size_t maxIternum=0,
			        typename Storage::StructureType decompositionType=Storage::GENERALSTRUCTURE,
			        ValueType precision=1.0,
			        bool absolutePrecision=true,
			        bool verbose=false)
	:parent(maxIternum,precision,absolutePrecision),
	 decompositionType_(decompositionType),
	 verbose_(verbose){}

	typename Storage::StructureType decompositionType_;
	bool verbose_;

	size_t& maxNumberOfIterations(){return parent::maxNumberOfIterations_;}
	const size_t& maxNumberOfIterations()const {return parent::maxNumberOfIterations_;}

	ValueType& precision(){return parent::precision_;}
	const ValueType& precision()const{return parent::precision_;}

	bool& isAbsolutePrecision(){return parent::absolutePrecision_;};//true for absolute precision, false for relative w.r.t. dual value
	const bool& isAbsolutePrecision()const{return parent::absolutePrecision_;};//true for absolute precision, false for relative w.r.t. dual value

	ValueType& minRelativeDualImprovement(){return parent::minRelativeDualImprovement_;}
	const ValueType& minRelativeDualImprovement()const{return parent::minRelativeDualImprovement_;}

	bool& fastComputations(){return parent::fastComputations_;}
	const bool& fastComputations()const{return parent::fastComputations_;}

	bool& canonicalNormalization(){return parent::canonicalNormalization_;};
	const bool& canonicalNormalization()const{return parent::canonicalNormalization_;};

	typename Storage::StructureType& decompositionType(){return decompositionType_;}
	const typename Storage::StructureType& decompositionType()const{return decompositionType_;}

	bool& verbose(){return verbose_;};
	const bool& verbose()const{return verbose_;};

#ifdef TRWS_DEBUG_OUTPUT
	  void print(std::ostream& fout)const
	  {
			fout << "maxNumberOfIterations="<<maxNumberOfIterations()<<std::endl;
			fout <<"precision="<<precision()<<std::endl;
			fout <<"isAbsolutePrecision="<<isAbsolutePrecision()<<std::endl;
			fout <<"minRelativeDualImprovement="<<minRelativeDualImprovement()<<std::endl;
			fout <<"fastComputations="<<fastComputations()<<std::endl;
			fout <<"canonicalNormalization="<<canonicalNormalization()<<std::endl;
			  if (decompositionType()==Storage::GENERALSTRUCTURE)
				  fout <<"decompositionType=" <<"GENERAL"<<std::endl;
			  else if (decompositionType()==Storage::GRIDSTRUCTURE)
				  fout <<"decompositionType=" <<"GRID"<<std::endl;
			  else
				  fout <<"decompositionType=" <<"UNKNOWN"<<std::endl;
			fout <<"verbose="<<verbose()<<std::endl;
	  }
#endif
};

//! [class trwsi]
/// TRWSi - tree-reweighted sequential message passing
/// Based on the paper:
/// V. Kolmogorov
/// Convergent tree-reweighted message passing for energy minimization. IEEE Trans. on PAMI, 28(10):1568–1583, 2006.
///
/// it provides:
/// * primal integer approximate solution for MRF energy minimization problem
/// * lower bound for a solution of the problem.
///
///
/// TODO: Code can be significantly speeded up!
///
/// Corresponding author: Bogdan Savchynskyy
///
///\ingroup inference

template<class GM, class ACC>
class TRWSi : public Inference<GM, ACC>
{
public:
  typedef ACC AccumulationType;
  typedef GM GraphicalModelType;
  OPENGM_GM_TYPE_TYPEDEFS;
  typedef trws_base::MaxSumTRWS<GM, ACC> Solver;
  typedef trws_base::DecompositionStorage<GM> Storage;
  typedef VerboseVisitor<TRWSi<GM, ACC> > VerboseVisitorType;
  typedef TimingVisitor<TRWSi<GM, ACC> > TimingVisitorType;
  typedef EmptyVisitor< TRWSi<GM, ACC> > EmptyVisitorType;

  typedef TRWSi_Parameter<GM> Parameter;

  TRWSi(const GraphicalModelType& gm, const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
		  ,std::ostream& fout=std::cout
#endif
  ):
						  _storage(gm,param.decompositionType_),
						  _solver(_storage,param
#ifdef TRWS_DEBUG_OUTPUT
								  ,(param.verbose_ ? fout : *OUT::nullstream::Instance()) //fout
#endif
						  ){
#ifdef TRWS_DEBUG_OUTPUT
	  std::ostream& out=(param.verbose_ ? fout : *OUT::nullstream::Instance());
	  out << "Parameters of the "<< name() <<" algorithm:"<<std::endl;
	  param.print(out);
#endif

	  if (param.maxNumberOfIterations_==0) throw
			  std::runtime_error("TRWSi: Maximal number of iterations (> 0) has to be specified!");
  }
  std::string name() const{ return "TRWSi"; }
  const GraphicalModelType& graphicalModel() const { return _storage.masterModel(); }
  InferenceTermination infer(){
	  _solver.infer();
	  return NORMAL;
  };

  template<class VISITOR> InferenceTermination infer(VISITOR & visitor){
	  trws_base::VisitorWrapper<VISITOR,TRWSi<GM, ACC> > visiwrap(&visitor,this);
	  _solver.infer(visiwrap);
	  return NORMAL;
  };

  InferenceTermination arg(std::vector<LabelType>& out, const size_t = 1) const
	  {
	  out = _solver.arg();
	  return opengm::NORMAL;}
  virtual ValueType bound() const{return _solver.bound();}
  virtual ValueType value() const{return _solver.value();}
  void getTreeAgreement(std::vector<bool>& out,std::vector<LabelType>* plabeling=0){_solver.getTreeAgreement(out,plabeling);}
  //const Storage& getDecompositionStorage()const{return _storage;}
  Storage& getDecompositionStorage(){return _storage;}
  const typename Solver::FactorProperties& getFactorProperties()const {return _solver.getFactorProperties();}
  private:
   Storage _storage;
   Solver _solver;
};

}
#endif

