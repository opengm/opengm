#ifndef DAOOPT_CALLER_HXX_
#define DAOOPT_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/daoopt.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class DAOOPTCaller : public InferenceCallerBase<IO, GM, ACC, DAOOPTCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::external::DAOOPT<GM> DAOOPT;
   typedef InferenceCallerBase<IO, GM, ACC, DAOOPTCaller<IO, GM, ACC> > BaseClass;
   typedef typename DAOOPT::VerboseVisitorType VerboseVisitorType;
   typedef typename DAOOPT::EmptyVisitorType EmptyVisitorType;
   typedef typename DAOOPT::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   DAOOPTCaller(IO& ioIn);
   virtual ~DAOOPTCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename DAOOPT::Parameter daooptParameter_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

};

template <class IO, class GM, class ACC>
inline DAOOPTCaller<IO, GM, ACC>::DAOOPTCaller(IO& ioIn)
   : BaseClass("DAOOPT", "detailed description of DAOOPT Parser...", ioIn) {
   addArgument(BoolArgument(daooptParameter_.nosearch, "n", "nosearch", "perform preprocessing, output stats, and exit"));
   addArgument(BoolArgument(daooptParameter_.nocaching, "", "nocaching", "disable context-based caching during search"));
   addArgument(BoolArgument(daooptParameter_.autoCutoff, "", "noauto", "don't determine cutoff automatically"));
   addArgument(BoolArgument(daooptParameter_.autoIter, "", "adaptive", "enable adaptive ordering scheme"));
   addArgument(BoolArgument(daooptParameter_.orSearch, "", "or", "use OR search (build pseudo tree as chain)"));
   addArgument(BoolArgument(daooptParameter_.par_solveLocal, "", "local", "solve all parallel subproblems locally"));
   addArgument(BoolArgument(daooptParameter_.par_preOnly, "", "pre", "perform preprocessing and generate subproblems only"));
   addArgument(BoolArgument(daooptParameter_.par_postOnly, "", "post", "read previously solved subproblems and compile solution"));

#if defined PARALLEL_DYNAMIC || defined PARALLEL_STATIC
#else
   addArgument(BoolArgument(daooptParameter_.rotate, "y", "rotate", "use breadth-rotating AOBB"));
#endif
   addArgument(IntArgument<>(daooptParameter_.ibound, "i", "ibound", "i-bound for mini bucket heuristics", daooptParameter_.ibound));
   addArgument(IntArgument<>(daooptParameter_.cbound, "j", "cbound", "context size bound for caching", daooptParameter_.cbound));
#if defined PARALLEL_DYNAMIC || defined PARALLEL_STATIC
   addArgument(IntArgument<>(daooptParameter_.cbound_worker, "k", "cbound-worker", "context size bound for caching in worker nodes", daooptParameter_.cbound_worker));
   addArgument(IntArgument<>(daooptParameter_.threads, "", "procs", "max. number of concurrent subproblem processes", daooptParameter_.threads));
#endif
   addArgument(IntArgument<>(daooptParameter_.order_iterations, "t", "orderIter", "iterations for finding ordering", daooptParameter_.order_iterations));
   addArgument(IntArgument<>(daooptParameter_.order_timelimit, "", "orderTime", "maximum time for finding ordering", daooptParameter_.order_timelimit));
   addArgument(IntArgument<>(daooptParameter_.order_tolerance, "", "orderTolerance", "allowed deviation from minfill suggested optimal", daooptParameter_.order_tolerance));
#if defined PARALLEL_DYNAMIC || defined PARALLEL_STATIC
   addArgument(IntArgument<>(daooptParameter_.cutoff_depth, "d", "cutoff-depth", "cutoff depth for central search", daooptParameter_.cutoff_depth));
   addArgument(IntArgument<>(daooptParameter_.cutoff_width, "w", "cutoff-width", "cutoff width for central search", daooptParameter_.cutoff_width));
   addArgument(IntArgument<>(daooptParameter_.nodes_init, "x", "init-nodes", "number of nodes (*10^5) for local initialization", daooptParameter_.nodes_init));
#endif
   addArgument(IntArgument<>(daooptParameter_.memlimit, "", "memlimit", "approx. memory limit for mini buckets (in MByte)", daooptParameter_.memlimit));
#if defined PARALLEL_DYNAMIC || defined PARALLEL_STATIC
   addArgument(IntArgument<>(daooptParameter_.cutoff_size, "l", "cutoff-size", "subproblem size cutoff for central search (* 10^5)", daooptParameter_.cutoff_size));
   addArgument(IntArgument<>(daooptParameter_.local_size, "u", "local-size", "minimum subproblem size (* 10^5)", daooptParameter_.local_size));
   addArgument(IntArgument<>(daooptParameter_.maxSubprob, "", "max-sub", "only generate the first few subproblems (for testing)", daooptParameter_.maxSubprob));
#endif
   addArgument(IntArgument<>(daooptParameter_.lds, "", "lds", "run initial LDS search with given limit (-1: disabled)", daooptParameter_.lds));
   addArgument(IntArgument<>(daooptParameter_.seed, "", "seed", "seed for random number generator, time() otherwise", daooptParameter_.seed));
#if defined PARALLEL_DYNAMIC || defined PARALLEL_STATIC
#else
   addArgument(IntArgument<>(daooptParameter_.rotateLimit, "z", "rotatelimit", "nodes per subproblem stack rotation (0: disabled)", daooptParameter_.rotateLimit));
#endif
   std::vector<int> possibleSubProbOrderValues;
   possibleSubProbOrderValues.push_back(0);
   possibleSubProbOrderValues.push_back(1);
   possibleSubProbOrderValues.push_back(2);
   possibleSubProbOrderValues.push_back(3);
   addArgument(IntArgument<>(daooptParameter_.subprobOrder, "r", "suborder", "subproblem order (0:width-inc 1:width-dec 2:heur-inc 3:heur-dec)", daooptParameter_.subprobOrder, possibleSubProbOrderValues));
#ifdef PARALLEL_STATIC
   addArgument(IntArgument<>(daooptParameter_.sampleDepth, "", "sampledepth", "Randomness branching depth for initial sampling", daooptParameter_.sampleDepth));
   //addArgument(IntArgument<>(daooptParameter_.sampleScheme, "", "maxIt", "sampling scheme (TBD)", daooptParameter_.sampleScheme));
   addArgument(IntArgument<>(daooptParameter_.sampleRepeat, "", "samplerepeat", "Number of sample sequence repeats", daooptParameter_.sampleRepeat));
#endif
   addArgument(IntArgument<>(daooptParameter_.maxWidthAbort, "", "max-width", "max. induced width to process, abort otherwise", daooptParameter_.maxWidthAbort));
#ifdef ENABLE_SLS
   addArgument(IntArgument<>(daooptParameter_.slsIter, "", "slsX", "Number of initial SLS iterations", daooptParameter_.slsIter));
   addArgument(IntArgument<>(daooptParameter_.slsTime, "", "slsT", "Time per SLS iteration", daooptParameter_.slsTime));
#endif
#ifdef PARALLEL_STATIC
   addArgument(IntArgument<>(daooptParameter_.aobbLookahead, "", "lookahead", "AOBB subproblem lookahead factor (multiplied by no. of problem variables)", daooptParameter_.aobbLookahead));
#endif

   addArgument(DoubleArgument<>(daooptParameter_.initialBound, "", "initial-bound", "initial lower bound on solution cost", daooptParameter_.initialBound));

   addArgument(StringArgument<>(daooptParameter_.runTag, "", "tag", "tag of the parallel run (to differentiate filenames etc.)"));
#ifdef PARALLEL_STATIC
   addArgument(StringArgument<>(daooptParameter_.sampleSizes, "", "samplesizes", "Sequence of sample sizes for complexity prediction (in 10^5 nodes)"));
#endif
   addArgument(StringArgument<>(daooptParameter_.in_problemFile, "f", "input-file", "path to problem file"));
   addArgument(StringArgument<>(daooptParameter_.in_evidenceFile, "", "evid-file", "path to optional evidence file"));
   addArgument(StringArgument<>(daooptParameter_.in_orderingFile, "", "ordering", "read elimination ordering from this file (first to last)"));
   addArgument(StringArgument<>(daooptParameter_.in_minibucketFile, "", "minibucket", "path to read/store mini bucket heuristic"));
   addArgument(StringArgument<>(daooptParameter_.in_subproblemFile, "s", "subproblem", "limit search to subproblem specified in file"));
   addArgument(StringArgument<>(daooptParameter_.in_boundFile, "b", "bound-file", "file with initial lower bound on solution cost"));
   addArgument(StringArgument<>(daooptParameter_.out_solutionFile, "c", "sol-file", "path to output optimal solution to"));
#if not (defined PARALLEL_DYNAMIC || defined PARALLEL_STATIC)
   addArgument(StringArgument<>(daooptParameter_.out_reducedFile, "", "reduce", "path to output the reduced network to (removes evidence and unary variables)"));
#endif
   addArgument(StringArgument<>(daooptParameter_.out_pstFile, "", "pst-file", "path to output the pseudo tree to, for plotting"));
}

template <class IO, class GM, class ACC>
inline DAOOPTCaller<IO, GM, ACC>::~DAOOPTCaller() {

}

template <class IO, class GM, class ACC>
inline void DAOOPTCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running DAOOPT caller" << std::endl;

   // flip autoCutoff
   daooptParameter_.autoCutoff = !daooptParameter_.autoCutoff;
   this-> template infer<DAOOPT, TimingVisitorType, typename DAOOPT::Parameter>(model, output, verbose, daooptParameter_);
}

template <class IO, class GM, class ACC>
const std::string DAOOPTCaller<IO, GM, ACC>::name_ = "DAOOPT";

} // namespace interface

} // namespace opengm

#endif /* DAOOPT_CALLER_HXX_ */
