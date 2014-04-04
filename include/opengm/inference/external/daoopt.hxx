#ifndef DAOOPT_HXX_
#define DAOOPT_HXX_

#include "opengm/inference/inference.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"


#include <Main.h>
#undef UNKNOWN

namespace daoopt{

   template<class V, class I>
   class OpengmVisitor : public daoopt::VisitorBase{
   public:
      OpengmVisitor(V& v, I& i) : visitor(v), inference(i) {};
      V& visitor;
      I& inference;
      virtual bool visit(){
         if(visitor(inference)==0) {return true;} 
         else {return false;}
      };
   };

}

namespace opengm {
   namespace external {

      /// DAOOPT
      /// DAOOPT inference algorithm class
      /// \ingroup inference
      /// \ingroup external_inference
      ///
      //    DAOOPT
      /// - cite :[?]
      /// - Maximum factor order : ?
      /// - Maximum number of labels : ?
      /// - Restrictions : ?
      /// - Convergent : ?
      template<class GM>
      class DAOOPT : public Inference<GM, opengm::Minimizer> {
      public:
         typedef GM                              GraphicalModelType;
         typedef opengm::Minimizer               AccumulationType;
         OPENGM_GM_TYPE_TYPEDEFS;
         typedef visitors::VerboseVisitor<DAOOPT<GM> > VerboseVisitorType;
         typedef visitors::EmptyVisitor<DAOOPT<GM> >   EmptyVisitorType;
         typedef visitors::TimingVisitor<DAOOPT<GM> >  TimingVisitorType;

         ///Parameter inherits from daoopt ProgramOptions
         struct Parameter : public daoopt::ProgramOptions {
            /// \brief Constructor
            Parameter() : daoopt::ProgramOptions() {
               // set default options, this is not done for all parameters by daoopt
               subprobOrder = 0;
               ibound = 10;
               cbound = 1000;
               cbound_worker = 1000;
               rotateLimit = 1000;
               order_iterations = 25;
               order_timelimit = -1;
               threads = -1;
               sampleDepth = 10;
               sampleRepeat = 1;
               aobbLookahead = 5;
            }
         };

         // construction
         DAOOPT(const GraphicalModelType& gm, const Parameter& para = Parameter());
         // destruction
         ~DAOOPT();
         // query
         std::string name() const;
         const GraphicalModelType& graphicalModel() const;
         // inference
         template<class VISITOR>
         InferenceTermination infer(VISITOR & visitor);
         InferenceTermination infer();
         InferenceTermination arg(std::vector<LabelType>&, const size_t& = 1) const;
         typename GM::ValueType bound() const;
         typename GM::ValueType value() const;

      protected:
         const GraphicalModelType& gm_;
         Parameter parameter_;
         daoopt::Main main_;
      };

      template<class GM>
      inline DAOOPT<GM>::DAOOPT(const typename DAOOPT<GM>::GraphicalModelType& gm, const Parameter& para)
         : gm_(gm), parameter_(para) {

         if(!main_.start()) {
            throw RuntimeError("Error starting DAOOPT main.");
         }

         // check options
         if (!parameter_.in_subproblemFile.empty() && parameter_.in_orderingFile.empty()) {
            throw RuntimeError("Error: Specifying a subproblem requires reading a fixed ordering from file.");
         }

         if (parameter_.subprobOrder < 0 || parameter_.subprobOrder > 3) {
            throw RuntimeError("Error: subproblem ordering has to be 0(width-inc), 1(width-dec), 2(heur-inc) or 3(heur-dec)");
         }

         if(parameter_.problemName.empty()) {
            //Extract the problem name
            if(parameter_.in_problemFile.empty()) {
               // set problem name to openGM
               parameter_.problemName = "openGM";
            } else {
               string& fname = parameter_.in_problemFile;
               size_t len, start, pos1, pos2;
               #if defined(WIN32)
                  pos1 = fname.find_last_of("\\");
               #elif defined(LINUX)
                  pos1 = fname.find_last_of("/");
               #endif
               pos2 = fname.find_last_of(".");
               if (pos1 == string::npos) { len = pos2; start = 0; }
               else { len = (pos2-pos1-1); start = pos1+1; }
               parameter_.problemName = fname.substr(start, len);
            }
         }

         // TODO set executable name (is this required by daoopt)???
         /*if(parameter_.executableName.empty()) {
            parameter_.executableName = ???
         }*/

         main_.setOptions(new daoopt::ProgramOptions(static_cast<daoopt::ProgramOptions>(parameter_)));

         if(!main_.outputInfo()) {
            throw RuntimeError("Error printing DAOOPT info.");
         }

         if(parameter_.in_problemFile.empty()) {
            daoopt::Problem* problem = new daoopt::Problem();
            if(!problem->convertOPENGM(gm_)) {
               throw RuntimeError("Error converting openGM to DAOOPT problem.");
            }
            main_.setProblem(problem);
         } else {
            if(!main_.loadProblem()) {
               throw RuntimeError("Error loading DAOOPT problem.");
            }
         }
      }

      template<class GM>
      inline DAOOPT<GM>::~DAOOPT() {

      }

      template<class GM>
      inline std::string DAOOPT<GM>::name() const {
         return "DAOOPT";
      }

      template<class GM>
      inline const typename DAOOPT<GM>::GraphicalModelType& DAOOPT<GM>::graphicalModel() const {
         return gm_;
      }

      template<class GM>
      inline InferenceTermination DAOOPT<GM>::infer() {
         EmptyVisitorType visitor;
         return this->infer(visitor);
      }

      template<class GM>
      template<class VISITOR>
      inline InferenceTermination DAOOPT<GM>::infer(VISITOR & visitor) {

         visitor.begin(*this);
         // TODO check for possible visitor injection method

         if(!main_.runSLS()) {
            throw RuntimeError("Error running DAOOPT SLS.");
         }

         if(!main_.findOrLoadOrdering()) {
            throw RuntimeError("Error running DAOOPT find/load ordering.");
         }

         if(!main_.initDataStructs()) {
            throw RuntimeError("Error initializing DAOOPT data structs.");
         }

         if(!main_.compileHeuristic()) {
            throw RuntimeError("Error compiling DAOOPT heuristic.");
         }

         if(!main_.runLDS()) {
            throw RuntimeError("Error running DAOOPT LDS.");
         }

         if(!main_.finishPreproc()) {
            throw RuntimeError("Error finishing DAOOPT preprocessing.");
         }

         daoopt::OpengmVisitor<VISITOR, DAOOPT<GM> > v(visitor,*this);
         if(!main_.runSearch(v)) {
            throw RuntimeError("Error running DAOOPT search.");
         }

         if(!main_.outputStats()) {
            throw RuntimeError("Error output DAOOPT stats.");
         }

         visitor.end(*this);
         return NORMAL;
      }

      template<class GM>
      inline InferenceTermination DAOOPT<GM>::arg(std::vector<LabelType>& arg, const size_t& n) const {
         const daoopt::Problem& problem = main_.getProblem();

         const std::vector<daoopt::val_t>& assignment = problem.getSolutionAssg();
         arg.assign(assignment.begin(), assignment.end()-1);

         return NORMAL;
      }

      template<class GM>
      inline typename GM::ValueType DAOOPT<GM>::bound() const {
         return AccumulationType::ineutral<ValueType>();
      }

      template<class GM>
      inline typename GM::ValueType DAOOPT<GM>::value() const {
         //std::vector<LabelType> c;
         //arg(c);
         //return gm_.evaluate(c);

         const daoopt::Problem& problem = main_.getProblem();
         const ValueType v =  static_cast<ValueType>(-problem.getSolutionCost());
         if(isnan(v))
            return  std::numeric_limits<ValueType>::infinity();
         else
            return v;
      }
   } // namespace external
} // namespace opengm

#endif /* DAOOPT_HXX_ */
