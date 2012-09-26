#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <string>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/astar.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/loc.hxx>

#include "nifty_iterator.hxx"
#include "inferencehelpers.hxx"
#include"../export_typedes.hxx"

using namespace boost::python;

namespace pybp{
   template< class PARAM>
   std::string paramAsString(const PARAM & param) {
      std::stringstream ss;
      ss<<"[ steps="<<param.maximumNumberOfSteps_;
      ss<<", damping="<<param.damping_;
      ss<<", convergenceBound="<<param.bound_<<" ]";
      ss<<", isAcyclic="<<param.isAcyclic_<<" ]";
      return ss.str();
   }
   template<class PARAM>
   void set
   (
      PARAM & p,
      const size_t maximumNumberOfSteps,
      const double damping,
      const double convergenceBound,
      const opengm::Tribool::State isAcyclic
   ){
      p.maximumNumberOfSteps_=maximumNumberOfSteps;
      p.damping_=damping;
      p.bound_=convergenceBound;
      p.isAcyclic_=isAcyclic;
   }
   
   template<class PARAM>
   void setIsAcyclic
   (
      PARAM & p,
      const opengm::Tribool::State isAcyclic
   ){
      p.isAcyclic_=isAcyclic;
   }
   template<class PARAM>
   opengm::Tribool::State getIsAcyclic
   (
      PARAM & p
   ){
      if (p.isAcyclic_.maybe())
         return opengm::Tribool::Maybe;
      else if (p.isAcyclic_==true)
         return opengm::Tribool::True;
      else
         return opengm::Tribool::False;
   }
}
namespace pytrbp{
   template< class PARAM>
   std::string paramAsString(const PARAM & param) {
      std::stringstream ss;
      ss<<"[ steps="<<param.maximumNumberOfSteps_;
      ss<<", damping="<<param.damping_;
      ss<<", convergenceBound="<<param.bound_<<" ]";
      ss<<", isAcyclic="<<param.isAcyclic_<<" ]";
      return ss.str();
   }
   template<class PARAM>
   void set
   (
      PARAM & p,
      const size_t maximumNumberOfSteps,
      const double damping,
      const double convergenceBound,
      const opengm::Tribool::State isAcyclic
   ){
      p.maximumNumberOfSteps_=maximumNumberOfSteps;
      p.damping_=damping;
      p.bound_=convergenceBound;
      p.isAcyclic_=isAcyclic;
   }
   
   template<class PARAM>
   void setIsAcyclic
   (
      PARAM & p,
      const opengm::Tribool::State isAcyclic
   ){
      p.isAcyclic_=isAcyclic;
   }
   template<class PARAM>
   opengm::Tribool::State getIsAcyclic
   (
      PARAM & p
   ){
      if (p.isAcyclic_.maybe())
         return opengm::Tribool::Maybe;
      else if (p.isAcyclic_==true)
         return opengm::Tribool::True;
      else
         return opengm::Tribool::False;
   }
}
namespace pyastar{   
   template<class PARAM,class INF>
   typename pyenums::AStarHeuristic getHeuristic(const PARAM & p){
      if(p.heuristic_==PARAM::DEFAULTHEURISTIC)
         return pyenums::DEFAULT_HEURISTIC;
      else if(p.heuristic_==PARAM::FASTHEURISTIC)
         return pyenums::FAST_HEURISTIC;
      else
         return pyenums::STANDARD_HEURISTIC;
   }
   template<class PARAM,class INF>
   void setHeuristic( PARAM & p,const pyenums::AStarHeuristic h){
      if(h==pyenums::DEFAULT_HEURISTIC)
         p.heuristic_=PARAM::DEFAULTHEURISTIC;
      else if(h==pyenums::FAST_HEURISTIC)
         p.heuristic_=PARAM::FASTHEURISTIC;
      else
         p.heuristic_=PARAM::STANDARDHEURISTIC;
   }
   
   template<class PARAM,class INF>
   void set
   ( 
      PARAM & p,
      const pyenums::AStarHeuristic h,
      const typename INF::ValueType bound ,
      const size_t maxHeapSize,
      const size_t numberOfOpt
   ){
      if(h==pyenums::DEFAULT_HEURISTIC)
         p.heuristic_=PARAM::DEFAULTHEURISTIC;
      else if(h==pyenums::FAST_HEURISTIC)
         p.heuristic_=PARAM::FASTHEURISTIC;
      else
         p.heuristic_=PARAM::STANDARDHEURISTIC;
      p.objectiveBound_=bound;
      p.maxHeapSize_=maxHeapSize;
      p.numberOfOpt_=numberOfOpt;
   }
}

namespace pyloc{

   template<class PARAM>
   inline void set
   (
      PARAM & p,
      const double phi,
      const size_t maxRadius,
      const size_t maxIterations,
      const size_t aStarThreshold
   ){
      p.phi_=phi;
      p.maxRadius_=maxRadius;
      p.maxIterations_=maxIterations;
      p.aStarThreshold_=aStarThreshold;
   }

}

template<class GM,class ACC>
void export_mp_based(){
import_array();    
typedef GM PyGm;
typedef typename PyGm::ValueType ValueType;
typedef typename PyGm::IndexType IndexType;
typedef typename PyGm::LabelType LabelType;

typedef opengm::BeliefPropagationUpdateRules<GM,ACC> UpdateRulesType;
typedef opengm::MessagePassing<GM, ACC,UpdateRulesType, opengm::MaxDistance> PyBp;
typedef typename PyBp::Parameter PyBpParameter;
typedef typename PyBp::VerboseVisitorType PyBpVerboseVisitor;

typedef opengm::TrbpUpdateRules<GM,ACC> UpdateRulesType2;
typedef opengm::MessagePassing<GM, ACC,UpdateRulesType2, opengm::MaxDistance> PyTrBp;
typedef typename PyTrBp::Parameter PyTrBpParameter;
typedef typename PyTrBp::VerboseVisitorType PyTrBpVerboseVisitor;

typedef opengm::AStar<PyGm, ACC>  PyAStar;
typedef typename PyAStar::Parameter PyAStarParameter;
typedef typename PyAStar::VerboseVisitorType PyAStarVerboseVisitor;

typedef opengm::LOC<GM, ACC>  PyLOC;
typedef typename PyLOC::Parameter PyLOCParameter;
typedef typename PyLOC::VerboseVisitorType PyLOCVerboseVisitor;


   class_<PyAStarParameter > ("AStarParameter", init< >() )
      .def_readwrite("obectiveBound", &PyAStarParameter::objectiveBound_,
      "AStar objective bound.\n\n"
      "  A good bound will speedup inference"
      )
      .def_readwrite("maxHeapSize", &PyAStarParameter::maxHeapSize_,
      "Maximum size of the heap which is used while inference"
      )
      .def_readwrite("numberOfOpt", &PyAStarParameter::numberOfOpt_,
      "Select which n best states should be searched for while inference:"
      )
      .add_property("heuristic", 
           &pyastar::getHeuristic<PyAStarParameter,PyAStar>, pyastar::setHeuristic<PyAStarParameter,PyAStar>,
      "heuristic can be:\n\n"
      "  -``opengm.AStarHeuristic.default`` :default AStar heuristc (default)\n\n"
      "  -``opengm.AStarHeuristic.standart`` : standart AStar heuristic \n\n"
      "  -``opengm.AStarHeuristic.fast`` : fast AStar heuristic for second order gm's"     
      )
      .def ("set", &pyastar::set<PyAStarParameter,PyAStar>, 
            (
            arg("heuristic")=pyenums::DEFAULT_HEURISTIC,
            arg("bound")= ACC::template neutral<ValueType>(),
            arg("maxHeapSize")=3000000,
            arg("numberOfOpt")=1
            ),
      "Set the parameters values.\n\n"
      "All values of the parameter have a default value.\n\n"
      "Args:\n\n"
      "  heuristic: Number of message passing updates\n\n"
      "     -``opengm.AStarHeuristic.default`` :default AStar heuristc (default)\n\n"
      "     -``opengm.AStarHeuristic.standart`` : stanart AStar heuristic \n\n"
      "     -``opengm.AStarHeuristic.fast`` : fast AStar heuristic for second order gm's\n\n"
      "  bound: AStar objective bound.\n\n"
      "     A good bound will speedup inference (default = neutral value)\n\n"
      "  maxHeapSize: Maximum size of the heap which is used while inference (default=3000000)\n\n"
      "  numberOfOpt: Select which n best states should be searched for while inference (default=1):\n\n"
      "Returns:\n"
      "  None\n\n"
      ) 
      ;


   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyAStarVerboseVisitor,"AStarVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyAStar,"AStar",
   "A star search algorithm:\n\n"
   "cite: Kappes, J. H. :\"Inference on Highly-Connected Discrete Graphical Models with Applications to Visual Object Recognition \"," 
   "Ph.D. Thesis 2011.\n\n"
   "Bergtholdt, M. & Kappes, J. H. & Schnoerr, C.:\"`Learning of Graphical Models and Efficient Inference for Object Class Recognition <http://hci.iwr.uni-heidelberg.de/Staff/jkappes/publications/dagm2006.pdf>`_\"," 
   "  DAGM 2006\n\n"
   "Bergtholdt, M. & Kappes, J. H. & Schmidt, S. & Schnoerr, C.: \"`A Study of Parts-Based Object Class Detection Using Complete Graphs <https://www.inf.tu-dresden.de/content/institutes/ki/is/HS_SS09_Papers/A_Study_of_Parts_Based_Object_Class_Detection_Using_Complete_Graphs.pdf>`_\"," 
   "  DAGM 2006\n\n"
   "limitations: graph must be small enough\n\n"
   "guarantees: global optimal\n\n"
   "The AStar-Algo transform the problem into a shortest path problem in an exponentially large graph.\n"
   "Due to the problem structure, this graph can be represented implicitly!\n"
   "To find the shortest path we perform a best first search and use a admissable tree-based heuristic\n"
   "to underestimate the cost to a goal node.\n"
   "This lower bound allows us to reduce the search to an manageable \n"
   "subspace of the exponentially large search-space. "
   );

   // export inference parameter
   class_<PyBpParameter > ("BpParameter", init< const size_t,const typename PyBp::ValueType,const typename PyBp::ValueType  >() )
      .def(init<const size_t,const typename PyBp::ValueType>())
      .def(init<const size_t>())
      .def(init<>())
      .def_readwrite("steps", &PyBpParameter::maximumNumberOfSteps_,
      "Number of message passing updates"
      )
      .def_readwrite("damping", &PyBpParameter::damping_,
      "Damping must be in [0,1]"
      )
      .def_readwrite("convergenceBound", &PyBpParameter::bound_,
      "Convergence bound stops message passing updates when message change is smaller than ``convergenceBound``"
      )
      .add_property("isAcyclic", &pytrbp::getIsAcyclic<PyBpParameter>, pytrbp::setIsAcyclic<PyBpParameter>,
      "isAcyclic can be:\n\n"
      "  -``opengm.Tribool.maybe`` : if its unknown that the gm is acyclic (default)\n\n"
      "  -``opengm.Tribool.true`` : if its known that the gm is acyclic (gm has no loops)\n\n"
      "  -``opengm.Tribool.false`` : if its known that the gm is not acyclic (gm has loops)\n\n"
      )
      .def ("set", &pytrbp::set<PyBpParameter>, 
            (
            arg("steps")=100,
            arg("damping")= 0,
            arg("convergenceBound")=0,
            arg("isAcyclic")=opengm::Tribool::Maybe
            ),
      "Set the parameters values.\n\n"
      "All values of the parameter have a default value.\n\n"
      "Args:\n\n"
      "  steps: Number of message passing updates (default=100)\n\n"
      "  damping: Damp the message.\n\n"
      "     Damping must be in [0,1] (default=0)\n\n"
      "  convergenceBound: Convergence bound stops message passing updates when the difference\n\n"
      "     between old and new messages is smaller than ``convergenceBound`` (default=0)\n\n"
      "  isAcyclic: isAcyclic can be:\n\n"
      "     -``opengm.Tribool.maybe`` : if its unknown that the gm is acyclic (default)\n\n"
      "     -``opengm.Tribool.true`` : if its known that the gm is acyclic / gm has no loops\n\n"
      "     -``opengm.Tribool.false`` : if its known that the gm is not acyclic /gm has loops\n\n"
      "Returns:\n"
      "  None\n\n"
      )
      ;
   // export inference verbose visitor via macro
   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyBpVerboseVisitor,"BpVerboseVisitor" );
   // export inference via macro
   //Cf. F. R. Kschischang, B. J. Frey and H.-A. Loeliger, "Factor Graphs and the Sum-Product Algorithm", IEEE Transactions on Information Theory 47:498-519, 2001.
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyBp,"Bp",
   "Belief Propagation (Bp):\n\n"
   "cite: Cf. F. R. Kschischang, B. J. Frey and H.-A. Loeliger:\"`Factor Graphs and the Sum-Product Algorithm <http://www.cs.utoronto.ca/~radford/csc2506/factor.pdf>`_\"," 
   " IEEE Transactions on Information Theory 47:498-519, 2001.\n\n"
   "limitations: -\n\n"
   "guarantees: -\n"
   );

      class_<PyTrBpParameter > ("TrBpParameter", init< const size_t,const typename PyTrBp::ValueType,const typename PyTrBp::ValueType  >() )
      .def(init<const size_t,const typename PyTrBp::ValueType>())
      .def(init<const size_t>())
      .def(init<>())
      .def_readwrite("steps", &PyTrBpParameter::maximumNumberOfSteps_,
      "Number of message passing updates"
      )
      .def_readwrite("damping", &PyTrBpParameter::damping_,
      "Damping must be in [0,1]"
      )
      .def_readwrite("convergenceBound", &PyTrBpParameter::bound_,
      "Convergence bound stops message passing updates when message change is smaller than ``convergenceBound``"
      )
      .add_property("isAcyclic", &pytrbp::getIsAcyclic<PyTrBpParameter>, pytrbp::setIsAcyclic<PyTrBpParameter>,
      "isAcyclic can be:\n\n"
      "     -``opengm.Tribool.maybe`` : if its unknown that the gm is acyclic (default)\n\n"
      "     -``opengm.Tribool.true`` : if its known that the gm is acyclic (gm has no loops)\n\n"
      "     -``opengm.Tribool.false`` : if its known that the gm is not acyclic (gm has loops)\n\n"
      )
      .def ("set", &pytrbp::set<PyTrBpParameter>, 
            (
            arg("steps")=100,
            arg("damping")= 0,
            arg("convergenceBound")=0,
            arg("isAcyclic")=opengm::Tribool::Maybe
            ),
      "Set the parameters values.\n\n"
      "All values of the parameter have a default value.\n\n"
      "Args:\n\n"
      "  steps: Number of message passing updates (default=100)\n\n"
      "  damping: Damp the message.\n\n"
      "     Damping must be in [0,1] (default=0)\n\n"
      "  convergenceBound: Convergence bound stops message passing updates when the difference\n\n"
      "     between old and new messages is smaller than ``convergenceBound`` (default=0)\n\n"
      "  isAcyclic: isAcyclic can be:\n\n"
      "     -``opengm.Tribool.maybe`` : if its unknown that the gm is acyclic (default)\n\n"
      "     -``opengm.Tribool.true`` : if its known that the gm is acyclic / gm has no loops\n\n"
      "     -``opengm.Tribool.false`` : if its known that the gm is not acyclic /gm has loops\n\n"
      "Returns:\n"
      "  None\n"
      )
      ;


   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyTrBpVerboseVisitor,"TrBpVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyTrBp,"TrBp","trbp docstring");

     class_<PyLOCParameter > ( "LOCParameter" , init< double ,size_t,size_t,size_t > (args("phi,maxRadius,maxIteration,aStarThreshold")))
   .def(init<>())
   .def_readwrite("phi", &PyLOCParameter::phi_,
   "Open parameter in (truncated) geometric distribution.\n"
   "The subgraph radius is sampled from that distribution"
   )
   .def_readwrite("maxRadius", &PyLOCParameter::maxRadius_,
   "Maximum subgraph radius.\n\n"
   "The subgraph radius is in [0,maxRadius]"
   )
   .def_readwrite("steps", &PyLOCParameter::maxIterations_,
   "Number of iterations. \n"
   "If steps is zero a suitable number is choosen)"
   )
   .def_readwrite("aStarThreshold", &PyLOCParameter::aStarThreshold_,
   "If there are more variables in the subgraph than ``aStarThreshold`` ,\n"
   "AStar is used to optimise the subgraph, otherwise Bruteforce is used."
   )
   .def ("set", &pyloc::set<PyLOCParameter>, 
            (
            arg("phi")=0.5,
            arg("maxRadius")=5,
            arg("steps")=0,
            arg("aStarThreshold")=10
            ), 
   "Set the parameters values.\n\n"
   "All values of the parameter have a default value.\n\n"
   "Args:\n\n"
   "  phi: Open parameter in (truncated) geometric distribution.\n\n"
   "     The subgraph radius is sampled from that distribution(default=0.5)\n\n"
   "  maxRadius: Maximum subgraph radius.\n\n"
   "     The subgraph radius is in [0,maxRadius] (default=5)\n\n"
   "  steps: Number of iterations. \n\n"
   "     If steps is zero a suitable number is choosen. (default=0)\n\n"
   "  aStarThreshold: If there are more variables in the subgraph than ``aStarThreshold`` ,\n\n"
   "     AStar is used to optimise the subgraph, otherwise Bruteforce is used.\n\n"
   "Returns:\n"
   "  None\n"
   )
   ;

   OPENGM_PYTHON_VERBOSE_VISITOR_EXPORTER(PyLOCVerboseVisitor,"LOCVerboseVisitor" );
   OPENGM_PYTHON_INFERENCE_EXPORTER(PyLOC,"LOC",
   "LOC:\n\n"
   "cite: K. Jung, P. Kohli and D. Shah:\"`Local Rules for Global MAP: When Do They Work? <http://research.microsoft.com/en-us/um/people/pkohli/papers/jks_nips09_TR.pdf>`_\"," 
   "NIPS 2009.\n\n"
   "limitations: -\n\n"
   "guarantees: epsilon approximation on planar graphs\n"
   );

}

template void export_mp_based<GmAdder, opengm::Minimizer>();
template void export_mp_based<GmAdder, opengm::Maximizer>();
template void export_mp_based<GmMultiplier, opengm::Minimizer>();
template void export_mp_based<GmMultiplier, opengm::Maximizer>();
