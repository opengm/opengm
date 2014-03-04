#pragma once
#ifndef OPENGM_MULTICUT_HXX
#define OPENGM_MULTICUT_HXX

#include <algorithm>
#include <vector>
#include <queue>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <limits> 
#ifdef WITH_BOOST
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>		
#else
#include <ext/hash_map> 
#include <ext/hash_set>
#endif

#include "opengm/datastructures/marray/marray.hxx"
#include "opengm/opengm.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/utilities/timer.hxx"

#include <ilcplex/ilocplex.h>
//ILOSTLBEGIN

namespace opengm {

/// \cond HIDDEN_SYMBOLS
class HigherOrderTerm
{
public:
   size_t factorID_;
   bool   potts_;
   size_t valueIndex_;
   std::vector<size_t> lpIndices_;
   HigherOrderTerm(size_t factorID, bool  potts, size_t valueIndex) 
      : factorID_(factorID), potts_(potts),valueIndex_(valueIndex) {}
   HigherOrderTerm() 
      : factorID_(0), potts_(false),valueIndex_(0) {}           
};
/// \endcond      

/// \brief Multicut Algorithm\n\n
/// [1] J. Kappes, M. Speth, B. Andres, G. Reinelt and C. Schnoerr, "Globally Optimal Image Partitioning by Multicuts", EMMCVPR 2011\n
/// [2] J. Kappes, M. Speth, G. Reinelt and C. Schnoerr, "Higher-order Segmentation via Multicuts", Technical Report (http://ipa.iwr.uni-heidelberg.de/ipabib/Papers/kappes-2013-multicut.pdf)\n
///
/// This code was also used in
/// [3] J. Kappes, M. Speth, G. Reinelt, and C. Schnoerr, “Towards Efficient and Exact MAP-Inference for Large Scale Discrete Computer Vision Problems via Combinatorial Optimization”. CVPR, 2013\n
/// [4] J. Kappes, B. Andres, F. Hamprecht, C. Schnoerr, S. Nowozin, D. Batra, S. Kim, B. Kausler, J. Lellmann, N. Komodakis, and C. Rother, “A Comparative Study of Modern Inference Techniques for Discrete Energy Minimization Problem”, CVPR, 2013.
///
/// Multicut-Algo :
/// - Cite: [1] and [2]
/// - Maximum factor order : potts (oo) generalized potts (4 - can be extended to N)
/// - Maximum number of labels : oo
/// - Restrictions : functions are arbitrary unary terms or generalized potts terms (positive or negative)
///                  all variables have the same labelspace (practical no theoretical restriction) 
///                  the number of states is at least as large as the order of a generalized potts function (practical no theoretical restriction)
/// - Convergent :   Converge to the global optima
///
/// see [2] for further details.
/// \ingroup inference 
template<class GM, class ACC>
class Multicut : public Inference<GM, ACC>
{
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef size_t LPIndexType;
   typedef visitors::VerboseVisitor<Multicut<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<Multicut<GM,ACC> > EmptyVisitorType;
   typedef visitors::TimingVisitor<Multicut<GM,ACC> > TimingVisitorType;


#ifdef WITH_BOOST
   typedef  boost::unordered_map<IndexType, LPIndexType> EdgeMapType;
   typedef  boost::unordered_set<IndexType> MYSET; 
#else 
   typedef __gnu_cxx::hash_map<IndexType, LPIndexType> EdgeMapType;
   typedef __gnu_cxx::hash_set<IndexType> MYSET; 
#endif


   struct Parameter{
   public:
      enum MWCRounding {NEAREST,DERANDOMIZED,PSEUDODERANDOMIZED};

      int numThreads_;
      bool verbose_;
      bool verboseCPLEX_;
      double cutUp_;
      double timeOut_;
      std::string workFlow_;
      size_t maximalNumberOfConstraintsPerRound_;
      double edgeRoundingValue_;
      MWCRounding MWCRounding_;
      size_t reductionMode_;

      /// \param numThreads number of threads that should be used (default = 0 [automatic])
      /// \param cutUp value which the optima at least has (helps to cut search-tree)
      Parameter
      (
         int numThreads=0,
         double cutUp=1.0e+75
         )
         : numThreads_(numThreads), verbose_(false),verboseCPLEX_(false), cutUp_(cutUp),
           timeOut_(36000000), maximalNumberOfConstraintsPerRound_(1000000),
           edgeRoundingValue_(0.00000001),MWCRounding_(NEAREST), reductionMode_(3)
         {};
   };

   virtual ~Multicut();
   Multicut(const GraphicalModelType&, Parameter para=Parameter());
   virtual std::string name() const {return "Multicut";}
   const GraphicalModelType& graphicalModel() const;
   virtual InferenceTermination infer();
   template<class VisitorType> InferenceTermination infer(VisitorType&);
   virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   ValueType bound() const;
   ValueType value() const;
   ValueType calcBound(){ return 0; }
   ValueType evaluate(std::vector<LabelType>&) const;

   template<class LPVariableIndexIterator, class CoefficientIterator>
   void addConstraint(LPVariableIndexIterator, LPVariableIndexIterator,
                        CoefficientIterator, const ValueType&, const ValueType&);
   std::vector<double> getEdgeLabeling() const;

   template<class IT>
   size_t getLPIndex(IT a, IT b) { return neighbours[a][b]; };

   size_t inferenceState_;
   size_t constraintCounter_;
private:
   enum ProblemType {INVALID, MC, MWC};

   const GraphicalModelType& gm_; 
   ProblemType problemType_;
   Parameter parameter_;
   double constant_;
   double bound_;
   const double infinity_;

   LabelType   numberOfTerminals_;
   IndexType   numberOfNodes_;
   LPIndexType numberOfTerminalEdges_;
   LPIndexType numberOfInternalEdges_;
   LPIndexType terminalOffset_;
   LPIndexType numberOfHigherOrderValues_;
   LPIndexType numberOfInterTerminalEdges_;

   std::vector<std::vector<size_t> >               workFlow_;
   std::vector<std::pair<IndexType,IndexType> >    edgeNodes_;

   /// For each variable it contains a map indexed by neighbord nodes giving the index to the LP-variable
   /// e.g. neighbours[a][b] = i means a has the neighbour b and the edge has the index i in the linear objective
   std::vector<EdgeMapType >   neighbours; 

   IloEnv         env_;
   IloModel       model_;
   IloNumVarArray x_;
   IloRangeArray  c_;
   IloObjective   obj_;
   IloNumArray    sol_;
   IloCplex       cplex_;

   bool           integerMode_;
   const double   EPS_;          //small number: for numerical issues constraints are still valid if the not up to EPS_


   void initCplex(); 

   size_t findCycleConstraints(IloRangeArray&, bool = true, bool = true);
   size_t findIntegerCycleConstraints(IloRangeArray&, bool = true);
   size_t findTerminalTriangleConstraints(IloRangeArray&);
   size_t findIntegerTerminalTriangleConstraints(IloRangeArray&, std::vector<LabelType>& conf);
   size_t findMultiTerminalConstraints(IloRangeArray&);
   size_t findOddWheelConstraints(IloRangeArray&);  
   size_t removeUnusedConstraints();            //TODO: implement
   size_t enforceIntegerConstraints();

   bool readWorkFlow(std::string);
  
   InferenceTermination partition(std::vector<LabelType>&, std::vector<std::list<size_t> >&, double = 0.5) const;
   ProblemType setProblemType();
   LPIndexType getNeighborhood(const LPIndexType, std::vector<EdgeMapType >&,std::vector<std::pair<IndexType,IndexType> >&, std::vector<HigherOrderTerm>&);

   template <class DOUBLEVECTOR>
   double shortestPath(const IndexType, const IndexType, const std::vector<EdgeMapType >&, const DOUBLEVECTOR&, std::vector<IndexType>&, const double = std::numeric_limits<double>::infinity(), bool = true) const;

   InferenceTermination derandomizedRounding(std::vector<LabelType>&) const;
   InferenceTermination pseudoDerandomizedRounding(std::vector<LabelType>&, size_t = 1000) const;
   double derandomizedRoundingSubProcedure(std::vector<LabelType>&,const std::vector<LabelType>&, const double) const;

   //PROTOCOLATION 

   enum{
      Protocol_ID_Solve              = 0,
      Protocol_ID_AddConstraints     = 1,
      Protocol_ID_RemoveConstraints  = 2,
      Protocol_ID_IntegerConstraints = 3,
      Protocol_ID_CC                 = 4, 
      Protocol_ID_TTC                = 5,
      Protocol_ID_MTC                = 6,
      Protocol_ID_OWC                = 7,
      Protocol_ID_Unknown            = 8  
   };
   
   enum{
      Action_ID_RemoveConstraints  = 0,
      Action_ID_IntegerConstraints = 1,
      Action_ID_CC                 = 10, 
      Action_ID_CC_I               = 11, 
      Action_ID_CC_IFD             = 12, 
      Action_ID_CC_FD              = 13, 
      Action_ID_CC_B               = 14, 
      Action_ID_CC_FDB             = 15, 
      Action_ID_TTC                = 20,    
      Action_ID_TTC_I              = 21,   
      Action_ID_MTC                = 30,    
      Action_ID_OWC                = 40
   };    
   
   std::vector<std::vector<double> > protocolateTiming_;
   std::vector<std::vector<size_t> > protocolateConstraints_;
 
};
 
template<class GM, class ACC>
typename Multicut<GM, ACC>::LPIndexType Multicut<GM, ACC>::getNeighborhood
(
   const LPIndexType numberOfTerminalEdges,
   std::vector<EdgeMapType >& neighbours,
   std::vector<std::pair<IndexType,IndexType> >& edgeNodes,
   std::vector<HigherOrderTerm>& higherOrderTerms
   )
{
   //Calculate Neighbourhood
   neighbours.resize(gm_.numberOfVariables());
   LPIndexType numberOfInternalEdges=0;
   LPIndexType numberOfAdditionalInternalEdges=0;
   // Add edges that have to be included
   for(size_t f=0; f<gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables()==2) { // Second Order Potts
         IndexType u = gm_[f].variableIndex(1);
         IndexType v = gm_[f].variableIndex(0);
         if(neighbours[u].find(v)==neighbours[u].end()) {
            neighbours[u][v] = numberOfTerminalEdges+numberOfInternalEdges;
            neighbours[v][u] = numberOfTerminalEdges+numberOfInternalEdges;
            edgeNodes.push_back(std::pair<IndexType,IndexType>(v,u));     
            ++numberOfInternalEdges;
         }
      }
   }
   for(size_t f=0; f<gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables()>2 && !gm_[f].isPotts()){ // Generalized Potts
         higherOrderTerms.push_back(HigherOrderTerm(f, false, 0));      
         for(size_t i=0; i<gm_[f].numberOfVariables();++i) {
            for(size_t j=0; j<i;++j) {
               IndexType u = gm_[f].variableIndex(i);
               IndexType v = gm_[f].variableIndex(j);
               if(neighbours[u].find(v)==neighbours[u].end()) {
                  neighbours[u][v] = numberOfTerminalEdges+numberOfInternalEdges;
                  neighbours[v][u] = numberOfTerminalEdges+numberOfInternalEdges;
                  edgeNodes.push_back(std::pair<IndexType,IndexType>(v,u));     
                  ++numberOfInternalEdges;
                  ++numberOfAdditionalInternalEdges;
               }
            }
         }
      }
   }
   //Add for higher order potts term only neccesary edges 
   for(size_t f=0; f<gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables()>2 && gm_[f].isPotts()) { //Higher order Potts
         higherOrderTerms.push_back(HigherOrderTerm(f, true, 0));  
         std::vector<LPIndexType> lpIndexVector;
         //Find spanning tree vor the variables nb(f) using edges that already exist.
         std::vector<bool> variableInSpanningTree(gm_.numberOfVariables(),true);
         for(size_t i=0; i<gm_[f].numberOfVariables();++i) {
            variableInSpanningTree[gm_[f].variableIndex(i)]=false;
         }     
         size_t connection = 2; 
         // 1 = find a spanning tree and connect higher order auxilary variable to this
         // 2 = find a spanning subgraph including at least all eges in the subset and connect higher order auxilary variable to this
         if(connection==2){
            // ADD ALL 
            for(size_t i=0; i<gm_[f].numberOfVariables();++i) {
               const IndexType u = gm_[f].variableIndex(i);  
               for(typename EdgeMapType::const_iterator it=neighbours[u].begin() ; it != neighbours[u].end(); ++it){
                  const IndexType v = (*it).first;
                  if(variableInSpanningTree[v] == false && u<v){
                    lpIndexVector.push_back((*it).second);
                  }
               }
            }
         }
         else if(connection==1){
            // ADD TREE
            for(size_t i=0; i<gm_[f].numberOfVariables();++i) {
               const IndexType u = gm_[f].variableIndex(i);  
               for(typename EdgeMapType::const_iterator it=neighbours[u].begin() ; it != neighbours[u].end(); ++it){
                  const IndexType v = (*it).first;
                  if(variableInSpanningTree[v] == false){
                     variableInSpanningTree[v] = true;
                     lpIndexVector.push_back((*it).second);
                  }
               }
            }
         }
         else{
            OPENGM_ASSERT(false);
         }
         higherOrderTerms.back().lpIndices_=lpIndexVector;

         // Check if edges need to be added to have a spanning subgraph
         //TODO 
      }
   }
   //std::cout << "Additional Edges: "<<numberOfAdditionalInternalEdges<<std::endl;
   return numberOfInternalEdges;
}

template<class GM, class ACC>
Multicut<GM, ACC>::~Multicut() {
   env_.end();
}

template<class GM, class ACC>
Multicut<GM, ACC>::Multicut
(
   const GraphicalModelType& gm,
   Parameter para
   ) : gm_(gm), parameter_(para) , bound_(-std::numeric_limits<double>::infinity()), infinity_(1e8), integerMode_(false),
       EPS_(1e-7)
{
   if(typeid(ACC) != typeid(opengm::Minimizer) || typeid(OperatorType) != typeid(opengm::Adder)) {
      throw RuntimeError("This implementation does only supports Min-Plus-Semiring.");
   } 
   if(parameter_.reductionMode_<0 ||parameter_.reductionMode_>3) {
      throw RuntimeError("Reduction Mode has to be 1, 2 or 3!");
   } 

   //Set Problem Type
   setProblemType();
   if(problemType_ == INVALID)
      throw RuntimeError("Invalid Model for Multicut-Solver! Solver requires a generalized potts model!");

   //Calculate Neighbourhood 
   std::vector<double> valuesHigherOrder;
   std::vector<HigherOrderTerm> higherOrderTerms;
   numberOfInternalEdges_ = getNeighborhood(numberOfTerminalEdges_, neighbours, edgeNodes_ ,higherOrderTerms);
   numberOfNodes_         = gm_.numberOfVariables();       
     
   //Build Objective Value 
   constant_=0;
   size_t valueSize;
   if(numberOfTerminals_==0) valueSize = numberOfInternalEdges_;
   else                      valueSize = numberOfTerminalEdges_+numberOfInternalEdges_+numberOfInterTerminalEdges_;
   std::vector<double> values (valueSize,0); 
 

   for(size_t f=0; f<gm_.numberOfFactors(); ++f) {
      if(gm_[f].numberOfVariables() == 0) {
         LabelType l = 0;
         constant_ +=  gm_[f](&l);
      }
      else if(gm_[f].numberOfVariables() == 1) {
         IndexType node = gm_[f].variableIndex(0);
         for(LabelType i=0; i<gm_.numberOfLabels(node); ++i) {
            for(LabelType j=0; j<gm_.numberOfLabels(node); ++j) {
               if(i==j) values[node*numberOfTerminals_+i] += (1.0/(numberOfTerminals_-1)-1) * gm_[f](&j);
               else     values[node*numberOfTerminals_+i] += (1.0/(numberOfTerminals_-1))   * gm_[f](&j);
            }
         }
      }
      else if(gm_[f].numberOfVariables() == 2) {
         if(gm_[f].numberOfLabels(0)==2 && gm_[f].numberOfLabels(1)==2){
            IndexType node0 = gm_[f].variableIndex(0);
            IndexType node1 = gm_[f].variableIndex(1);
            LabelType cc[] = {0,0}; ValueType a = gm_[f](cc);
            cc[0]=1;cc[1]=1;        ValueType b = gm_[f](cc);
            cc[0]=0;cc[1]=1;        ValueType c = gm_[f](cc);
            cc[0]=1;cc[1]=0;        ValueType d = gm_[f](cc);

            values[neighbours[gm_[f].variableIndex(0)][gm_[f].variableIndex(1)]] += ((c+d-a-a) - (b-a))/2.0; 
            values[node0*numberOfTerminals_+0] += ((b-a)-(-d+c))/2.0;
            values[node1*numberOfTerminals_+0] += ((b-a)-( d-c))/2.0;
            constant_ += a;
         }else{
            LabelType cc0[] = {0,0};
            LabelType cc1[] = {0,1};
            values[neighbours[gm_[f].variableIndex(0)][gm_[f].variableIndex(1)]] += gm_[f](cc1) - gm_[f](cc0); 
            constant_ += gm_[f](cc0);
         }
      }
   }
   for(size_t h=0; h<higherOrderTerms.size();++h){
      if(higherOrderTerms[h].potts_) {
         const IndexType f = higherOrderTerms[h].factorID_; 
         higherOrderTerms[h].valueIndex_= valuesHigherOrder.size();
         OPENGM_ASSERT(gm_[f].numberOfVariables() > 2);
         std::vector<LabelType> cc0(gm_[f].numberOfVariables(),0);
         std::vector<LabelType> cc1(gm_[f].numberOfVariables(),0); 
         cc1[0] = 1;
         valuesHigherOrder.push_back(gm_[f](cc1.begin()) - gm_[f](cc0.begin()) ); 
         constant_ += gm_[f](cc0.begin());
      }
      else{
         const IndexType f = higherOrderTerms[h].factorID_;
         higherOrderTerms[h].valueIndex_= valuesHigherOrder.size();
         if(gm_[f].numberOfVariables() == 3) {
            size_t i[] = {0, 1, 2 }; 
            valuesHigherOrder.push_back(gm_[f](i)); 
            i[0]=0; i[1]=0; i[2]=1;
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=0; i[1]=1; i[2]=0;
            valuesHigherOrder.push_back(gm_[f](i)); 
            i[0]=1; i[1]=0; i[2]=0;
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=0; i[1]=0; i[2]=0;
            valuesHigherOrder.push_back(gm_[f](i));
         }
         else if(gm_[f].numberOfVariables() == 4) {
            size_t i[] = {0, 1, 2, 3 };//0
            if(numberOfTerminals_>=4){
               valuesHigherOrder.push_back(gm_[f](i));
            }else{
               valuesHigherOrder.push_back(0.0);
            }
            if(numberOfTerminals_>=3){
               i[0]=0; i[1]=0; i[2]=1; i[3] = 2;//1
               valuesHigherOrder.push_back(gm_[f](i));
               i[0]=0; i[1]=1; i[2]=0; i[3] = 2;//2
               valuesHigherOrder.push_back(gm_[f](i));
               i[0]=0; i[1]=1; i[2]=1; i[3] = 2;//4
               valuesHigherOrder.push_back(gm_[f](i));
            }else{
               valuesHigherOrder.push_back(0.0);
               valuesHigherOrder.push_back(0.0);
               valuesHigherOrder.push_back(0.0);
            }
            i[0]=0; i[1]=0; i[2]=0; i[3] = 1;//7
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=0; i[1]=1; i[2]=2; i[3] = 0;//8
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=0; i[1]=1; i[2]=1; i[3] = 0;//12
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=1; i[1]=0; i[2]=2; i[3] = 0;//16
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=1; i[1]=0; i[2]=1; i[3] = 0;//18
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=0; i[1]=0; i[2]=1; i[3] = 0;//25
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=1; i[1]=2; i[2]=0; i[3] = 0;//32
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=1; i[1]=1; i[2]=0; i[3] = 0;//33
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=0; i[1]=1; i[2]=0; i[3] = 0;//42
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=1; i[1]=0; i[2]=0; i[3] = 0;//52
            valuesHigherOrder.push_back(gm_[f](i));
            i[0]=0; i[1]=0; i[2]=0; i[3] = 0;//63
            valuesHigherOrder.push_back(gm_[f](i));
         }
         else{
            throw RuntimeError("Generalized Potts Terms of an order larger than 4 a currently not supported. If U really need them let us know!");
         }
      }
   }

   //count auxilary variables
   numberOfHigherOrderValues_ = valuesHigherOrder.size();

   // build LP 
   //std::cout << "Higher order auxilary variables " << numberOfHigherOrderValues_ << std::endl;
   //std::cout << "TerminalEdges " << numberOfTerminalEdges_ << std::endl;
   OPENGM_ASSERT( numberOfTerminalEdges_ == gm_.numberOfVariables()*numberOfTerminals_ );
   //std::cout << "InternalEdges " << numberOfInternalEdges_ << std::endl;

   OPENGM_ASSERT(values.size() == numberOfTerminalEdges_+numberOfInternalEdges_+numberOfInterTerminalEdges_);
   IloInt N = values.size() + numberOfHigherOrderValues_;
   model_ = IloModel(env_);
   x_     = IloNumVarArray(env_);
   c_     = IloRangeArray(env_);
   obj_   = IloMinimize(env_);
   sol_   = IloNumArray(env_,N);

   // set variables and objective
   x_.add(IloNumVarArray(env_, N, 0, 1, ILOFLOAT));

   IloNumArray    obj(env_,N);
   for (size_t i=0; i< values.size();++i) {
      if(values[i]==0)
         obj[i] = 0;//1e-50; //for numerical reasons
      else
         obj[i] = values[i];
   }
   {
      size_t count =0;
      for (size_t i=0; i<valuesHigherOrder.size();++i) {
         obj[values.size()+count++] = valuesHigherOrder[i];
      }
      OPENGM_ASSERT(count == numberOfHigherOrderValues_);
   }
   obj_.setLinearCoefs(x_,obj);
 
   // set constraints 
   size_t constraintCounter = 0;
   // multiway cut constraints
   if(problemType_ == MWC) {
      // From each internal-node only one terminal-edge should be 0
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var) {
         c_.add(IloRange(env_, numberOfTerminals_-1, numberOfTerminals_-1));
         for(LabelType i=0; i<gm_.numberOfLabels(var); ++i) {
            c_[constraintCounter].setLinearCoef(x_[var*numberOfTerminals_+i],1);
         }
         ++constraintCounter;
      }
      // Inter-terminal-edges have to be 1
      for(size_t i=0; i<(size_t)(numberOfTerminals_*(numberOfTerminals_-1)/2); ++i) {
         c_.add(IloRange(env_, 1, 1));
         c_[constraintCounter].setLinearCoef(x_[numberOfTerminalEdges_+numberOfInternalEdges_+i],1);
         ++constraintCounter;
      }
   }

   
   // higher order constraints
   size_t count = 0;
   for(size_t i=0; i<higherOrderTerms.size(); ++i) {
      size_t factorID = higherOrderTerms[i].factorID_;
      size_t numVar   = gm_[factorID].numberOfVariables();
      OPENGM_ASSERT(numVar>2);

      if(higherOrderTerms[i].potts_) {
         double b = higherOrderTerms[i].lpIndices_.size();
         

         // Add only one constraint is sufficient with {0,1} constraints
         // ------------------------------------------------------------
         // ** -|E|+1 <= -|E|*y_H+\sum_{e\in H} y_e <= 0 
         if(parameter_.reductionMode_ % 2 == 1){
            c_.add(IloRange(env_, -b+1 , 0));
            for(size_t i1=0; i1<higherOrderTerms[i].lpIndices_.size();++i1) {
               const LPIndexType edgeID = higherOrderTerms[i].lpIndices_[i1]; 
               c_[constraintCounter].setLinearCoef(x_[edgeID],1);
            }
            c_[constraintCounter].setLinearCoef(x_[values.size()+count],-b);
            constraintCounter += 1;
         }
         // In general this additional contraints and more local constraints leeds to tighter relaxations
         // ---------------------------------------------------------------------------------------------
         if(parameter_.reductionMode_ % 4 >=2){ 
            // ** y_H <= sum_{e \in H} y_e
            c_.add(IloRange(env_, -2.0*b, 0));
            for(size_t i1=0; i1<higherOrderTerms[i].lpIndices_.size();++i1) {
               const LPIndexType edgeID = higherOrderTerms[i].lpIndices_[i1]; 
               c_[constraintCounter].setLinearCoef(x_[edgeID],-1);
            }
            c_[constraintCounter].setLinearCoef(x_[values.size()+count],1);
            constraintCounter += 1;
         
            // ** forall e \in H : y_H>=y_e
            for(size_t i1=0; i1<higherOrderTerms[i].lpIndices_.size();++i1) {
               const LPIndexType edgeID = higherOrderTerms[i].lpIndices_[i1]; 
               c_.add(IloRange(env_, 0 , 1));
               c_[constraintCounter].setLinearCoef(x_[edgeID],-1);
               c_[constraintCounter].setLinearCoef(x_[values.size()+count],1); 
               constraintCounter += 1;
            }        
         }
         count++;
      }else{
         if(numVar==3) {
            OPENGM_ASSERT(higherOrderTerms[i].valueIndex_<=valuesHigherOrder.size());
            LPIndexType edgeIDs[3];
            edgeIDs[0] = neighbours[gm_[factorID].variableIndex(0)][gm_[factorID].variableIndex(1)];
            edgeIDs[1] = neighbours[gm_[factorID].variableIndex(0)][gm_[factorID].variableIndex(2)];
            edgeIDs[2] = neighbours[gm_[factorID].variableIndex(1)][gm_[factorID].variableIndex(2)];
               
            const unsigned int P[] = {0,1,2,4,7,8,12,16,18,25,32,33,42,52,63};
            double c[3];  

            c_.add(IloRange(env_, 1, 1));
            size_t lvc=0;
            for(size_t p=0; p<5; p++){
               if(true || valuesHigherOrder[higherOrderTerms[i].valueIndex_+p]!=0){   
                  c_[constraintCounter].setLinearCoef(x_[values.size()+count+lvc],1);
                  ++lvc;
               }
            }
            ++constraintCounter;  

            for(size_t p=0; p<5; p++){
               if(true || valuesHigherOrder[higherOrderTerms[i].valueIndex_+p]!=0){
                  double ub = 2.0;
                  double lb = 0.0;
                  unsigned int mask = 1;
                  for(size_t n=0; n<3; n++){
                     if(P[p] & mask){
                        c[n] = -1.0;
                        ub--;
                        lb--; 
                     }
                     else{
                        c[n] = 1.0; 
                     }
                     mask = mask << 1;
                  }
                  c_.add(IloRange(env_, lb, ub));
                  for(size_t n=0; n<3; n++){
                     c_[constraintCounter].setLinearCoef(x_[edgeIDs[n]],c[n]);
                  }
                  c_[constraintCounter].setLinearCoef(x_[values.size()+count],-1);
                  ++constraintCounter;  
               
                  for(size_t n=0; n<3; n++){
                     if(c[n]>0){
                        c_.add(IloRange(env_, 0, 1));
                        c_[constraintCounter].setLinearCoef(x_[edgeIDs[n]],1);
                        c_[constraintCounter].setLinearCoef(x_[values.size()+count],-1);
                        ++constraintCounter;     
                     }else{
                        c_.add(IloRange(env_, -1, 0));
                        c_[constraintCounter].setLinearCoef(x_[edgeIDs[n]],-1);
                        c_[constraintCounter].setLinearCoef(x_[values.size()+count],-1);
                        ++constraintCounter;     
                     }     
                  }
                  ++count;
               }
            }
         }
         else if(numVar==4) {                  
            OPENGM_ASSERT(higherOrderTerms[i].valueIndex_<=valuesHigherOrder.size());
            LPIndexType edgeIDs[6];
            edgeIDs[0] = neighbours[gm_[factorID].variableIndex(0)][gm_[factorID].variableIndex(1)];
            edgeIDs[1] = neighbours[gm_[factorID].variableIndex(0)][gm_[factorID].variableIndex(2)];
            edgeIDs[2] = neighbours[gm_[factorID].variableIndex(1)][gm_[factorID].variableIndex(2)];
            edgeIDs[3] = neighbours[gm_[factorID].variableIndex(0)][gm_[factorID].variableIndex(3)];
            edgeIDs[4] = neighbours[gm_[factorID].variableIndex(1)][gm_[factorID].variableIndex(3)];
            edgeIDs[5] = neighbours[gm_[factorID].variableIndex(2)][gm_[factorID].variableIndex(3)];
          
               
            const unsigned int P[] = {0,1,2,4,7,8,12,16,18,25,32,33,42,52,63};
            double c[6];

            c_.add(IloRange(env_, 1, 1));
            size_t lvc=0;
            for(size_t p=0; p<15; p++){
               if(true ||valuesHigherOrder[higherOrderTerms[i].valueIndex_+p]!=0){   
                  c_[constraintCounter].setLinearCoef(x_[values.size()+count+lvc],1);
                  ++lvc;
               }
            }
            ++constraintCounter;  


            for(size_t p=0; p<15; p++){
               double ub = 5.0;
               double lb = 0.0;
               unsigned int mask = 1;
               for(size_t n=0; n<6; n++){
                  if(P[p] & mask){
                     c[n] = -1.0;
                     ub--;
                     lb--; 
                  }
                  else{
                     c[n] = 1.0; 
                  }
                  mask = mask << 1;
               }
               c_.add(IloRange(env_, lb, ub));
               for(size_t n=0; n<6; n++){
                  c_[constraintCounter].setLinearCoef(x_[edgeIDs[n]],c[n]);
               }
               c_[constraintCounter].setLinearCoef(x_[values.size()+count],-1);
               ++constraintCounter;  
               
               for(size_t n=0; n<6; n++){
                  if(c[n]>0){
                     c_.add(IloRange(env_, 0, 1));
                     c_[constraintCounter].setLinearCoef(x_[edgeIDs[n]],1);
                     c_[constraintCounter].setLinearCoef(x_[values.size()+count],-1);
                     ++constraintCounter;     
                  }else{
                     c_.add(IloRange(env_, -1, 0));
                     c_[constraintCounter].setLinearCoef(x_[edgeIDs[n]],-1);
                     c_[constraintCounter].setLinearCoef(x_[values.size()+count],-1);
                     ++constraintCounter;     
                  }     
               }
               ++count;
            }  
         }
         else{
            OPENGM_ASSERT(false);
         }
      }
   } 


   model_.add(obj_);
   if(constraintCounter>0) {
      model_.add(c_);
   }

   // initialize solver
   cplex_ = IloCplex(model_);

}

template<class GM, class ACC>
typename Multicut<GM, ACC>::ProblemType Multicut<GM, ACC>::setProblemType() {
   problemType_ = MC;
   for(size_t f=0; f<gm_.numberOfFactors();++f) {
      if(gm_[f].numberOfVariables()==1) {
         problemType_ = MWC;
      }
      if(gm_[f].numberOfVariables()>1) {
         for(size_t i=0; i<gm_[f].numberOfVariables();++i) {
            if(gm_[f].numberOfLabels(i)<gm_.numberOfVariables()) {
               problemType_ = MWC;
            }
         }
      }
      if(gm_[f].numberOfVariables()==2 && gm_[f].numberOfLabels(0)==2 && gm_[f].numberOfLabels(1)==2){
         problemType_ = MWC; //OK - can be reparmetrized
      }
      else if(gm_[f].numberOfVariables()>1 && !gm_[f].isGeneralizedPotts()) {
         problemType_ = INVALID;
         break;
      }
   } 
 
   // set member variables
   if(problemType_ == MWC) {
      numberOfTerminals_ = gm_.numberOfLabels(0); 
      numberOfInterTerminalEdges_ = (numberOfTerminals_*(numberOfTerminals_-1))/2; 
      numberOfTerminalEdges_ = 0;
      for(IndexType i=0; i<gm_.numberOfVariables(); ++i) {
         for(LabelType j=0; j<gm_.numberOfLabels(i); ++j) {
            ++numberOfTerminalEdges_;
         }
      } 
   }
   else{
      numberOfTerminalEdges_ = 0;
      numberOfTerminals_     = 0;
      numberOfInterTerminalEdges_ = 0;
   } 
   return problemType_;
}

//**********************************************
//**
//** Functions that find violated Constraints
//**
//**********************************************

template<class GM, class ACC>
size_t Multicut<GM, ACC>::removeUnusedConstraints()
{ 
   std::cout << "Not Implemented " <<std::endl ; 
   return 0;
}



template<class GM, class ACC>
size_t Multicut<GM, ACC>::enforceIntegerConstraints()
{
   size_t N=numberOfTerminalEdges_;
   if (N==0) N = numberOfInternalEdges_;

   for(size_t i=0; i<N; ++i)
      model_.add(IloConversion(env_, x_[i], ILOBOOL));
   for(size_t i=0; i<numberOfHigherOrderValues_; ++i)
      model_.add(IloConversion(env_, x_[numberOfTerminalEdges_+numberOfInternalEdges_+numberOfInterTerminalEdges_+i], ILOBOOL));
   integerMode_ = true;

   return N+numberOfHigherOrderValues_;
}

/// Find violated terminal triangle constrains
///  * Only for Multi Way Cut
///  * can be used for fractional and integer case
///  * check |E_I|*3*|L| constrains
///
///  (a,b)-(a,l)-(b,l) must be consistent for all l in L
template<class GM, class ACC>
size_t Multicut<GM, ACC>::findTerminalTriangleConstraints(IloRangeArray& constraint)
{
   OPENGM_ASSERT(problemType_ == MWC);
   if(!(problemType_ == MWC)) return 0;
   size_t tempConstrainCounter = constraintCounter_;

   size_t u,v;
   for(size_t i=0; i<numberOfInternalEdges_;++i) {
      u = edgeNodes_[i].first;//[0];
      v = edgeNodes_[i].second;//[1];
      for(size_t l=0; l<numberOfTerminals_;++l) {
         if(-sol_[numberOfTerminalEdges_+i]+sol_[u*numberOfTerminals_+l]+sol_[v*numberOfTerminals_+l]<-EPS_) {
            constraint.add(IloRange(env_, 0 , 2));
            constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],-1);
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+l],1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+l],1);
            ++constraintCounter_;
         }
         if(sol_[numberOfTerminalEdges_+i]-sol_[u*numberOfTerminals_+l]+sol_[v*numberOfTerminals_+l]<-EPS_) {
            constraint.add(IloRange(env_, 0 , 2));
            constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],1);
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+l],-1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+l],1);
            ++constraintCounter_;
         }
         if(sol_[numberOfTerminalEdges_+i]+sol_[u*numberOfTerminals_+l]-sol_[v*numberOfTerminals_+l]<-EPS_) {
            constraint.add(IloRange(env_, 0 , 2));
            constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],1);
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+l],1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+l],-1);
            ++constraintCounter_;
         }
      }
      if(constraintCounter_-tempConstrainCounter >= parameter_.maximalNumberOfConstraintsPerRound_)
         break;
   }
   return constraintCounter_-tempConstrainCounter;
}

/// Find violated multi terminal constrains
///  * Only for Multi Way Cut
///  * can be used for fractional and integer case
///  * check |E_I| constrains
///
///  x(u,v) >= \sum_{s\in S \subset T} x(u,t)-x(v,t)
template<class GM, class ACC>
size_t Multicut<GM, ACC>::findMultiTerminalConstraints(IloRangeArray& constraint)
{
   OPENGM_ASSERT(problemType_ == MWC);
   if(!(problemType_ == MWC)) return 0;
   size_t tempConstrainCounter = constraintCounter_;

   size_t u,v;
   for(size_t i=0; i<numberOfInternalEdges_;++i) {
      u = edgeNodes_[i].first;//[0];
      v = edgeNodes_[i].second;//[1];
      std::vector<size_t> terminals1;
      std::vector<size_t> terminals2;
      double sum1 = 0;
      double sum2 = 0;
      for(size_t l=0; l<numberOfTerminals_;++l) {
         if(sol_[u*numberOfTerminals_+l]-sol_[v*numberOfTerminals_+l] > EPS_) {
            terminals1.push_back(l);
            sum1 += sol_[u*numberOfTerminals_+l]-sol_[v*numberOfTerminals_+l];
         }
         if(sol_[v*numberOfTerminals_+l]-sol_[u*numberOfTerminals_+l] > EPS_) {
            terminals2.push_back(l);
            sum2 +=sol_[v*numberOfTerminals_+l]-sol_[u*numberOfTerminals_+l];
         }
      }
      if(sol_[numberOfTerminalEdges_+i]-sum1<-EPS_) {
         constraint.add(IloRange(env_, 0 , 200000));
         constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],1);
         for(size_t k=0; k<terminals1.size(); ++k) {
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+terminals1[k]],-1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+terminals1[k]],1);
         }
         ++constraintCounter_;
      }
      if(sol_[numberOfTerminalEdges_+i]-sum2<-EPS_) {
         constraint.add(IloRange(env_, 0 , 200000));
         constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],1);
         for(size_t k=0; k<terminals2.size(); ++k) {
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+terminals2[k]],1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+terminals2[k]],-1);
         }
         ++constraintCounter_;
      } 
      if(constraintCounter_-tempConstrainCounter >= parameter_.maximalNumberOfConstraintsPerRound_)
         break;
   } 
   return constraintCounter_-tempConstrainCounter;
}

/// Find violated integer terminal triangle constrains
///  * Only for Multi Way Cut
///  * can be used for integer case only
///  * check |E_I|*3*|L| constrains
///
///  (a,b)-(a,l)-(b,l) must be consistent for all l in L
template<class GM, class ACC>
size_t Multicut<GM, ACC>::findIntegerTerminalTriangleConstraints(IloRangeArray& constraint, std::vector<LabelType>& conf)
{ 
   OPENGM_ASSERT(integerMode_);
   OPENGM_ASSERT(problemType_ == MWC);
   if(!(problemType_ == MWC)) return 0;
   size_t tempConstrainCounter = constraintCounter_;

   size_t u,v;
   for(size_t i=0; i<numberOfInternalEdges_;++i) {
      u = edgeNodes_[i].first;//[0];
      v = edgeNodes_[i].second;//[1];
      if(sol_[numberOfTerminalEdges_+i]<EPS_ && (conf[u]!=conf[v]) ) {
         if(sol_[numberOfTerminalEdges_+i]-sol_[u*numberOfTerminals_+conf[u]]+sol_[v*numberOfTerminals_+conf[u]]<=0) {
            constraint.add(IloRange(env_, 0 , 10));
            constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],1);
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+conf[u]],-1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+conf[u]],1);
            ++constraintCounter_;
         }
         if(sol_[numberOfTerminalEdges_+i]-sol_[u*numberOfTerminals_+conf[u]]+sol_[v*numberOfTerminals_+conf[u]]<=0) {
            constraint.add(IloRange(env_, 0 , 10));
            constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],1);
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+conf[u]],1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+conf[u]],-1);
            ++constraintCounter_;
         }
         if(sol_[numberOfTerminalEdges_+i]-sol_[u*numberOfTerminals_+conf[v]]+sol_[v*numberOfTerminals_+conf[v]]<=0) {
            constraint.add(IloRange(env_, 0 , 10));
            constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],1);
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+conf[v]],-1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+conf[v]],1);
            ++constraintCounter_;
         }
         if(sol_[numberOfTerminalEdges_+i]+sol_[u*numberOfTerminals_+conf[v]]-sol_[v*numberOfTerminals_+conf[v]]<=0) {
            constraint.add(IloRange(env_, 0 , 10));
            constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],1);
            constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+conf[v]],1);
            constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+conf[v]],-1);
            ++constraintCounter_;
         }
      }
      if(sol_[numberOfTerminalEdges_+i]>1-EPS_ && (conf[u]==conf[v]) ) {
         constraint.add(IloRange(env_, 0 , 10));
         constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],-1);
         constraint[constraintCounter_].setLinearCoef(x_[u*numberOfTerminals_+conf[u]],1);
         constraint[constraintCounter_].setLinearCoef(x_[v*numberOfTerminals_+conf[v]],1);
         ++constraintCounter_;
      }
      if(constraintCounter_-tempConstrainCounter >= parameter_.maximalNumberOfConstraintsPerRound_)
         break;
   }
   return constraintCounter_-tempConstrainCounter;
}

/// Find violate cycle constrains
///  * add at most |E_I| constrains
///
///  
template<class GM, class ACC>
size_t Multicut<GM, ACC>::findCycleConstraints(
   IloRangeArray& constraint,
   bool addOnlyFacetDefiningConstraints,
   bool usePreBounding
)
{ 
   std::vector<LabelType> partit;
   std::vector<std::list<size_t> > neighbours0;

   size_t tempConstrainCounter = constraintCounter_;
 
   if(usePreBounding){
      partition(partit,neighbours0,1-EPS_);
   }
  
   std::map<std::pair<IndexType,IndexType>,size_t> counter;
   for(size_t i=0; i<numberOfInternalEdges_;++i) {
      
      IndexType u = edgeNodes_[i].first;//[0];
      IndexType v = edgeNodes_[i].second;//[1]; 
      
      if(usePreBounding && partit[u] != partit[v])
         continue;

      OPENGM_ASSERT(i+numberOfTerminalEdges_ == neighbours[u][v]);
      OPENGM_ASSERT(i+numberOfTerminalEdges_ == neighbours[v][u]);
      
      if(sol_[numberOfTerminalEdges_+i]>EPS_){
         //search for cycle
         std::vector<IndexType> path;
         const double pathLength = shortestPath(u,v,neighbours,sol_,path,sol_[numberOfTerminalEdges_+i],addOnlyFacetDefiningConstraints);
         if(sol_[numberOfTerminalEdges_+i]-EPS_>pathLength){
            OPENGM_ASSERT(path.size()>2);
            constraint.add(IloRange(env_, 0  , 1000000000)); 
            //negative zero seemed to be required for numerical reasons, even CPlex handel this by its own, too.
            constraint[constraintCounter_].setLinearCoef(x_[numberOfTerminalEdges_+i],-1);
            for(size_t n=0;n<path.size()-1;++n){
               constraint[constraintCounter_].setLinearCoef(x_[neighbours[path[n]][path[n+1]]],1);
            }
            ++constraintCounter_; 
         }
      } 
      if(constraintCounter_-tempConstrainCounter >= parameter_.maximalNumberOfConstraintsPerRound_)
         break;
   }
   return constraintCounter_-tempConstrainCounter;
}

template<class GM, class ACC>
size_t Multicut<GM, ACC>::findOddWheelConstraints(IloRangeArray& constraints){
   size_t tempConstrainCounter = constraintCounter_;
   std::vector<IndexType> var2node(gm_.numberOfVariables(),std::numeric_limits<IndexType>::max());
   for(size_t center=0; center<gm_.numberOfVariables();++center){
      var2node.assign(gm_.numberOfVariables(),std::numeric_limits<IndexType>::max());
      size_t N = neighbours[center].size();
      std::vector<IndexType> node2var(N);
      std::vector<EdgeMapType> E(2*N);
      std::vector<double>     w;
      typename EdgeMapType::const_iterator it;
      size_t id=0;
      for(it=neighbours[center].begin() ; it != neighbours[center].end(); ++it) {
         IndexType var = (*it).first;
         node2var[id]  = var;
         var2node[var] = id++;
      } 
     
      for(it=neighbours[center].begin() ; it != neighbours[center].end(); ++it) { 
         const IndexType var1 = (*it).first;
         const LPIndexType u = var2node[var1];   
         typename EdgeMapType::const_iterator it2;
         for(it2=neighbours[var1].begin() ; it2 != neighbours[var1].end(); ++it2) {
            const IndexType var2 = (*it2).first; 
            const LPIndexType v = var2node[var2];   
            if( v !=  std::numeric_limits<IndexType>::max()){
               if(u<v){
                  E[2*u][2*v+1]=w.size();
                  E[2*v+1][2*u]=w.size(); 
                  E[2*u+1][2*v]=w.size();
                  E[2*v][2*u+1]=w.size();
                  double weight = 0.5-sol_[neighbours[var1][var2]]+0.5*(sol_[neighbours[center][var1]]+sol_[neighbours[center][var2]]); 
                  //OPENGM_ASSERT(weight>-1e-7);
                  if(weight<0) weight=0;
                  w.push_back(weight);
                 
               }
            }
         }
      }
     
      //Search for odd wheels
      for(size_t n=0; n<N;++n) {
         std::vector<IndexType> path;
         const double pathLength = shortestPath(2*n,2*n+1,E,w,path,1e22,false);
         if(pathLength<0.5-EPS_*path.size()){// && (path.size())>3){
            OPENGM_ASSERT((path.size())>3);
            OPENGM_ASSERT(((path.size())/2)*2 == path.size() );

            constraints.add(IloRange(env_, -100000 , (path.size()-2)/2 ));
            for(size_t k=0;k<path.size()-1;++k){
               constraints[constraintCounter_].setLinearCoef(x_[neighbours[center][node2var[path[k]/2]]],-1);
            }
            for(size_t k=0;k<path.size()-1;++k){
               const IndexType u= node2var[path[k]/2];
               const IndexType v= node2var[path[k+1]/2];
               constraints[constraintCounter_].setLinearCoef(x_[neighbours[u][v]],1);
            }
            ++constraintCounter_; 
         } 
         if(constraintCounter_-tempConstrainCounter >= parameter_.maximalNumberOfConstraintsPerRound_)
            break;
      }
 
      //Reset var2node
      for(it=neighbours[center].begin() ; it != neighbours[center].end(); ++it) {
         var2node[(*it).first] = std::numeric_limits<IndexType>::max();
      }
    
   }
   
   return constraintCounter_-tempConstrainCounter;
}  
  
/// Find violate integer cycle constrains
///  * can be used for integer case only
///  * add at most |E_I| constrains
///
///  
template<class GM, class ACC>
size_t Multicut<GM, ACC>::findIntegerCycleConstraints(
   IloRangeArray& constraint,
   bool addFacetDefiningConstraintsOnly
)
{
   OPENGM_ASSERT(integerMode_);
   std::vector<LabelType> partit(numberOfNodes_,0);
   std::vector<std::list<size_t> > neighbours0;
   partition(partit,neighbours0);
   size_t tempConstrainCounter = constraintCounter_;
  
   //Find Violated Cycles inside a Partition
   size_t u,v;
   for(size_t i=0; i<numberOfInternalEdges_;++i) {
      u = edgeNodes_[i].first;//[0];
      v = edgeNodes_[i].second;//[1];
      OPENGM_ASSERT(partit[u] >= 0);
      if(sol_[numberOfTerminalEdges_+i]>=EPS_ && partit[u] == partit[v]) {
         //find shortest path from u to v by BFS
         std::queue<size_t> nodeList;
         std::vector<size_t> path(numberOfNodes_,std::numeric_limits<size_t>::max());
         size_t n = u;
         while(n!=v) {
            std::list<size_t>::iterator it;
            for(it=neighbours0[n].begin() ; it != neighbours0[n].end(); ++it) {
               if(path[*it]==std::numeric_limits<size_t>::max()) {
                  //Check if this path is chordless 
                  if(addFacetDefiningConstraintsOnly) {
                     bool isCordless = true;
                     size_t s = n;
                     const size_t c = *it;
                     while(s!=u){
                        s = path[s];
                        if(s==u && c==v) continue;
                        if(neighbours[c].find(s)!=neighbours[c].end()) {
                           isCordless = false;
                           break;
                        } 
                     }
                     if(isCordless){
                        path[*it]=n;
                        nodeList.push(*it);
                     }
                  }
                  else{
                     path[*it]=n;
                     nodeList.push(*it);
                  }
               }
            }
            if(nodeList.size()==0)
               break;
            n = nodeList.front(); nodeList.pop();
         }
         if(path[v] != std::numeric_limits<size_t>::max()){
            if(!integerMode_){//check if it is realy violated
               double w=0;
               while(n!=u) {
                  w += sol_[neighbours[n][path[n]]];
                  n=path[n];
               }
               if(sol_[neighbours[u][v]]-EPS_<w)//constraint is not violated
                  continue;
            }

            constraint.add(IloRange(env_, 0 , 1000000000));
            constraint[constraintCounter_].setLinearCoef(x_[neighbours[u][v]],-1);
            while(n!=u) {
               constraint[constraintCounter_].setLinearCoef(x_[neighbours[n][path[n]]],1);
               n=path[n];
            }
            ++constraintCounter_;
         } 
         if(constraintCounter_-tempConstrainCounter >= parameter_.maximalNumberOfConstraintsPerRound_)
            break;
      }
      if(constraintCounter_-tempConstrainCounter >= parameter_.maximalNumberOfConstraintsPerRound_)
         break;
   }
   return constraintCounter_-tempConstrainCounter;
}
//************************************************

template <class GM, class ACC>
InferenceTermination
Multicut<GM,ACC>::infer()
{
   EmptyVisitorType mcv;
   return infer(mcv);
}

template <class GM, class ACC>
void
Multicut<GM,ACC>::initCplex()
{

   cplex_.setParam(IloCplex::Threads, parameter_.numThreads_);
   cplex_.setParam(IloCplex::CutUp, parameter_.cutUp_);
   cplex_.setParam(IloCplex::MIPDisplay,0);
   cplex_.setParam(IloCplex::BarDisplay,0);
   cplex_.setParam(IloCplex::NetDisplay,0);
   cplex_.setParam(IloCplex::SiftDisplay,0);
   cplex_.setParam(IloCplex::SimDisplay,0);

   cplex_.setParam(IloCplex::EpOpt,1e-9);
   cplex_.setParam(IloCplex::EpRHS,1e-8); //setting this to 1e-9 seemed to be to agressive!
   cplex_.setParam(IloCplex::EpInt,0);
   cplex_.setParam(IloCplex::EpAGap,0);
   cplex_.setParam(IloCplex::EpGap,0);

   if(parameter_.verbose_ == true && parameter_.verboseCPLEX_) {
      cplex_.setParam(IloCplex::MIPDisplay,2);
      cplex_.setParam(IloCplex::BarDisplay,1);
      cplex_.setParam(IloCplex::NetDisplay,1);
      cplex_.setParam(IloCplex::SiftDisplay,1);
      cplex_.setParam(IloCplex::SimDisplay,1);
   }
}


template <class GM, class ACC>
template<class VisitorType>
InferenceTermination
Multicut<GM,ACC>::infer(VisitorType& mcv)
{
   std::vector<LabelType> conf(gm_.numberOfVariables());
   initCplex();
   //cplex_.setParam(IloCplex::RootAlg, IloCplex::Primal);
    
   if(problemType_ == INVALID){ 
      throw RuntimeError("Error:  Model can not be solved!"); 
   }
   else if(!readWorkFlow(parameter_.workFlow_)){//Use given workflow if posible
      std::cout << "Warning: can not parse workflow : " << parameter_.workFlow_ <<std::endl;
      std::cout << "Using default workflow ";
      if(problemType_ == MWC){
         std::cout << "(TTC)(MTC)(IC)(CC-IFD,TTC-I)" <<std::endl;
         readWorkFlow("(TTC)(MTC)(IC)(CC-IFD,TTC-I)");
      }
      else if(problemType_ == MC){
         std::cout << "(CC-FDB)(IC)(CC-I)" <<std::endl;
         readWorkFlow("(CC-FDB)(IC)(CC-I)");
      }
      else{
         throw RuntimeError("Error:  Model can not be solved!"); 
      }
   }

   Timer timer,timer2;
   timer.tic();     
   mcv.begin(*this);    
   size_t workingState = 0;
   while(workingState<workFlow_.size()){
      protocolateTiming_.push_back(std::vector<double>(20,0));
      protocolateConstraints_.push_back(std::vector<size_t>(20,0));
      std::vector<double>& T = protocolateTiming_.back();
      std::vector<size_t>& C = protocolateConstraints_.back();

      // Check for timeout
      timer.toc();
      if(timer.elapsedTime()>parameter_.timeOut_) {
         break;
      }
      //check for integer constraints   
      for (size_t it=1; it<10000000000; ++it) { 
         cplex_.setParam(IloCplex::Threads, parameter_.numThreads_); 
         cplex_.setParam(IloCplex::TiLim, parameter_.timeOut_-timer.elapsedTime());
         timer2.tic();
         if(!cplex_.solve()) {
            std::cout << "failed to optimize. " <<cplex_.getStatus()<< std::endl; 
            if(cplex_.getStatus() != IloAlgorithm::Unbounded){
               //Serious problem -> exit
               mcv(*this);  
               return NORMAL;
            }  
            else{ 
               //undbounded ray - most likely numerical problems
            }
         }
         if(cplex_.getStatus()!= IloAlgorithm::Unbounded){
            if(!integerMode_)
               bound_ = cplex_.getObjValue()+constant_;
            else{
               bound_ = cplex_.getBestObjValue()+constant_;
               if(!cplex_.solveFixed()) {
                  std::cout << "failed to fixed optimize." << std::endl; 
                  mcv(*this);
                  return NORMAL;
               }
            } 
         }
         else{
            //bound is not set - todo
         }
         cplex_.getValues(sol_, x_);
         timer2.toc();
         T[Protocol_ID_Solve] += timer2.elapsedTime();
         if(mcv(*this)!=0){
            workingState = workFlow_.size(); // go to the end of the workflow
            break;
         }         
 
         //std::cout << "... done."<<std::endl;
         
         //Find Violated Constraints
         IloRangeArray constraint = IloRangeArray(env_);
         constraintCounter_ = 0;
         
         //std::cout << "Search violated constraints ..." <<std::endl; 
       

         size_t cycleConstraints = std::numeric_limits<size_t>::max();
         bool   constraintAdded = false;
         for(std::vector<size_t>::iterator it=workFlow_[workingState].begin() ; it != workFlow_[workingState].end(); it++ ){
            size_t n = 0;
            size_t protocol_ID = Protocol_ID_Unknown;
            timer2.tic();
            if(*it == Action_ID_TTC){
               if(parameter_.verbose_) std::cout << "* Add  terminal triangle constraints: " << std::flush;
               n = findTerminalTriangleConstraints(constraint);
               if(parameter_.verbose_) std::cout << n << std::endl;
               protocol_ID = Protocol_ID_TTC;
            } 
            else if(*it == Action_ID_TTC_I){ 
               if(!integerMode_){
                  throw RuntimeError("Error: Calling integer terminal triangle constraint (TTC-I) seperation provedure before switching in integer mode (IC)"); 
               }
               if(parameter_.verbose_) std::cout << "* Add integer terminal triangle constraints: " << std::flush;
               arg(conf);
               n = findIntegerTerminalTriangleConstraints(constraint, conf);
               if(parameter_.verbose_) std::cout << n  << std::endl;
               protocol_ID = Protocol_ID_TTC;
        
            }
            else if(*it == Action_ID_MTC){
               if(parameter_.verbose_) std::cout << "* Add multi terminal constraints: " << std::flush;
               n = findMultiTerminalConstraints(constraint);
               if(parameter_.verbose_) std::cout <<  n << std::endl;
               protocol_ID = Protocol_ID_MTC;
        
            }
            else if(*it == Action_ID_CC_I){
               if(!integerMode_){
                  throw RuntimeError("Error: Calling integer cycle constraints (CC-I) seperation provedure before switching in integer mode (IC)"); 
               }
               if(parameter_.verbose_) std::cout << "Add integer cycle constraints: " << std::flush;
               n = findIntegerCycleConstraints(constraint, false);
               if(parameter_.verbose_) std::cout << n  << std::endl; 
               protocol_ID = Protocol_ID_CC;
            } 
            else if(*it == Action_ID_CC_IFD){ 
               if(!integerMode_){
                  throw RuntimeError("Error: Calling facet defining integer cycle constraints (CC-IFD) seperation provedure before switching in integer mode (IC)"); 
               }
               if(parameter_.verbose_) std::cout << "Add facet defining integer cycle constraints: " << std::flush;
               n = findIntegerCycleConstraints(constraint, true);
               if(parameter_.verbose_) std::cout << n  << std::endl; 
               protocol_ID = Protocol_ID_CC;
            }
            else if(*it == Action_ID_CC){
               if(parameter_.verbose_) std::cout << "Add cycle constraints: " << std::flush;     
               n = findCycleConstraints(constraint, false, false);
               cycleConstraints=n;
               if(parameter_.verbose_) std::cout  << n << std::endl; 
               protocol_ID = Protocol_ID_CC;
            } 
            else if(*it == Action_ID_CC_FD){
               if(parameter_.verbose_) std::cout << "Add facet defining cycle constraints: " << std::flush;     
               n = findCycleConstraints(constraint, true, false);
               cycleConstraints=n;
               if(parameter_.verbose_) std::cout  << n << std::endl; 
               protocol_ID = Protocol_ID_CC;
            } 
            else if(*it == Action_ID_CC_FDB){
               if(parameter_.verbose_) std::cout << "Add facet defining cycle constraints (with bounding): " << std::flush;     
               n = findCycleConstraints(constraint, true, true);
               cycleConstraints=n;
               if(parameter_.verbose_) std::cout  << n << std::endl; 
               protocol_ID = Protocol_ID_CC;
            }
            else if(*it == Action_ID_CC_B){
               if(parameter_.verbose_) std::cout << "Add cycle constraints (with bounding): " << std::flush;     
               n = findCycleConstraints(constraint, false, true);
               cycleConstraints=n;
               if(parameter_.verbose_) std::cout  << n << std::endl; 
               protocol_ID = Protocol_ID_CC;
            } 
            else  if(*it == Action_ID_RemoveConstraints){ 
               if(parameter_.verbose_) std::cout << "Remove unused constraints: " << std::flush;            
               n = removeUnusedConstraints();
               if(parameter_.verbose_) std::cout  << n  << std::endl; 
               protocol_ID = Protocol_ID_RemoveConstraints;  
            }
            else  if(*it == Action_ID_IntegerConstraints){
               if(integerMode_) continue;
               if(parameter_.verbose_) std::cout << "Add  integer constraints: " << std::flush;
               n = enforceIntegerConstraints();
               if(parameter_.verbose_) std::cout  << n << std::endl; 
               protocol_ID = Protocol_ID_IntegerConstraints;  
            } 
            else  if(*it == Action_ID_OWC){
               if(cycleConstraints== std::numeric_limits<size_t>::max()){
                  std::cout << "WARNING: using Odd Wheel Contraints without Cycle Constrains! -> Use CC,OWC!"<<std::endl;
                  n=0;
               }
               else if(cycleConstraints==0){             
                  if(parameter_.verbose_) std::cout << "Add odd wheel constraints: " << std::flush;
                  n = findOddWheelConstraints(constraint);
                  if(parameter_.verbose_) std::cout  << n << std::endl;
               }
               else{
                  //since cycle constraints are found we have to search for more violated cycle constraints first
               }
               protocol_ID = Protocol_ID_OWC;  
            }
            else{
               std::cout <<"Unknown Inference State "<<*it<<std::endl;
            } 
            timer2.toc();
            T[protocol_ID] += timer2.elapsedTime();
            C[protocol_ID] += n;
            if(n>0) constraintAdded = true;
         }
         //std::cout <<"... done!"<<std::endl;
         
       
         
         if(!constraintAdded){
            //Switch to next working state
            ++workingState;
            if(workingState<workFlow_.size())
               if(parameter_.verbose_) std::cout <<std::endl<< "** Switching into next state ( "<< workingState << " )**" << std::endl;
            break;
         }
         else{
            timer2.tic();
            //Add Constraints
            model_.add(constraint);
            //cplex_.addCuts(constraint);
            timer2.toc();
            T[Protocol_ID_AddConstraints] += timer2.elapsedTime();
         }
         
         // Check for timeout
         timer.toc();
         if(timer.elapsedTime()>parameter_.timeOut_) {
            break;
         }
         
      } //end inner loop over one working state
   } //end loop over all working states
   
   mcv.end(*this); 
   if(parameter_.verbose_){
      std::cout << "end normal"<<std::endl;
      std::cout <<"Protokoll:"<<std::endl;
      std::cout <<"=========="<<std::endl;
      std::cout << "  i  |   SOLVE  |   ADD    |    CC    |    OWC   |    TTC   |    MTV   |     IC    " <<std::endl;
      std::cout << "-----+----------+----------+----------+----------+----------+----------+-----------" <<std::endl;
   }
   std::vector<size_t> IDS;
   IDS.push_back(Protocol_ID_Solve);
   IDS.push_back(Protocol_ID_AddConstraints);
   IDS.push_back(Protocol_ID_CC);
   IDS.push_back(Protocol_ID_OWC);
   IDS.push_back(Protocol_ID_TTC);
   IDS.push_back(Protocol_ID_MTC);
   IDS.push_back(Protocol_ID_IntegerConstraints);
 
   if(parameter_.verbose_){
      for(size_t i=0; i<protocolateTiming_.size(); ++i){
         std::cout << std::setw(5)<<   i  ;
         for(size_t n=0; n<IDS.size(); ++n){
            std::cout << "|"<< std::setw(10) << setiosflags(std::ios::fixed)<< std::setprecision(4) << protocolateConstraints_[i][IDS[n]];
         }
         std::cout << std::endl; 
         std::cout << "     "  ; 
         for(size_t n=0; n<IDS.size(); ++n){ 
            std::cout << "|"<< std::setw(10) << setiosflags(std::ios::fixed)<< std::setprecision(4) << protocolateTiming_[i][IDS[n]];
         }
         std::cout << std::endl;
         std::cout << "-----+----------+----------+----------+----------+----------+----------+-----------" <<std::endl;
      }
      std::cout << "sum  ";
      double t_all=0;
      for(size_t n=0; n<IDS.size(); ++n){
         double t_one=0;
         for(size_t i=0; i<protocolateTiming_.size(); ++i){
            t_one += protocolateTiming_[i][IDS[n]];
         }
         std::cout << "|"<< std::setw(10) << setiosflags(std::ios::fixed)<< std::setprecision(4) << t_one;
         t_all += t_one;
      }
      std::cout << " | " <<t_all <<std::endl;
      std::cout << "-----+----------+----------+----------+----------+----------+----------+-----------" <<std::endl;
   }
   return NORMAL;
}

template <class GM, class ACC>
InferenceTermination
Multicut<GM,ACC>::arg
(
   std::vector<typename Multicut<GM,ACC>::LabelType>& x,
   const size_t N
   ) const
{
   if(N!=1) {
      return UNKNOWN;
   }
   else{
      if(problemType_ == MWC) {
         if(parameter_.MWCRounding_== parameter_.NEAREST){
            x.resize(gm_.numberOfVariables());
            for(IndexType node = 0; node<gm_.numberOfVariables(); ++node) {
               double v = sol_[numberOfTerminals_*node+0];
               x[node] = 0;
               for(LabelType i=0; i<gm_.numberOfLabels(node); ++i) {
                  if(sol_[numberOfTerminals_*node+i]<v) {
                     x[node]=i;
                  }
               }
            }
            return NORMAL;
         }
         else if(parameter_.MWCRounding_==parameter_.DERANDOMIZED){
            return derandomizedRounding(x);
         }
         else if(parameter_.MWCRounding_==parameter_.PSEUDODERANDOMIZED){
            return pseudoDerandomizedRounding(x,1000);
         }
         else{
            return UNKNOWN;
         }
      }
      else{
         std::vector<std::list<size_t> > neighbours0;
         InferenceTermination r =  partition(x, neighbours0,parameter_.edgeRoundingValue_);
         return r;
      }
   }
}


template <class GM, class ACC>
InferenceTermination
Multicut<GM,ACC>::pseudoDerandomizedRounding
(
   std::vector<typename Multicut<GM,ACC>::LabelType>& x,
   size_t bins
   ) const
{
   std::vector<bool>      processedBins(bins+1,false);
   std::vector<LabelType> temp;
   double                 value = 1000000000000.0;
   std::vector<LabelType> labelorder1(numberOfTerminals_,numberOfTerminals_-1);
   std::vector<LabelType> labelorder2(numberOfTerminals_,numberOfTerminals_-1);
   for(LabelType i=0; i<numberOfTerminals_-1;++i){
      labelorder1[i]=i;
      labelorder2[i]=numberOfTerminals_-2-i;
   } 
   for(size_t i=0; i<numberOfTerminals_*gm_.numberOfVariables();++i){
      size_t bin;
      double t,d;
      if(sol_[i]<=0){
         bin=0;
         t=0;
      }
      else if(sol_[i]>=1){
         bin=bins;
         t=1;
      }
      else{
         bin = (size_t)(sol_[i]*bins)%bins;
         t = sol_[i];
      }
      if(!processedBins[bin]){
         processedBins[bin]=true;
         if(value>(d=derandomizedRoundingSubProcedure(temp,labelorder1,sol_[i]))){
            value=d;
            x=temp;
         }
         if(value>(d=derandomizedRoundingSubProcedure(temp,labelorder2,sol_[i]))){
            value=d;
            x=temp;
         }
      }
   }
   return NORMAL;
}

template <class GM, class ACC>
InferenceTermination
Multicut<GM,ACC>::derandomizedRounding
(
   std::vector<typename Multicut<GM,ACC>::LabelType>& x
   ) const
{
   std::vector<LabelType> temp;
   double                 value = 1000000000000.0;
   std::vector<LabelType> labelorder1(numberOfTerminals_,numberOfTerminals_-1);
   std::vector<LabelType> labelorder2(numberOfTerminals_,numberOfTerminals_-1);
   for(LabelType i=0; i<numberOfTerminals_-1;++i){
      labelorder1[i]=i;
      labelorder2[i]=numberOfTerminals_-2-i;
   }
   // Test primitives
   double d;
   if(value>(d=derandomizedRoundingSubProcedure(temp,labelorder1,1e-8))){
      value=d;
      x=temp;
   }
   if(value>(d=derandomizedRoundingSubProcedure(temp,labelorder2,1e-8))){
      value=d;
      x=temp;
   } 
   if(value>(d=derandomizedRoundingSubProcedure(temp,labelorder1,1-1e-8))){
      value=d;
      x=temp;
   }
   if(value>(d=derandomizedRoundingSubProcedure(temp,labelorder2,1-1e-8))){
      value=d;
      x=temp;
   }
   for(size_t i=0; i<numberOfTerminals_*gm_.numberOfVariables();++i){
      if( sol_[i]>1e-8 &&  sol_[i]<1-1e-8){
         if(value>(d=derandomizedRoundingSubProcedure(temp,labelorder1,sol_[i]))){
            value=d;
            x=temp;
         }
         if(value>(d=derandomizedRoundingSubProcedure(temp,labelorder2,sol_[i]))){
            value=d;
            x=temp;
         }
      }
   }
   return NORMAL;
}

template <class GM, class ACC>
double
Multicut<GM,ACC>::derandomizedRoundingSubProcedure
(
   std::vector<typename Multicut<GM,ACC>::LabelType>& x,
   const std::vector<typename Multicut<GM,ACC>::LabelType>& labelorder,
   const double threshold
   ) const
{ 
   const LabelType lastLabel = labelorder.back();
   const IndexType numVar    = gm_.numberOfVariables();

   x.assign(numVar,lastLabel);
  
   for(size_t i=0; i<labelorder.size()-1; ++i){
      for(IndexType v=0; v<numVar; ++v){
         if(x[v]==lastLabel && sol_[numberOfTerminals_*v+i]<=threshold){
            x[v]=labelorder[i];
         }
      }
   }
   return gm_.evaluate(x);
}




template <class GM, class ACC>
InferenceTermination
Multicut<GM,ACC>::partition
(
   std::vector<LabelType>& partit,
   std::vector<std::list<size_t> >& neighbours0,
   double threshold
   ) const
{

   partit.resize(numberOfNodes_,0);
   neighbours0.resize(numberOfNodes_, std::list<size_t>());

   size_t counter=0;
   for(size_t i=0; i<numberOfInternalEdges_; ++i) {
      IndexType u = edgeNodes_[i].first;//variableIndex(0);
      IndexType v = edgeNodes_[i].second;//variableIndex(1);
      if(sol_[numberOfTerminalEdges_+counter] <= threshold) {
         neighbours0[u].push_back(v);
         neighbours0[v].push_back(u);
      }
      ++counter;
   }

   LabelType p=0;
   std::vector<bool> assigned(numberOfNodes_,false);
   for(size_t node=0; node<numberOfNodes_; ++node) {
      if(assigned[node])
         continue;
      else{
         std::list<size_t> nodeList;
         partit[node]   = p;
         assigned[node] = true;
         nodeList.push_back(node);
         while(!nodeList.empty()) {
            size_t n=nodeList.front(); nodeList.pop_front();
            std::list<size_t>::const_iterator it;
            for(it=neighbours0[n].begin() ; it != neighbours0[n].end(); ++it) {
               if(!assigned[*it]) {
                  partit[*it] = p;
                  assigned[*it] = true;
                  nodeList.push_back(*it);
               }
            }
         }
         ++p;
      }
   }
   return NORMAL;
}


template <class GM, class ACC>
typename Multicut<GM,ACC>::ValueType
Multicut<GM,ACC>::evaluate
(
   std::vector<LabelType>& conf
   ) const
{

   return gm_.evaluate(conf);
}

template<class GM, class ACC>
inline const typename Multicut<GM, ACC>::GraphicalModelType&
Multicut<GM, ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC>
typename GM::ValueType Multicut<GM, ACC>::bound() const
{
   return bound_;
}

template<class GM, class ACC>
typename GM::ValueType Multicut<GM, ACC>::value() const
{
   std::vector<LabelType> c;
   arg(c);  
   ValueType value = gm_.evaluate(c);
   return value;
}

template<class GM, class ACC>
bool Multicut<GM, ACC>::readWorkFlow(std::string s)
{
   if(s[0]!='(' || s[s.size()-1] !=')')
      return false;
   workFlow_.clear();
   size_t n=0;
   std::string sepString;
   if(s.size()<2)
      return false;
   while(n<s.size()){
      if(s[n]==',' || s[n]==')'){//End of sepString
         if(sepString.compare("CC")==0)
            workFlow_.back().push_back(Action_ID_CC);
         else if(sepString.compare("CC-I")==0)
            workFlow_.back().push_back(Action_ID_CC_I);
         else if(sepString.compare("CC-IFD")==0)
            workFlow_.back().push_back(Action_ID_CC_IFD);
         else if(sepString.compare("CC-B")==0)
            workFlow_.back().push_back(Action_ID_CC_B);
         else if(sepString.compare("CC-FDB")==0)
            workFlow_.back().push_back(Action_ID_CC_FDB);
         else if(sepString.compare("CC-FD")==0)
            workFlow_.back().push_back(Action_ID_CC_FD);
         else if(sepString.compare("TTC")==0)
            workFlow_.back().push_back(Action_ID_TTC);
         else if(sepString.compare("TTC-I")==0)
            workFlow_.back().push_back(Action_ID_TTC_I);
         else if(sepString.compare("MTC")==0)
            workFlow_.back().push_back(Action_ID_MTC);
         else if(sepString.compare("OWC")==0)
            workFlow_.back().push_back(Action_ID_OWC);
         else if(sepString.compare("IC")==0)
            workFlow_.back().push_back(Action_ID_IntegerConstraints);
         else
            std::cout << "WARNING:  Unknown Seperation Procedure ' "<<sepString<<"' is skipped!"<<std::endl;
         sepString.clear();
      }
      else if(s[n]=='('){//New Round
         workFlow_.push_back(std::vector<size_t>()); 
      }
      else{
         sepString += s[n];
      }
      ++n;
   }
   return true;
}
  

///
/// computed sigle shortest path by the Dijkstra algorithm with following modifications:
/// * stop when target node (endNode) is reached
/// * optional avoid chordal paths (cordless = true)
/// * avoid paths that are longer than a threshold (maxLength)
template<class GM, class ACC>
template<class DOUBLEVECTOR>
inline double Multicut<GM, ACC>::shortestPath(
   const IndexType startNode, 
   const IndexType endNode, 
   const std::vector<EdgeMapType >& E, //E[n][i].first/.second are the i-th neighbored node and weight-index (for w), respectively. 
   const DOUBLEVECTOR& w,
   std::vector<IndexType>& shortestPath,
   const double maxLength,
   bool cordless
) const
{ 
   
   const IndexType numberOfNodes = E.size();
   const double    inf           = std::numeric_limits<double>::infinity();
   const IndexType nonePrev      = endNode;

   std::vector<IndexType>  prev(numberOfNodes,nonePrev);
   std::vector<double>     dist(numberOfNodes,inf);
   MYSET                   openNodes;
   
   openNodes.insert(startNode);
   dist[startNode]=0.0;

   while(!openNodes.empty()){ 
      IndexType node;
      //Find smallest open node
      {
         typename MYSET::iterator it, itMin;
         double v = std::numeric_limits<double>::infinity();
         for(it = openNodes.begin(); it!= openNodes.end(); ++it){
            if( dist[*it]<v ){
               v = dist[*it];
               itMin = it;
            }
         }
         node = *itMin;
         openNodes.erase(itMin);
      }
      // Check if target is reached
      if(node == endNode)
         break;
      // Update all neigbors of node
      {
         typename EdgeMapType::const_iterator it;
         for(it=E[node].begin() ; it != E[node].end(); ++it) {
            const IndexType node2      = (*it).first;  //second edge-node
            const LPIndexType weighId  = (*it).second; //index in weigh-vector w
            double cuttedWeight        = std::max(0.0,w[weighId]); //cut up negative edge-weights
            const double weight2       = dist[node]+cuttedWeight;
           

            if(dist[node2] > weight2 && weight2 < maxLength){
               //check chordality
               bool updateNode = true;
               if(cordless) {
                  IndexType s = node;
                  while(s!=startNode){
                     s= prev[s];
                     if(s==startNode && node2==endNode) continue;
                     if(neighbours[node2].find(s)!=neighbours[node2].end()) {
                        updateNode = false; // do not update node if path is chordal
                        break;
                     } 
                  }
               } 
               if(updateNode){
                  prev[node2] = node;
                  dist[node2] = weight2;
                  openNodes.insert(node2);
               } 
            }
         }
      }
   }
   
   if(prev[endNode] != nonePrev){//find path?
      shortestPath.clear();
      shortestPath.push_back(endNode);
      IndexType n = endNode;
      do{
         n=prev[n];
         shortestPath.push_back(n);
      }while(n!=startNode);
      OPENGM_ASSERT(shortestPath.back() == startNode);
   }
   
   return dist[endNode];
}


template<class GM, class ACC>
std::vector<double>
Multicut<GM, ACC>::getEdgeLabeling
() const
{
   std::vector<double> l(numberOfInternalEdges_,0);
   for(size_t i=0; i<numberOfInternalEdges_; ++i) {
      l[i] = sol_[numberOfTerminalEdges_+i];
   }
   return l;
}

// WARNING: this function is considered experimental.
// variable indices refer to variables of the LP set up
// in the constructor of the class template LPCplex,
// NOT to the variables of the graphical model.
template<class GM, class ACC>
template<class LPVariableIndexIterator, class CoefficientIterator>
void Multicut<GM, ACC>::addConstraint
(
   LPVariableIndexIterator viBegin,
   LPVariableIndexIterator viEnd,
   CoefficientIterator coefficient,
   const ValueType& lowerBound,
   const ValueType& upperBound)
{
   IloRange constraint(env_, lowerBound, upperBound);
   while(viBegin != viEnd) {
      constraint.setLinearCoef(x_[*viBegin], *coefficient);
      ++viBegin;
      ++coefficient;
   }
   model_.add(constraint);
   // this update of model_ does not require a
   // re-initialization of the object cplex_.
   // cplex_ is initialized in the constructor.
}

} // end namespace opengm

#endif
