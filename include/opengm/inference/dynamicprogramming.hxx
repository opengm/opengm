#pragma once
#ifndef OPENGM_DYNAMICPROGRAMMING_HXX
#define OPENGM_DYNAMICPROGRAMMING_HXX

#include <typeinfo>
#include <limits>
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {

  /// DynamicProgramming
  ///\ingroup inference
  /// \ingroup messagepassing_inference
  template<class GM, class ACC>
  class DynamicProgramming : public Inference<GM, ACC> {
  public:
    typedef ACC AccumulationType;
    typedef ACC AccumulatorType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef LabelType  MyStateType;
    typedef ValueType  MyValueType;
    typedef visitors::VerboseVisitor<DynamicProgramming<GM, ACC> > VerboseVisitorType;
    typedef visitors::EmptyVisitor<DynamicProgramming<GM, ACC> >   EmptyVisitorType;
    typedef visitors::TimingVisitor<DynamicProgramming<GM, ACC> >  TimingVisitorType;
    struct Parameter {
      std::vector<IndexType> roots_;
    };

    DynamicProgramming(const GraphicalModelType&, const Parameter& = Parameter());
    ~DynamicProgramming();

    std::string name() const;
    const GraphicalModelType& graphicalModel() const;
    InferenceTermination infer();
    template< class VISITOR>
    InferenceTermination infer(VISITOR &);
    InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
    
    
    void getNodeInfo(const IndexType Inode, std::vector<ValueType>& values, std::vector<std::vector<LabelType> >& substates, std::vector<IndexType>& nodes) const;
    

  private:
    const GraphicalModelType& gm_;
    Parameter para_;
    MyValueType* valueBuffer_;
    MyStateType* stateBuffer_;
    std::vector<MyValueType*> valueBuffers_;
    std::vector<MyStateType*> stateBuffers_;
    std::vector<size_t> nodeOrder_; 
    std::vector<size_t> orderedNodes_;
    bool inferenceStarted_;
  };

  template<class GM, class ACC>
  inline std::string
  DynamicProgramming<GM, ACC>::name() const {
    return "DynamicProgramming";
  }

  template<class GM, class ACC>
  inline const typename DynamicProgramming<GM, ACC>::GraphicalModelType&
  DynamicProgramming<GM, ACC>::graphicalModel() const {
    return gm_;
  }

  template<class GM, class ACC>
  DynamicProgramming<GM, ACC>::~DynamicProgramming()
  {
    free(valueBuffer_);
    free(stateBuffer_);
  }
  
  template<class GM, class ACC>
  inline DynamicProgramming<GM, ACC>::DynamicProgramming
  (
  const GraphicalModelType& gm, 
  const Parameter& para
  ) 
  :  gm_(gm), inferenceStarted_(false)
  {
    OPENGM_ASSERT(gm_.isAcyclic());
    para_ = para;
    
    // Set nodeOrder 
    std::vector<size_t> numChildren(gm_.numberOfVariables(),0);
    std::vector<size_t> nodeList;
    size_t orderCount = 0;
    size_t varCount   = 0;
    nodeOrder_.resize(gm_.numberOfVariables(),std::numeric_limits<std::size_t>::max());
    size_t rootCounter=0;
    while(varCount < gm_.numberOfVariables() && orderCount < gm_.numberOfVariables()){
      if(rootCounter<para_.roots_.size()){
        nodeOrder_[para_.roots_[rootCounter]] = orderCount++;
        nodeList.push_back(para_.roots_[rootCounter]);
        ++rootCounter;
      }
      else if(nodeOrder_[varCount]==std::numeric_limits<std::size_t>::max()){
        nodeOrder_[varCount] = orderCount++;
        nodeList.push_back(varCount);
      }
      ++varCount;
      while(nodeList.size()>0){
        size_t node = nodeList.back();
        nodeList.pop_back();
        for(typename GM::ConstFactorIterator it=gm_.factorsOfVariableBegin(node); it !=gm_.factorsOfVariableEnd(node); ++it){
          const typename GM::FactorType& factor = gm_[(*it)];
          if( factor.numberOfVariables() == 2 ){
            if( factor.variableIndex(1) == node && nodeOrder_[factor.variableIndex(0)]==std::numeric_limits<std::size_t>::max() ){
              nodeOrder_[factor.variableIndex(0)] = orderCount++;
              nodeList.push_back(factor.variableIndex(0));
              ++numChildren[node];
            }
            if( factor.variableIndex(0) == node && nodeOrder_[factor.variableIndex(1)]==std::numeric_limits<std::size_t>::max() ){
              nodeOrder_[factor.variableIndex(1)] = orderCount++;
              nodeList.push_back(factor.variableIndex(1));
              ++numChildren[node];                       
            }
          }
        }
      }
    }

    // Allocate memmory
    size_t memSizeValue = 0;
    size_t memSizeState = 0;
    for(size_t i=0; i<gm_.numberOfVariables();++i){
      memSizeValue += gm_.numberOfLabels(i);
      memSizeState += gm.numberOfLabels(i) * numChildren[i];
    }
    valueBuffer_ = (MyValueType*) malloc(memSizeValue*sizeof(MyValueType));
    stateBuffer_ = (MyStateType*) malloc(memSizeState*sizeof(MyStateType));
    valueBuffers_.resize(gm_.numberOfVariables());
    stateBuffers_.resize(gm_.numberOfVariables()); 
    
    MyValueType* valuePointer =  valueBuffer_;
    MyStateType* statePointer =  stateBuffer_;
    for(size_t i=0; i<gm_.numberOfVariables();++i){
      valueBuffers_[i] = valuePointer;
      valuePointer += gm.numberOfLabels(i);
      stateBuffers_[i] = statePointer;
      statePointer +=  gm.numberOfLabels(i) * numChildren[i];
    }
    
    orderedNodes_.resize(gm_.numberOfVariables(),std::numeric_limits<std::size_t>::max());
    for(size_t i=0; i<gm_.numberOfVariables(); ++i)
      orderedNodes_[nodeOrder_[i]] = i;
    
  }
  
  template<class GM, class ACC>
  inline InferenceTermination 
  DynamicProgramming<GM, ACC>::infer(){
    EmptyVisitorType v;
    return infer(v);
  }
  
  template<class GM, class ACC>
  template<class VISITOR>
  inline InferenceTermination 
  DynamicProgramming<GM, ACC>::infer
  (
  VISITOR & visitor
  ){
     visitor.begin(*this);
     inferenceStarted_ = true;
    for(size_t i=1; i<=gm_.numberOfVariables();++i){
      const size_t node = orderedNodes_[gm_.numberOfVariables()-i];
      // set buffer neutral
      for(size_t n=0; n<gm_.numberOfLabels(node); ++n){
        OperatorType::neutral(valueBuffers_[node][n]);
      }
      // accumulate messages
      size_t childrenCounter = 0;
      for(typename GM::ConstFactorIterator it=gm_.factorsOfVariableBegin(node); it !=gm_.factorsOfVariableEnd(node); ++it){
        const typename GM::FactorType& factor = gm_[(*it)];

        // unary
        if(factor.numberOfVariables()==1 ){
          for(size_t n=0; n<gm_.numberOfLabels(node); ++n){
            const ValueType fac = factor(&n);
            OperatorType::op(fac, valueBuffers_[node][n]); 
          } 
        }
        
        //pairwise
        if( factor.numberOfVariables()==2 ){
          size_t vec[] = {0,0};
          if(factor.variableIndex(0) == node && nodeOrder_[factor.variableIndex(1)]>nodeOrder_[node] ){
            const size_t node2 = factor.variableIndex(1);
            MyStateType s;
            MyValueType v,v2;
            for(vec[0]=0; vec[0]<gm_.numberOfLabels(node); ++vec[0]){
              ACC::neutral(v);
              for(vec[1]=0; vec[1]<gm_.numberOfLabels(node2); ++vec[1]){ 
                const ValueType fac = factor(vec);
                OperatorType::op(fac,valueBuffers_[node2][vec[1]],v2) ;
                if(ACC::bop(v2,v)){
                  v=v2;
                  s=vec[1];
                }
              }
              stateBuffers_[node][childrenCounter*gm_.numberOfLabels(node)+vec[0]] = s;
              OperatorType::op(v,valueBuffers_[node][vec[0]]);
            }  
            ++childrenCounter;
            
          }
          if(factor.variableIndex(1) == node && nodeOrder_[factor.variableIndex(0)]>nodeOrder_[node]){ 
            const size_t node2 = factor.variableIndex(0);
            MyStateType s;
            MyValueType v,v2;
            for(vec[1]=0; vec[1]<gm_.numberOfLabels(node); ++vec[1]){
              ACC::neutral(v);
              for(vec[0]=0; vec[0]<gm_.numberOfLabels(node2); ++vec[0]){
                const ValueType fac = factor(vec);
                OperatorType::op(fac,valueBuffers_[node2][vec[0]],v2); 
                if(ACC::bop(v2,v)){
                  v=v2;
                  s=vec[0];
                }
              }  
              stateBuffers_[node][childrenCounter*gm_.numberOfLabels(node)+vec[1]] = s;
              OperatorType::op(v,valueBuffers_[node][vec[1]]); 
            }
            ++childrenCounter;                      
          }
        }
        // higher order
        if( factor.numberOfVariables()>2 ){
           throw std::runtime_error("This implementation of Dynamic Programming does only support second order models so far, but could be extended.");
        }

      } 
    }
    visitor.end(*this);
    return NORMAL;
  }

  template<class GM, class ACC>
  inline InferenceTermination DynamicProgramming<GM, ACC>::arg
  (
  std::vector<LabelType>& arg, 
  const size_t n
  ) const {
    if(n > 1) {
       arg.assign(gm_.numberOfVariables(), 0);
      return UNKNOWN;
    } 
    else {
       if(inferenceStarted_) {
         std::vector<size_t> nodeList;
         arg.assign(gm_.numberOfVariables(), std::numeric_limits<LabelType>::max() );
         size_t var = 0;
         while(var < gm_.numberOfVariables()){
           if(arg[var]==std::numeric_limits<LabelType>::max()){
             MyValueType v; ACC::neutral(v);
             for(size_t i=0; i<gm_.numberOfLabels(var); ++i){
               if(ACC::bop(valueBuffers_[var][i], v)){
                 v = valueBuffers_[var][i];
                 arg[var]=i;
               }
             }
             nodeList.push_back(var);
           }
           ++var;
           while(nodeList.size()>0){
             size_t node = nodeList.back();
             size_t childrenCounter = 0;
             nodeList.pop_back();
             for(typename GM::ConstFactorIterator it=gm_.factorsOfVariableBegin(node); it !=gm_.factorsOfVariableEnd(node); ++it){
               const typename GM::FactorType& factor = gm_[(*it)];
               if(factor.numberOfVariables()==2 ){
                 if(factor.variableIndex(1)==node && nodeOrder_[factor.variableIndex(0)] > nodeOrder_[node] ){
                   arg[factor.variableIndex(0)] = stateBuffers_[node][childrenCounter*gm_.numberOfLabels(node)+arg[node]];
                   nodeList.push_back(factor.variableIndex(0));
                   ++childrenCounter;
                 }
                 if(factor.variableIndex(0)==node && nodeOrder_[factor.variableIndex(1)] > nodeOrder_[node] ){
                   arg[factor.variableIndex(1)] = stateBuffers_[node][childrenCounter*gm_.numberOfLabels(node)+arg[node]];
                   nodeList.push_back(factor.variableIndex(1));
                   ++childrenCounter;
                 }
               }
             }
           }
         }
         return NORMAL;
       } else {
          arg.assign(gm_.numberOfVariables(), 0);
          return UNKNOWN;
       }
    }
  }

  template<class GM, class ACC>
  inline void DynamicProgramming<GM, ACC>::getNodeInfo(const IndexType Inode, std::vector<ValueType>& values, std::vector<std::vector<LabelType> >& substates, std::vector<IndexType>& nodes) const{
    values.clear();
    substates.clear();
    nodes.clear();
    values.resize(gm_.numberOfLabels(Inode));
    substates.resize(gm_.numberOfLabels(Inode));
    std::vector<LabelType> arg;
    bool firstround = true;
    std::vector<size_t> nodeList;
    for(IndexType i=0;i<gm_.numberOfLabels(Inode); ++i){
      arg.assign(gm_.numberOfVariables(), std::numeric_limits<LabelType>::max() );
      arg[Inode]=i;
      values[i]=valueBuffers_[Inode][i];
      nodeList.push_back(Inode);
      if(i!=0){
        firstround=false;
      }
      
      while(nodeList.size()>0){
        size_t node = nodeList.back();
        size_t childrenCounter = 0;
        nodeList.pop_back();
        for(typename GM::ConstFactorIterator it=gm_.factorsOfVariableBegin(node); it !=gm_.factorsOfVariableEnd(node); ++it){
          const typename GM::FactorType& factor = gm_[(*it)];
          if(factor.numberOfVariables()==2 ){
            if(factor.variableIndex(1)==node && nodeOrder_[factor.variableIndex(0)] > nodeOrder_[node] ){
              arg[factor.variableIndex(0)] = stateBuffers_[node][childrenCounter*gm_.numberOfLabels(node)+arg[node]];
              substates[i].push_back(stateBuffers_[node][childrenCounter*gm_.numberOfLabels(node)+arg[node]]);
              if(firstround==true){              
                nodes.push_back(factor.variableIndex(0));
              }
              nodeList.push_back(factor.variableIndex(0));
              ++childrenCounter;             
            }
            if(factor.variableIndex(0)==node && nodeOrder_[factor.variableIndex(1)] > nodeOrder_[node] ){
              arg[factor.variableIndex(1)] = stateBuffers_[node][childrenCounter*gm_.numberOfLabels(node)+arg[node]];
              substates[i].push_back(stateBuffers_[node][childrenCounter*gm_.numberOfLabels(node)+arg[node]]);
              if(firstround==true){
                nodes.push_back(factor.variableIndex(1));
              }
              nodeList.push_back(factor.variableIndex(1)); 
              ++childrenCounter;                                 
            }
          }
        }
      }
    }
  }
  
  
} // namespace opengm

#endif // #ifndef OPENGM_DYNAMICPROGRAMMING_HXX

