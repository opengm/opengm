#ifndef OPENGM_CGC_VISITORS
#define OPENGM_CGC_VISITORS

#include <opengm/opengm.hxx>

template<class CGC>
class CgcStateVisitor{
public:
   typedef CGC InfType;
   typedef typename InfType::AccumulationType AccumulationType;
   typedef typename InfType::GraphicalModelType GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;

   CgcStateVisitor(
      const GraphicalModelType & gm,
      std::list< std::vector<typename GraphicalModelType::LabelType> >& l,
      const size_t visitNth=10,const size_t skipN=10
   )
      :  gm_(gm),
         visitNth_(visitNth),
         visitNr_(0),
         skipN_(skipN),
         l_(l)
   {
       OPENGM_CHECK_OP(visitNth,>=,1," ");

   }

   void begin(InfType & inf ,const ValueType val,const ValueType bound){

   }
   void end(InfType & inf ,const ValueType val,const ValueType bound){
      
   }

   void operator()(InfType & inf ,const ValueType val,const ValueType bound){
      const bool inRecursive2Coloring = inf.inRecursive2Coloring();
      const bool inGreedy2Coloring    = inf.inRecursive2Coloring();
      if(visitNr_>=skipN_ && ( (visitNr_-skipN_)==0 ||  (visitNr_-skipN_) % visitNth_==0) ){
         // get arg
         inf.arg(argBuffer_);
         l_.push_back(argBuffer_);
      }
      else {
          throw std::runtime_error("no!");
      }
      ++visitNr_;
   }
   

private:
   GraphicalModelType gm_;
   size_t visitNth_;
   size_t visitNr_;
   size_t skipN_;

   std::vector<LabelType> argBuffer_;
   std::list< std::vector<LabelType> >& l_;
};


#endif // OPENGM_CGC_VISITORS