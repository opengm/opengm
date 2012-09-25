#ifndef OPENGM_VERBOSE_VISITOR_HXX
#define OPENGM_VERBOSE_VISITOR_HXX

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>

#include <opengm/opengm.hxx>
#include <opengm/utilities/timer.hxx>
#include <opengm/inference/visitors/visitorspecializations.hxx>

namespace opengm {

   /// \cond HIDDEN_SYMBOLS
   namespace detail_visitor{
     
   struct VerboseVisitor{};
   struct TimingVisitor{};
   struct EmptyVisitor{};
   
   template<size_t STATE,size_t FORMAT>
   class Print{
      static const size_t stepSpace_=8;
      static const size_t valueSpace_=9;
      static const size_t boundSpace_=9;
      static const size_t extraNumber1Space_=9;
      static const size_t extraNumber2Space_=9;
   private:
      typedef opengm::meta::And<
         opengm::meta::EqualNumber<FORMAT,0>::value,
         opengm::meta::EqualNumber<STATE,1>::value 
      > SameLineAtBegin; 
      typedef opengm::meta::Or<
         opengm::meta::EqualNumber<FORMAT,1>::value  ,
         opengm::meta::Not<
            opengm::meta::EqualNumber<STATE,1>::value
         >::value
      > NewLineAtEnd;
      typedef opengm::meta::And<
         opengm::meta::EqualNumber<FORMAT,0>::value,
         opengm::meta::EqualNumber<STATE,2>::value 
      > NewLineAtBeginEnd;
   public:
      template<class S,class V,class B>
         static void print(const S ,const V ,const B );
      template<class S,class V,class B,class E1>
         static void print(const S ,const V ,const B ,const std::string & ,const E1 );
      template<class S,class V,class B,class E1,class E2>
         static void print(const S ,const V ,const B ,const std::string & ,const E1 ,const std::string & ,const E2 );
      
   };
   
   template<bool MULTI_LINE>
   class PrintFormated{
   public:
      template<class S,class V,class B>
         static void printAtBegin(const S ,const V ,const B bound);
      template<class S,class V,class B>
         static void printAtVisit(const S ,const V ,const B bound);
      template<class S,class V,class B>
         static void printAtEnd(const S ,const V ,const B bound);

      template<class S,class V,class B,class E1>
         static void printAtBegin(const S ,const V ,const B ,const std::string & ,const E1 );
      template<class S,class V,class B,class E1>
         static void printAtVisit(const S ,const V ,const B ,const std::string & ,const E1 );
      template<class S,class V,class B,class E1>
         static void printAtEnd(const S ,const V ,const B ,const std::string & ,const E1 );
      
      template<class S,class V,class B,class E1,class E2>
         static void printAtBegin(const S ,const V ,const B ,const std::string & ,const E1 ,const std::string & ,const E2 );
      template<class S,class V,class B,class E1,class E2>
         static void printAtVisit(const S ,const V ,const B ,const std::string & ,const E1 ,const std::string & ,const E2 );
      template<class S,class V,class B,class E1,class E2>
         static void printAtEnd(const S ,const V ,const B ,const std::string & ,const E1 ,const std::string & ,const E2 );
   };
      
   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   struct VisitorImplementation;
   /// empty visitors
   template<class INFERENCE_TYPE>
   class VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>{
   public:
      VisitorImplementation();
      
      void visit(const INFERENCE_TYPE &)const;
      void beginVisit(const INFERENCE_TYPE &)const;
      void endVisit(const INFERENCE_TYPE &)const;
      
      template<class E1>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1)const;
      template<class E1>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1)const;
      template<class E1>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1)const;
      
      template<class E1,class E2>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2)const;
      template<class E1,class E2>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2)const;
      template<class E1,class E2>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2)const;
      
      template<class E1,class E2,class E3>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3)const;
      template<class E1,class E2,class E3>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3)const;
      template<class E1,class E2,class E3>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3)const;
      
      
      
      template<class T1,class T2>
         void visit(T1 a1 ,T2 a2) const;
      template<class T1,class T2,class E1>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1) const;
      template<class T1,class T2,class E1,class E2>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) const;
      template<class T1,class T2,class E1,class E2,class E3>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) const;
      
      
      template<class T1,class T2>
         void beginVisit(T1 a1 ,T2 a2) const;
      template<class T1,class T2,class E1>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1) const;
      template<class T1,class T2,class E1,class E2>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) const;
      template<class T1,class T2,class E1,class E2,class E3>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) const;
      
      
      template<class T1,class T2>
         void endVisit(T1 a1 ,T2 a2) const;
      template<class T1,class T2,class E1>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1) const;
      template<class T1,class T2,class E1,class E2>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) const;
      template<class T1,class T2,class E1,class E2,class E3>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) const;
   };
   /// verbose visitors
   template<class INFERENCE_TYPE>
   class VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>{
   public:
      VisitorImplementation(const size_t = 1,bool = true);
      void assign(const size_t = 1,bool = true);
      
      void visit(const INFERENCE_TYPE &);
      void beginVisit(const INFERENCE_TYPE &);
      void endVisit(const INFERENCE_TYPE &);
      
      template<class E1>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1);
      template<class E1>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1);
      template<class E1>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1);
      
      template<class E1,class E2>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2);
      template<class E1,class E2>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2);
      template<class E1,class E2>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2);
      
      template<class E1,class E2,class E3>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3);
      template<class E1,class E2,class E3>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3);
      template<class E1,class E2,class E3>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3);
      
      template<class T1,class T2>
         void visit(T1 a1 ,T2 a2) ;
      template<class T1,class T2,class E1>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1) ;
      template<class T1,class T2,class E1,class E2>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) ;
      template<class T1,class T2,class E1,class E2,class E3>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) ;
      
      template<class T1,class T2>
         void beginVisit(T1 a1 ,T2 a2) ;
      template<class T1,class T2,class E1>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1) ;
      template<class T1,class T2,class E1,class E2>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) ;
      template<class T1,class T2,class E1,class E2,class E3>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) ;
      
      template<class T1,class T2>
         void endVisit(T1 a1 ,T2 a2) ;
      template<class T1,class T2,class E1>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1) ;
      template<class T1,class T2,class E1,class E2>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) ;
      template<class T1,class T2,class E1,class E2,class E3>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) ;
      
   private:
      size_t visitNth_;
      size_t visitNumber_;
      bool multiline_;
   };
   /// verbose timing visitors
   template<class INFERENCE_TYPE>
   class VisitorImplementation<INFERENCE_TYPE,TimingVisitor>{
   public:
      typedef typename INFERENCE_TYPE::ValueType ValueType;
      typedef ValueType BoundType;
      typedef opengm::DefaultTimingType TimeType;
      typedef size_t IterationType;
      
      VisitorImplementation(const size_t visitNth=1,const size_t = 0,bool = false,bool = false);
      void assign(const size_t visitNth=1,const size_t = 0,bool = false,bool = false);
      
      const std::vector<TimeType > &      getTimes() const;
      const std::vector<ValueType > &     getValues() const;
      const std::vector<BoundType > &     getBounds() const;
      const std::vector<IterationType> &  getIterations() const;
      
      void visit(const INFERENCE_TYPE &);
      void beginVisit(const INFERENCE_TYPE &);
      void endVisit(const INFERENCE_TYPE &);
      
      template<class E1>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1);
      template<class E1>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1);
      template<class E1>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1);
      
      template<class E1,class E2>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2);
      template<class E1,class E2>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2);
      template<class E1,class E2>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2);
      
      template<class E1,class E2,class E3>
      void visit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3);
      template<class E1,class E2,class E3>
      void beginVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3);
      template<class E1,class E2,class E3>
      void endVisit(const INFERENCE_TYPE &,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3);
      
      template<class T1,class T2>
         void visit(T1 a1 ,T2 a2) ;
      template<class T1,class T2,class E1>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1) ;
      template<class T1,class T2,class E1,class E2>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) ;
      template<class T1,class T2,class E1,class E2,class E3>
         void visit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) ;
      
      template<class T1,class T2>
         void beginVisit(T1 a1 ,T2 a2) ;
      template<class T1,class T2,class E1>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1) ;
      template<class T1,class T2,class E1,class E2>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) ;
      template<class T1,class T2,class E1,class E2,class E3>
         void beginVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) ;
      
      template<class T1,class T2>
         void endVisit(T1 a1 ,T2 a2) ;
      template<class T1,class T2,class E1>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1) ;
      template<class T1,class T2,class E1,class E2>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2) ;
      template<class T1,class T2,class E1,class E2,class E3>
         void endVisit(T1 a1 ,T2 a2,const std::string &,const E1,const std::string &,const E2,const std::string &,const E3) ;
   
   public:
      typedef std::map<std::string, std::vector<double> >  LogMapType;
      // call only once in "beginVisit"
      void reserveMapVector(const std::string & name) {
         logs_[name].reserve(times_.capacity());
      }
      void pushBackStringLogData(const std::string & name,const double data) {
         logs_[name].push_back(data);
      }
      const LogMapType & getLogsMap()const{
         return logs_;
      }

   private:
      LogMapType logs_;
      size_t visitNth_;
      size_t visitNumber_;
      std::vector<TimeType > times_;
      std::vector<ValueType > values_;
      std::vector<BoundType > bounds_;
      
      std::vector<IterationType> iterations_;
      opengm::Timer timer_;
      bool verbose_;
      bool multiline_;
   };

   /// implementation of the generic visitor interface
   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   class Visitor
   :  public VisitorImplementation<INFERENCE_TYPE,VISITOR_TYPE>{
      typedef INFERENCE_TYPE InferenceType;
      typedef typename InferenceType::AccumulationType AccumulationType;
      typedef typename InferenceType::GraphicalModelType GraphicalModelType;
   public:
      Visitor();
      
      template<class T1,class T2,class E1>
      void operator() (const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1) {
         this->visit(inference,t1,t2,n1,e1);
      }
      template<class T1,class T2,class E1,class E2>
      void operator() (const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2) {
         this->visit(inference,t1,t2,n1,e1,n2,e2);
      }
      template<class T1,class T2,class E1,class E2,class E3>
      void operator() (const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2,const std::string & n3 ,const E2 e3) {
         this->visit(inference,t1,t2,n1,e1,n2,e2,n3,e3);
      }
      
      template<class E1>
      void operator() (const  INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1) {
         this->visit(inference,n1,e1);
      }
      template<class E1,class E2>
      void operator() (const INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2) {
         this->visit(inference,n1,e1,n2,e2);
      }
      template<class E1,class E2,class E3>
      void operator() (const  INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2,const std::string & n3 ,const E2 e3) {
         this->visit(inference,n1,e1,n2,e2,n3,e3);
      }

      
      void operator()(); 
      void operator() (const  INFERENCE_TYPE & inference );
      template<class T1>
         void operator()(const  INFERENCE_TYPE & ,const  T1); 
      template<class T1,class T2>
         void operator()(const  INFERENCE_TYPE & ,const  T1,const  T2 ); 
      template<class T1,class T2,class T3>
         void operator()(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 & );        
      template<class T1,class T2,class T3,class T4>
         void operator()(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 & );        
      template<class T1,class T2,class T3,class T4,class T5>
         void operator()(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &);
      template<class T1,class T2,class T3,class T4,class T5,class T6>
         void operator()(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &);
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7>
         void operator()(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 & );
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8>
         void operator()(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 &,const  T8 & );
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9>
         void operator()(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 &,const  T8 & ,const T9 &);

      void begin(); 
      
      
      
      template<class T1,class T2,class E1>
      void begin (const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1) {
         this->beginVisit(inference,t1,t2,n1,e1);
      }
      template<class T1,class T2,class E1,class E2>
      void begin(const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2) {
         this->beginVisit(inference,t1,t2,n1,e1,n2,e2);
      }
      template<class T1,class T2,class E1,class E2,class E3>
      void begin(const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2,const std::string & n3 ,const E2 e3) {
         this->beginVisit(inference,t1,t2,n1,e1,n2,e2,n3,e3);
      }
      
      template<class E1>
      void begin(const  INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1) {
         this->beginVisit(inference,n1,e1);
      }
      template<class E1,class E2>
      void begin(const INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2) {
         this->beginVisit(inference,n1,e1,n2,e2);
      }
      template<class E1,class E2,class E3>
      void begin(const  INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2,const std::string & n3 ,const E2 e3) {
         this->beginVisit(inference,n1,e1,n2,e2,n3,e3);
      }
      
      
      void begin (const  INFERENCE_TYPE & inference );
      template<class T1>
         void begin(const  INFERENCE_TYPE & ,const  T1); 
      template<class T1,class T2>
         void begin(const  INFERENCE_TYPE & ,const  T1,const  T2 ); 
      template<class T1,class T2,class T3>
         void begin(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &);        
      template<class T1,class T2,class T3,class T4>
         void begin(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 & );        
      template<class T1,class T2,class T3,class T4,class T5>
         void begin(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &);
      template<class T1,class T2,class T3,class T4,class T5,class T6>
         void begin(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &);
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7>
         void begin(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 & );
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8>
         void begin(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 &,const  T8 & );
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9>
         void begin(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 &,const  T8 & ,const T9 &);

      void end(); 
      
      template<class T1,class T2,class E1>
      void end (const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1) {
         this->endVisit(inference,t1,t2,n1,e1);
      }
      template<class T1,class T2,class E1,class E2>
      void end(const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2) {
         this->endVisit(inference,t1,t2,n1,e1,n2,e2);
      }
      template<class T1,class T2,class E1,class E2,class E3>
      void end(const  INFERENCE_TYPE & inference ,const T1 t1,const T2 t2,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2,const std::string & n3 ,const E2 e3) {
         this->endVisit(inference,t1,t2,n1,e1,n2,e2,n3,e3);
      }
      
      template<class E1>
      void end(const  INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1) {
         this->endVisit(inference,n1,e1);
      }
      template<class E1,class E2>
      void end(const INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2) {
         this->endVisit(inference,n1,e1,n2,e2);
      }
      template<class E1,class E2,class E3>
      void end(const  INFERENCE_TYPE & inference ,const std::string & n1 ,const E1 e1,const std::string & n2 ,const E2 e2,const std::string & n3 ,const E2 e3) {
         this->endVisit(inference,n1,e1,n2,e2,n3,e3);
      }
      
      void end (const  INFERENCE_TYPE & inference );
      template<class T1>
         void end(const  INFERENCE_TYPE & ,const  T1); 
      template<class T1,class T2>
         void end(const  INFERENCE_TYPE & ,const  T1,const  T2 ); 
      template<class T1,class T2,class T3>
         void end(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &);        
      template<class T1,class T2,class T3,class T4>
         void end(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 & );        
      template<class T1,class T2,class T3,class T4,class T5>
         void end(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &);
      template<class T1,class T2,class T3,class T4,class T5,class T6>
         void end(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &);
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7>
         void end(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 & );
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8>
         void end(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 &,const  T8 & );
      template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9>
         void end(const  INFERENCE_TYPE & ,const  T1,const  T2 ,const T3 &,const T4 &,const T5 &,const T6 &,const T7 &,const  T8 & ,const T9 &); 
   };
   }// end namespace detail_visitor
   /// \endcond
   
   /// base class for empty visitor
   template<class INFERENCE_TYPE>
   class EmptyVisitorBase
   : public detail_visitor::Visitor<INFERENCE_TYPE,detail_visitor::EmptyVisitor>{
   public:
      EmptyVisitorBase();
   };
   
   /// base class for verbose visitor
   template<class INFERENCE_TYPE>
   class VerboseVisitorBase
   : public detail_visitor::Visitor<INFERENCE_TYPE,detail_visitor::VerboseVisitor>{
   public:
      VerboseVisitorBase(const size_t =1,const bool=true);
   };
   
   /// base class for timing visitor
   template<class INFERENCE_TYPE>
   class TimingVisitorBase
   :  public detail_visitor::Visitor<INFERENCE_TYPE,detail_visitor::TimingVisitor>{
   public:
      TimingVisitorBase(const size_t = 1,size_t = 0,bool = false,bool = true);
   };
   
   /// default empty visitor
   template<class INFERENCE_TYPE>
   class EmptyVisitor
   : public EmptyVisitorBase<INFERENCE_TYPE>{
   public:
      EmptyVisitor();
   };
   
   /// default verbose visitor
   template<class INFERENCE_TYPE>
   class VerboseVisitor
   : public VerboseVisitorBase<INFERENCE_TYPE>{
   public:
      VerboseVisitor(const size_t=1,bool=true);
   };
   
   /// default timing visitor
   template<class INFERENCE_TYPE>
   class TimingVisitor
   :  public TimingVisitorBase<INFERENCE_TYPE>{
   public:
      TimingVisitor(const size_t = 1,size_t = 0,bool = false,bool = true);
   };
   
   // constructor
   template<class INFERENCE_TYPE>
   EmptyVisitorBase<INFERENCE_TYPE>::EmptyVisitorBase() 
   :  detail_visitor::Visitor<INFERENCE_TYPE,detail_visitor::EmptyVisitor>() {
   }
   
   /// constructor
   /// \param visitNth print each nth visit
   /// \param do multi-line or single-line cout's
   template<class INFERENCE_TYPE>
   VerboseVisitorBase<INFERENCE_TYPE>::VerboseVisitorBase
   (
      const size_t visitNth,
      bool multilineCout
   ) 
   :  detail_visitor::Visitor<INFERENCE_TYPE,detail_visitor::VerboseVisitor>() {
      this->assign(visitNth,multilineCout);
   }
   
   /// constructor
   /// \param visitNth print each nth visit
   /// \param reserve reserve memory for n visits
   /// \param verbose do verbose visiting ?
   /// \param do multi-line or single-line cout's
   template<class INFERENCE_TYPE>
   TimingVisitorBase<INFERENCE_TYPE>::TimingVisitorBase
   (
      const size_t visitNth,
      size_t reserve,
      bool verbose,
      bool multilineCout
   ) 
   :  detail_visitor::Visitor<INFERENCE_TYPE,detail_visitor::TimingVisitor>() {
      this->assign(visitNth,reserve,verbose,multilineCout);
   }
   
   /// constructor
   template<class INFERENCE_TYPE>
   EmptyVisitor<INFERENCE_TYPE>::EmptyVisitor() 
   :  EmptyVisitorBase<INFERENCE_TYPE> () {
   }
   
   /// constructor
   /// \param visitNth print each nth visit
   /// \param do multi-line or single-line cout's
   template<class INFERENCE_TYPE>
   VerboseVisitor<INFERENCE_TYPE>::VerboseVisitor
   (
      const size_t visitNth,
      bool multilineCout
   ) 
   :  VerboseVisitorBase<INFERENCE_TYPE>(visitNth,multilineCout) {
   }
   
   /// constructor
   /// \param visitNth print each nth visit
   /// \param reserve reserve memory for n visits
   /// \param verbose do verbose visiting ?
   /// \param do multi-line or single-line cout's
   template<class INFERENCE_TYPE>
   TimingVisitor<INFERENCE_TYPE>::TimingVisitor
   (
      const size_t visitNth,
      size_t reserve,
      bool verbose,
      bool multilineCout
   ) 
   :  TimingVisitorBase<INFERENCE_TYPE>(visitNth,reserve,verbose,multilineCout) {
   }
   
  
   // implementation of the classes in detail_visitor
   /// \cond HIDDEN_SYMBOLS
   namespace detail_visitor{ 
         
   template<size_t STATE,size_t FORMAT>   
   template<class S,class V,class B>
   inline void 
   Print<STATE,FORMAT>::print
   (
      const S step,
      const V value,
      const B bound
   ) {
      std::cout<<((NewLineAtBeginEnd::value )? "\n" : "")
      <<((SameLineAtBegin::value )? "\r" : "")
      << (opengm::meta::EqualNumber<STATE,0>::value  ? "Begin : " : (opengm::meta::EqualNumber<STATE,1>::value  ? "Step  : " : "End   : "))
      << std::scientific<<std::setw(Print::stepSpace_)
      << (opengm::meta::EqualNumber<STATE,1>::value ? step : step)
      << "  Value : "  <<std::scientific<<std::setw(Print::valueSpace_)<<value
      << "  Bound : "  <<std::scientific<<std::setw(Print::boundSpace_)<<bound
      << ((NewLineAtEnd::value )? "\n":"")<<std::flush;
   }
   
   template<size_t STATE,size_t FORMAT>
   template<class S,class V,class B,class E1>
   inline void 
   Print<STATE,FORMAT>::print
   (
      const S step,
      const V value,
      const B bound,
      const std::string & extraName1,
      const E1 extra1
   ) {
      std::cout<<((NewLineAtBeginEnd::value )? "\n" : "")
      <<((SameLineAtBegin::value )? "\r" : "")
      << (opengm::meta::EqualNumber<STATE,0>::value  ? "Begin : " : (opengm::meta::EqualNumber<STATE,1>::value  ? "Step  : " : "End   : "))
      << std::scientific<<std::setw(Print::stepSpace_)
      << (opengm::meta::EqualNumber<STATE,1>::value ? step : step)
      << "  Value : "  <<std::scientific<<std::setw(Print::valueSpace_)<<value
      << "  Bound : "  <<std::scientific<<std::setw(Print::boundSpace_)<<bound
      << "  "<<extraName1<<" : "  <<std::scientific<<std::setw(Print::extraNumber1Space_)<<extra1
      << ((NewLineAtEnd::value )? "\n":"")<<std::flush;
   }
   
   template<size_t STATE,size_t FORMAT>
   template<class S,class V,class B,class E1,class E2>
   inline void 
   Print<STATE,FORMAT>::print
   (
      const S step,
      const V value,
      const B bound,
      const std::string & extraName1,
      const E1 extra1,
      const std::string & extraName2,
      const E2 extra2
   ) {
      std::cout<<((NewLineAtBeginEnd::value )? "\n" : "")
      <<((SameLineAtBegin::value )? "\r" : "")
      << (opengm::meta::EqualNumber<STATE,0>::value  ? "Begin : " : (opengm::meta::EqualNumber<STATE,1>::value  ? "Step  : " : "End   : "))
      << std::scientific<<std::setw(Print::stepSpace_)
      << (opengm::meta::EqualNumber<STATE,1>::value ? step : step)
      << "  Value : "  <<std::scientific<<std::setw(Print::valueSpace_)<<value
      << "  Bound : "  <<std::scientific<<std::setw(Print::boundSpace_)<<bound
      << "  "<<extraName1<<" : "  <<std::scientific<<std::setw(Print::extraNumber1Space_)<<extra1
      << "  "<<extraName2<<" : "  <<std::scientific<<std::setw(Print::extraNumber2Space_)<<extra2
      << ((NewLineAtEnd::value )? "\n":"")<<std::flush;
   }

   template<bool MULTI_LINE>   
   template<class S,class V,class B>
   inline void 
   PrintFormated<MULTI_LINE>::printAtBegin
   (
      const S step,
      const V value,
      const B bound
   ) {
      Print<0,MULTI_LINE>::print(step,value,bound);
   }
   
   template<bool MULTI_LINE>
   template<class S,class V,class B>
   inline void 
   PrintFormated<MULTI_LINE>::printAtVisit
   (
      const S step,
      const V value,
      const B bound
   ) {
      Print<1,MULTI_LINE>::print(step,value,bound);
   }
   
   template<bool MULTI_LINE>
   template<class S,class V,class B>
   inline void 
   PrintFormated<MULTI_LINE>::printAtEnd
   (
      const S step,
      const V value,
      const B bound
   ) {
      Print<2,MULTI_LINE>::print(step,value,bound);
   }
   
   template<bool MULTI_LINE>
   template<class S,class V,class B,class E1>
   inline void 
   PrintFormated<MULTI_LINE>::printAtBegin
   (
      const S step,
      const V value,
      const B bound,
      const std::string & extraName1,
      const E1 extra1
   ) {
      Print<0,MULTI_LINE>::print(step,value,bound,extraName1,extra1);
   }
   
   template<bool MULTI_LINE>
   template<class S,class V,class B,class E1>
   inline void 
   PrintFormated<MULTI_LINE>::printAtVisit
   (
      const S step,
      const V value,
      const B bound,
      const std::string & extraName1,
      const E1 extra1
   ) {
      Print<1,MULTI_LINE>::print(step,value,bound,extraName1,extra1);
   }
   
   template<bool MULTI_LINE>
   template<class S,class V,class B,class E1>
   inline void 
   PrintFormated<MULTI_LINE>::printAtEnd
   (
      const S step,
      const V value,
      const B bound,
      const std::string & extraName1,
      const E1 extra1
   ) {
      Print<2,MULTI_LINE>::print(step,value,bound,extraName1,extra1);
   }
   
   template<bool MULTI_LINE>
   template<class S,class V,class B,class E1,class E2>
   inline void 
   PrintFormated<MULTI_LINE>::printAtBegin
   (
      const S step,
      const V value,
      const B bound,
      const std::string & extraName1,
      const E1 extra1,
      const std::string & extraName2,
      const E2 extra2
   ) {
      Print<0,MULTI_LINE>::print(step,value,bound,extraName1,extra1,extraName2,extra2);
   }
   
   template<bool MULTI_LINE>
   template<class S,class V,class B,class E1,class E2>
   inline void 
   PrintFormated<MULTI_LINE>::printAtVisit
   (
      const S step,
      const V value,
      const B bound,
      const std::string & extraName1,
      const E1 extra1,
      const std::string & extraName2,
      const E2 extra2
   ) {
      Print<1,MULTI_LINE>::print(step,value,bound,extraName1,extra1,extraName2,extra2);
   }
   
   template<bool MULTI_LINE>
   template<class S,class V,class B,class E1,class E2>
   inline void 
   PrintFormated<MULTI_LINE>::printAtEnd
   (
      const S step,
      const V value,
      const B bound,
      const std::string & extraName1,
      const E1 extra1,
      const std::string & extraName2,
      const E2 extra2
   ) {
      Print<2,MULTI_LINE>::print(step,value,bound,extraName1,extra1,extraName2,extra2);
   }
   
   // implementation empty visitor
   template<class INFERENCE_TYPE>
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::VisitorImplementation() {
   }
   
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::visit 
   (
      const INFERENCE_TYPE & inf
   )const{ 
   }
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::beginVisit 
   (
      const INFERENCE_TYPE & inf
   )const{ 
   }
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::endVisit 
   (
      const INFERENCE_TYPE & inf
   )const{ 
   }
      
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1)const{
      
   }
   
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1)const{
      
   }
   
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1)const{
      
   }
   
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2)const{
      
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2)const{
      
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2)const{
      
   }
   
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3)const{
      
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3)const{
      
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3)const{
      
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::visit 
   (
      T1 a1 ,
      T2 a2
   )const{ 
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::visit 
   (
      T1 a1 ,
      T2 a2 ,
      const std::string & name1,
      const E1 e1
   )const{ 
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::visit 
   (
      T1 a1 ,
      T2 a2 ,
      const std::string & name1,
      const E1 e1,
      const std::string & name21,
      const E2 e2
   )const{ 
   }

   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::beginVisit
   (
      T1 a1 ,
      T2 a2
   )const{ 
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::beginVisit 
   (
      T1 a1 ,
      T2 a2 ,
      const std::string & name1,
      const E1 e1
   )const{ 
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::beginVisit 
   (
      T1 a1 ,
      T2 a2 ,
      const std::string & name1,
      const E1 e1,
      const std::string & name21,
      const E2 e2
   )const{ 
   }

   
   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::endVisit    
   (
      T1 a1 ,
      T2 a2
   )const{ 
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::endVisit 
   (
      T1 a1 ,
      T2 a2 ,
      const std::string & name1,
      const E1 e1
   )const{ 
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,EmptyVisitor>::endVisit 
   (
      T1 a1 ,
      T2 a2 ,
      const std::string & name1,
      const E1 e1,
      const std::string & name2,
      const E2 e2
   )const{ 
   }
   
   // implementation verbose visitor   
   template<class INFERENCE_TYPE>
   inline 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::VisitorImplementation
   (
      const size_t visitNth,
      bool multiline
   )
   :  visitNth_(visitNth),
      visitNumber_(0),
      multiline_(multiline) {
   }
   
   template<class INFERENCE_TYPE>
   inline void
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::assign
   (
      const size_t visitNth,
      bool multiline
   ) {  
      visitNth_=visitNth;
      visitNumber_=0;
      multiline_=multiline;
   }
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::visit 
   (
      const INFERENCE_TYPE & inf
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtVisit(visitNumber_,inf.value(),inf.bound());
         else
            PrintFormated<false>::printAtVisit(visitNumber_, inf.value(), inf.bound());
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::beginVisit 
   (
      const INFERENCE_TYPE & inf
   ) { 
      if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,inf.value(),inf.bound());
      else
         PrintFormated<false>::printAtBegin(visitNumber_, inf.value(), inf.bound());
   }
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::endVisit 
   (
      const INFERENCE_TYPE & inf
   ) {
      if(multiline_)
            PrintFormated<true>::printAtEnd(visitNumber_,inf.value(),inf.bound());
      else
         PrintFormated<false>::printAtEnd(visitNumber_, inf.value(), inf.bound());
   }
   
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtVisit(visitNumber_,inf.value(),inf.bound(),n1,e1);
         else
            PrintFormated<false>::printAtVisit(visitNumber_, inf.value(), inf.bound(),n1,e1);
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1) {
      if(multiline_)
         PrintFormated<true>::printAtBegin(visitNumber_,inf.value(),inf.bound(),n1,e1);
      else
         PrintFormated<false>::printAtBegin(visitNumber_, inf.value(), inf.bound(),n1,e1);
   }
   
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1) {
      if(multiline_)
         PrintFormated<true>::printAtEnd(visitNumber_,inf.value(),inf.bound(),n1,e1);
      else
         PrintFormated<false>::printAtEnd(visitNumber_, inf.value(), inf.bound(),n1,e1);
   }
   
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtVisit(visitNumber_,inf.value(),inf.bound(),n1,e1,n2,e2);
         else
            PrintFormated<false>::printAtVisit(visitNumber_, inf.value(), inf.bound(),n1,e1.n2,e2);
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2) {
      if(multiline_)
         PrintFormated<true>::printAtBegin(visitNumber_,inf.value(),inf.bound(),n1,e1,n2,e2);
      else
         PrintFormated<false>::printAtBegin(visitNumber_, inf.value(), inf.bound(),n1,e1,n2,e2);
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2) {
      if(multiline_)
         PrintFormated<true>::printAtEnd(visitNumber_,inf.value(),inf.bound(),n1,e1,n2,e2);
      else
         PrintFormated<false>::printAtEnd(visitNumber_, inf.value(), inf.bound(),n1,e1,n2,e2);
   }
   
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtVisit(visitNumber_,inf.value(),inf.bound(),n1,e1,n2,e2,n3,e3);
         else
            PrintFormated<false>::printAtVisit(visitNumber_, inf.value(), inf.bound(),n1,e1.n2,e2,n3,e3);
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3) {
      if(multiline_)
         PrintFormated<true>::printAtBegin(visitNumber_,inf.value(),inf.bound(),n1,e1,n2,e2,n3,e3);
      else
         PrintFormated<false>::printAtBegin(visitNumber_, inf.value(), inf.bound(),n1,e1,n2,e2,n3,e3);
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3) {
      if(multiline_)
         PrintFormated<true>::printAtEnd(visitNumber_,inf.value(),inf.bound(),n1,e1,n2,e2,n3,e3);
      else
         PrintFormated<false>::printAtEnd(visitNumber_, inf.value(), inf.bound(),n1,e1,n2,e2.n3,e3);
   }
   
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::visit
   (
      T1 value ,
      T2 bound
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtVisit(visitNumber_,value,bound);
         else
            PrintFormated<false>::printAtVisit(visitNumber_, value, bound);
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::visit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtVisit(visitNumber_,value,bound,name1,e1);
         else
            PrintFormated<false>::printAtVisit(visitNumber_, value, bound,name1,e1);
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::visit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1,
      const std::string & name2,
      const E2 e2
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtVisit(visitNumber_,value,bound,name1,e1,name2,e2);
         else
            PrintFormated<false>::printAtVisit(visitNumber_, value, bound,name1,e1,name2,e2);
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::beginVisit
   (
      T1 value ,
      T2 bound
   ) { 
      if(multiline_)
         PrintFormated<true>::printAtBegin(visitNumber_,value,bound);
      else
         PrintFormated<false>::printAtBegin(visitNumber_, value, bound);
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::beginVisit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound,name1,e1);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound,name1,e1);
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::beginVisit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1,
      const std::string & name2,
      const E2 e2
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound,name1,e1,name2,e2);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound,name1,e1,name2,e2);
      }
      ++visitNumber_;
   }

   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::endVisit
   (
      T1 value ,
      T2 bound
   ) {
      if(multiline_)
         PrintFormated<true>::printAtEnd(visitNumber_,value,bound);
      else
         PrintFormated<false>::printAtEnd(visitNumber_, value, bound);
   }
   
      template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::endVisit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1
   ) {
      if(multiline_)
         PrintFormated<true>::printAtEnd(visitNumber_,value,bound,name1,e1);
      else
         PrintFormated<false>::printAtEnd(visitNumber_, value, bound,name1,e1);
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,VerboseVisitor>::endVisit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1,
      const std::string & name2,
      const E2 e2
   ) {
      if(multiline_)
         PrintFormated<true>::printAtEnd(visitNumber_,value,bound,name1,e1,name2,e2);
      else
         PrintFormated<false>::printAtEnd(visitNumber_, value, bound,name1,e1,name2,e2);
   }
   
   //implementation of the timing visitor
   template<class INFERENCE_TYPE>
   inline const std::vector<typename VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::TimeType > & 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::getTimes() const{
      return times_;
   }
   
   template<class INFERENCE_TYPE>
   inline const std::vector<typename VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::ValueType > & 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::getValues() const{
      return values_;
   }
   
   template<class INFERENCE_TYPE>
   inline const std::vector<typename VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::BoundType > & 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::getBounds() const{
      return bounds_;
   }
   
   template<class INFERENCE_TYPE>
   inline const std::vector<typename VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::IterationType> & 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::getIterations() const{
      return iterations_;
   }

   template<class INFERENCE_TYPE>
   inline
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::VisitorImplementation
   (
      const size_t visitNth,
      const size_t reserve,
      bool verbose,
      bool multiline
   )
   :  visitNth_(visitNth),
      visitNumber_(0),
      verbose_(verbose),
      multiline_(multiline) {
      if(reserve!=0) {
         times_.reserve(reserve);
         values_.reserve(reserve);
         bounds_.reserve(reserve);
         iterations_.reserve(reserve);
      }
   }
   
   template<class INFERENCE_TYPE>
   inline void
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::assign
   (
      const size_t visitNth,
      const size_t reserve,
      bool verbose,
      bool multiline
   ) {  
      visitNth_=visitNth;
      visitNumber_=0;
      verbose_=verbose;
      multiline_=multiline;
      if(reserve!=0) {
         times_.reserve(reserve);
         values_.reserve(reserve);
         bounds_.reserve(reserve);
         iterations_.reserve(reserve);;
      }
   }
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::visit 
   (
      const INFERENCE_TYPE & inf
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         timer_.toc();
         times_.push_back(timer_.elapsedTime());
         const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
         values_.push_back(value);
         bounds_.push_back(bound);
         iterations_.push_back(visitNumber_);
         timer_.reset();
         if(verbose_) {
            if(multiline_)
               PrintFormated<true>::printAtVisit(visitNumber_,value,bound);
            else
               PrintFormated<false>::printAtVisit(visitNumber_, value, bound);
         }
         timer_.tic();
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::beginVisit 
   (
      const INFERENCE_TYPE & inf
   ) { 
      const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound);
      }
      
      times_.push_back(0.0);
      values_.push_back(value);
      bounds_.push_back(bound);
      iterations_.push_back(IterationType(0));
      timer_.tic();
   }
   
   template<class INFERENCE_TYPE>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::endVisit 
   (
      const INFERENCE_TYPE & inf
   ) {
      timer_.toc();
      times_.push_back(timer_.elapsedTime());
      const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
      values_.push_back(value);
      bounds_.push_back(bound);
      iterations_.push_back(visitNumber_);
      timer_.reset();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtEnd(visitNumber_,value,bound);
         else
            PrintFormated<false>::printAtEnd(visitNumber_, value, bound);
      };
   }
   
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1) {
      if(visitNumber_ % visitNth_ == 0) {
         timer_.toc();
         times_.push_back(timer_.elapsedTime());
         const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
         values_.push_back(value);
         bounds_.push_back(bound);
         this->pushBackStringLogData(n1,e1);
         iterations_.push_back(visitNumber_);
         timer_.reset();
         if(verbose_) {
            if(multiline_)
               PrintFormated<true>::printAtVisit(visitNumber_,value,bound,n1,e1);
            else
               PrintFormated<false>::printAtVisit(visitNumber_, value, bound,n1,e1);
         }
         timer_.tic();
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1) {
      const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound,n1,e1);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound,n1,e1);
      }
      times_.push_back(0);
      values_.push_back(value);
      bounds_.push_back(bound);
      this->reserveMapVector(n1);
      this->pushBackStringLogData(n1,e1);
      iterations_.push_back(IterationType(0)); 
      timer_.tic();
   }
   
   template<class INFERENCE_TYPE>
   template<class E1>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1) {
      timer_.toc();
      times_.push_back(timer_.elapsedTime());
      const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
      values_.push_back(value);
      bounds_.push_back(bound);
      this->pushBackStringLogData(n1,e1);
      iterations_.push_back(visitNumber_);
      timer_.reset();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtEnd(visitNumber_,value,bound,n1,e1);
         else
            PrintFormated<false>::printAtEnd(visitNumber_, value, bound,n1,e1);
      };
   }
   
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2) {
      if(visitNumber_ % visitNth_ == 0) {
         timer_.toc();
         times_.push_back(timer_.elapsedTime());
         const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
         values_.push_back(value);
         bounds_.push_back(bound);
         this->pushBackStringLogData(n1,e1);
         this->pushBackStringLogData(n2,e2);
         iterations_.push_back(visitNumber_);
         timer_.reset();
         if(verbose_) {
            if(multiline_)
               PrintFormated<true>::printAtVisit(visitNumber_,value,bound,n1,e1,n2,e2);
            else
               PrintFormated<false>::printAtVisit(visitNumber_, value, bound,n1,e1,n2,e2);
         }
         timer_.tic();
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2) {
      const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound,n1,e1,n2,e2);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound,n1,e1,n2,e2);
      }
      times_.push_back(0);
      values_.push_back(value);
      bounds_.push_back(bound);
      this->reserveMapVector(n1);
      this->pushBackStringLogData(n1,e1);
      this->reserveMapVector(n2);
      this->pushBackStringLogData(n2,e2);
      iterations_.push_back(IterationType(0));
      timer_.tic();
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2) {
      timer_.toc();;
      times_.push_back(timer_.elapsedTime());
      const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
      values_.push_back(value);
      bounds_.push_back(bound);
      this->pushBackStringLogData(n1,e1);
      this->pushBackStringLogData(n2,e2);
      iterations_.push_back(visitNumber_);
      timer_.reset();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtEnd(visitNumber_,value,bound,n1,e1,n2,e2);
         else
            PrintFormated<false>::printAtEnd(visitNumber_, value, bound,n1,e1,n2,e2);
      };
   }
   
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::visit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3) {
      if(visitNumber_ % visitNth_ == 0) {
         timer_.toc();
         times_.push_back(timer_.elapsedTime());
         const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
         values_.push_back(value);
         bounds_.push_back(bound);
         this->pushBackStringLogData(n1,e1);
         this->pushBackStringLogData(n2,e2);
         this->pushBackStringLogData(n3,e3);
         iterations_.push_back(visitNumber_);
         timer_.reset();
         if(verbose_) {
            if(multiline_)
               PrintFormated<true>::printAtVisit(visitNumber_,value,bound,n1,e1,n2,e2,n3,e3);
            else
               PrintFormated<false>::printAtVisit(visitNumber_, value, bound,n1,e1,n2,e2,n3,e3);
         }
         timer_.tic();
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::beginVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3) {
      const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound,n1,e1,n2,e2,n3,e3);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound,n1,e1,n2,e2,n3,e3);
      }
      times_.push_back(0);
      values_.push_back(value);
      bounds_.push_back(bound);
      this->reserveMapVector(n1);
      this->pushBackStringLogData(n1,e1);
      this->reserveMapVector(n2);
      this->pushBackStringLogData(n2,e2);
      this->reserveMapVector(n3);
      this->pushBackStringLogData(n3, e3);
      iterations_.push_back(IterationType(0));
      timer_.tic();
   }
   
   template<class INFERENCE_TYPE>
   template<class E1,class E2,class E3>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::endVisit(const INFERENCE_TYPE & inf,const std::string & n1,const E1 e1,const std::string & n2,const E2 e2,const std::string & n3,const E3 e3) {
      timer_.toc();;
      times_.push_back(timer_.elapsedTime());
      const typename INFERENCE_TYPE::ValueType value=inf.value(),bound=inf.bound();
      values_.push_back(value);
      bounds_.push_back(bound);
      this->pushBackStringLogData(n1,e1);
      this->pushBackStringLogData(n2,e2);
      this->pushBackStringLogData(n3,e3);
      iterations_.push_back(visitNumber_);
      timer_.reset();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtEnd(visitNumber_,value,bound,n1,e1,n2,e2,n3,e3);
         else
            PrintFormated<false>::printAtEnd(visitNumber_, value, bound,n1,e1,n2,e2,n3,e3);
      };
   }
   
   
   
   
   
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::visit
   (
      T1 value ,
      T2 bound
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         timer_.toc();
         times_.push_back(timer_.elapsedTime());
         values_.push_back(value);
         bounds_.push_back(bound);
         iterations_.push_back(visitNumber_);
         timer_.reset();
         if(verbose_) {
            if(multiline_)
               PrintFormated<true>::printAtVisit(visitNumber_,value,bound);
            else
               PrintFormated<false>::printAtVisit(visitNumber_, value, bound);
         }
         timer_.tic();
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::visit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         timer_.toc();
         times_.push_back(timer_.elapsedTime());
         values_.push_back(value);
         bounds_.push_back(bound);
         this->pushBackStringLogData(name1,e1);
         iterations_.push_back(visitNumber_);
         timer_.reset();
         if(verbose_) {
            if(multiline_)
               PrintFormated<true>::printAtVisit(visitNumber_,value,bound,name1,e1);
            else
               PrintFormated<false>::printAtVisit(visitNumber_, value, bound,name1,e1);
         }
         timer_.tic();
      }
      ++visitNumber_;
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::visit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1,
      const std::string & name2,
      const E2 e2
   ) {
      if(visitNumber_ % visitNth_ == 0) {
         timer_.toc();
         times_.push_back(timer_.elapsedTime());
         values_.push_back(value);
         bounds_.push_back(bound);
         this->pushBackStringLogData(name1,e1);
         this->pushBackStringLogData(name2,e2);
         iterations_.push_back(visitNumber_);
         timer_.reset();
         if(verbose_) {
            if(multiline_)
               PrintFormated<true>::printAtVisit(visitNumber_,value,bound,name1,e1,name2,e2);
            else
               PrintFormated<false>::printAtVisit(visitNumber_, value, bound,name1,e1,name2,e2);
         }
         timer_.tic();
      }
      ++visitNumber_;
   }
      

   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::beginVisit
   (
      T1 value ,
      T2 bound
   ) {
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound);
      }
      
      times_.push_back(0);
      values_.push_back(value);
      bounds_.push_back(bound);
      iterations_.push_back(IterationType(0));
      timer_.tic();
   }
   
      template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::beginVisit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1
   ) {
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound,name1,e1);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound,name1,e1);
      } 
      times_.push_back(0);
      values_.push_back(value);
      bounds_.push_back(bound);
      this->reserveMapVector(name1);
      this->pushBackStringLogData(name1,e1);
      iterations_.push_back(IterationType(0));
      timer_.tic();
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::beginVisit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1,
      const std::string & name2,
      const E2 e2
   ) {
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtBegin(visitNumber_,value,bound,name1,e1);
         else
            PrintFormated<false>::printAtBegin(visitNumber_, value, bound,name1,e1);
      }
      times_.push_back(0);
      values_.push_back(value);
      bounds_.push_back(bound);
      this->reserveMapVector(name1);
      this->pushBackStringLogData(name1,e1);
      this->reserveMapVector(name2);
      this->pushBackStringLogData(name2,e2);
      iterations_.push_back(IterationType(0));
      timer_.tic();
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::endVisit
   (
      T1 value ,
      T2 bound
   ) {
      timer_.toc();
      times_.push_back(timer_.elapsedTime());
      values_.push_back(value);
      bounds_.push_back(bound);
      iterations_.push_back(visitNumber_);
      timer_.reset();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtEnd(visitNumber_,value,bound);
         else
            PrintFormated<false>::printAtEnd(visitNumber_, value, bound);
      };
   }
   
      template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::endVisit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1
   ) {
      timer_.toc();
      times_.push_back(timer_.elapsedTime());
      values_.push_back(value);
      bounds_.push_back(bound);
      this->pushBackStringLogData(name1,e1);
      iterations_.push_back(visitNumber_);
      timer_.reset();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtEnd(visitNumber_,value,bound,name1,e1);
         else
            PrintFormated<false>::printAtEnd(visitNumber_, value, bound,name1,e1);
      };
   }
   
   template<class INFERENCE_TYPE>
   template<class T1,class T2,class E1,class E2>
   inline void 
   VisitorImplementation<INFERENCE_TYPE,TimingVisitor>::endVisit 
   (
      T1 value ,
      T2 bound ,
      const std::string & name1,
      const E1 e1,
      const std::string & name2,
      const E2 e2
   ) {
      timer_.toc();
      times_.push_back(timer_.elapsedTime());
      values_.push_back(value);
      bounds_.push_back(bound);
      this->pushBackStringLogData(name1,e1);
      this->pushBackStringLogData(name2,e2);
      iterations_.push_back(visitNumber_);
      timer_.reset();
      if(verbose_) {
         if(multiline_)
            PrintFormated<true>::printAtEnd(visitNumber_,value,bound,name1,e1,name2,e2);
         else
            PrintFormated<false>::printAtEnd(visitNumber_, value, bound,name1,e1,name2,e2);
      };
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   Visitor<INFERENCE_TYPE,VISITOR_TYPE>::Visitor()
   :  VisitorImplementation<INFERENCE_TYPE,VISITOR_TYPE>() {
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()() {
      const typename INFERENCE_TYPE::ValueType n= INFERENCE_TYPE::AccumulationType:: template neutral<typename INFERENCE_TYPE::ValueType>(); 
      const typename INFERENCE_TYPE::ValueType in= INFERENCE_TYPE::AccumulationType:: template ineutral<typename INFERENCE_TYPE::ValueType>();
      this->visit(n,in);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference
   ) {
      this->visit(inference);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1 value
   ) {
      const typename INFERENCE_TYPE::ValueType in= INFERENCE_TYPE::AccumulationType:: template ineutral<typename INFERENCE_TYPE::ValueType>(); 
      this->visit(value,in);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound
   ) {
      this->visit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3
   ) {
      this->visit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4
   ) {
      this->visit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5
   ) {
      this->visit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6
   ) {
      this->visit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7
   ) {
      this->visit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7,
      const  T8 & a8
   ) {
      this->visit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::operator()
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7,
      const  T8 & a8,
      const  T9 & a9
   ) {
      this->visit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin() {
      const typename INFERENCE_TYPE::ValueType n= INFERENCE_TYPE::AccumulationType:: template neutral<typename INFERENCE_TYPE::ValueType>();
      const typename INFERENCE_TYPE::ValueType in= INFERENCE_TYPE::AccumulationType:: template ineutral<typename INFERENCE_TYPE::ValueType>();
      this->beginVisit(n,in);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference
   ) {
      this->beginVisit(inference);
   }
   
   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value
   ) {
      const typename INFERENCE_TYPE::ValueType in= INFERENCE_TYPE::AccumulationType:: template ineutral<typename INFERENCE_TYPE::ValueType>(); 
      this->begin(value,in);
   }


   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound
   ) {
      this->beginVisit(value,bound);
   }


   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3
   ) {
      this->beginVisit(value,bound);
   }


   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4
   ) {
      this->beginVisit(value,bound);
   }


   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5
   ) {
      this->beginVisit(value,bound);
   }


   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6
   ) {
      this->beginVisit(value,bound);
   }


   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7
   ) {
      this->beginVisit(value,bound);
   }


   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7,
      const  T8 & a8
   ) {
      this->beginVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::begin
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7,
      const  T8 & a8,
      const  T9 & a9
   ) {
      this->beginVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end() {
      const typename INFERENCE_TYPE::ValueType n= INFERENCE_TYPE::AccumulationType:: template neutral<typename INFERENCE_TYPE::ValueType>();
      const typename INFERENCE_TYPE::ValueType in= INFERENCE_TYPE::AccumulationType:: template ineutral<typename INFERENCE_TYPE::ValueType>();
      this->endVisit(n,in);
   }


   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference
   ) {
      
      this->endVisit(inference);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>      
   template<class T1>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1 value
   ) {
      const typename INFERENCE_TYPE::ValueType in= INFERENCE_TYPE::AccumulationType:: template ineutral<typename INFERENCE_TYPE::ValueType>(); 
      this->endVisit(value,in);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound
   ) {
      this->endVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3> 
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3
   ) {
      this->endVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4
   ) {
      this->endVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5
   ) {
      this->endVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6
   ) {
      this->endVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7
   ) {
      this->endVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7,
      const  T8 & a8
   ) {
      this->endVisit(value,bound);
   }

   template<class INFERENCE_TYPE,class VISITOR_TYPE>
   template<class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9>
   inline void Visitor<INFERENCE_TYPE,VISITOR_TYPE>::end
   (
      const  INFERENCE_TYPE & inference,
      const  T1  value,
      const  T2  bound,
      const  T3 & a3,
      const  T4 & a4,
      const  T5 & a5,
      const  T6 & a6,
      const  T7 & a7,
      const  T8 & a8,
      const  T9 & a9
   ) {
      this->endVisit(value,bound);
   }

   } // namespace detail_visitor
   
   /// \endond

} // namespace opengm

#endif
