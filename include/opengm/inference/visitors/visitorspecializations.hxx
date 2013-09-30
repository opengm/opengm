#pragma once
#ifndef OPENGM_VISITORSPECIALIZATIONS_HXX
#define OPENGM_VISITORSPECIALIZATIONS_HXX

/// \cond HIDDEN_SYMBOLS

namespace opengm {

   template<class INF>
      class VerboseVisitor;
   template<class INF>
      class TimingVisitor;
   template<class INF>
      class VerboseVisitorBase;
   template<class INF>
      class TimingVisitorBase;
   template<class GM,class ACC>
      class LazyFlipper;
   //template<class INF>
   //   class SelfFusion;



   template< class GM,class ACC >
   class VerboseVisitor< LazyFlipper<GM,ACC> >
   : public VerboseVisitorBase< LazyFlipper<GM,ACC> > {
   public:
      VerboseVisitor(const size_t visitNth=1, bool multiline=true )
      : VerboseVisitorBase< LazyFlipper<GM,ACC> >(visitNth,multiline) {
      }  
      void operator() (
         const LazyFlipper<GM,ACC> & lf,
         const typename LazyFlipper<GM,ACC>::ValueType value,
         const typename LazyFlipper<GM,ACC>::ValueType bound,
         const size_t subgraphSize,
         const size_t subgraphForrestSize
      ) {
         this->visit(value,bound,"subgraph-size",subgraphSize,"forrest-size",subgraphForrestSize);
      }
      void begin(
         const LazyFlipper<GM,ACC> & lf,
         const typename LazyFlipper<GM,ACC>::ValueType value,
         const typename LazyFlipper<GM,ACC>::ValueType bound,
         const size_t subgraphSize,
         const size_t subgraphForrestSize
      ) {
         this->beginVisit(value,bound,"subgraph-size",subgraphSize,"forrest-size",subgraphForrestSize);
      }
      void end (
         const LazyFlipper<GM,ACC> & lf,
         const typename LazyFlipper<GM,ACC>::ValueType value,
         const typename LazyFlipper<GM,ACC>::ValueType bound,
         const size_t subgraphSize,
         const size_t subgraphForrestSize
      ) {
         this->endVisit(value,bound,"subgraph-size",subgraphSize,"forrest-size",subgraphForrestSize);
      }
   };
   
   template< class GM,class ACC >
   class TimingVisitor< LazyFlipper<GM,ACC> >
   : public TimingVisitorBase< LazyFlipper<GM,ACC> > {
   public:
      TimingVisitor(
         const size_t visitNth=1,
         size_t reserve=0,
         bool verbose=false,
         bool multilineCout=true
      )
      :TimingVisitorBase< LazyFlipper<GM,ACC> >(visitNth,reserve,verbose,multilineCout) {
      }  
      void operator()(
         const LazyFlipper<GM,ACC> & lf,
         const typename LazyFlipper<GM,ACC>::ValueType value,
         const typename LazyFlipper<GM,ACC>::ValueType bound,
         const size_t subgraphSize,
         const size_t subgraphForrestSize
      ) {
         this->visit(value,bound,"subgraph-size",subgraphSize,"forrest-size",subgraphForrestSize);
      }
      void begin(
         const LazyFlipper<GM,ACC> & lf,
         const typename LazyFlipper<GM,ACC>::ValueType value,
         const typename LazyFlipper<GM,ACC>::ValueType bound,
         const size_t subgraphSize,
         const size_t subgraphForrestSize
      ) {
         this->beginVisit(value,bound,"subgraph-size",subgraphSize,"forrest-size",subgraphForrestSize);
      }
      void end(
         const LazyFlipper<GM,ACC> & lf,
         const typename LazyFlipper<GM,ACC>::ValueType value,
         const typename LazyFlipper<GM,ACC>::ValueType bound,
         const size_t subgraphSize,
         const size_t subgraphForrestSize
      ) {
         this->endVisit(value,bound,"subgraph-size",subgraphSize,"forrest-size",subgraphForrestSize);
      }
   };
}

/// \endcond

#endif // #ifndef OPENGM_VISITORSPECIALIZATIONS_HXX
