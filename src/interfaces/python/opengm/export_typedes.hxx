#ifndef EXPORT_TYPEDES_HXX
#define	EXPORT_TYPEDES_HXX

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/utilities/tribool.hxx>

template<class V,class I,class O,class F>
struct GmGen{
   typedef opengm::DiscreteSpace<I,I> SpaceType;
   typedef opengm::GraphicalModel<V,O,F,SpaceType,false> type;
};
template<class V,class I>
struct ETLGen{
   typedef opengm::ExplicitFunction<V ,I,I> type;
};

template<class V,class I>
struct FTLGen{
   typedef typename opengm::meta::TypeListGenerator<
      opengm::ExplicitFunction<V,I,I>,
      opengm::PottsFunction<V,I,I>
   >::type type;
};


typedef float GmValueType;
typedef opengm::UInt64Type GmIndexType;
typedef GmGen<
   GmValueType,
   GmIndexType,
   opengm::Adder ,
   FTLGen<GmValueType,GmIndexType>::type
>::type   GmAdder;

typedef GmAdder::FactorType FactorGmAdder;

typedef GmGen<
   GmValueType,
   GmIndexType,
   opengm::Multiplier ,
   FTLGen<GmValueType,GmIndexType>::type
>::type   GmMultiplier;

typedef GmMultiplier::FactorType FactorGmMultiplier;

typedef opengm::IndependentFactor<GmValueType,GmIndexType,GmIndexType> GmIndependentFactor;

namespace pyenums{
   enum AStarHeuristic{
      DEFAULT_HEURISTIC=0,
      FAST_HEURISTIC=1,
      STANDARD_HEURISTIC=2
   };
   enum IcmMoveType{
      SINGLE_VARIABLE=0,
      FACTOR=1
   };
   enum GibbsVariableProposal{
      RANDOM=0,
      CYCLIC=1
   };
      namespace libdai{
         #ifdef WITH_LIBDAI
         enum UpdateRule{
            PARALL=0,
            SEQFIX=1,
            SEQRND=2,
            SEQMAX=3
         };
         #endif
   }
}

#endif	/* EXPORT_TYPEDES_HXX */

