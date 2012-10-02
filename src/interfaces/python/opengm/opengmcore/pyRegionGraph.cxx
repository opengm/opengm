#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/utilities/tribool.hxx>
#include <map>
#include <vector>
#include "../export_typedes.hxx"
#include "../converter.hxx"

using namespace boost::python;

struct Coordinate{
   typedef opengm::UInt64Type IndexType;
   IndexType x_;
   IndexType y_;
   Coordinate(const IndexType x=0,const IndexType y=0):x_(x),y_(y){
      
   }
   bool operator<(const Coordinate & c)const{
      if (x_<c.x_)
         return true;
      else if  (x_>c.x_)
         return false;
      else return y_<c.y_;
   }
};

struct BoundaryKey{
   typedef opengm::UInt64Type IndexType;
   IndexType r1_;
   IndexType r2_;
   BoundaryKey(const IndexType r1=0,const IndexType r2=0):r1_(r1),r2_(r2){
      r1_=r1<r2 ? r1:r2;
      r2_=r1<r2 ? r2:r1;
   }
   bool operator<(const BoundaryKey & c)const{
      if (r1_<c.r1_)
         return true;
      else if  (r1_>c.r1_)
         return false;
      else return r2_<c.r2_;
   }
   bool operator==(const BoundaryKey & c)const{
      if (r1_==c.r1_)
         return true;
      else return r2_==c.r2_;
   }
};




struct RegionGraph{
   typedef opengm::UInt64Type IndexType;
   typedef opengm::RandomAccessSet<Coordinate>  CSet;
   typedef opengm::RandomAccessSet<IndexType>   ASet;
   
   typedef std::map<IndexType,CSet> CSetMap;
   typedef typename CSetMap::const_iterator CSetMapIter;
   typedef std::map<BoundaryKey,CSet> BCSetMap;
   typedef typename BCSetMap::const_iterator BCSetMapIter;
   typedef std::vector<CSet> CSetVector;
   typedef typename CSetVector::const_iterator CSetVectorIter;
   typedef std::vector<ASet> ASetVector;
   // seg
   NumpyView<IndexType,2> seg_;
   // 
   size_t numRegions_;
   size_t numBoundaries_;
   // adjacencie
   ASetVector regionsBoundaries_;
   ASetVector boundariesRegions_;
   // coordinates
   CSetVector boundaryPixels_;
   CSetVector regionPixels_;
   
   
   
   template<class MAP,class COORDINATE,class KEY>
   void mapInsert(MAP & map,const COORDINATE & val,const COORDINATE & val2,const KEY & key){
      if(map.find(key)==map.end()){
         CSet set;
         set.insert(val);
         set.insert(val2);
         map.insert(std::pair<KEY,CSet>(key,set));
      }
      else{
         CSet & set=map.find(key)->second;
         set.insert(val);
         set.insert(val2);
      }    
   }
   
   
   
   RegionGraph( NumpyView<IndexType,2> seg,const IndexType numRegion,const bool verb):
   seg_(seg),
   numRegions_(numRegion),
   numBoundaries_(0){
      size_t dx=seg_.shape(0);
      size_t dy=seg_.shape(1);
      
      regionPixels_.resize(numRegions_);
      regionsBoundaries_.resize(numRegions_);
      
      BCSetMap boundariesPixelsMap;
      //find regions coordinates:
      //find all boundaries
      for(size_t x=0;x<dx;++x){
         for(size_t y=0;y<dy;++y){
            IndexType ri=seg_(x,y);
            regionPixels_[ri].insert(Coordinate(x,y));
            // detect boundary
            if(x+1<dx){
               IndexType ri2=seg_(x+1,y);
               if(ri!=ri2){
                  //size_t a=ri<ri2 ? ri:ri2;
                  //size_t b=ri<ri2 ? ri2:ri;
                  size_t a=ri,b=ri2;
                  mapInsert(boundariesPixelsMap,Coordinate(x,y),Coordinate(x+1,y),BoundaryKey(a,b));
               }
            }
            if(y+1<dy){
               IndexType ri2=seg_(x,y+1);
               if(ri!=ri2){
                  size_t a=ri,b=ri2;
                  mapInsert(boundariesPixelsMap,Coordinate(x,y),Coordinate(x,y+1),BoundaryKey(a,b));
               }
            }
         }
      }
      numBoundaries_=boundariesPixelsMap.size();
      boundaryPixels_.resize(numBoundaries_);
      boundariesRegions_.resize(numBoundaries_);
      if(verb==true){
         std::cout<<"number of Regions: "<<numRegions_<<"\n";
         std::cout<<"number of Boundaries: "<<numBoundaries_<<"\n";
      }
      // loop over all boundaries
      size_t bi=0;
      for(BCSetMapIter iter=boundariesPixelsMap.begin();iter!=boundariesPixelsMap.end();++iter,++bi){
         
         OPENGM_ASSERT(iter->second.size()!=0)
         
         boundaryPixels_[bi]=iter->second;
         const size_t r1=iter->first.r1_;
         const size_t r2=iter->first.r2_;
         boundariesRegions_[bi].insert(r1);
         boundariesRegions_[bi].insert(r2);
         
         //std::cout<<"bi="<<bi<<" bregsize:"<<boundariesRegions_[bi].size()<<"\n";
         
         regionsBoundaries_[r1].insert(bi);
         regionsBoundaries_[r2].insert(bi);
      }
      std::cout<<"bi="<<bi<<"\n";
   }
   
   IndexType numberOfRegions()const{return numRegions_;}
   IndexType numberOfBoundaries()const{return numBoundaries_;}
   IndexType numberOfAdjacentBoundaries(const IndexType ri)const{return regionsBoundaries_[ri].size(); }
   IndexType adjacentBoundary(const IndexType ri,const IndexType b)const{return regionsBoundaries_[ri][b]; }
   IndexType numberOfAdjacentRegions(const IndexType bi)const{return boundariesRegions_[bi].size(); }
   IndexType adjacentRegion(const IndexType bi,const IndexType b)const{return boundariesRegions_[bi][b]; }
   
   IndexType boundarySize(const IndexType bi)const{return boundaryPixels_[bi].size();}
   boost::python::list boundaryPixels( const IndexType bi)const{
      boost::python::list list;
      for(size_t i=0;i<boundaryPixels_[bi].size();++i){
         list.append(make_tuple(int(boundaryPixels_[bi][i].x_),int(boundaryPixels_[bi][i].y_)));
      }
      return list;
   }
   
};


void export_rag() {
   import_array();

   class_<RegionGraph > ("RegionGraph", init<NumpyView<opengm::UInt64Type,2>, const opengm::UInt64Type,const bool > ())
  .def("numberOfRegions", &RegionGraph::numberOfRegions)
  .def("numberOfBoundaries", &RegionGraph::numberOfBoundaries)
  .def("numberOfAdjacentBoundaries", &RegionGraph::numberOfAdjacentBoundaries)
  .def("adjacentBoundary", &RegionGraph::adjacentBoundary)
  .def("numberOfAdjacentRegions", &RegionGraph::numberOfAdjacentRegions)
  .def("adjacentRegion", &RegionGraph::numberOfBoundaries)
  .def("boundarySize", &RegionGraph::boundarySize)
  .def("boundaryPixels", &RegionGraph::boundaryPixels)
   ;

}



