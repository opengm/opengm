template<class GM_ADDER,class GM_MULT>
class VariableIndexGenerator{
    // typedefs
    typedef typename GM_ADDER::ValueType ValueType;
    typedef typename GM_ADDER::IndexType IndexType;
    typedef typename GM_ADDER::LabelType LabelType;

    typedef typename GM_ADDER::FunctionIdentifier  FidAdderType;
    typedef typename GM_MULT::FunctionIdentifier   FidMultiplierType;
    typedef std::vector<FidAdderType>       FidVectorAdderType;
    typedef std::vector<FidMultiplierType>  FidVectorMultiplierType;

public:
    virtual IndexType addFactors(const FidVectorAdderType      & fids,GM_ADDER & gm) =0;
    virtual IndexType addFactors(const FidVectorMultiplierType & fids,GM_ADDER & gm) =0;
};




template<class GM_ADDER,class GM_MULT>
class GridVariableIndexGenerator{
    // typedefs
    typedef typename GM_ADDER::ValueType ValueType;
    typedef typename GM_ADDER::IndexType IndexType;
    typedef typename GM_ADDER::LabelType LabelType;

    typedef typename GM_ADDER::FunctionIdentifier  FidAdderType;
    typedef typename GM_MULT::FunctionIdentifier   FidMultiplierType;
    typedef std::vector<FidAdderType>       FidVectorAdderType;
    typedef std::vector<FidMultiplierType>  FidVectorMultiplierType;

public:
    template<class SHAPE_ITERATOR>
    GridVariableIndexGenerator(
        SHAPE_ITERATOR begin,
        SHAPE_ITERATOR end,
        const size_t neighbourhood,
        const bool numpyViOrder
    ):
    shape_(begin,end),
    strides_(std::distance(begin,end))
    neighbourhood_(neighbourhood),
    numpyViOrder_(numpyViOrder)
    {
        const size_t dim_=shape.size();
        if(numpyViOrder_){
            if(dim==2){
                strides_[0]=shape_[1];
                strides_[1]=1;
            }
            else{
                strides_[0]=shape_[1]*shape_[2];
                strides_[1]=shape_[2];
                strides_[2]=1;
            }

        }
        else{
            if(dim==2){
                strides_[0]=1;
                strides_[1]=shape_[0];
            }
            else{
                strides_[0]=1;
                strides_[1]=shape_[0];
                strides_[2]=shape_[0]*shape_[1];
            }
        }
    }

    IndexType getVis(const IndexType x,const IndexType y)const{
        return x*strides_[0]+x*strides_[1];
    }
    IndexType getVis(const x vi0,const y vi1,const z vi2)const{
        return x*strides_[0]+y*strides_[1]+z*strides_[2];
    }

    template<class FID_VEC>
    typename const FID_VEC::value_type & getFid(const FID_VEC fids,size_t & fidIndex){
        const size_t i=fidIndex;
        ++fidIndex;
        return fids[  i<fids.size() ? i:fids.size()-1 ];
    }


    template<class GM>
    IndexType addFactorsGeneric(const std::vector<typename GM::FunctionIdentifier> & fids,GM & gm){
        const size_t dim=shape_.size();
        factorIndex=0;
        size_t fidIndex=0;
        IndexType vis[3]={0,0};
        IndexType c[3]={0,0,0};
        if(dim==2){
            bool n8=neighbourhood_==8;
            // neighbour
            const size_t numNeighbour=(n8==true ? 3:2);
            IndexType nh[3][2];
            nh[0][0]=1; nh[0][1]=0;  // right neighbour
            nh[1][0]=0; nh[1][1]=1;  // down neighbour 
            nh[2][0]=1; nh[2][1]=1;  // right-down neighbour
            const size_t numNeighbour=(n8==true ? 3:2);
            for(c[0]=0;c[0]<shape_[0];++c[0])
            for(c[1]=0;c[1]<shape_[1];++c[1]){
                // walk over Neighbourhood
                for(n=0 ; n<numNeighbour; ++n){
                    // check if neighbour pixel is in the image
                    if(  c[0]+nh[n][0] <shape_[0]  && c[1]+nh[n][1] <shape_[1]  ){
                        // compute variable indices
                        vis[0]=getVis(c[0],c[1]);
                        vis[1]=getVis(c[0]+nh[n][0],c[1]+nh[n][1]);
                        // add factor (getFid will increment the fidIndex by its own)
                        factorIndex=gm.addFactor(getFid(fids,fidIndex),vis,vis+2);
                    }
                }
            }
        }
        if(dim==3){
            bool n26=neighbourhood_==26;
            // neighbour
            const size_t numNeighbour=(n26==true ? 7:3);
            IndexType nh[7][3];
            nh[0][0]=1; nh[0][1]=0; nh[0][2]=0; 
            nh[0][0]=0; nh[0][1]=1; nh[0][2]=0; 
            nh[0][0]=0; nh[0][1]=0; nh[0][2]=1; 
            nh[0][0]=1; nh[0][1]=1; nh[0][2]=0;
            nh[0][0]=0; nh[0][1]=1; nh[0][2]=1;
            nh[0][0]=1; nh[0][1]=0; nh[0][2]=1; 
            nh[0][0]=1; nh[0][1]=1; nh[0][2]=1; 

            for(c[0]=0;c[0]<shape_[0];++c[0])
            for(c[1]=0;c[1]<shape_[1];++c[1])
            for(c[2]=0;c[2]<shape_[2];++c[2]){
                // walk over Neighbourhood
                for(n=0 ; n<numNeighbour; ++n){
                    // check if neighbour pixel is in the image
                    if(  c[0]+nh[n][0] <shape_[0]  && c[1]+nh[n][1] <shape_[1]  && c[2]+nh[n][2] <shape_[2]){
                        // compute variable indices
                        vis[0]=getVis(c[0],c[1]);
                        vis[1]=getVis(c[0]+nh[n][0],c[1]+nh[n][1],c[2]+nh[n][2]);
                        // add factor (getFid will increment the fidIndex by its own)
                        factorIndex=gm.addFactor(getFid(fids,fidIndex),vis,vis+2);
                    }
                }
            }
        }
    }
    std::vector<IndexType> shape_;
    std::vector<IndexType> strides_;
    neighbourhood_;
    bool numpyViOrder_;
}; 