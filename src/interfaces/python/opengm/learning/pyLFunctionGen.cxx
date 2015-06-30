#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>


#include "opengm/graphicalmodel/weights.hxx"
#include "opengm/functions/learnable/lpotts.hxx"
#include "opengm/functions/learnable/lunary.hxx"
#include "opengm/functions/learnable/lsum_of_experts.hxx"

#include "../opengmcore/functionGenBase.hxx"


namespace bp = boost::python;
namespace op = opengm::python;
namespace ol = opengm::learning;
namespace ofl = opengm::functions::learnable;
namespace opengm{



    template<class GM_ADDER,class GM_MULT>
    class LPottsFunctionGen :
    public FunctionGeneratorBase<GM_ADDER,GM_MULT>
    {
    public:       
        typedef typename GM_ADDER::ValueType ValueType;
        typedef typename GM_ADDER::IndexType IndexType;
        typedef typename GM_ADDER::LabelType LabelType;
        typedef ol::Weights<ValueType> WeightType;
        typedef  ofl::LPotts<ValueType, IndexType, LabelType> FType;

        LPottsFunctionGen(
            WeightType & weights,
            const size_t numFunctions,
            const size_t numLabels,
            op::NumpyView<ValueType, 2> features,
            op::NumpyView<IndexType, 1> weightIds,
            const bool addConstFeature
        ):
        FunctionGeneratorBase<GM_ADDER,GM_MULT>(),
        weights_(weights),
        numFunctions_(numFunctions),
        numLabels_(numLabels),
        features_(features.view()),
        weightIds_(weightIds.begin(), weightIds.end()),
        addConstFeature_(addConstFeature)
        {
            OPENGM_CHECK_OP(features.shape(0), == , numFunctions, "wrong shape");
            OPENGM_CHECK_OP(features.shape(1)+int(addConstFeature), == , weightIds.shape(0), "wrong shape");
        }
 

        template<class GM>
        std::vector< typename GM::FunctionIdentifier > * addFunctionsGeneric(GM & gm)const{

            typedef typename GM::FunctionIdentifier Fid;
            typedef std::vector<Fid> FidVector;
            FidVector * fidVector = new FidVector(numFunctions_);

            const size_t nFeat =features_.shape(1);
            std::vector<ValueType> fFeat(nFeat+int(addConstFeature_));
            for(size_t  i=0;i<numFunctions_;++i){
                for(size_t f=0; f<nFeat; ++f){
                    fFeat[f] = features_(i,f);
                }
                if(addConstFeature_){
                    fFeat[nFeat] = 1.0;
                }
                const FType f(weights_, numLabels_, weightIds_, fFeat);
                (*fidVector)[i] = gm.addFunction(f);
            }   
            return fidVector;
        }

        virtual std::vector< typename GM_ADDER::FunctionIdentifier > * addFunctions(GM_ADDER & gm)const{
            return this-> template addFunctionsGeneric<GM_ADDER>(gm);
        }
        virtual std::vector< typename GM_MULT::FunctionIdentifier >  * addFunctions(GM_MULT & gm)const{
            throw RuntimeError("Wrong Operator for Learning");
            return NULL;
        }
    private:
        WeightType & weights_;
        size_t numFunctions_;
        size_t numLabels_;
        marray::Marray<ValueType>  features_;
        std::vector<size_t>  weightIds_; 
        bool addConstFeature_;
    };



    template<class GM_ADDER,class GM_MULT>
    class LUnarySharedFeatFunctionGen :
    public FunctionGeneratorBase<GM_ADDER,GM_MULT>
    {
    public:       
        typedef typename GM_ADDER::ValueType ValueType;
        typedef typename GM_ADDER::IndexType IndexType;
        typedef typename GM_ADDER::LabelType LabelType;
        typedef ol::Weights<ValueType> WeightType;
        typedef  ofl::LUnary<ValueType, IndexType, LabelType> FType;

        LUnarySharedFeatFunctionGen(
            WeightType & weights,
            const size_t numFunctions,
            const size_t numLabels,
            op::NumpyView<ValueType, 2> & features,
            op::NumpyView<IndexType, 2> & weightIds,
            const bool makeFirstEntryConst,
            const bool addConstFeature
        ):
        FunctionGeneratorBase<GM_ADDER,GM_MULT>(),
        weights_(weights),
        numFunctions_(numFunctions),
        numLabels_(numLabels),
        features_(features.view()),
        //weightIds_(weightIds),
        makeFirstEntryConst_(makeFirstEntryConst),
        addConstFeature_(addConstFeature)
        {
            //std::cout<<"constructor\n";

            //std::cout<<"    features (1000,1)"<<features(1000,1)<<"\n";
            //std::cout<<"    features_(1000,1)"<<features_(1000,1)<<"\n";
            OPENGM_CHECK_OP(features.shape(0), == , numFunctions, "wrong shape");
            OPENGM_CHECK_OP(weightIds.shape(1), == , features.shape(1) + int(addConstFeature), "wrong shape");
            OPENGM_CHECK_OP(weightIds.shape(0)+int(makeFirstEntryConst), == ,numLabels, "wrong shape");


            const size_t nFeat =features_.shape(1);
            const size_t nWPerL = nFeat+int(addConstFeature_);
            const size_t wShape[2] = {numLabels_- int(makeFirstEntryConst_) ,nWPerL};

            wIds_ = marray::Marray<size_t>(wShape, wShape+2);

            //std::cout<<"assignment\n";
            //std::cout<<"passed wi shape "<<weightIds.shape(0)<<" "<<weightIds.shape(1)<<" given "<<wShape[0]<<" "<<wShape[1]<<"\n";
            //std::cout<<"wIds_  shape "<<wIds_.shape(0)<<" "<<wIds_.shape(1)<<"\n";

            for(size_t ll=0; ll<wShape[0]; ++ll){
                for(size_t wi=0; wi<wShape[1]; ++wi){
                    //std::cout<<"ll "<<ll<<" wi "<<wi<<"\n";
                    size_t passed =  weightIds(ll,wi);
                    //std::cout<<"passed "<<passed<<"\n";
                    wIds_(ll,wi) = passed;
                }  
            }
            //std::cout<<"constructor done\n";
        }
 

        template<class GM>
        std::vector< typename GM::FunctionIdentifier > * addFunctionsGeneric(GM & gm)const{
            //std::cout<<"&** features_(1000,1)"<<features_(1000,1)<<"\n";



            typedef typename GM::FunctionIdentifier Fid;
            typedef std::vector<Fid> FidVector;
            FidVector * fidVector = new FidVector(numFunctions_);


            const size_t nFeat =features_.shape(1);
            const size_t nWPerL = nFeat+int(addConstFeature_);
            marray::Marray<ValueType> fFeat(&nWPerL,&nWPerL+1);


            // copy the weights once!
            const size_t wShape[2] = {numLabels_- int(makeFirstEntryConst_) ,nWPerL};
            marray::Marray<size_t> _weightIds(wShape, wShape+2);

            //for(size_t ll=0; ll<wShape[0]; ++ll)
            //for(size_t wi=0; wi<wShape[1]; ++wi){
            //    _weightIds(ll,wi) = weightIds_(ll,wi);
            //}    


            for(size_t  i=0;i<numFunctions_;++i){
                // copy the features for that instance
                for(size_t f=0; f<nFeat; ++f){
                    //std::cout<<"added feat:"<<features_(i,f)<<"\n";
                    fFeat(f) = features_(i,f);
                }
                if(addConstFeature_){
                    fFeat(nFeat) = 1.0;
                }
                FType f(weights_, numLabels_, wIds_, fFeat, makeFirstEntryConst_);

                //std::cout<<"INTERNAL TEST\n";
                //for(size_t l=0;l<numLabels_; ++l){
                //    std::cout<<"l "<<l<<" f(l) = "<<f(&l)<<"\n";
                //}

                (*fidVector)[i] = gm.addFunction(f);
            }   
            return fidVector;
        }

        virtual std::vector< typename GM_ADDER::FunctionIdentifier > * addFunctions(GM_ADDER & gm)const{
            return this-> template addFunctionsGeneric<GM_ADDER>(gm);
        }
        virtual std::vector< typename GM_MULT::FunctionIdentifier >  * addFunctions(GM_MULT & gm)const{
            throw RuntimeError("Wrong Operator for Learning");
            return NULL;
        }
    private:
        WeightType & weights_;
        size_t numFunctions_;
        size_t numLabels_;

        marray::Marray<ValueType> features_;
        //op::NumpyView<ValueType, 2>  features_;
        op::NumpyView<IndexType, 2>  weightIds_;
        bool makeFirstEntryConst_;
        bool addConstFeature_;
        marray::Marray<size_t> wIds_;
    };


    template<class GM_ADDER,class GM_MULT>
    FunctionGeneratorBase<GM_ADDER,GM_MULT> * lunarySharedFeatFunctionGen(
        ol::Weights<typename GM_ADDER::ValueType> & weights,
        const size_t numFunctions,
        const size_t numLabels,
        opengm::python::NumpyView<typename GM_ADDER::ValueType,2> features,
        opengm::python::NumpyView<typename GM_ADDER::IndexType,2> weightIds,
        const bool makeFirstEntryConst,
        const bool addConstFeature
    ){
        FunctionGeneratorBase<GM_ADDER,GM_MULT> * ptr = 
            new LUnarySharedFeatFunctionGen<GM_ADDER,GM_MULT>(weights,numFunctions,numLabels,
                                                              features,weightIds,makeFirstEntryConst,
                                                              addConstFeature);
        return ptr;
    }


    template<class GM_ADDER,class GM_MULT>
    FunctionGeneratorBase<GM_ADDER,GM_MULT> * lpottsFunctionGen(
        ol::Weights<typename GM_ADDER::ValueType> & weights,
        const size_t numFunctions,
        const size_t numLabels,
        opengm::python::NumpyView<typename GM_ADDER::ValueType,2> features,
        opengm::python::NumpyView<typename GM_ADDER::IndexType,1> weightIds,
        const bool addConstFeature
    ){
        FunctionGeneratorBase<GM_ADDER,GM_MULT> * ptr = 
            new LPottsFunctionGen<GM_ADDER,GM_MULT>(weights,numFunctions,numLabels,features,weightIds, addConstFeature);
        return ptr;
    }












    template<class GM_ADDER,class GM_MULT>  
    void export_lfunction_generator(){
        typedef LPottsFunctionGen<GM_ADDER, GM_MULT> FGen;

         bp::def("_lpottsFunctionsGen",&lpottsFunctionGen<GM_ADDER,GM_MULT>,
                bp::return_value_policy<bp::manage_new_object>(),
            (
                bp::arg("weights"),
                bp::arg("numFunctions"),
                bp::arg("numLabels"),
                bp::arg("features"),
                bp::arg("weightIds"),
                bp::arg("addConstFeature")
            )
        );

         bp::def("_lunarySharedFeatFunctionsGen",&lunarySharedFeatFunctionGen<GM_ADDER,GM_MULT>,
                bp::with_custodian_and_ward_postcall<0, 4, bp::return_value_policy<bp::manage_new_object> >(),
            (
                bp::arg("weights"),
                bp::arg("numFunctions"),
                bp::arg("numLabels"),
                bp::arg("features"),
                bp::arg("weightIds"),
                bp::arg("makeFirstEntryConst"),
                bp::arg("addConstFeature")
            )
        );

    }









}


template void opengm::export_lfunction_generator<op::GmAdder,op::GmMultiplier>();
