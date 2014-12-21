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
        features_(features),
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
        op::NumpyView<ValueType, 2>  features_;
        std::vector<size_t>  weightIds_; 
        bool addConstFeature_;
    };


    template<class GM_ADDER,class GM_MULT>
    FunctionGeneratorBase<GM_ADDER,GM_MULT> * lpottsFunctionGen(
        ol::Weights<typename GM_ADDER::ValueType> weights,
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
    void export_function_generator_lpotts(){
        typedef LPottsFunctionGen<GM_ADDER, GM_MULT> FGen;

         bp::def("_lpottsFunctionsGen",&lpottsFunctionGen<GM_ADDER,GM_MULT>,bp::return_value_policy<bp::manage_new_object>(),
            (
                bp::arg("weights"),
                bp::arg("numFunctions"),
                bp::arg("numLabels"),
                bp::arg("features"),
                bp::arg("weightIds"),
                bp::arg("addConstFeature")
            ),
      "factory function to generate a lpotts function generator object which can be passed to ``gm.addFunctions(functionGenerator)``");

    }









}


template void opengm::export_function_generator_lpotts<op::GmAdder,op::GmMultiplier>();
