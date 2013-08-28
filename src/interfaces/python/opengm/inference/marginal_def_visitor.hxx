#ifndef MARGINAL_DEF_VISITOR
#define MARGINAL_DEF_VISITOR

#include <boost/python.hpp>
#include <sstream>
#include <string>
#include "gil.hxx"
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


#include <opengm/inference/inference.hxx>
#include <opengm/opengm.hxx>


template<class INF>
class MarginalSuite: public boost::python::def_visitor<MarginalSuite<INF> >{
public:
    friend class boost::python::def_visitor_access;
    typedef typename INF::GraphicalModelType                    GraphicalModelType;
    typedef typename GraphicalModelType::IndependentFactorType  IndependentFactorType;
    typedef typename IndependentFactorType::ShapeIteratorType   IFactorShapeType;
    typedef typename INF::Parameter                             ParameterType;
    typedef typename GraphicalModelType::IndexType              IndexType;
    typedef typename GraphicalModelType::LabelType              LabelType;
    typedef typename GraphicalModelType::ValueType              ValueType;

    template <class classT>
    void visit(classT& c) const{ 
        c
            // marginals for a single var and single
            .def("_marginals",&marginals)
            .def("_factorMarginals",&factorMarginals)
        ;
    }


    static boost::python::object marginals(
        const INF & inf,
        opengm::python::NumpyView<IndexType> vis
    ){
        const GraphicalModelType & gm     = inf.graphicalModel();
        const IndexType numberOfVariables = gm.numberOfVariables();
        const LabelType numLabels         = gm.numberOfLabels(vis(0));
        const size_t numPassedVis         = vis.size();

        // allocate 2d array
        boost::python::object obj    = opengm::python::get2dArray<ValueType>(numPassedVis,numLabels);
        opengm::python::NumpyView<ValueType,2> numpyArray(obj);

        {
            releaseGIL rgil;
            // allocate ifactor to store marginal and inference termination
            IndependentFactorType ifactor;
            opengm::InferenceTermination infTerm;
            for(size_t i=0;i<numPassedVis;++i){
                const IndexType vi      = vis[i];
                const LabelType nLabels = gm.numberOfLabels(vi);
                if(nLabels!=numLabels){
                    throw opengm::RuntimeError("all variables in ``vis`` must have the same number of Labels");
                }
                // get marginal
                infTerm = inf.marginal(vi,ifactor);
                if(infTerm == opengm::UNKNOWN){
                    throw opengm::RuntimeError("this inference class does not support marginalization");
                }
                // write marginals into numpy array
                for(LabelType l=0;l<numLabels;++l){
                    numpyArray(i,l)=ifactor(&l);
                }

            }
        }
        return obj;
    }

    static boost::python::object factorMarginals(
        const INF & inf,
        opengm::python::NumpyView<IndexType> fis
    ){
        const GraphicalModelType & gm     = inf.graphicalModel();
        const size_t order                = gm[fis(0)].numberOfVariables();
        const size_t numPassedFac         = fis.size();
        const size_t outNDim              = order+1;

        // fill out shape
        opengm::FastSequence<LabelType> outShape(outNDim);
        //opengm::FastSequence<LabelType> facShape(numVariables);
        outShape[0]=numPassedFac;

        for(size_t v=0;v<order;++v){
            outShape[v+1]=gm[fis(0)].numberOfLabels(v);
        }

        // allocate nd array
        boost::python::object obj    = opengm::python::getArray<ValueType>(outShape.begin(),outShape.end());
        opengm::python::NumpyView<ValueType,2> numpyArray(obj);
        std::vector<LabelType> outCoord(outNDim);
        OPENGM_ASSERT(numpyArray.dimension()==order+1)
        {
            releaseGIL rgil;
            // allocate ifactor to store factorMarginal and inference termination
            IndependentFactorType ifactor;
            opengm::InferenceTermination infTerm;
            // go over all factor(-indices) in fis
            for(size_t i=0;i<numPassedFac;++i){
                const IndexType fi      = fis[i];
                const LabelType nVar    = gm[fi].numberOfVariables();
                // check that order is the same for all factors
                if(nVar!=order){
                    throw opengm::RuntimeError("all factors in ``fis`` must have the same order");
                }
                // check that shape is the same for all factors
                for(size_t v=0;v<order;++v){
                    if(gm[fi].numberOfLabels(v)!=outShape[v+1]){
                        throw opengm::RuntimeError("all factors in ``fis`` must have the same shape ");
                    }
                }
                // get factor marginal
                infTerm = inf.factorMarginal(fi,ifactor);
                if(infTerm == opengm::UNKNOWN){
                    throw opengm::RuntimeError("this inference class does not support marginalization");
                }
                // fill marginal for factor in fi 
                opengm::ShapeWalker<IFactorShapeType> walker(ifactor.shapeBegin(),ifactor.size());
                OPENGM_ASSERT(ifactor.numberOfVariables()==order)
                OPENGM_ASSERT(outCoord.size()==numpyArray.dimension())
                // first out coordinate is the index of the fis-range
                outCoord[0]=i;  
                for (size_t e=0;e<ifactor.size();++e) {
                    ValueType marginalValue=ifactor.function()(e);
                    for(size_t c=0;c<order;++c){
                        outCoord[c+1]=walker.coordinateTuple()[c];
                    }
                    numpyArray(outCoord.begin())=marginalValue;
                    ++walker;
                }
            }
        }
        return obj;
    }
};










#endif // MARGINAL_DEF_VISITOR 