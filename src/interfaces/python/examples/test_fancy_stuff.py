import opengm
import numpy
np = numpy
#---------------------------------------------------------------
# MinSum  with SelfFusion
#---------------------------------------------------------------
numpy.random.seed(42)

#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/3096.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/175032.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/291000.bmp.h5","gm")
gm=opengm.loadGm("/home/tbeier/datasets/image-seg/148026.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-450/gm_knott_3d_102.h5","gm")#(ROTTEN)
gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-300/gm_knott_3d_078.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-150/gm_knott_3d_038.h5","gm")

#---------------------------------------------------------------
# Minimize
#---------------------------------------------------------------
#get an instance of the optimizer / inference-algorithm



print gm





N = np.arange(0.0, 10, 0.5)
R = np.arange(0.1, 0.99, 0.1)

print N
print R


for n in N:
    for r in R:
        print n,r

with opengm.Timer("with new method", verbose=False) as timer:

    infParam = opengm.InfParam(
        numStopIt=0,
        numIt=40,
        generator='qpboBased'
    )
    inf=opengm.inference.IntersectionBased(gm, parameter=infParam)
    # inf.setStartingPoint(arg)
    # start inference (in this case verbose infernce)
    visitor=inf.verboseVisitor(printNth=1,multiline=False)
    inf.infer(visitor)
    arg = inf.arg()


    proposalParam = opengm.InfParam(
        randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=1.000000001, ignoreSeed=True),
        stopWeight=0.0,
        reduction=0.85,
        setCutToZero=False
    )

    infParam = opengm.InfParam(
        numStopIt=20,
        numIt=20,
        generator='randomizedHierarchicalClustering',
        proposalParam=proposalParam
    )



    inf=opengm.inference.IntersectionBased(gm, parameter=infParam)
    inf.setStartingPoint(arg)
    # start inference (in this case verbose infernce)
    visitor=inf.verboseVisitor(printNth=1,multiline=False)
    inf.infer(visitor)
    arg = inf.arg()




    infParam = opengm.InfParam(
        numStopIt=0,
        numIt=40,
        generator='qpboBased'
    )
    inf=opengm.inference.IntersectionBased(gm, parameter=infParam)
    inf.setStartingPoint(arg)
    # start inference (in this case verbose infernce)
    visitor=inf.verboseVisitor(printNth=1,multiline=False)
    inf.infer(visitor)
    arg = inf.arg()

timer.interval


with opengm.Timer("with multicut method"):

    infParam = opengm.InfParam(
        workflow="(IC)(TTC-I,CC-I)"
    )
    inf=opengm.inference.Multicut(gm, parameter=infParam)
    # inf.setStartingPoint(arg)
    # start inference (in this case verbose infernce)
    visitor=inf.verboseVisitor(printNth=1,multiline=False)
    inf.infer(visitor)
    arg = inf.arg()
