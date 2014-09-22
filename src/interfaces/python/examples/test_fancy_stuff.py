import opengm
import numpy
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








with opengm.Timer("with new method"):

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
        numStopIt=100,
        numIt=400,
        generator='randomizedHierarchicalClustering',
        proposalParam=proposalParam
    )


    #proposalParam = opengm.InfParam(
    #    randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=0.100000001,ignoreSeed=False),
    #    seedFraction = 0.01
    #)
    #infParam = opengm.InfParam(
    #    numStopIt=10,
    #    numIt=40,
    #    generator='randomizedWatershed',
    #    proposalParam=proposalParam
    #)


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


with opengm.Timer("with multicut method"):

    infParam = opengm.InfParam(
        #workflow="(MTC)(CC)"
    )
    inf=opengm.inference.Multicut(gm, parameter=infParam)
    # inf.setStartingPoint(arg)
    # start inference (in this case verbose infernce)
    visitor=inf.verboseVisitor(printNth=1,multiline=False)
    inf.infer(visitor)
    arg = inf.arg()
