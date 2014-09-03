import opengm
import numpy
#---------------------------------------------------------------
# MinSum  with SelfFusion
#---------------------------------------------------------------
numpy.random.seed(42)


gm=opengm.loadGm("/home/tbeier/models/mrf-inpainting/house-gm.h5","gm")
#---------------------------------------------------------------
# Minimize
#---------------------------------------------------------------
#get an instance of the optimizer / inference-algorithm

inf = opengm.inference.CheapInitialization(gm)
inf.infer()
arg = inf.arg()
print gm.evaluate(arg)






infParam = opengm.InfParam(
    numIt=2000,
    generator='upDown'
)
inf=opengm.inference.FusionBased(gm, parameter=infParam)
inf.setStartingPoint(arg)
# start inference (in this case verbose infernce)
visitor=inf.verboseVisitor(printNth=1,multiline=True)
inf.infer(visitor)
arg = inf.arg()
