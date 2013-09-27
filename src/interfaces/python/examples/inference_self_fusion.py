import opengm
import numpy
import matplotlib.pyplot as plt
import sys




def plotInfRes(v):
	val= v.getValues()
	t= v.getTimes()
	a=t.copy()
	tt=numpy.cumsum(a)
	#tt-=tt[0]
	plt.plot(tt,val)

	print "t0 tt0 tt-1 ",t[0],tt[0],tt[-1]
	tt=None
	t=None
	#plt.show()
numpy.random.seed(42)

#gm = opengm.TestModels.chain3(nVar=100,nLabels=10)
gm = opengm.TestModels.chainN(nVar=1000,nLabels=2,order=4,nSpecialUnaries=0,beta=1.3)
#gm = opengm.TestModels.secondOrderGrid(20,50,50)

print gm


fs=['bp_lf_fusion']#,'lazy_flipper_fusion','qpbo_fusion']



bpParam 	= opengm.InfParam(damping=0.999,steps=400)
gibbsParam 	= opengm.InfParam(steps=100000000,tempMin=0.0001,tempMax=0.1)


toFuse='bp'
infParam=opengm.InfParam()
fuseNth=1
printNth=1
if toFuse == 'gibbs':
	fuseNth=1000000
	infParam=gibbsParam
	printNth=100
elif toFuse == 'bp':
	fuseNth=10
	infParam=bpParam
	printNth==1
	
with opengm.Timer(fs[0]):
	param  = opengm.InfParam(
		fusionSolver=fs[0],
		toFuseInf=toFuse,
		infParam=infParam,
		maxSubgraphSize=2,
		fuseNth=fuseNth,
		bpSteps=10000
	)


if True:
	selfFusion = opengm.inference.SelfFusion(gm,parameter=param)
	visitor = selfFusion.timingVisitor()
	selfFusion.infer(visitor)
	plotInfRes(visitor)
	visitor = None

if False:
	bp = opengm.inference.BeliefPropagation(gm,parameter=bpParam)
	visitor = bp.timingVisitor(visitNth=1)
	bp.infer(visitor)
	plotInfRes(visitor)
	visitor = None
if False:
	gibbs = opengm.inference.Gibbs(gm,parameter=gibbsParam)
	visitor = gibbs.timingVisitor(visitNth=100)
	gibbs.infer(visitor)
	plotInfRes(visitor)
	visitor = None




plt.show()