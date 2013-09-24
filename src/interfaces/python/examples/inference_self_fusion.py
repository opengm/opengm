import opengm
import numpy

numpy.random.seed(42)

#gm = opengm.TestModels.chain3(nVar=100,nLabels=10)
gm = opengm.TestModels.chainN(nVar=100,nLabels=5,order=6,nSpecialUnaries=5,beta=0.5)
#gm = opengm.TestModels.secondOrderGrid(20,50,50)

print gm


fs=['lazy_flipper_fusion']#,'lazy_flipper_fusion','qpbo_fusion']


for fsn in fs:

	toFuse='bp'
	infParam=opengm.InfParam()
	fuseNth=1
	printNth=1
	if toFuse == 'gibbs':
		fuseNth=100
		infParam=opengm.InfParam(steps=1000000,tempMin=0.0001,tempMax=0.1)
		printNth=100
	elif toFuse == 'bp':
		fuseNth=1
		infParam=opengm.InfParam(damping=0.9,steps=20)
		printNth==1
		
	with opengm.Timer(fsn):
		param  = opengm.InfParam(
			fusionSolver=fsn,
			toFuseInf=toFuse,
			infParam=infParam,
			maxSubgraphSize=4,
			fuseNth=fuseNth
		)


		selfFusion = opengm.inference.SelfFusion(gm,parameter=param)
		selfFusion.infer(selfFusion.verboseVisitor(multiline=True,printNth=printNth))


