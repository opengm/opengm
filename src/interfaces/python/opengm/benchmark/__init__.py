import opengm
import os

import numpy

try:
	import matplotlib.pyplot as plt
	from matplotlib import pyplot
	from matplotlib import pylab
except:
	pass


class ModelResult(object):
	def __init__(self):
		print opengm.configuration





def filenamesFromDir(path,ending='.h5'):
	return [path+f for f in os.listdir(path) if f.endswith(ending)]


def plotInfRes(v):

	

	val= v.getValues()
	t= v.getTimes()
	a=t.copy()
	tt=numpy.cumsum(a)
	#tt-=tt[0]
	p=pylab.plot(tt,val)

	print "t0 tt0 tt-1 ",t[0],tt[0],tt[-1]
	tt=None
	t=None
	return p


def makePath(p):
	if not os.path.exists(p):
	    os.makedirs(p)



def makePathEnding(f):
	if f.endswith("/"):
		return f
	else :
		return f+"/"


def storeSingleResult(result,outFolder,dataSetName,solverName,gmName):


	basePath    =  makePathEnding(outFolder)
	dataSetPath =  makePathEnding(basePath+dataSetName)
	solverPath  =  makePathEnding(dataSetPath+solverName)

	makePath(solverPath)



def runBenchmark(fNames,solvers,outFolder,dataSetName,plot=False):

	nFiles  = len(fNames)
	nSolver = len(solvers)

	result = dict()

	for fNr,fName in enumerate(fNames):

		#if fNr!=1:
		#	continue
		print fNr+1,"/",nFiles,":",fName
		print "load gm"
		if isinstance(fName,str):
			print "from string"
			gm = opengm.loadGm(fName)
		else :
			print "from gm"
			gm = fName
		print gm

		if plot:
			pr=[]
		names=[]
		#fig, ax = plt.subplots()


		fileResult=dict()
		#fileResult[fn]=fName

		for sNr,solver in enumerate(solvers) :

			


			(sName,sClass,sParam)=solver
			print sName
			inf=sClass(gm=gm,parameter=sParam)
			tv=inf.timingVisitor(verbose=True,multiline=False,visitNth=1)
			inf.infer(tv)

			# store results
			solverResult=dict()

			solverResult['values'] 		= tv.getValues()
			solverResult['times'] 		= tv.getTimes()
			solverResult['bounds'] 		= tv.getBounds()
			solverResult['iterations'] 	= tv.getIterations()
			solverResult['name']		= sName
			solverResult['arg']			= inf.arg()
			solverResult['gmName']		= fName


			# write to file
			storeSingleResult(result=tv,outFolder=outFolder,dataSetName=dataSetName,solverName=sName,gmName=fName)


			# write into result dict
			fileResult[sName] 			= solverResult

			if plot:
				pr.append(plotInfRes(tv))
			print sName
			names.append(sName)

		result[fName]=fileResult

		print names
		if plot:
			plt.legend( names,loc= 5)
			#plt.legend(pr,names)
			plt.show()


		#return result



if __name__ == '__main__':

	import opengm

	
	#fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/geo-surf-7/')
	#fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/cell-tracking/')
	#fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/color-seg/')
	#fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/mrf-inpainting/')
	#fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/scene-decomposition/')
	
	fNames = filenamesFromDir('/home/tbeier/models/color-seg-n4')
	#print fNames

	Ip = opengm.InfParam
	oi = opengm.inference 




	#gms = [ opengm.TestModels.secondOrderGrid(100,100,25) ]

	#pyplot.yscale('log')
	#pyplot.xscale('log')

	#print help(oi.DualDecompositionSubgradient)
	s=200
	solvers =[
		#('pbp',oi.Pbp,Ip(steps=10,pruneLimit=1.1)),
		#('bp', 					oi.BeliefPropagation, 			Ip(steps=10,damping=0.0) )	,
		#('random-fusion-lf2-a',oi.RandomFusion,Ip(steps=400,fusionSolver='lf2')),
		#('lf2', 					oi.LazyFlipper,			Ip(maxSubgraphSize=3) ) ,
		#('lf3', 					oi.LazyFlipper,			Ip(maxSubgraphSize=3) ) ,
		#('loc',oi.Loc,Ip(steps=s,solver='lf2',maxBlockRadius=10,maxTreeRadius=30,phi=0.0001,treeRuns=20)),
		#('localOpt-lf2',  oi.ChainedInf,	
		#	Ip(
		#		solvers=(oi.CheapInitialization,oi.LazyFlipper),
		#		parameters=(Ip(),Ip(maxSubgraphSize=1))
		#	)
		#),
		#('localOpt-lf3',  oi.ChainedInf,	
		#	Ip(
		#		solvers=(oi.CheapInitialization,oi.LazyFlipper),
		#		parameters=(Ip(),Ip(maxSubgraphSize=3))
		#	)
		#),
		#('localOpt-loc',  oi.ChainedInf,	
		#	Ip(
		#		solvers=(oi.CheapInitialization,oi.Loc),
		#		parameters=(Ip(),Ip(steps=s,solver='lf2',maxBlockRadius=3,maxTreeRadius=30,phi=0.0001,treeRuns=20))
		#	)
		#),
		#('random-fusion-lf2-b',oi.RandomFusion,Ip(steps=20,fusionSolver='lf2')),
		#('random-fusion-qpbo-a',oi.RandomFusion,Ip(steps=20,fusionSolver='qpbo')),
		#('random-fusion-qpbo-b',oi.RandomFusion,Ip(steps=100,fusionSolver='qpbo')),
		#('loc-block-ad3',  oi.Loc,Ip(steps=s,solver='ad3',maxBlockRadius=5,maxTreeRadius=0,phi=0.0001) ) ,
		#('loc-block-5-30-lf2-tr-1',  oi.Loc,Ip(steps=s,solver='lf2',maxBlockRadius=5,maxTreeRadius=30,phi=0.0001,treeRuns=1) ) ,
		#('loc-block-5-30-lf2-tr-2',  oi.Loc,Ip(steps=s,solver='lf3',maxBlockRadius=5,maxTreeRadius=20,phi=0.0001,treeRuns=2) ) ,
		#('loc-block-5-30-lf2-tr-5',  oi.Loc,Ip(steps=s,solver='lf2',maxBlockRadius=10,maxTreeRadius=30,phi=0.0001,treeRuns=5) ) ,
		#('loc-block-10-30-lf2-tr-10',  oi.Loc,Ip(steps=s,solver='lf2',maxBlockRadius=50,maxTreeRadius=3000,phi=0.0001,treeRuns=-5) ) ,
		#('loc-block-10-30-lf2-tr-10',  oi.Loc,Ip(steps=s,solver='lf2',maxBlockRadius=10,maxTreeRadius=30,phi=0.0001,treeRuns=20) ) ,
		#('loc-block-10-30-lf2-tr- -2',  oi.Loc,Ip(steps=s,solver='lf2',maxBlockRadius=10,maxTreeRadius=30,phi=0.0001,treeRuns=-1) ) ,
		#('loc-block-10-30-lf2-tr- -4',  oi.Loc,Ip(steps=s,solver='lf2',maxBlockRadius=10,maxTreeRadius=30,phi=0.0001,treeRuns=-4) ) ,
		#('loc-block-10-50-lf2',  oi.Loc,Ip(steps=s*2,solver='lf2',maxBlockRadius=10,maxTreeRadius=50,phi=0.0001) ) ,
		#('loc-block-20-lf2',  oi.Loc,Ip(steps=s,solver='lf2',maxBlockRadius=20,maxTreeRadius=0,phi=0.0001) ) ,
		#('loc-block-40-lf2',  oi.Loc,Ip(steps=s/2,solver='lf2',maxBlockRadius=40,maxTreeRadius=0,phi=0.0001) ) ,
		#('loc-block-lf3',  oi.Loc,Ip(steps=2000,solver='lf3',maxBlockRadius=3,maxTreeRadius=0,phi=0.0001) ) ,
		#('loc-tree-20', oi.Loc,Ip(steps=2000,solver='ad3',maxBlockRadius=0,maxTreeRadius=20,phi=0.0001) ) ,
		#('loc-block-tree-20', oi.Loc,Ip(steps=1000,solver='ad3',maxBlockRadius=3,maxTreeRadius=20,phi=0.0001) ) ,
		#('loc-tree-40', oi.Loc,Ip(steps=s/3,solver='ad3',maxBlockRadius=0,maxTreeRadius=40,phi=0.0001) ) ,
		#('loc-tree-100', oi.Loc,Ip(steps=s,solver='ad3',maxBlockRadius=0,maxTreeRadius=100,phi=0.1) ) ,
		#('loc-block-tree-40', oi.Loc,Ip(steps=s,solver='ad3',maxBlockRadius=4,maxTreeRadius=40,phi=0.0001) )

		#('self-fusion-gibbs', oi.SelfFusion 	   	 , Ip(fuseNth=1000,toFuseInf='gibbs',fusionSolver='bp_lf_fusion', infParam=Ip(steps=10000000,tempMin=0.00001,tempMax=1.00) ) )	,
		
		#('dd-sg_dp_st', 		oi.DualDecompositionSubgradient 	   	 , Ip(maximalNumberOfIterations=300,decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) )	,
		#('dd-sg_dp_st_auto', 		oi.DualDecompositionSubgradient 	   	 , Ip(decompositionId='spanningtrees',stepsizeStride=1.0,subProbParam=(True,False)) )	,
		#('dd-sg_dp_st', 		oi.DualDecompositionSubgradient 	   	 , Ip(decompositionId='tree') )	,

		#('self-fusion-dd-sg_dp_t', 		oi.SelfFusion 	   	 , Ip(toFuseInf='gibbs'   ,fusionSolver='bp_lf_fusion', infParam=Ip( ) )	,
		('self-fusion-dd-sg_dp_st_auto_LF_2', 		oi.SelfFusion 	   	 , Ip(toFuseInf='dd_sg_dp',maxSubgraphSize=2   ,fusionSolver='lf_fusion', infParam=Ip(maximalNumberOfIterations=300,decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) ) )	,
		('self-fusion-dd-sg_dp_st_auto_QPBO', 		oi.SelfFusion 	   	 , Ip(toFuseInf='dd_sg_dp',maxSubgraphSize=2   ,fusionSolver='qpbo_fusion', infParam=Ip(maximalNumberOfIterations=300,decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) ) )	,
		#('self-fusion-dd-sg_dp_st_auto_LF_3', 		oi.SelfFusion 	   	 , Ip(toFuseInf='dd_sg_dp',maxSubgraphSize=3   ,fusionSolver='lf_fusion', infParam=Ip(maximalNumberOfIterations=30,decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) ) )	,
		#('self-fusion-dd-sg_dp_st_auto_BP_LF_2', 		oi.SelfFusion 	  , Ip(toFuseInf='dd_sg_dp',maxSubgraphSize=2   ,fusionSolver='bp_lf_fusion', infParam=Ip(maximalNumberOfIterations=30,decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) ) )	,
		#('self-fusion-dd-sg_dp_st_auto_BP_LF_3', 		oi.SelfFusion 	  , Ip(toFuseInf='dd_sg_dp',maxSubgraphSize=3   ,fusionSolver='bp_lf_fusion', infParam=Ip(maximalNumberOfIterations=30,decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) ) )	,
		#('self-fusion-dd-sg_dp_st_auto_QPBO', 		oi.SelfFusion 	   	 , Ip(toFuseInf='dd_sg_dp'   ,fusionSolver='qpbo_fusion', infParam=Ip(maximalNumberOfIterations=30,decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) ) )	,
		#('self-fusion-dd-sg_dp_st_auto', 		oi.SelfFusion 	   	 , Ip(toFuseInf='dd_sg_dp'   ,fusionSolver='lf_fusion', infParam=Ip(decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) ) )	,
		#('self-fusion-dd-sg_dp_st', 		oi.SelfFusion 	   	 , Ip(toFuseInf='dd_sg_dp'   ,fusionSolver='lf_fusion', infParam=Ip(decompositionId='spanningtrees',stepsizeStride=3.0) ) )	,
		#('loc-block-0.1', 					oi.Loc,					Ip(steps=1000000,solver='ad3',maxRadius=100,maxSubgraphSize=60,phi=0.1) ) ,
		#('loc-tree-0.1', 					oi.Loc,					Ip(autoStop=0,pFastHeuristic=0,steps=500000,solver='dp',maxRadius=500000,maxSubgraphSize=400000,phi=0.1) ) , 
		#('loc-block-0.5', 					oi.Loc,					Ip(steps=2000,solver='ad3',maxRadius=10,maxSubgraphSize=100,phi=0.5) ) ,
		#('loc-tree-0.5', 					oi.Loc,					Ip(autoStop=1000,pFastHeuristic=0,steps=100000,solver='dp',maxRadius=5000,maxSubgraphSize=1500,phi=0.5) ) , 
		#('sf-bp', 					oi.SelfFusion, 			Ip(fuseNth=2,toFuseInf='bp'   ,fusionSolver='cplex_fusion', infParam=Ip(steps=50,damping=0.95) ) )	,
		#('bp', 					oi.BeliefPropagation, 			Ip(steps=10,damping=0.0) )	,
		#('lf2', 					oi.LazyFlipper,			Ip(maxSubgraphSize=1) ) ,
		#('lf3', 					oi.LazyFlipper,			Ip(maxSubgraphSize=3) ) ,
		#('lf4', 					oi.LazyFlipper,			Ip(maxSubgraphSize=4) ) ,
		#('icm',   oi.Icm, 			Ip() ) ,
		#('lf3-loc',  oi.ChainedInf,	
		#	Ip(
		#		solvers=(oi.LazyFlipper,oi.Loc),
		#		parameters=(Ip(maxSubgraphSize=3),Ip(steps=s,solver='lf2',maxBlockRadius=5,maxTreeRadius=60,phi=0.0001,treeRuns=10))
		#	) 
		#),
		#('lf4-loc',  oi.ChainedInf,	
		#	Ip(
		#		solvers=(oi.LazyFlipper,oi.Loc,oi.LazyFlipper),
		#		parameters=(Ip(maxSubgraphSize=4),Ip(steps=s,solver='lf2',maxBlockRadius=4,maxTreeRadius=600,phi=0.0001,treeRuns=10),Ip(maxSubgraphSize=4))
		#	) 
		#),
	

	]

	outFolder = '/home/tbeier/Desktop/benchmark_result'

	result=runBenchmark(fNames=fNames,solvers=solvers,dataSetName='geo-surf-3',outFolder=outFolder,plot=True)
	print result

