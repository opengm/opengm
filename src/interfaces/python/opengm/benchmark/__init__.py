import opengm
import os
import matplotlib.pyplot as plt
import numpy
from matplotlib import pyplot
from matplotlib import pylab

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

def runBenchmark(fNames,solvers):

	nFiles  = len(fNames)
	nSolver = len(solvers)


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

		pr=[]
		names=[]
		#fig, ax = plt.subplots()

		for sNr,solver in enumerate(solvers) :
			(sName,sClass,sParam)=solver
			print sName
			inf=sClass(gm=gm,parameter=sParam)
			tv=inf.timingVisitor(verbose=True,multiline=False,visitNth=1)

			inf.infer(tv)
			pr.append(plotInfRes(tv))
			print sName
			names.append(sName)


		print names
		plt.legend( names,loc= 5)
		#plt.legend(pr,names)
		plt.show()





if __name__ == '__main__':

	import opengm

	
	#fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/geo-surf-3/')
	#fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/cell-tracking/')
	fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/color-seg/')
	#fNames  =  filenamesFromDir('/home/tbeier/Desktop/models/mrf-inpainting/')
	#print fNames

	Ip = opengm.InfParam
	oi = opengm.inference 




	#gms = [ opengm.TestModels.secondOrderGrid(100,100,25) ]

	#pyplot.yscale('log')
	#pyplot.xscale('log')

	#print help(oi.DualDecompositionSubgradient)
	s=2000
	solvers =[

		('loc-block-', oi.Loc,Ip(steps=s,solver='ad3',maxBlockRadius=4,maxTreeRadius=0,phi=0.0001) ) ,
		('loc-tree-20', oi.Loc,Ip(steps=s*4,solver='ad3',maxBlockRadius=0,maxTreeRadius=20,phi=0.0001) ) ,
		('loc-block-tree-20', oi.Loc,Ip(steps=s,solver='ad3',maxBlockRadius=4,maxTreeRadius=20,phi=0.0001) ) ,
		#('loc-tree-40', oi.Loc,Ip(steps=s/3,solver='ad3',maxBlockRadius=0,maxTreeRadius=40,phi=0.0001) ) ,
		#('loc-tree-100', oi.Loc,Ip(steps=s,solver='ad3',maxBlockRadius=0,maxTreeRadius=100,phi=0.1) ) ,
		#('loc-block-tree-40', oi.Loc,Ip(steps=s,solver='ad3',maxBlockRadius=4,maxTreeRadius=40,phi=0.0001) )

		#('self-fusion-gibbs', oi.SelfFusion 	   	 , Ip(fuseNth=1000,toFuseInf='gibbs',fusionSolver='bp_lf_fusion', infParam=Ip(steps=10000000,tempMin=0.00001,tempMax=1.00) ) )	,
		
		#('dd-sg_dp_st', 		oi.DualDecompositionSubgradient 	   	 , Ip(decompositionId='spanningtrees',numberOfBlocks=2) )	,
		#('dd-sg_dp_st_auto', 		oi.DualDecompositionSubgradient 	   	 , Ip(decompositionId='spanningtrees',stepsizeStride=1.0,subProbParam=(True,False)) )	,
		#('dd-sg_dp_st', 		oi.DualDecompositionSubgradient 	   	 , Ip(decompositionId='tree') )	,

		#('self-fusion-dd-sg_dp_t', 		oi.SelfFusion 	   	 , Ip(toFuseInf='gibbs'   ,fusionSolver='bp_lf_fusion', infParam=Ip( ) )	,
		#('self-fusion-dd-sg_dp_st_auto_LF_2', 		oi.SelfFusion 	   	 , Ip(toFuseInf='dd_sg_dp',maxSubgraphSize=2   ,fusionSolver='lf_fusion', infParam=Ip(maximalNumberOfIterations=30,decompositionId='spanningtrees',stepsizeStride=3.0,subProbParam=(True,False) ) ) )	,
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
		

		#('lf2', 					oi.LazyFlipper,			Ip(maxSubgraphSize=2) ) ,
		#('icm',   oi.Icm, 			Ip() ) ,
		#('cinf',  oi.ChainedInf,	
		#	Ip(solvers=(oi.Icm,oi.LazyFlipper),parameters=(Ip(),Ip(maxSubgraphSize=2))) 
		#)
	]


	runBenchmark(fNames=fNames,solvers=solvers)


