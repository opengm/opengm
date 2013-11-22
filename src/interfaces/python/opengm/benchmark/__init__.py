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

