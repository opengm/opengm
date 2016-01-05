import opengm
import vigra
import numpy
import sys



if __name__ == "__main__":
	args = sys.argv

	if len(args)!=8 :
		print "Usage: ",  args[0] , " infile outfile red green blue T lambda"
		sys.exit(0)

	img = vigra.readImage(args[1])

	if img.shape[2]!=3:
		print "Image must be RGB"
		sys.exit(0)

	T 	 = float(args[6])
	beta = float(args[7])

	

	imgFlat = img.reshape([-1,3]).view(numpy.ndarray)
	numVar  = imgFlat.shape[0]


	gm = opengm.gm(numpy.ones(numVar,dtype=opengm.label_type)*2)

	protoColor = numpy.array([args[3],args[4],args[5]],dtype=opengm.value_type).reshape([3,-1])
	protoColor = numpy.repeat(protoColor,numVar,axis=1).swapaxes(0,1)
	diffArray  = numpy.sum(numpy.abs(imgFlat - protoColor),axis=1)
	unaries    = numpy.ones([numVar,2],dtype=opengm.value_type)
	unaries[:,0]=T
	unaries[:,1]=diffArray

	print diffArray

	gm.addFactors(gm.addFunctions(unaries),numpy.arange(numVar))


	regularizer=opengm.pottsFunction([2,2],0.0,beta)
	gridVariableIndices=opengm.secondOrderGridVis(img.shape[0],img.shape[1])

	fid=gm.addFunction(regularizer)
	gm.addFactors(fid,gridVariableIndices)

	print gm

	inf=opengm.inference.GraphCut(gm)
	inf.infer()
	arg=inf.arg().reshape(img.shape[0:2])

	vigra.impex.writeImage(args[2])