import opengm
import vigra
import numpy
import sys



if __name__ == "__main__":
	args = sys.argv

	if len(args)!=11 :
		print "Usage: ",  args[0] , " infile outfile red1 green1 blue1 red2 green2 blue2 T lambda"
		sys.exit(0)

	img = vigra.readImage(args[1])

	if img.shape[2]!=3:
		print "Image must be RGB"
		sys.exit(0)

	T 	 = float(args[9])
	beta = float(args[10])

	

	imgFlat = img.reshape([-1,3]).view(numpy.ndarray)
	numVar  = imgFlat.shape[0]


	gm = opengm.gm(numpy.ones(numVar,dtype=opengm.label_type)*3)

	protoColor1 = numpy.array([args[3],args[4],args[5]],dtype=opengm.value_type).reshape([3,-1])
	protoColor1 = numpy.repeat(protoColor1,numVar,axis=1).swapaxes(0,1)

	protoColor2 = numpy.array([args[6],args[7],args[8]],dtype=opengm.value_type).reshape([3,-1])
	protoColor2 = numpy.repeat(protoColor2,numVar,axis=1).swapaxes(0,1)

	diffArray1 = numpy.sum(numpy.abs(imgFlat - protoColor1),axis=1)
	diffArray2 = numpy.sum(numpy.abs(imgFlat - protoColor2),axis=1)
	unaries    = numpy.ones([numVar,3],dtype=opengm.value_type)
	unaries[:,0]=T
	unaries[:,1]=diffArray1
	unaries[:,2]=diffArray2


	gm.addFactors(gm.addFunctions(unaries),numpy.arange(numVar))


	regularizer=opengm.pottsFunction([3,3],0.0,beta)
	gridVariableIndices=opengm.secondOrderGridVis(img.shape[0],img.shape[1])

	fid=gm.addFunction(regularizer)
	gm.addFactors(fid,gridVariableIndices)

	if True :
		inf=opengm.inference.TrwsExternal(gm)
	else:
		inf=opengm.inference.BeliefPropagation(gm)

	inf.infer()
	arg=inf.arg().reshape(img.shape[0:2])
	vigra.impex.writeImage(arg,args[2])