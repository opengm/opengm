import opengm
import numpy
import matplotlib.pyplot as plt
import sys



numpy.random.seed(42)

nLabels =  3
shape 	=  [100,10]
nVar 	=  shape[0]*shape[1]



gm = opengm.TestModels.chain3(nVar=nVar,nLabels=nLabels)


print(gm)


fusionMover=opengm.inference.adder.minimizer.FusionMover(gm)

sa=numpy.random.randint(low=0, high=nLabels, size=nVar).astype(opengm.label_type)

for x in range(10):
	sb=numpy.random.randint(low=0, high=nLabels, size=nVar).astype(opengm.label_type)
	r = fusionMover.fuse(sa,sb,'lf2')
	sa=r[0]
	print(r[1],r[2],r[3])

#print sa
#print sb
#print (sa!=sb).astype(opengm.label_type)

#r=fusionMover.fuse(sa,sb,'qpbo')

print(r)