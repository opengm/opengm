import opengm
import numpy 

def myFunc( labels):
    p=1.0
    s=0
    for l in labels:
        p*=l
        s+=l
    return p+s


pf=opengm.PythonFunction(myFunc,[2,2])

print pf.shape
print pf[0,0]
print pf[1,1]
for c in opengm.shapeWalker(pf.shape):
    print c," ",pf[c]



unaries=numpy.random.rand(5 , 5,2)
potts=opengm.PottsFunction([2,2],0.0,0.1)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)


viewFunction=opengm.modelViewFunction(gm[0])
for c in opengm.shapeWalker(viewFunction.shape):
    print c," ",viewFunction[c]

viewFunction=opengm.modelViewFunction(gm[25])
print viewFunction.shape
print gm[25].shape

for c in opengm.shapeWalker(viewFunction.shape):
    print c," ",viewFunction[c]



print gm.numberOfLabels(0)