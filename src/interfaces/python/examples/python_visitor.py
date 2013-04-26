import opengm
import numpy
shape=[40,40]
unaries=numpy.random.rand(shape[0], shape[1],2).astype(numpy.float32)
potts=opengm.PottsFunction([2,2],0.0,0.4)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)


inf=opengm.inference.Icm(gm)


class PyCallback(object):
    def __init__(self,shape):
        self.shape=shape
    def begin(self,inference):
        self.visitNr=1
        self.gm=inference.gm()
        self.labelVector=opengm.LabelVector()
        self.labelVector.resize(self.gm.numberOfVariables)

    def end(self,inference):
        print "end"
    def visit(self,inference):
        self.labelVector=inference.arg(output=self.labelVector,returnAsVector=True)
        print "energy ",self.gm.evaluate(self.labelVector)
        self.visitNr+=1


callback=PyCallback(shape)
visitor=inf.pythonVisitor(callback,visitNth=1)



inf.infer(visitor)
# get the result states
argmin=inf.arg()
print "argminEnergy",gm.evaluate(argmin)
# print the argmin (on the grid)
#print argmin.reshape(shape)
