import opengm
import numpy
shape=[10,10]
unaries=numpy.random.rand(shape[0], shape[1],2)
potts=opengm.PottsFunction([2,2],0.0,0.2)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)


inf=opengm.inference.Icm(gm)


class PyCallback(object):
    def __init__(self):
        pass
    def begin(self,inference):
        print "begin"
    def end(self,inference):
        print "end"
    def visit(self,inference):
        print "visit"
        arg=inference.arg()
        gm=inference.gm()
        print "energy ",gm.evaluate(arg)


callback=PyCallback()
visitor=inf.pythonVisitor(callback,visitNth=1)



inf.infer(visitor)
# get the result states
argmin=inf.arg()
print "argminEnergy",gm.evaluate(argmin)
# print the argmin (on the grid)
#print argmin.reshape(shape)
