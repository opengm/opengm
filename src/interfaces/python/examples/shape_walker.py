import opengm
import numpy

# some graphical model 
# -with 3 variables with 4 labels.
# -with 2 random 2-order functions
# -connected to 2 factors
gm=opengm.gm([4]*3)
f=numpy.random.rand(4,4).astype(numpy.float32)
gm.addFactor(gm.addFunction(f),[0,1])
f=numpy.random.rand(4,4).astype(numpy.float32)
gm.addFactor(gm.addFunction(f),[1,2])

# iterate over all factors of the graphical model
for factor in gm.factors():
    # iterate over all labelings with a "shape walker"
    for coord in opengm.shapeWalker(f.shape):
        print " f[",coord,"]=",factor[coord]
