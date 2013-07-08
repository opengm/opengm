import opengm
import numpy

gm=opengm.gm([2,2,3,3,4,4,4],operator='adder')
functionIds=[]

#---------------------------------------------------------------
# Numpy Ndarray
# (is stored in a different multi array function within opengm)
#---------------------------------------------------------------

f=numpy.random.rand(2,2,3,4)
fid=gm.addFunction(f)
gm.addFactor(fid,[0,1,2,4])
print "\nexplicit function: \n",f

#---------------------------------------------------------------
# Sparse Function
#--------------------------------------------------------------- 

# fill sparse function "by hand"
f=opengm.SparseFunction(shape=[3,4,4],defaultValue=1)
# fill diagonale with zeros
for d in xrange(4):
    f[[d,d,d]]=0
print "\nsparse function: \n",f
fid=gm.addFunction(f)
functionIds.append(fid)
gm.addFactor(fid,[3,4,5])

# fill sparse function from dense function
f=opengm.SparseFunction()
f.assignDense(numpy.identity(4),defaultValue=0)
fid=gm.addFunction(f)
functionIds.append(fid)
gm.addFactor(fid,[4,5])
print "\nsparse function: \n",f


#---------------------------------------------------------------
# Potts Function
#--------------------------------------------------------------- 
f=opengm.PottsFunction(shape=[2,4],valueEqual=0.0,valueNotEqual=1.0)
fid=gm.addFunction(f)
functionIds.append(fid)
gm.addFactor(fid,[0,5])
print "\npotts function: \n",f

#---------------------------------------------------------------
# Truncated Absolute Difference Function
#--------------------------------------------------------------- 
f=opengm.TruncatedAbsoluteDifferenceFunction(shape=[3,4],truncate=2,weight=0.2,)
fid=gm.addFunction(f)
functionIds.append(fid)
gm.addFactor(fid,[2,5])
print "\ntruncated absolute difference function: \n",f


#---------------------------------------------------------------
# Truncated  Squared Difference Function
#--------------------------------------------------------------- 
f=opengm.TruncatedSquaredDifferenceFunction(shape=[3,4],truncate=2,weight=2.0)
fid=gm.addFunction(f)
functionIds.append(fid)
gm.addFactor(fid,[2,5])
print "\ntruncated  squared difference function: \n",f

for factor,factorIndex in gm.factorsAndIds():
    print "\ngm[",factorIndex,"] : ",factor
    print "Value Table: \n",numpy.array(factor)