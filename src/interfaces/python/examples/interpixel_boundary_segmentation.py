import opengm
import numpy 
import sys

class TopologicalCoordinateToIndex:
   def __init__(self ,geometricGridSize) :
      self.gridSize=geometricGridSize
   def convert(self, tx,ty) :
      return tx / 2 + (ty / 2)*(self.gridSize[0]) + ((ty + ty % 2) / 2)*(self.gridSize[0] - 1)
      
def printSolution(data,solution,coordinateHelper):
   for  x   in range (data.shape[0]*2 - 1):
      sys.stdout.write("___")
   sys.stdout.write("\n")   
   for  y   in range (data.shape[1]*2 - 1):
      sys.stdout.write("|")
      for  x   in range (data.shape[0]*2 - 1):
         if x % 2 == 0 and y % 2 == 0:
            sys.stdout.write("   ")
         elif x % 2 == 0 and y % 2 == 1 :
            if solution[coordinateHelper.convert(x,y)]==1:
               sys.stdout.write("___")
            else:
               sys.stdout.write("   ")
         elif   x % 2 == 1 and y % 2 == 0  : 
            if solution[coordinateHelper.convert(x,y)]==1:
               sys.stdout.write(" | ")
            else:    
               sys.stdout.write("   ")
         elif x % 2 == 1 and y % 2 == 1:
               sys.stdout.write(" * ")
      sys.stdout.write("|\n")
   for  x   in range (data.shape[1]*2 - 1):
      sys.stdout.write("___")
   sys.stdout.write("\n")   
                 
# model parameter
gridSize=[3,3] # size of grid
beta=0.7     # bias to choose between under- and over-segmentation   
high=100       # closedness-enforcing soft-constraint value for forbidden configurations

# size of the topological grid
tGridSize=[2*gridSize[0] -1,2*gridSize[1] -1]
nrOfVariables=gridSize[1]*(gridSize[0]-1)+gridSize[0]*(gridSize[1]-1)
cToVi=TopologicalCoordinateToIndex(gridSize)
# some random data on a grid
data=numpy.random.random(gridSize[0]*gridSize[1]).astype(numpy.float32).reshape(gridSize[0],gridSize[1])
# construct gm
numberOfLabels=numpy.ones(nrOfVariables,dtype=opengm.label_type)*2
gm=opengm.graphicalModel(numberOfLabels)

# 4th closedness-function
fClosedness=numpy.zeros( pow(2,4),dtype=numpy.float32).reshape(2,2,2,2)
for x1 in range(2):
   for x2 in range(2):
      for x3 in range(2):
         for x4 in range(2):
            labelsum=x1+x2+x3+x4
            if labelsum is not 2 and labelsum is not 0 :
               fClosedness[x1,x2,x3,x4]=high          
fidClosedness=gm.addFunction(fClosedness)
# for each boundary in the grid, i.e. for each variable 
# of the model, add one 1st order functions 
# and one 1st order factor
# and for each junction of four inter-pixel edges on the grid, 
# one factor is added that connects the corresponding variable 
# indices and refers to the closedness-function
for yt in range(tGridSize[1]):
   for xt in range(tGridSize[0]):
      # unaries
      if (xt % 2 + yt % 2) == 1 :
         gradient = abs(  data[xt / 2, yt / 2]- data[xt/ 2 + xt % 2, yt / 2 + yt % 2])
         f=numpy.array([beta*gradient , (1.0-beta)*(1.0-gradient)])
         gm.addFactor(gm.addFunction(f),[cToVi.convert(xt,yt)])
      # high order factors (4.th order)   
      if xt % 2 + yt % 2 == 2 :
         vi=[cToVi.convert(xt + 1, yt),cToVi.convert(xt - 1, yt),cToVi.convert(xt, yt + 1), cToVi.convert(xt, yt - 1)]
         vi=sorted(vi);
         gm.addFactor(fidClosedness,vi)

inf=opengm.inference.LazyFlipper(gm,parameter=opengm.InfParam(maxSubgraphSize=4))
inf.inference.infer()
arg=inf.inference.arg()
printSolution(data,arg,cToVi)
