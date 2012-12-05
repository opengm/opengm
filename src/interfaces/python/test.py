import numpy
import opengm

def makeGrid(dimx,dimy,labels,beta,acc="min"):
   nos=numpy.ones(dimx*dimy,dtype=numpy.uint64)*labels
   if acc=="min":
      gm=opengm.adder.GraphicalModel(nos)
   else:
      gm=opengm.multiplier.GraphicalModel(nos)
   for vi in range(dimx*dimy):
      f1=numpy.random.random((labels,)).astype(numpy.float32)
      fid1=gm.addFunction(f1)
      gm.addFactor(fid1,(vi,))
   f2=numpy.ones(labels*labels,dtype=numpy.float32).reshape(labels,labels)*beta
   for l in range(labels):
      f2[l,l]=0
   fid2=gm.addFunction(f2)
   for y in range(dimy):   
         for x in range(dimx):
            if x+1 <dimx-1:
               gm.addFactor(fid2, [x+y*dimx,x+1+y*dimx])
            if y+1 <dimy-1:
               gm.addFactor(fid2, [x+y*dimx,x+(y+1)*dimx])
   return gm 
   
def checkSolution(gm,argOpt,arg,acc="min",tolerance=None,check=True):
   valOpt=gm.evaluate(argOpt)
   val=gm.evaluate(arg)
   numtol=0.00000000001
   if check :
      if acc=="min":
         if tolerance==None:
            tol=numtol
            assert(val-tol <= valOpt)
         else :
            tol=(valOpt)*tolerance
            assert(val-tol <= valOpt)
      if acc=="max":
         if tolerance==None:
            tol=numtol
            assert(val-tol >= valOpt)
         else :
            tol=(valOpt)*tolerance + numtol
            assert(val-tol >= valOpt)
            
def checkInference(gm,solver ,argOpt, optimal=False,tolerance=None,acc="min"):
   solver.infer()
   arg=solver.arg()
   checkSolution(gm,argOpt,arg,acc,tolerance,optimal)

class TestHdf5:
   def test_hdf5(self):
      assert(True)      
class TestUtilities:
   def test_vector(self):
      assert(True)
   def test_enums(self):
      assert(True)
class TestSpace:
   def test_space(self):
      assert(True)     
class TestGm:     
   def test_constructor_numpy(self):
      numberOfStates=numpy.ones(3,dtype=numpy.uint64)
      numberOfStates[0]=2
      numberOfStates[1]=3
      numberOfStates[2]=4
      gm=opengm.graphicalModel(numberOfStates)
      assert(gm.numberOfVariables==3)
      assert(gm.numberOfLabels(0)==2)
      assert(gm.numberOfLabels(1)==3)
      assert(gm.numberOfLabels(2)==4)
   def test_constructor_list(self):
      numberOfStates=[2,3,4]
      gm=opengm.graphicalModel(numberOfStates,operator="adder")
      assert(gm.numberOfVariables==3)
      assert(gm.numberOfLabels(0)==2)
      assert(gm.numberOfLabels(1)==3)
      assert(gm.numberOfLabels(2)==4)
   def test_assign_numpy(self):
      numberOfStates=numpy.ones(3,dtype=numpy.uint64)
      numberOfStates[0]=2
      numberOfStates[1]=3
      numberOfStates[2]=4
      gm=opengm.adder.GraphicalModel()
      gm.assign(numberOfStates)
      assert(gm.numberOfVariables==3)
      assert(gm.numberOfLabels(0)==2)
      assert(gm.numberOfLabels(1)==3)
      assert(gm.numberOfLabels(2)==4)
   def test_assign_list(self):
      numberOfStates=[2,3,4]
      gm=opengm.adder.GraphicalModel()
      gm.assign(numberOfStates)
      assert(gm.numberOfVariables==3)
      assert(gm.numberOfLabels(0)==2)
      assert(gm.numberOfLabels(1)==3)
      assert(gm.numberOfLabels(2)==4)
   def test_copy(self):
      numberOfStates=[2,3,4]
      gmA=opengm.adder.GraphicalModel(numberOfStates)
      gm=gmA
      assert(gm.numberOfVariables==3)
      assert(gm.numberOfLabels(0)==2)
      assert(gm.numberOfLabels(1)==3)
      assert(gm.numberOfLabels(2)==4)
      #TODO add real copy here 
   def test_space(self):
      numberOfStates=[2,3,4]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      assert(gm.space().numberOfVariables==3)
      assert(gm.space()[0]==2)
      assert(gm.space()[1]==3)
      assert(gm.space()[2]==4)
   def test_print(self):
      numberOfStates=[2,3,4]
      gm=opengm.adder.GraphicalModel()
      print gm
      print gm.space()
      gm.assign(numberOfStates)
      print gm 
      print gm.space()
   def test_add_factor_numpy(self):
      numberOfStates=[2,2,2,2]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      assert(gm.space().numberOfVariables==4)
      assert(gm.numberOfFactors==0)
      f1=numpy.ones(2,dtype=numpy.float32).reshape(2)
      f2=numpy.ones(4,dtype=numpy.float32).reshape(2,2)
      if1=gm.addFunction(f1)
      if2=gm.addFunction(f2)
      vis1=numpy.array([0], numpy.uint64)
      vis2=numpy.array([0,1], numpy.uint64)
      gm.addFactor(if1,vis1)
      assert(gm.numberOfFactors==1)
      gm.addFactor(if2,vis2)
      assert(gm.numberOfFactors==2)
      assert(gm[0].variableIndices[0]==0)
      assert(gm[1].variableIndices[0]==0)
      assert(gm[1].variableIndices[1]==1)
   def test_add_factor_tuple(self):
      numberOfStates=[2,2,2,2]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      assert(gm.space().numberOfVariables==4)
      assert(gm.numberOfFactors==0)
      f1=numpy.ones(2,dtype=numpy.float32).reshape(2)
      f2=numpy.ones(4,dtype=numpy.float32).reshape(2,2)
      if1=gm.addFunction(f1)
      if2=gm.addFunction(f2)
      gm.addFactor(if1,(0,))
      gm.addFactor(if2,(0,1))
      gm.addFactor(if1,(long(0),))
      gm.addFactor(if2,(long(0),int(1)))
      assert(gm.numberOfFactors==4)
      assert(gm[0].variableIndices[0]==0)
      assert(gm[1].variableIndices[0]==0)
      assert(gm[1].variableIndices[1]==1)
   def test_add_factor_list(self):
      numberOfStates=[2,2,2,2]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      assert(gm.space().numberOfVariables==4)
      assert(gm.numberOfFactors==0)
      f1=numpy.ones(2,dtype=numpy.float32).reshape(2)
      f2=numpy.ones(4,dtype=numpy.float32).reshape(2,2)
      if1=gm.addFunction(f1)
      if2=gm.addFunction(f2)
      gm.addFactor(if1,[0])
      gm.addFactor(if2,[0,1])
      assert(gm.numberOfFactors==2)
      assert(gm[0].variableIndices[0]==0)
      assert(gm[1].variableIndices[0]==0)
      assert(gm[1].variableIndices[1]==1)
   def test_add_function(self):
      numberOfStates=[2,3,4]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      f1=numpy.ones(6*4, numpy.float32)
      for i in range(2*3*4):
         f1[i]=i
      f1=f1.reshape(2,3,4)
      idf=gm.addFunction(f1)
      gm.addFactor(idf,(0,1,2))
      nf1=gm[0].asNumpy();
      assert(len(f1.shape)==len(nf1.shape))
      for i in range(len(f1.shape)):
         assert(f1.shape[i]==nf1.shape[i])
      for k in range(f1.shape[2]):
         for j in range(f1.shape[1]):
            for i in range(f1.shape[0]):
               assert(gm[0][ numpy.array( [i,j,k],dtype=numpy.uint64)  ]==f1[i,j,k])
               assert(gm[0][(i,j,k)]==f1[i,j,k])
               assert(gm[0][(i,j,k)]==nf1[i,j,k])
   def test_evaluate(self):
      numberOfStates=[2,2,2,2]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      f1=numpy.ones(2,dtype=numpy.float32).reshape(2)
      f2=numpy.ones(4,dtype=numpy.float32).reshape(2,2)
      for i in range(3):
         gm.addFactor(gm.addFunction(f1),[i])
      for i in range(2):
         gm.addFactor(gm.addFunction(f2),[i,i+1])
      sequenceList=[0,1,0,1]
      valueList=gm.evaluate(sequenceList)
      assert(float(valueList)==float(gm.numberOfFactors))
      sequenceNumpy=numpy.array([0,1,0,1],dtype=numpy.uint64)
      valueNumpy=gm.evaluate(sequenceNumpy)
      assert(float(valueNumpy)==float(gm.numberOfFactors))
      assert(float(valueNumpy)==float(valueList)) 
class TestFactor:     
   def test_factor_shape(self):
      numberOfStates=[2,3,4]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      f1=numpy.ones(6*4, numpy.float32).reshape(2,3,4)
      idf=gm.addFunction(f1)
      gm.addFactor(idf,(0,1,2))
      nf1=gm[0].asNumpy();
      for i in range(3):
         assert(gm[0].shape[i]==numberOfStates[i])
         assert(gm[0].shape.asNumpy()[i]==numberOfStates[i])
         assert(gm[0].shape.asList()[i] ==numberOfStates[i])
         assert(gm[0].shape.asTuple()[i]==numberOfStates[i])
   def test_factor_vi(self):
      numberOfStates=[2,3,4]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      f1=numpy.ones(6*4, numpy.float32).reshape(2,3,4)
      idf=gm.addFunction(f1)
      gm.addFactor(idf,(0,1,2))
      nf1=gm[0].asNumpy();
      for i in range(3):
         assert(gm[0].variableIndices[i]  == i)
         assert(gm[0].variableIndices.asNumpy()[i]==i)
         assert(gm[0].variableIndices.asList()[i] ==i)
         assert(gm[0].variableIndices.asTuple()[i]==i)
   def test_factor_properties(self):
      numberOfStates=[2,2,2,2]
      gm=opengm.adder.GraphicalModel(numberOfStates)
      assert(gm.space().numberOfVariables==4)
      assert(gm.numberOfFactors==0)
      f1=numpy.array([2,3], numpy.float32)
      f2=numpy.array([1,2,3,4], numpy.float32).reshape(2,2)
      if1=gm.addFunction(f1)
      if2=gm.addFunction(f2)
      gm.addFactor(if1,(0,))
      gm.addFactor(if2,(0,1))
      nf0=gm[0].asNumpy()
      nf1=gm[1].asNumpy()
      for i in range(f1.shape[0]):
         assert(nf0[i]==gm[0][(i,)])
         assert(nf0[i]==f1[i])
      for i in range(f2.shape[0]):
         for j in range(f2.shape[1]):
            assert(nf1[i,j]==gm[1][(i,j)])
            assert(nf1[i,j]==f2[i,j])       
      assert(gm[0].min()==2)
      assert(gm[0].max()==3)  
      assert(gm[0].sum()==5)
      assert(gm[0].product()==6)
      assert(gm[0][(0,)] == 2)
      assert(gm[0][(1,)] == 3)
      assert(gm[1].min()==1)
      assert(gm[1].max()==4)  
      assert(gm[1].sum()==1+2+3+4)
      assert(gm[1].product()==1*2*3*4)
class TestInference:
   def __init__(self):
      self.gm=makeGrid(3,4,2,0.8)
      param=opengm.inference.adder.minimizer.BruteforceParameter( )
      inf=opengm.inference.adder.minimizer.Bruteforce(self.gm,param)
      inf.infer()
      self.argOpt=inf.arg()
   def runAlgTester(self,algName,optimal=False,parameter=None):
      if parameter is None:
         parameter=opengm.inferenceParameter(self.gm,alg=algName)
      inf=opengm.inferenceAlgorithm(self.gm,alg=algName,parameter=parameter)
      checkInference(self.gm,inf,self.argOpt,optimal=optimal)
   def test_bp(self):
      infname='bp'
      param=opengm.inferenceParameter(self.gm,alg=infname)
      param.set(steps=100,damping=0.5,isAcyclic=opengm.Tribool.false)
      param.isAcyclic=opengm.Tribool.false
      self.runAlgTester(infname,parameter=param)
   def test_trbp(self):
      infname='trbp'
      param=opengm.inferenceParameter(self.gm,alg=infname)
      param.set(isAcyclic=opengm.Tribool.false,steps=100)
      param.isAcyclic=opengm.Tribool.maybe
      self.runAlgTester(infname,parameter=param)
   def test_icm(self):
      param=opengm.inferenceParameter(self.gm,alg='icm')
      param.moveType=opengm.IcmMoveType.variable
      self.runAlgTester('icm',parameter=param)
      param.moveType=opengm.IcmMoveType.factor
      self.runAlgTester('icm',parameter=param)
   def test_gibbs(self):
      infname='gibbs'
      param=opengm.inferenceParameter(self.gm,alg=infname)
      param.set(steps=2000)
      param.isAcyclic=opengm.Tribool.maybe
      self.runAlgTester(infname,parameter=param)
   def test_astar(self):
      infname='astar'
      param=opengm.inferenceParameter(self.gm,alg=infname)
      param.set(heuristic=opengm.AStarHeuristic.standard,maxHeapSize=2000000,numberOfOpt=1)
      self.runAlgTester(infname,optimal=True,parameter=param)
      param.heuristic=opengm.AStarHeuristic.fast
      self.runAlgTester(infname,optimal=True,parameter=param)
      param.heuristic=opengm.AStarHeuristic.default
      self.runAlgTester(infname,optimal=True,parameter=param) 
   def test_loc(self):
      self.runAlgTester('loc')
   def test_lazyflipper(self):
      param=opengm.inferenceParameter(self.gm,alg='lf')
      param.maxSubgraphSize=2
      self.runAlgTester('lf',parameter=param)
      param.maxSubgraphSize=3
      self.runAlgTester('lf',parameter=param) 
   def test_graphcut(self):
      self.runAlgTester('gc',optimal=True)
   def test_alpha_beta_swap(self):
      self.runAlgTester('ab-swap')
   def test_alpha_expansion(self):
      self.runAlgTester('ae')


if opengm.configuration.withLibdai:
   class TestLibdaiWrapper:
      def __init__(self):
         self.gm=makeGrid(3,4,2,0.8)
         param=opengm.inference.adder.minimizer.BruteforceParameter( )
         inf=opengm.inference.adder.minimizer.Bruteforce(self.gm,param)
         inf.infer()
         self.argOpt=inf.arg()
      def runAlgTester(self,algName,optimal=False,parameter=None):
         if parameter is None:
            parameter=opengm.inferenceParameter(self.gm,alg=algName)
         inf=opengm.inferenceAlgorithm(self.gm,alg=algName,parameter=parameter)
         checkInference(self.gm,inf,self.argOpt,optimal=optimal)
      def test_bp(self):
         infname='libdai-bp'
         param=opengm.inferenceParameter(self.gm,alg=infname)
         param.set(steps=100,updateRule=opengm.BpUpdateRule.parall,tolerance=0.0001,damping=0.01)
         self.runAlgTester(infname,parameter=param)
      def test_fractional_bp(self):
         infname='libdai-fbp'
         param=opengm.inferenceParameter(self.gm,alg=infname)
         param.set(steps=100,updateRule=opengm.BpUpdateRule.parall,tolerance=0.0001,damping=0.01)
         self.runAlgTester(infname,parameter=param)
      def test_trbp(self):
         infname='libdai-trbp'
         param=opengm.inferenceParameter(self.gm,alg=infname)
         param.set(steps=100,updateRule=opengm.BpUpdateRule.parall,tolerance=0.0001,damping=0.01,ntrees=0)
         self.runAlgTester(infname,parameter=param)
      def test_junction_tree(self):
         infname='libdai-jt'
         param=opengm.inferenceParameter(self.gm,alg=infname)
         param.set()
         self.runAlgTester(infname,parameter=param)
         param.set(updateRule=opengm.JunctionTreeUpdateRule.shsh,heuristic=opengm.JunctionTreeHeuristic.minfill)
         
if opengm.configuration.withCplex:
   class TestCplex:
      def __init__(self):
         self.gm=makeGrid(3,4,2,0.8)
         param=opengm.inference.adder.minimizer.BruteforceParameter( )
         inf=opengm.inference.adder.minimizer.Bruteforce(self.gm,param)
         inf.infer()
         self.argOpt=inf.arg()
      def runAlgTester(self,algName,optimal=False,parameter=None):
         if parameter is None:
            parameter=opengm.inferenceParameter(self.gm,alg=algName)
         inf=opengm.inferenceAlgorithm(self.gm,alg=algName,parameter=parameter)
         checkInference(self.gm,inf,self.argOpt,optimal=optimal)
      def test_cplex(self):
         infname='lpcplex'
         param=opengm.inferenceParameter(self.gm,alg=infname)
         #param.set(steps=100,updateRule=opengm.BpUpdateRule.parall,tolerance=0.0001,damping=0.01)
         self.runAlgTester(infname,parameter=param)