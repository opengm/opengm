class Inference:
   """
      High level Inference Interface Class
   """
   def __init__(self,gm,alg,impl=None,parameter=None,accumulator=None,constructSolver=True):
      """
      High level Inference Interface

      Keyword arguments:
      
      gm -- the graphical model to optimize
      
      alg -- algorithm  name as string :
         for example:
         * ``\'graph-cut\'`` / ``\'gc\'``
         A complete list is given below in an automatic generated documentation

      impl -- Implementation of the solver:
         this argument depends on the choise of ``alg``.

         For ``alg=\'gc\'`` ``impl`` might be ``boost-push-relabel``.

         A complete list is given below in an automatic generated documentation
      
      parameter -- Parameter for the inference algorithm 
         this argument depends on the choise of ``alg`` and ``impl``

         A complete list of the parameters for each algorithm (and implementation)

         is given below in an automatic generated documentation
      accumulator -- accumulator of the inference algorithm:
         * ``\'minimizer\'``  (default for a gm with an Adder as operator)
         * ``\'maximizer\'``  (default for a gm with a Multiplier as operator)

      Construct a a bp solver and optimize the model::
      
         #get an instance of the optimizer / inference-algorithm
         inf=opengm.Inference(gm,alg='bp',impl='opengm',parameter=opengm.InfParam(steps=10,damping=0.5,convergenceBound=0.001))
         # start inference (in this case unverbose infernce)
         inf.infer(verbose=False)
         # get the result states
         argmin=inf.arg()

      Jump to the automatic generated documentation for :
         - Min-Sum_
            - adder-minimizer-icm_
         - Max-Sum_
         - Min-Prod_
         - Max-Prod_

      Automatic generated documentation:
      """
      self.gm=gm
      self.accumulator=accumulator
      self.cppParameter=None
      self.inference=None
      if accumulator is None :
         self.accumulator = defaultAccumulator(gm)
      possibleInferenceFamilyClasses=_CppInferenceAlgorithms.inferenceDict[self.gm.operator][self.accumulator]
      if alg in possibleInferenceFamilyClasses :
         inferenceFamilyClasses=possibleInferenceFamilyClasses[alg]
      else:
         supportedAlgorithmsStr=_CppInferenceAlgorithms.supportedAlgorithmsStr(gm.operator,accumulator)
         raise NameError("alg "+alg + " is unknon:\n")

      # try to get impl (if impl is not given)
      if impl is None:
         if len(inferenceFamilyClasses)==1:
            for key in inferenceFamilyClasses:
               solverItem=inferenceFamilyClasses[key]
               self.cppInferenceClass=solverItem[0]
               self.cppParameterClass=solverItem[1]
         elif  'opengm' in inferenceFamilyClasses:
               solverItem=inferenceFamilyClasses['opengm']
               self.cppInferenceClass=solverItem[0]
               self.cppParameterClass=solverItem[1]
         else:
            for key in inferenceFamilyClasses:
               solverItem=inferenceFamilyClasses[key]
               self.cppInferenceClass=solverItem[0]
               self.cppParameterClass=solverItem[1]
               break
      # impl is given
      else :
         if impl in inferenceFamilyClasses :
            solverItem=inferenceFamilyClasses[impl]
            self.cppInferenceClass=solverItem[0]
            self.cppParameterClass=solverItem[1]
         else:
            supportedAlgorithmsStr=_CppInferenceAlgorithms.supportedAlgorithmsStr(gm.operator,accumulator)
            raise NameError("impl "+impl + " is unknon:\n")

      #set up parameter
      if parameter is not None:
         self.cppParameter=to_native_class_converter(givenValue=parameter,nativeClass=self.cppParameterClass)
         """
         # set up parameter
         if isinstance(parameter,InfParam):
            # parameter is given as mata parameter calls (InfParam)
            self.cppParameter=_paramFromMetaParam(metaParam=parameter,paramClass=self.cppParameterClass)

            self.cppParameter=to_native_class_converter(givenValue=parameter,nativeClass=self.cppParameterClass)

         elif isinstance(parameter,self.cppParameterClass): 
            self.cppParameter = parameter
         else:
            TypeError("wrong parameter class:  try: "+ str(self.cppParameterClass))
         if constructSolver :
            self.inference=self.cppInferenceClass(gm,self.cppParameter)
         """
      else :
         self.cppParameter=self.cppParameterClass()
         self.cppParameter.set()

      if constructSolver :
         self.inference=self.cppInferenceClass(gm,self.cppParameter)

   def verboseVisitor(self,printNth=1,multiline=True):
      return self.inference.verboseVisitor(printNth,multiline)

   def pythonVisitor(self,callbackObject,visitNth):
      return self.inference.pythonVisitor(callbackObject,visitNth)

   def infer(self,visitor=None):
      if visitor is None:
         return self.inference.infer()
      else:
         return self.inference.infer(visitor)

   def arg(self,returnAsVector=False,output=None):
      return self.inference.arg(returnAsVector,output)

   def name(self):
      return self.inference.name()