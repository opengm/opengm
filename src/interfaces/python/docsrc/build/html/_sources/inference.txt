Infer/Optimize a Graphical Model
-------------------------------- 
Getting the inference object and the corresponding parameter object is the same
for all algorithms. The following algorithms are implemented within OpenGM Python

- Belief propagation (``'bp'``) Semirings:  (min ,+), (max,+), (min,*), (max,*) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='bp')
   
- Treereweighted Bp (``'trbp'``) Semirings:  (min ,+), (max,+), (min,*), (max,*) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='trbp')

- ICM (``'icm'``)  (min ,+), (max,+), (min,*), (max,*) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='icm')
 
- Gibbs (``'gibbs'``)  (min ,+), (max,*) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='gibbs')
 
- AStar (``'astar'``) (min ,+), (max,+), (min,*), (max,*) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='astar')
 
- LOC (``'loc'``) (min ,+), (max,+), (min,*), (max,*) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='loc')
 
- Lazy Flipper (``'lf'``):  (min ,+), (max,+), (min,*), (max,*) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='lf')
 
- Graph Cut (``'gc'``):  (min ,+) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='gc')
 
- Alpha-Beta Swap(``'ab-swap'``):  (min ,+) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='ab-swap')
 
- Alpha Expansion (``'a-expansion'``):  (min ,+) ::
   
   inf=opengm.inferenceAlgorithm(gm=gm,alg='a-expansion')
 
   
To infer a gm one need to do the following seps.

Get the parameter::
   
   # get an instance of the inference algorithm's parameter object
   param=opengm.inferenceParameter(self.gm,alg='bp',accumulator='minimizer')
   # the same parameter but for inference with a maximizer as accumulator
   param=opengm.inferenceParameter(self.gm,alg='bp',accumulator='maximizer')
   
If no accumulator is given the most suitable accumulator is choosen.
For a gm with an Adder(+) as operator the accumulator will be a minimizer.
For  =a gm with an Multiplier(*) as operator the accumulator will be a maximizer::
   
   #get an instance of the inference algorithm's parameter object
   param=opengm.inferenceParameter(self.gm,alg='bp')

   
Set up the parameter::
   
   # set up the paraemters (this is different for all the inference algorithms)
   param.set( )

Construct the inference / solver object::

   #get an instance of the optimizer / inference-algorithm
   inf=opengm.inferenceAlgorithm(gm=gm,alg='bp',parameter=param,accumulator='minimizer')
   
If no accumulator is given the most suitable accumulator is choosen.
For a gm with an Adder(+) as operator the accumulator will be a minimizer.
For  =a gm with an Multiplier(*) as operator the accumulator will be a maximizer::
      
   #get an instance of the optimizer / inference-algorithm
   inf=opengm.inferenceAlgorithm(gm=gm,alg='bp',parameter=param)

If no parameter is choosen a parameter with construced with the default parameters will be passed to 
the inference object::    

   #get an instance of the optimizer / inference-algorithm
   inf=opengm.inferenceAlgorithm(gm=gm,alg='bp')
   
Start inference::

   inf.infer()
   
   
Start inference (in this case verbose infernce):: 

   inf.infer(verbose=True)
   
Start inference (in this case verbose infernce where each 10. step is printed)::

   inf.infer(verbose=True,printNth=10)
   
Start inference (print verbose information in new lines)::

   inf.infer(verbose=True,printNth=10,multiline=True)
   
Start inference (print verbose information in the same line ,more compact verbose information)::

   inf.infer(verbose=True,printNth=10,multiline=False)
   
   
Get the inference result ::   
   
   # get the result states
   argmin=inf.arg()


Belief propagation (Bp)
+++++++++++++++++++++++
Assuming a graphical model with the name gm exists,the following code will
minimize a gm with Belief Propagation::
   
   #get an instance of the inference algorithm's parameter object
   param=opengm.inferenceParameter(self.gm,alg='bp',accumulator='minimizer')
   # set up the paraemters
   param.set(steps=100,damping=0.5,convergenceBound=0.00000001)
   #get an instance of the optimizer / inference-algorithm
   inf=opengm.inferenceAlgorithm(gm=gm,alg='bp',parameter=param,accumulator='minimizer')
   # start inference (in this case verbose infernce)
   inf.infer(verbose=true)
   # get the result states
   argmin=inf.arg()


Treereweighted Belief propagation (Trbp)
+++++++++++++++++++++++++++++++++++++++++
.. note::

   TODO

ICM
+++++++++++++++++++
.. note::

   TODO
Gibbs
+++++++++++++++++++
.. note::

   TODO
AStar
+++++++++++++++++++
.. note::

   TODO
LOC
+++++++++++++++++++
.. note::

   TODO
Lazy Flipper
+++++++++++++++++++
.. note::

   TODO
Graph Cut
+++++++++++++++++++
.. note::

   TODO
Alpha Beta Swap
+++++++++++++++++++
.. note::

   TODO
Alpha Expansion
+++++++++++++++++++
.. note::

   TODO
CPlex
+++++++++++++++++++
.. note::

   TODO
