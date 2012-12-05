#from opengmcore import _opengmcore.adder as adder
import opengmcore._opengmcore.adder as adder2
from opengmcore import *
#from opengmcore import *
#import version 
from __version__ import version

import inference
import hdf5
import sys
import types
import numpy

configuration=OpengmConfiguration()


def graphicalModel(numberOfLabels,operator='adder'):
   """Factory function to construct a graphical model.

   Keyword arguments:
   
   numberOfLabels -- number of label sequence (can be a list or  a 1d numpy.ndarray)
   
   operator -- operator of the graphical model. Can be 'adder' or 'multiplier' (default: 'adder')
   
   Construct a gm with ``\'adder\'`` as operator::
   
      gm=opengm.graphicalModel([2,2,2,2,2],operator='adder')
      # or just
      gm=opengm.graphicalModel([2,2,2,2,2])
      
   Construct a gm with ``\'multiplier\'`` as operator::  
   
      gm=opengm.graphicalModel([2,2,2,2,2],operator='multiplier')
      
   """
   if operator=='adder' :
      return adder.GraphicalModel(numberOfLabels)
   elif operator=='multiplier' :
      return multiplier.GraphicalModel(numberOfLabels)
   else:
      raise NameError('operator must be \'adder\' or \'multiplier\'') 


def inferenceParameter(gm ,alg, accumulator=None):
   """Factory function to construct a parameter object for an inference object.

   Keyword arguments:
   
   gm -- a graphical model
   
   alg -- the inference algorithm ('bp','trbp','icm','astar','lf','a-expansion','ab-swap','graphcut',Lazyflipper',...)
   
   accumulator -- accumulator ('minimizer','maximizer') (default: depends on gm operator)
      If the operator of the gm is ``\'adder\'`` the default
      accumulator is ``\'minimizer\'`` .
      If the operator of the gm is ``\'multiplier\'`` the default
      accumulator is ``\'maximizer\'`` .

   Construct a parameter object::
   
      #TODO
      
   """
   alg=alg.lower()
   #evaluate accumulator
   operator=gm.operator
   if accumulator is None :
      if operator=='adder' :
         accumulator='minimizer'
      elif operator=='multiplier':
         accumulator='maximizer'
      else :
         print "selected operator : ", operator
         raise NameError('operator must be \'adder\' or \'multiplier\'') 
   else:
      accumulator=accumulator.lower()
      if accumulator == 'min' or accumulator =='minimizer':
         accumulator='minimizer'
      elif accumulator == 'max' or accumulator=='maximizer':
         accumulator='maximizer'
      else :
         print "selected accumulator: ", accumulator
         raise NameError('accumulator must be \'minimizer\' or \'maximizer\' (or just \'min\' or \'max\' )')   
   #evaluate inference algorithm      
   name = 'inference.'+operator+'.'+accumulator
   #dirty hack!
   if alg=='bp' or alg=='beliefpropagation':
      return eval(name).BpParameter()
   elif alg=='trbp' or alg=='tr-beliefpropagation':
      return eval(name).TrBpParameter()
   elif alg=='icm':
      return eval(name).IcmParameter()
   elif alg=='gibbs':
      return eval(name).GibbsParameter()
   elif alg=='astar' or alg=='a-star':
      return eval(name).AStarParameter()
   elif alg=='loc':
      return eval(name).LOCParameter()
   elif alg=='lf' or alg=="lazyflipper" or alg=="lazy-flipper":
      return eval(name).LazyFlipperParameter()
   elif alg=='gc' or alg=='graphcut' or alg=='graph-cut':
      return eval(name).GraphCutBoostKolmogorovParameter()
   elif alg=='abs' or alg=='ab-swap' or alg=="alphabetaswap" or alg=="alpha-beta-swap" or alg=="alphabeta-swap":
      return eval(name).AlphaBetaSwapBoostKolmogorovParameter()
   elif alg=='ae' or alg=='a-expansion' or alg=="alphaexpansion" or alg=="alpha-expansion" :
      return eval(name).AlphaExpansionBoostKolmogorovParameter()
   elif configuration.withLibdai :
      if alg=='libdai-bp':
         return eval(name).LibDaiBpParameter()
      elif alg=='libdai-trbp' :
         return eval(name).LibDaiTrBpParameter()
      elif alg=='libdai-fbp' or alg=='libdai-fractional-bp':
         return eval(name).LibDaiFractionalBpParameter()
      elif alg=='libdai-jt' or alg=='libdai-junction-tree':
         return eval(name).LibDaiJunctionTreeParameter()
      elif alg=='libdai-dlgbp' or alg=='libdai-double-loop-generalized-bp':
         return eval(name).LibDaiDoubleLoopGbpParameter()
   else:
      raise NameError( 'alg \'' + alg + '\' is unknown') 

         
def inferenceAlgorithm(gm ,alg, accumulator=None,parameter=None,dtype=numpy.float32):
   """Factory function to construct an inference object.

   Keyword arguments:
   
   gm -- a graphical model
   
   alg -- the inference algorithm ('bp','trbp','icm','astar','lf','a-expansion','ab-swap','graphcut',Lazyflipper',...)
   
   accumulator -- accumulator ('minimizer','maximizer') (default: depends on gm operator)
      If the operator of the gm is ``\'adder\'`` the default
      accumulator is ``\'minimizer\'`` .
      If the operator of the gm is ``\'multiplier\'`` the default
      accumulator is ``\'maximizer\'`` .
   parameter -- parameter object of the inference object
      Construct a parameter object with ``inferenceParameter(gm ,alg, accumulator=None)``
   Construct an inference object::
   
      #TODO
      
   """
   alg=alg.lower()
   #evaluate accumulator
   operator=gm.operator
   if accumulator is None :
      if operator=='adder' :
         accumulator='minimizer'
      elif operator=='multiplier':
         accumulator='maximizer'
      else :
         print "selected operator : ", operator
         raise NameError('operator must be \'adder\' or \'multiplier\'') 
   else:
      accumulator=accumulator.lower()
      if accumulator == 'min' or accumulator == 'minimizer':
         accumulator='minimizer'
      elif accumulator == 'max' or accumulator == 'maximizer':
         accumulator='maximizer'
      else :
         print "selected accumulator: ", accumulator
         raise NameError('operator must be \'minimizer\' or \'maximizer\' (or just \'min\' or \'max\' )')   
   #evaluate inference algorithm      
   name = 'inference.'+operator+'.'+accumulator
   #dirty hack!
   if alg=='bp' or alg=='beliefpropagation':
      if parameter == None: return eval(name).Bp(gm)
      else: return eval(name).Bp(gm,parameter)
   if alg=='trwbp' or alg=='trbp':
      if parameter is None: return eval(name).TrBp(gm)
      else: return eval(name).TrBp(gm,parameter)
   elif alg=='icm':
      if parameter is None:return eval(name).Icm(gm)
      else: return eval(name).Icm(gm,parameter)
   elif alg=='gibbs':
      if parameter is None:return eval(name).Gibbs(gm)
      else: return eval(name).Gibbs(gm,parameter)     
   elif alg=='astar' or alg=='a-star':
      if parameter is None:return eval(name).AStar(gm)
      else: return eval(name).AStar(gm,parameter) 
   elif alg=='loc':
      if parameter is None:return eval(name).LOC(gm)
      else: return eval(name).LOC(gm,parameter) 
   elif alg=='lf' or alg=="lazyflipper" or alg=="lazy-flipper":
      if parameter is None:return eval(name).LazyFlipper(gm)
      else: return eval(name).LazyFlipper(gm,parameter) 
   elif alg=='gc' or alg=='graphcut' or alg=='graph-cut':
      if parameter is None:return eval(name).GraphCutBoostKolmogorov(gm)
      else: return eval(name).GraphCutBoostKolmogorov(gm,parameter) 
   elif alg=='abs' or alg=='ab-swap' or alg=="alphabetaswap" or alg=="alpha-beta-swap" or alg=="alphabeta-swap":
      if parameter is None:return eval(name).AlphaBetaSwapBoostKolmogorov(gm)
      else: return eval(name).AlphaBetaSwapBoostKolmogorov(gm,parameter) 
   elif alg=='ae' or alg=='a-expansion' or alg=="alphaexpansion" or alg=="alpha-expansion" :
      if parameter is None:return eval(name).AlphaExpansionBoostKolmogorov(gm)
      else: return eval(name).AlphaExpansionBoostKolmogorov(gm,parameter) 
   elif configuration.withLibdai :
      if alg=='libdai-bp' or alg=='libdai-beliefpropagation':
         if parameter is None:return eval(name).LibDaiBp(gm)
         else: return eval(name).LibDaiBp(gm,parameter) 
      elif alg=='libdai-trbp' :
         if parameter is None:return eval(name).LibDaiTrBp(gm)
         else: return eval(name).LibDaiTrBp(gm,parameter) 
      elif alg=='libdai-fbp' or alg=='libdai-fractional-bp':
         if parameter is None:return eval(name).LibDaiFractionalBp(gm)
         else: return eval(name).LibDaiFractionalBp(gm,parameter) 
      elif alg=='libdai-jt' or alg=='libdai-junction-tree':
         if parameter is None:return eval(name).LibDaiJunctionTree(gm)
         else: return eval(name).LibDaiJunctionTree(gm,parameter) 
      elif alg=='libdai-dlgbp' or alg=='libdai-double-loop-generalized-bp':
         if parameter is None:return eval(name).LibDaiDoubleLoopGbp(gm)
         else: return eval(name).LibDaiDoubleLoopGbp(gm,parameter) 
   else:
      raise NameError( 'alg \'' + alg + '\' is unknown') 
      

      
def pottsFunction(shape,diag=0.0,nondiag=1.0,dtype=numpy.float32):
   """Factory function to construct a numpy array which encodes a potts-function.

   Keyword arguments:
   
   shape -- shape / number of of labels of the potts-function 
   
   diag  -- value if labels are equal (defualt :) 

   nondiag -- value if labels are not equal

   dtype -- data type of the numpy array
   
   get a potts-function ::
   
      f=opengm.pottsFunction([2,2],diag=0.0,nondiag=1.0)
      
   """
   f=numpy.ones(shape,dtype=numpy.float32)
   f.fill(nondiag)
   numpy.fill_diagonal(f, diag)
   return f

def differenceFunction(shape,norm=2,weight=1.0,truncate=None,dtype=numpy.float32):
   """Factory function to construct a numpy array which encodes a difference-function.
   The difference can be of any norm (1,2,...) and can be truncated or untruncated.

   Keyword arguments:
   
   shape -- shape / number of of labels of the potts-function 
   
   weight  -- weight which is multiplied to the norm

   truncate -- truncate all values where the norm is bigger than truncate

   dtype -- data type of the numpy array
   
   get a truncated squared difference function ::
   
      f=opengm.differenceFunction([2,4],weight=0.5,truncate=5)
      
   """
   assert len(shape)==2
   f=numpy.ones(shape,dtype=numpy.float32)
   if shape[0]<shape[1]:
      yVal=numpy.arange(0,shape[1])
      for x in range(shape[0]):
         f[x,:]=(numpy.abs(x-yVal)**norm)
   else :
      xVal=numpy.arange(0,shape[0])
      for y in range(shape[1]):
         f[:,y]=(numpy.abs(xVal-y)**norm)
   if truncate!=None:
      f[numpy.where(f>truncate)]=truncate      
   f*=weight
   return f