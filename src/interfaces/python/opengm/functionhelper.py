import numpy

def pottsFunction(shape,equal=0.0,nonequal=1.0,dtype=numpy.float32):
   """Factory function to construct a numpy array which encodes a potts-function.

   Keyword arguments:
   
   shape -- shape / number of of labels of the potts-function 
   
   equal  -- value if labels are equal (defualt :) 

   nonequal -- value if labels are not equal

   dtype -- data type of the numpy array
   
   get a potts-function ::
   
      f=opengm.pottsFunction([2,2],equal=0.0,nonequal=1.0)
      
   """
   f=numpy.empty(shape,dtype=dtype)
   f.fill(nonequal)
   numpy.fill_diagonal(f, equal)
   return f

def relabeledPottsFunction(shape,relabelings,equal=0.0,nonequal=1.0,dtype=numpy.float32):
   """Factory function to construct a numpy array which encodes a potts-function.
   The labelings on which the potts function is computed are given by relabelings
   Keyword arguments:
   
   shape -- shape / number of of labels of the potts-function 

   relabelings -- a list of relabelings for the 2 variables

   equal  -- value if labels are equal (defualt :) 

   nonequal -- value if labels are not equal

   dtype -- data type of the numpy array
   
   get a potts-function ::
   
      f=opengm.pottsFunction([4,3],[[0,2,3,5],[4,2,5,3]],equal=0.0,nonequal=1.0)
      
   """
   assert len(shape)==2
   assert len(relabelings)==2
   assert len(relabelings[0])==shape[0]
   assert len(relabelings[1])==shape[1]
   f=numpy.empty(shape,dtype=dtype)
   f[:]=nonequal
   rl1=relabelings[0]
   rl2=relabelings[1]
   for x in range(shape[0]):
      for y in range(shape[1]):
         if(rl1[x]==rl2[y]):
            f[x,y]=equal

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
   assert len(relabelings)==2
   assert len(relabelings[0])==shape[0]
   assert len(relabelings[1])==shape[1]
   f=numpy.empty(shape,dtype=dtype)
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

def relabeledDifferenceFunction(shape,relabelings,norm=2,weight=1.0,truncate=None,dtype=numpy.float32):
   """Factory function to construct a numpy array which encodes a difference-function.
   The difference can be of any norm (1,2,...) and can be truncated or untruncated.
   The labelings on which the potts function is computed are given by relabelings
   Keyword arguments:
   Keyword arguments:
   
   shape -- shape / number of of labels of the potts-function 
   
   weight  -- weight which is multiplied to the norm

   truncate -- truncate all values where the norm is bigger than truncate

   dtype -- data type of the numpy array
   
   get a truncated squared difference function ::
   
      f=opengm.differenceFunction([2,4],[[1,2],[2,3,4,5]],weight=0.5,truncate=5)  
   """
   assert len(shape)==2
   f=numpy.empty(shape,dtype=dtype)
   if shape[0]<shape[1]:
      rl1=relabelings[0]
      yVal=numpy.array(relabelings[1])
      for x in range(shape[0]):
         f[x,:]=(numpy.abs(rl1[x]-yVal)**norm)
   else :
      rl2=relabelings[1]
      xVal=numpy.array(relabelings[2])
      for y in range(shape[1]):
         f[:,y]=(numpy.abs(xVal-rl2[y])**norm)
   if truncate!=None:
      f[numpy.where(f>truncate)]=truncate      
   f*=weight
   return f   